import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
import torch.distributed as dist

# =============================================================================
# PART 1: DDP NETWORK SETUP (The Communication Layer)
# =============================================================================
def setup_ddp_file(rank, world_size, sync_file):
    """
    Connects the GPUs using a file-system sync to bypass network firewall issues.
    """
    sync_uri = f"file://{sync_file}"
    dist.init_process_group(
        backend="nccl", 
        init_method=sync_uri, 
        rank=rank, 
        world_size=world_size
    )
    # Lock this specific process to its assigned GPU ID
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Safely destroys the connection to prevent GPU memory lock."""
    dist.destroy_process_group()

# =============================================================================
# PART 2: THE DATA ENGINE (Augmentation & CPU Worker Optimization)
# =============================================================================
def get_dataloaders(rank, world_size, data_dir, batch_size, image_size):
    """
    Reads, heavily augments, and splits the images evenly between GPUs.
    Now utilizes multiple CPU cores to prevent GPU starvation.
    """
    # 1. AGGRESSIVE AUGMENTATION: Forces the model to learn features, not memorize.
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), # Randomly shifts colors, contrast, and rotations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation images MUST remain clean (No augmentation, just resize and normalize)
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Read and split dataset (80% Train, 20% Val)
    full_dataset = datasets.ImageFolder(root=data_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    sub_train, sub_val = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    sub_train.dataset.transform = train_transforms
    sub_val.dataset.transform = val_transforms
    
    # Distributed Samplers (The Traffic Cops that prevent GPUs from reading the same image)
    train_sampler = DistributedSampler(sub_train, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(sub_val, num_replicas=world_size, rank=rank, shuffle=False)

    # -------------------------------------------------------------------------
    # WAKING UP THE XEON CPU (Fixing the 0% GPU Utilization Bottleneck)
    # -------------------------------------------------------------------------
    train_loader = DataLoader(
        sub_train, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=4,               # 8 background CPU threads per GPU fetching images
        pin_memory=True,             # Locks data in RAM for instant GPU transfer
        prefetch_factor=2,           # Each CPU worker prepares 4 batches in advance
        persistent_workers=True      # Keeps CPU workers alive between epochs to save time
    )
    
    val_loader = DataLoader(
        sub_val, 
        batch_size=batch_size, 
        sampler=val_sampler, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader, val_loader, train_sampler, len(full_dataset.classes)

# =============================================================================
# PART 3: MODEL CONSTRUCTION (The Safe Download Barrier Added)
# =============================================================================
def get_model(rank, num_classes):
    """
    Downloads the massive model safely, modifies the head, and wraps it in DDP.
    """
    weights = models.EfficientNet_V2_L_Weights.DEFAULT
    
    # SAFE DOWNLOAD PROTOCOL: GPU 0 downloads first while GPU 1 waits at the barrier.
    if rank == 0:
        model = models.efficientnet_v2_l(weights=weights)
        
    dist.barrier() # Synchronization point: Both GPUs wait here until download is done
    
    # GPU 1 loads the safely downloaded file from the SSD instantly
    if rank != 0:
        model = models.efficientnet_v2_l(weights=weights)
    
    # Modify the classification head for the specific number of classes (120 for Stanford Dogs)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # Wrap the model in DistributedDataParallel
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    return model

# =============================================================================
# PART 4: THE CORE TRAINING ENGINE
# =============================================================================
def main():
    # Automatically get GPU details provided by the 'torchrun' command
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Hyperparameters
    BATCH_SIZE = 16         
    IMAGE_SIZE = 384        
    LEARNING_RATE = 0.00001
    EPOCHS = 15            # Set to 15 to accommodate the aggressive augmentation
    SYNC_FILE = "/home/mimo/projects/computer_vision/project1/shared_sync_file"
    TRAIN_DATA_DIR = "./data/train"

    # Initialize communication
    setup_ddp_file(rank, world_size, SYNC_FILE)
    
    if rank == 0:
        print("="*60)
        print(f"🚀 TRAINING EFFICIENTNETV2-L (AdamW + CPU Unleashed)")
        print("="*60)

    # Prepare Data and Model
    train_loader, val_loader, train_sampler, num_classes = get_dataloaders(rank, world_size, TRAIN_DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    model = get_model(rank, num_classes)
    
    criterion = nn.CrossEntropyLoss().to(rank)
    
    # OPTIMIZER FIX: Using AdamW with Weight Decay to heavily penalize memorization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # Scaler handles "Mixed Precision" (16-bit math) to save VRAM and increase speed
    scaler = torch.amp.GradScaler('cuda')

    # SAVE BEST LOGIC: Track the highest validation accuracy
    best_val_acc = 0.0

    # ================= THE EPOCH LOOP =================
    for epoch in range(EPOCHS):
        # We MUST tell the sampler the epoch number, or it uses the same image order every time
        train_sampler.set_epoch(epoch)
        
        # --- TRAINING PHASE ---
        model.train()
        start_time = time.time()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad(): 
            for images, labels in val_loader:
                images, labels = images.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    v_loss = criterion(outputs, labels)
                
                val_loss += v_loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        # --- GATHER METRICS FROM BOTH GPUS ---
        metrics = torch.tensor([running_loss, correct_train, total_train, val_loss, correct_val, total_val]).to(rank)
        dist.reduce(metrics, dst=0, op=dist.ReduceOp.SUM) 
        
        # --- PRINT RESULTS & SAVE BEST MODEL ---
        # Only GPU 0 prints to avoid duplicate console output
        if rank == 0:
            e_loss = metrics[0].item() / metrics[2].item()
            e_acc = (metrics[1].item() / metrics[2].item()) * 100
            ev_loss = metrics[3].item() / metrics[5].item()
            ev_acc = (metrics[4].item() / metrics[5].item()) * 100
            
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Time: {elapsed:.1f}s")
            print(f"    ↳ Train Loss: {e_loss:.4f} | Train Acc: {e_acc:.2f}%")
            print(f"    ↳ Val Loss:   {ev_loss:.4f} | Val Acc:   {ev_acc:.2f}%")
            
            # ONLY SAVE IF VALIDATION ACCURACY IMPROVED
            if ev_acc > best_val_acc:
                best_val_acc = ev_acc
                torch.save(model.module.state_dict(), "stanford_dogs_effnetV2L_BEST.pth")
                print(f"    🌟 New Best Model Saved! (Score: {best_val_acc:.2f}%)")
            
            print("-" * 60)
            
    # Safely close the GPU connection
    cleanup_ddp()

# =============================================================================
# PART 5: SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    main()