import os
import time
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
import torch.distributed as dist
from tqdm import tqdm

# =============================================================================
# PART 1: DISTRIBUTED DATA PARALLEL (DDP) NETWORK LAYER
# =============================================================================
def setup_ddp_file(rank, world_size, sync_file):
    """Connects both GPUs together using a shared local file system."""
    sync_uri = f"file://{sync_file}"
    dist.init_process_group(
        backend="nccl", 
        init_method=sync_uri, 
        rank=rank, 
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Safely closes the distributed communication paths."""
    dist.destroy_process_group()

# =============================================================================
# PART 2: DATA EXTRACTION & DYNAMIC AUGMENTATION ENGINE
# =============================================================================
def get_dataloaders(rank, world_size, data_dir, batch_size, image_size, num_workers):
    """Reads, augments, and splits the images evenly between GPUs using custom worker counts."""
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir)
    train_size = int(0.90 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    sub_train, sub_val = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    sub_train.dataset.transform = train_transforms
    sub_val.dataset.transform = val_transforms
    
    train_sampler = DistributedSampler(sub_train, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(sub_val, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        sub_train, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers,       
        pin_memory=True,             
        prefetch_factor=2,           
        persistent_workers=True      
    )
    
    val_loader = DataLoader(
        sub_val, 
        batch_size=batch_size, 
        sampler=val_sampler, 
        num_workers=num_workers, 
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader, val_loader, train_sampler, len(full_dataset.classes)

# =============================================================================
# PART 3: MODEL CONFIGURATION (EfficientNetV2-M Blueprint)
# =============================================================================
def get_model(rank, num_classes):
    """Downloads EfficientNetV2-M weights safely and wraps inside DDP."""
    weights = models.EfficientNet_V2_M_Weights.DEFAULT
    
    if rank == 0:
        model = models.efficientnet_v2_m(weights=weights)
        
    dist.barrier() 
    
    if rank != 0:
        model = models.efficientnet_v2_m(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    return model

# =============================================================================
# PART 4: MAIN PIPELINE RUNTIME
# =============================================================================
def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Hyperparameters & Infrastructure Configurations
    BATCH_SIZE = 8         
    IMAGE_SIZE = 384        
    LEARNING_RATE = 0.000001
    NUM_WORKERS = 8         # Number of CPU threads per GPU
    EPOCHS = 25            
    SYNC_FILE = "/home/mimo/projects/computer_vision/project1/shared_sync_file"
    TRAIN_DATA_DIR = "./data/train"
    
    # Automatically capture the actual computer network name
    pc_name = socket.gethostname()

    setup_ddp_file(rank, world_size, SYNC_FILE)
    
    if rank == 0:
        print("="*70)
        print(f"🖥️  WORKSTATION HOSTNAME : {pc_name}")
        print(f"🚀 ENGINE               : EFFICIENTNETV2-M DDP CLUSTER")
        print(f"📐 IMAGE RESOLUTION     : {IMAGE_SIZE}x{IMAGE_SIZE}")
        print(f"👥 TOTAL CPU WORKERS    : {NUM_WORKERS * world_size} ({NUM_WORKERS} per GPU)")
        print(f"⚙️  BATCH CONFIGURATION  : {BATCH_SIZE} per GPU | Total: {BATCH_SIZE * world_size}")
        print(f"📉 INITIAL LEARNING RATE: {LEARNING_RATE}")
        print("="*70)

    train_loader, val_loader, train_sampler, num_classes = get_dataloaders(
        rank, world_size, TRAIN_DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS
    )
    model = get_model(rank, num_classes)
    
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0

    # -------------------------------------------------------------------------
    # CORE TRAINING LOOP
    # -------------------------------------------------------------------------
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        
        # --- TRAINING MODE ---
        model.train()
        start_time = time.time()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        # Extract the current learning rate from the optimizer configuration
        current_lr = optimizer.param_groups[0]['lr']
        
        # Initialize tqdm slide bar only on GPU 0 to prevent printing multiple loops
        if rank == 0:
            pbar = tqdm(
                enumerate(train_loader), 
                total=len(train_loader),
                desc=f"Epoch [{epoch+1}/{EPOCHS}] Train",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}"
            )
            datasource = pbar
        else:
            datasource = enumerate(train_loader)
            
        for batch_idx, (images, labels) in datasource:
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
            
            # Live updates for the progress bar metrics
            if rank == 0:
                current_batch_loss = loss.item()
                pbar.set_postfix({
                    "Batch": f"{batch_idx+1}/{len(train_loader)}",
                    "Loss": f"{current_batch_loss:.4f}",
                    "LR": f"{current_lr:.6f}"
                })
                
        # --- VALIDATION MODE ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        if rank == 0:
            val_pbar = tqdm(
                enumerate(val_loader), 
                total=len(val_loader),
                desc=f"Epoch [{epoch+1}/{EPOCHS}] Val  ",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
                leave=False # Disappears after validation completes to keep terminal clean
            )
            val_datasource = val_pbar
        else:
            val_datasource = enumerate(val_loader)
            
        with torch.no_grad(): 
            for batch_idx, (images, labels) in val_datasource:
                images, labels = images.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    v_loss = criterion(outputs, labels)
                
                val_loss += v_loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
                if rank == 0:
                    val_pbar.set_postfix({"Batch": f"{batch_idx+1}/{len(val_loader)}"})

        # Collect data metrics from both parallel execution environments
        metrics = torch.tensor([running_loss, correct_train, total_train, val_loss, correct_val, total_val]).to(rank)
        dist.reduce(metrics, dst=0, op=dist.ReduceOp.SUM) 
        
        # --- CONSOLE METRICS ANALYSIS SUMMARY (Rank 0 Only) ---
        if rank == 0:
            e_loss = metrics[0].item() / metrics[2].item()
            e_acc = (metrics[1].item() / metrics[2].item()) * 100
            ev_loss = metrics[3].item() / metrics[5].item()
            ev_acc = (metrics[4].item() / metrics[5].item()) * 100
            
            elapsed = time.time() - start_time
            
            print(f"\n📊 Summary for Epoch {epoch+1} on [{pc_name}] (Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}, Workers: {NUM_WORKERS*world_size})")
            print(f"    ↳ Time: {elapsed:.1f}s | Current LR: {current_lr:.6f}")
            print(f"    ↳ Training Loss:   {e_loss:.4f} | Training Accuracy:   {e_acc:.2f}%")
            print(f"    ↳ Validation Loss: {ev_loss:.4f} | Validation Accuracy: {ev_acc:.2f}%")
            
            if ev_acc > best_val_acc:
                best_val_acc = ev_acc
                torch.save(model.module.state_dict(), "efficientnet_v2m_stanford_dogs_best.pth")
                print(f"    🌟 Benchmark Breakthrough! Saving State: efficientnet_v2m_stanford_dogs_best.pth (Score: {best_val_acc:.2f}%)")
            
            print("-" * 70)
            
    cleanup_ddp()

if __name__ == "__main__":
    main()