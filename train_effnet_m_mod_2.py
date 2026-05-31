import os
import time
import math
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
    """Connects both GPUs using a shared local file."""
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{sync_file}",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# =============================================================================
# PART 2: NUMA AFFINITY — pin each rank to its own CPU socket
# =============================================================================
def pin_to_numa(rank):
    """
    Dell 7910 has two Xeon 2680v4 sockets:
      Socket 0 → cores  0-13  (rank 0 / GPU 0)
      Socket 1 → cores 14-27  (rank 1 / GPU 1)
    Pinning stops DataLoader workers from crossing the QPI link,
    which would otherwise add latency on every single batch load.
    """
    if rank == 0:
        os.sched_setaffinity(0, range(0, 14))
    else:
        os.sched_setaffinity(0, range(14, 28))

# =============================================================================
# PART 3: SUBSET WRAPPER — independent transform per split (bug fix from v2)
# =============================================================================
class TransformSubset(torch.utils.data.Dataset):
    """
    Gives each data split its own transform without touching the shared
    parent ImageFolder object. The original script wrote to sub.dataset.transform
    which overwrote the same object for both train and val.
    """
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# =============================================================================
# PART 4: DATA LOADING
# =============================================================================
def get_dataloaders(rank, world_size, data_dir, batch_size, image_size, num_workers):

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir)
    train_size   = int(0.90 * len(full_dataset))
    val_size     = len(full_dataset) - train_size

    sub_train_raw, sub_val_raw = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    sub_train = TransformSubset(sub_train_raw, train_transforms)
    sub_val   = TransformSubset(sub_val_raw,   val_transforms)

    train_sampler = DistributedSampler(sub_train, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(sub_val,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        sub_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True
    )

    # FIX 6: val batch is much larger — no gradients means much more VRAM free
    val_loader = DataLoader(
        sub_val,
        batch_size=batch_size * 2,   # 64 per GPU (32 * 2)
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True
    )

    return train_loader, val_loader, train_sampler, len(full_dataset.classes)

# =============================================================================
# PART 5: MODEL
# =============================================================================
def get_model(rank, num_classes):
    weights = models.EfficientNet_V2_M_Weights.DEFAULT

    if rank == 0:
        model = models.efficientnet_v2_m(weights=weights)
    dist.barrier()
    if rank != 0:
        model = models.efficientnet_v2_m(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    model = model.to(rank)

    # FIX 7 + 8: gradient_as_bucket_view eliminates an extra memory copy per bucket
    # static_graph=True is safe because EfficientNet's graph never changes
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        static_graph=True,
        gradient_as_bucket_view=True   # eliminates redundant gradient memory copy
    )

    return model

# =============================================================================
# PART 6: MAIN PIPELINE
# =============================================================================
def main():
    rank       = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # ── Hyperparameters ──────────────────────────────────────────────────────
    # FIX 1: batch size raised from 8 to 32
    # 24 GB VRAM + 384px + AMP FP16 comfortably fits 32 per GPU
    # 4× more gradient signal per step = significantly better GPU utilisation
    BATCH_SIZE    = 32

    IMAGE_SIZE    = 384

    # FIX 4: workers raised from 8 to 12
    # Dual Xeon 2680v4 = 56 total cores. 12 workers per GPU = 24 total,
    # leaving 32 cores for the OS, main process, and GPU kernel launches
    NUM_WORKERS   = 12

    LEARNING_RATE = 1e-4
    EPOCHS        = 30
    WARMUP_EPOCHS = 3
    WEIGHT_DECAY  = 0.05
    LABEL_SMOOTH  = 0.1
    GRAD_CLIP     = 1.0
    SYNC_FILE     = "/home/mimo/projects/computer_vision/project1/shared_sync_file"
    TRAIN_DATA_DIR = "./data/train"

    pc_name = socket.gethostname()

    # FIX 5: NUMA affinity — pin before any data loading starts
    pin_to_numa(rank)

    # FIX 2: cuDNN benchmark — auto-tunes convolution kernels for 384×384 inputs
    # Safe because our input size never changes during training
    torch.backends.cudnn.benchmark = True

    # FIX 3: TF32 matmul — RTX 3090 (Ampere) supports TF32 which is ~8× faster
    # than FP32 for matmul with negligible accuracy difference
    torch.set_float32_matmul_precision('high')

    setup_ddp_file(rank, world_size, SYNC_FILE)

    if rank == 0:
        print("=" * 70)
        print(f"🖥️  WORKSTATION HOSTNAME : {pc_name}")
        print(f"🚀 ENGINE               : EFFICIENTNETV2-M DDP CLUSTER v3 (MAX SPEED)")
        print(f"📐 IMAGE RESOLUTION     : {IMAGE_SIZE}x{IMAGE_SIZE}")
        print(f"👥 TOTAL CPU WORKERS    : {NUM_WORKERS * world_size} ({NUM_WORKERS} per GPU)")
        print(f"⚙️  BATCH CONFIG         : {BATCH_SIZE} per GPU | Total: {BATCH_SIZE * world_size}")
        print(f"📉 PEAK LR              : {LEARNING_RATE} | Warmup: {WARMUP_EPOCHS} epochs")
        print(f"🎯 LABEL SMOOTHING      : {LABEL_SMOOTH}")
        print(f"✂️  GRADIENT CLIPPING    : {GRAD_CLIP}")
        print(f"⚡ cudnn.benchmark      : ON")
        print(f"⚡ TF32 matmul          : ON")
        print(f"⚡ NUMA affinity        : ON (rank 0→socket0, rank 1→socket1)")
        print(f"⚡ gradient_as_bucket   : ON")
        print("=" * 70)

    train_loader, val_loader, train_sampler, num_classes = get_dataloaders(
        rank, world_size, TRAIN_DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS
    )
    model = get_model(rank, num_classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH).to(rank)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # FIX 8 (Python): math is now imported at top of file, not inside lr_lambda.
    # lr_lambda is called once per training step — importing inside it ran
    # thousands of redundant import calls across 30 epochs.
    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-6 / LEARNING_RATE, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        # ── TRAINING MODE ────────────────────────────────────────────────────
        model.train()
        start_time = time.time()
        running_loss, correct_train, total_train = 0.0, 0, 0

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
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss  += loss.item() * images.size(0)
            _, predicted   = outputs.max(1)
            total_train   += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            if rank == 0:
                pbar.set_postfix({
                    "Batch": f"{batch_idx+1}/{len(train_loader)}",
                    "Loss":  f"{loss.item():.4f}",
                    "LR":    f"{scheduler.get_last_lr()[0]:.7f}"
                })

        # ── VALIDATION MODE ──────────────────────────────────────────────────
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        if rank == 0:
            val_pbar = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Epoch [{epoch+1}/{EPOCHS}] Val  ",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
                leave=False
            )
            val_datasource = val_pbar
        else:
            val_datasource = enumerate(val_loader)

        with torch.no_grad():
            for batch_idx, (images, labels) in val_datasource:
                images = images.to(rank, non_blocking=True)
                labels = labels.to(rank, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    v_loss  = criterion(outputs, labels)

                val_loss      += v_loss.item() * images.size(0)
                _, predicted   = outputs.max(1)
                total_val     += labels.size(0)
                correct_val   += predicted.eq(labels).sum().item()

                if rank == 0:
                    val_pbar.set_postfix({"Batch": f"{batch_idx+1}/{len(val_loader)}"})

        # Aggregate metrics across both GPUs
        metrics = torch.tensor([
            running_loss, correct_train, total_train,
            val_loss,     correct_val,   total_val
        ], dtype=torch.float64).to(rank)
        dist.reduce(metrics, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            e_loss  = metrics[0].item() / metrics[2].item()
            e_acc   = (metrics[1].item() / metrics[2].item()) * 100
            ev_loss = metrics[3].item() / metrics[5].item()
            ev_acc  = (metrics[4].item() / metrics[5].item()) * 100
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]

            print(f"\n📊 Epoch {epoch+1} | [{pc_name}] | {IMAGE_SIZE}px | {NUM_WORKERS*world_size} workers")
            print(f"    ↳ Time: {elapsed:.1f}s | LR: {current_lr:.7f}")
            print(f"    ↳ Train  — Loss: {e_loss:.4f} | Acc: {e_acc:.2f}%")
            print(f"    ↳ Val    — Loss: {ev_loss:.4f} | Acc: {ev_acc:.2f}%")

            if ev_acc > best_val_acc:
                best_val_acc = ev_acc
                torch.save(
                    model.module.state_dict(),
                    "efficientnet_v2m_best_v3.pth"
                )
                print(f"    🌟 New best! Saved efficientnet_v2m_best_v3.pth  (Val Acc: {best_val_acc:.2f}%)")

            print("-" * 70)

    cleanup_ddp()

if __name__ == "__main__":
    main()