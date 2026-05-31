import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
import torch.distributed as dist

def setup_ddp_file(rank, world_size):
    # مسار الملف المشترك لربط الكروت بعيداً عن جدار الحماية (File-System DDP)
    sync_file = "file:///home/mimo/projects/computer_vision/project1/shared_sync_file"
    dist.init_process_group(
        backend="nccl", 
        init_method=sync_file, 
        rank=rank, 
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    setup_ddp_file(rank, world_size)
    
    # 1. إعدادات صارمة للسرعة القصوى
    BATCH_SIZE = 512      # 128 صورة لكل كارت (الإجمالي 256)
    IMAGE_SIZE = 224
    LEARNING_RATE = 0.0005

    EPOCHS = 5
    
    if rank == 0:
        print("="*60)
        print(f"🚀 INITIALIZING FILE-SYSTEM DDP WITH VALIDATION (World Size: {world_size})")
        print("="*60)

    # 2. خطوط معالجة الصور
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. إعداد الداتا (قراءة من الهارد مع تقسيم الداتا بين الكارتين)
    TRAIN_DATA_DIR = "./data/train"
    full_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    sub_train, sub_val = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    sub_train.dataset.transform = train_transforms
    sub_val.dataset.transform = val_transforms
    num_classes = len(full_dataset.classes)

    # 4. الموزعات الآلية (Samplers) - استخدام num_replicas للعمل بدون أخطاء
    train_sampler = DistributedSampler(sub_train, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(sub_val, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(sub_train, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(sub_val, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0, pin_memory=True)

    # 5. الموديل والتغليف الذكي
    resnet_weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=resnet_weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    # 6. دورة التدريب والتحقق الاحترافية
    for epoch in range(EPOCHS):
        # ضروري جداً لضمان تغيير ترتيب الصور في كل دورة
        train_sampler.set_epoch(epoch)
        
        # ==========================================
        # أ. مرحلة التدريب (Training Phase)
        # ==========================================
        model.train()
        start_time = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)
            
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
            
        # ==========================================
        # ب. مرحلة التحقق (Validation Phase)
        # ==========================================
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(rank, non_blocking=True)
                val_labels = val_labels.to(rank, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    val_outputs = model(val_images)
                    v_loss = criterion(val_outputs, val_labels)
                    
                val_loss += v_loss.item() * val_images.size(0)
                _, val_predicted = val_outputs.max(1)
                total_val += val_labels.size(0)
                correct_val += val_predicted.eq(val_labels).sum().item()

        # ==========================================
        # ج. مزامنة وتجميع الأرقام من الكارتين للطباعة
        # ==========================================
        
        # 1. تجميع أرقام التدريب
        loss_tensor = torch.tensor(running_loss).to(rank)
        correct_tensor = torch.tensor(correct_train).to(rank)
        total_tensor = torch.tensor(total_train).to(rank)
        
        dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_tensor, dst=0, op=dist.ReduceOp.SUM)
        
        # 2. تجميع أرقام التحقق
        v_loss_tensor = torch.tensor(val_loss).to(rank)
        v_correct_tensor = torch.tensor(correct_val).to(rank)
        v_total_tensor = torch.tensor(total_val).to(rank)
        
        dist.reduce(v_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(v_correct_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(v_total_tensor, dst=0, op=dist.ReduceOp.SUM)
        
        # الطباعة والحفظ من الكارت الرئيسي (Rank 0) فقط لمنع التكرار
        if rank == 0:
            epoch_loss = loss_tensor.item() / total_tensor.item()
            epoch_acc = (correct_tensor.item() / total_tensor.item()) * 100
            
            epoch_v_loss = v_loss_tensor.item() / v_total_tensor.item()
            epoch_v_acc = (v_correct_tensor.item() / v_total_tensor.item()) * 100
            
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1}/{EPOCHS}] -> Time: {elapsed:.1f}s")
            print(f"    ↳ Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
            print(f"    ↳ Val Loss:   {epoch_v_loss:.4f} | Val Acc:   {epoch_v_acc:.2f}%")
            print("-" * 60)
            
            # حفظ الأوزان الصافية في النهاية
            if epoch == EPOCHS - 1:
                torch.save(model.module.state_dict(), "stanford_dogs_ddp_resnet50.pth")
                print("💾 Model weights saved successfully!")

    cleanup_ddp()

if __name__ == "__main__":
    main()