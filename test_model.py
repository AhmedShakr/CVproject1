import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def test_model():
    print("="*50)
    print("🔍 INITIATING FINAL MODEL EVALUATION")
    print("="*50)

    # 1. إعدادات الصور (نفس إعدادات التحقق)
    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. تحميل بيانات الاختبار
    TEST_DATA_DIR = "./data/test" # تأكد من وجود مجلد الاختبار
    test_dataset = datasets.ImageFolder(root=TEST_DATA_DIR, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. بناء الموديل وتحميل الأوزان التي دربناها بالـ DDP
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = len(test_dataset.classes)
    
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # تحميل الأوزان الصافية
    model.load_state_dict(torch.load("stanford_dogs_ddp_resnet50.pth", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 4. بدء الاختبار
    correct = 0
    total = 0
    
    print(f"[*] Streaming {len(test_dataset)} testing images...\n")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    final_accuracy = (correct / total) * 100
    print("="*50)
    print(f"🏆 TRUE UNBIASED TEST ACCURACY: {final_accuracy:.2f}%")
    print("="*50)

if __name__ == "__main__":
    test_model()