import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# =============================================================================
# CONFIGURATIONS
# =============================================================================
MODEL_WEIGHTS_PATH = "efficientnet_v2m_stanford_dogs_best_93.pth"
TRAIN_DATA_DIR = "/home/mimo/projects/computer_vision/project1/data/train"
TEST_DATA_DIR = "/home/mimo/projects/computer_vision/project1/data/test"
IMAGE_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# LOAD ARCHITECTURE & WEIGHTS
# =============================================================================
def load_trained_model(weights_path, num_classes):
    model = models.efficientnet_v2_m(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    state_dict = torch.load(weights_path, map_location=DEVICE)
    clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model

# =============================================================================
# THE 5-WAY GEOMETRIC TTA GENERATOR
# =============================================================================
def generate_5_tta_variants(image_tensor):
    """
    Outputs 5 geometric look-angles per image.
    Color, brightness, and contrast remain 100% untouched.
    """
    hf = transforms.RandomHorizontalFlip(p=1.0)
    
    scale_105 = transforms.Compose([transforms.Resize((int(IMAGE_SIZE*1.05), int(IMAGE_SIZE*1.05))), transforms.CenterCrop(IMAGE_SIZE)])
    scale_115 = transforms.Compose([transforms.Resize((int(IMAGE_SIZE*1.15), int(IMAGE_SIZE*1.15))), transforms.CenterCrop(IMAGE_SIZE)])

    v1  = image_tensor                                # 1. Original Clean Image
    v2  = hf(image_tensor)                            # 2. Horizontal Flip
    v3  = scale_105(image_tensor)                     # 3. 5% Zoom
    v4  = hf(v3)                                      # 4. 5% Zoom + Flip
    v5  = scale_115(image_tensor)                     # 5. 15% Zoom

    return torch.stack([v1, v2, v3, v4, v5])

# =============================================================================
# 5-WAY TTA DATASET EVALUATOR
# =============================================================================
def evaluate_dataset_with_5_way_tta(model, test_loader):
    print("🧪 Running 5-Way Pure Geometric TTA Evaluation Loop...")
    criterion = nn.CrossEntropyLoss()
    test_loss, correct_test, total_test = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            batch_logits = []
            for i in range(images.size(0)):
                single_img = images[i]
                tta_variants = generate_5_tta_variants(single_img) # Shape: [5, 3, 384, 384]
                
                outputs = model(tta_variants) # Shape: [5, 120]
                mean_logits = outputs.mean(dim=0) # Average the 5 angles -> Shape: [120]
                batch_logits.append(mean_logits)
            
            final_outputs = torch.stack(batch_logits)
            
            loss = criterion(final_outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            _, predicted = final_outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()

    final_loss = test_loss / total_test
    final_accuracy = (correct_test / total_test) * 100
    
    print("\n" + "="*50)
    print("📊 5-WAY GEOMETRIC TTA METRICS")
    print("="*50)
    print(f"📂 Target Directory   : {TEST_DATA_DIR}")
    print(f"📉 5-Way Blended Loss: {final_loss:.4f}")
    print(f"🏆 Final TTA Accuracy : {final_accuracy:.2f}%")
    print("="*50 + "\n")

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"❌ Error: Train dataset path not found.")
        exit()
        
    train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR)
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    model = load_trained_model(MODEL_WEIGHTS_PATH, num_classes)
    
    if os.path.exists(TEST_DATA_DIR):
        test_dataset = datasets.ImageFolder(root=TEST_DATA_DIR, transform=base_transforms)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        evaluate_dataset_with_5_way_tta(model, test_loader)
    else:
        print(f"❌ Error: Test directory path not found.")