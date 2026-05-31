import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

# =============================================================================
# CONFIGURATIONS
# =============================================================================
MODEL_WEIGHTS_PATH = "efficientnet_v2m_stanford_dogs_best_93.pth"
TRAIN_DATA_DIR = "/home/mimo/projects/computer_vision/project1/data/train"
TEST_DATA_DIR = "/home/mimo/projects/computer_vision/project1/data/test"
IMAGE_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base clean transform to load images into tensor space safely
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
# THE 10-WAY TEST-TIME AUGMENTATION GENERATOR
# =============================================================================
def generate_10_tta_variants(image_tensor):
    """
    Takes a single image tensor [3, 384, 384] and outputs a batch of 10 
    distinct mathematical look-angles and lighting variations.
    """
    # Define TTA transform variations
    hf = transforms.RandomHorizontalFlip(p=1.0)
    
    scale_105 = transforms.Compose([transforms.Resize((int(IMAGE_SIZE*1.05), int(IMAGE_SIZE*1.05))), transforms.CenterCrop(IMAGE_SIZE)])
    scale_115 = transforms.Compose([transforms.Resize((int(IMAGE_SIZE*1.15), int(IMAGE_SIZE*1.15))), transforms.CenterCrop(IMAGE_SIZE)])
    
    bright_up = transforms.ColorJitter(brightness=(1.2, 1.2))
    bright_down = transforms.ColorJitter(brightness=(0.8, 0.8))
    contrast_up = transforms.ColorJitter(contrast=(1.2, 1.2))

    # Construct the 10 paths
    v1  = image_tensor                                # 1. Original Clean Image
    v2  = hf(image_tensor)                            # 2. Horizontal Flip
    v3  = scale_105(image_tensor)                     # 3. Slight Scale (5%)
    v4  = hf(v3)                                      # 4. Slight Scale + Flip
    v5  = scale_115(image_tensor)                     # 5. Deep Scale (15%)
    v6  = hf(v5)                                      # 6. Deep Scale + Flip
    v7  = bright_up(image_tensor)                     # 7. Brightness Shift Up
    v8  = bright_down(image_tensor)                   # 8. Brightness Shift Down
    v9  = contrast_up(image_tensor)                   # 9. Contrast Enhancement
    v10 = hf(bright_up(image_tensor))                 # 10. Brightness Up + Horizontal Flip

    # Stack into a combined batch tensor shape: [10, 3, 384, 384]
    return torch.stack([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])

# =============================================================================
# 10-WAY TTA DATASET EVALUATOR
# =============================================================================
def evaluate_dataset_with_10_way_tta(model, test_loader):
    print("🧪 Running 10-Way TTA Evaluation Loop (Geometric + Photometric)...")
    criterion = nn.CrossEntropyLoss()
    test_loss, correct_test, total_test = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Process each sample inside the dataset loader batch independently
            batch_logits = []
            for i in range(images.size(0)):
                single_img = images[i]
                # Generate 10 variations for this specific image -> [10, 3, 384, 384]
                tta_variants = generate_10_tta_variants(single_img)
                
                # Execute forward pass through EfficientNetV2-M for all 10 views
                outputs = model(tta_variants) # Matrix Shape: [10, 120]
                
                # Compute the soft prediction average across the 10 look-angles
                mean_logits = outputs.mean(dim=0) # Shape: [120]
                batch_logits.append(mean_logits)
            
            # Stack back into a batch structure
            final_outputs = torch.stack(batch_logits)
            
            loss = criterion(final_outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            _, predicted = final_outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()

    final_loss = test_loss / total_test
    final_accuracy = (correct_test / total_test) * 100
    
    print("\n" + "="*50)
    print("📊 10-WAY ADVANCED TTA METRICS")
    print("="*50)
    print(f"📂 Target Directory   : {TEST_DATA_DIR}")
    print(f"📉 10-Way Blended Loss: {final_loss:.4f}")
    print(f"🏆 Final TTA Accuracy : {final_accuracy:.2f}%")
    print("="*50 + "\n")

# =============================================================================
# PIPELINE EXECUTION ENTRYPOINT
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
        
        evaluate_dataset_with_10_way_tta(model, test_loader)
    else:
        print(f"❌ Error: Test directory path not found.")