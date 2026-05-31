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

# Base transform for loading images cleanly
base_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# LOAD ARCHITECTURE & WEIGHTS
# =============================================================================
def load_trained_model(weights_path, num_classes):
    print(f"⚙️  Loading EfficientNetV2-M architecture...")
    model = models.efficientnet_v2_m(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    print(f"💾 Loading trained weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=DEVICE)
    
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model

# =============================================================================
# TEST-TIME AUGMENTATION (TTA) EVALUATION ENGINE
# =============================================================================
def evaluate_dataset_with_tta(model, test_loader):
    """Evaluates the test set using a 3-way TTA ensemble strategy."""
    print("🧪 Running evaluation with 3-Way TTA (Original + Flip + Scale Shift)...")
    criterion = nn.CrossEntropyLoss()
    test_loss, correct_test, total_test = 0.0, 0, 0
    
    # Define on-the-fly geometric augmentations for TTA
    hf_transform = transforms.RandomHorizontalFlip(p=1.0)
    scale_transform = transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),
        transforms.CenterCrop(IMAGE_SIZE)
    ])
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # --- Pass 1: Original Clean Images ---
            outputs_orig = model(images)
            
            # --- Pass 2: Horizontal Flips ---
            flipped_images = torch.stack([hf_transform(img) for img in images])
            outputs_flip = model(flipped_images)
            
            # --- Pass 3: Scaled and Center Cropped Images ---
            scaled_images = torch.stack([scale_transform(img) for img in images])
            outputs_scale = model(scaled_images)
            
            # ENSEMBLE: Average the soft prediction logit spaces
            final_outputs = (outputs_orig + outputs_flip + outputs_scale) / 3.0
            
            loss = criterion(final_outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            _, predicted = final_outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()

    final_loss = test_loss / total_test
    final_accuracy = (correct_test / total_test) * 100
    
    print("\n" + "="*50)
    print("📊 UNSEEN TEST DATASET METRICS WITH TTA")
    print("="*50)
    print(f"📂 Target Directory   : {TEST_DATA_DIR}")
    print(f"📉 Average Blended Loss: {final_loss:.4f}")
    print(f"🏆 Final TTA Accuracy : {final_accuracy:.2f}%")
    print("="*50 + "\n")

# =============================================================================
# SINGLE IMAGE TTA INFERENCE
# =============================================================================
def predict_single_image_tta(model, image_path, class_names):
    """Predicts a single image by ensembling multiple look-angles."""
    if not os.path.exists(image_path):
        return

    img = Image.open(image_path).convert('RGB')
    orig_tensor = base_transforms(img).to(DEVICE)
    
    hf_transform = transforms.RandomHorizontalFlip(p=1.0)
    scale_transform = transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),
        transforms.CenterCrop(IMAGE_SIZE)
    ])
    
    # Generate variations
    flip_tensor = hf_transform(orig_tensor)
    scale_tensor = scale_transform(orig_tensor)
    
    # Stack into a mini-batch [3, 3, 384, 384]
    tta_batch = torch.stack([orig_tensor, flip_tensor, scale_tensor])
    
    with torch.no_grad():
        outputs = model(tta_batch)
        # Average across the batch dimension
        averaged_logits = outputs.mean(dim=0, keepdim=True)
        probabilities = torch.nn.functional.softmax(averaged_logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_breed = class_names[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    print("="*50)
    print("🎯 SINGLE IMAGE TTA PREDICTION RESULTS")
    print("="*50)
    print(f"🐕 Predicted : {predicted_breed.replace('_', ' ').title()}")
    print(f"🔥 Confidence: {confidence_score:.2f}%")
    print("="*50 + "\n")

# =============================================================================
# EXECUTION PIPELINE
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"❌ Error: Train path missing.")
        exit()
        
    train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR)
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    model = load_trained_model(MODEL_WEIGHTS_PATH, num_classes)
    
    if os.path.exists(TEST_DATA_DIR):
        test_dataset = datasets.ImageFolder(root=TEST_DATA_DIR, transform=base_transforms)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        evaluate_dataset_with_tta(model, test_loader)
    else:
        print(f"❌ Error: Test directory missing.")

    SAMPLE_IMAGE = "test_dog.jpg" 
    predict_single_image_tta(model, SAMPLE_IMAGE, class_names)