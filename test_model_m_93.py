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

# =============================================================================
# 1. VISUAL PIPELINE (Transforms)
# =============================================================================
# Clean verification transforms only: No random crops, flips, or augmentations.
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# 2. LOAD ARCHITECTURE & WEIGHTS
# =============================================================================
def load_trained_model(weights_path, num_classes):
    """Loads the EfficientNetV2-M architecture and injects the saved weights."""
    print(f"⚙️  Loading EfficientNetV2-M architecture...")
    model = models.efficientnet_v2_m(weights=None)
    
    # Recreate the exact linear head (120 classes)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    print(f"💾 Loading trained weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=DEVICE)
    
    # Clean up DDP prefixes if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    model = model.to(DEVICE)
    model.eval() # Set to evaluation mode
    return model

# =============================================================================
# 3. FULL DATASET ACCURACY VERIFIER
# =============================================================================
def evaluate_dataset(model, test_loader):
    """Runs through the dedicated test directory to verify final accuracy."""
    print("🧪 Evaluating your 93.64% model on the unseen test dataset...")
    criterion = nn.CrossEntropyLoss()
    test_loss, correct_test, total_test = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()

    final_loss = test_loss / total_test
    final_accuracy = (correct_test / total_test) * 100
    
    print("\n" + "="*50)
    print("📊 UNSEEN TEST DATASET METRICS")
    print("="*50)
    print(f"📂 Target Directory: {TEST_DATA_DIR}")
    print(f"📉 Average Loss    : {final_loss:.4f}")
    print(f"🏆 Final Accuracy  : {final_accuracy:.2f}%")
    print("="*50 + "\n")

# =============================================================================
# 4. SINGLE IMAGE INFERENCE ENGINE
# =============================================================================
def predict_single_image(model, image_path, class_names):
    """Predicts the breed of a single external image file."""
    if not os.path.exists(image_path):
        return

    img = Image.open(image_path).convert('RGB')
    img_tensor = test_transforms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_breed = class_names[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    print("="*50)
    print("🎯 SINGLE IMAGE PREDICTION RESULTS")
    print("="*50)
    print(f"🖼️  Image Path : {image_path}")
    print(f"🐕 Predicted   : {predicted_breed.replace('_', ' ').title()}")
    print(f"🔥 Confidence  : {confidence_score:.2f}%")
    print("="*50 + "\n")

# =============================================================================
# RUNNING PIPELINE
# =============================================================================
if __name__ == "__main__":
    # Extract class names from train folder to preserve numerical index mapping
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"❌ Error: Training data directory not found at {TRAIN_DATA_DIR}. Class index cannot be mapped.")
        exit()
        
    train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR)
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    # Load the model weights
    model = load_trained_model(MODEL_WEIGHTS_PATH, num_classes)
    
    # Verify and load test dataset
    if os.path.exists(TEST_DATA_DIR):
        test_dataset = datasets.ImageFolder(root=TEST_DATA_DIR, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        # Run full dataset evaluation
        evaluate_dataset(model, test_loader)
    else:
        print(f"❌ Error: Test directory not found at {TEST_DATA_DIR}")

    # Optional: Run single image prediction if file exists
    SAMPLE_IMAGE = "test_dog.jpg" 
    predict_single_image(model, SAMPLE_IMAGE, class_names)