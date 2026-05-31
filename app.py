import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =============================================================================
# 1. PAGE CONFIGURATIONS & UI TEXT
# =============================================================================
st.set_page_config(
    page_title="Stanford Dogs Classifier",
    page_icon="🐕",
    layout="centered"
)

st.title("🐕 Smart Dog Breed Explorer")
st.write("Upload a dog image, and our **EfficientNetV2-M** model reinforced with **5-Way TTA** will identify the breed with a production-grade **94.13%** accuracy.")

# =============================================================================
# 2. STATIC 120 CLASS MATRIX (Eliminates the need for a physical train folder)
# =============================================================================
CLASS_NAMES = [
    'n02085620-Chihuahua', 'n02085782-Japanese_spaniel', 'n02085936-Maltese_dog', 'n02086079-Pekinese', 
    'n02086240-Shih-Tzu', 'n02086646-Blenheim_spaniel', 'n02086910-papillon', 'n02087046-toy_terrier', 
    'n02087394-Rhodesian_ridgeback', 'n02088094-Afghan_hound', 'n02088238-basset', 'n02088364-beagle', 
    'n02088466-bloodhound', 'n02088632-bluetick', 'n02089078-black-and-tan_coonhound', 'n02089867-Walker_hound', 
    'n02089973-English_foxhound', 'n02090379-redbone', 'n02090622-borzoi', 'n02090721-Irish_wolfhound', 
    'n02091032-Italian_greyhound', 'n02091134-whippet', 'n02091244-Ibizan_hound', 'n02091467-Norwegian_elkhound', 
    'n02091635-otterhound', 'n02091831-Saluki', 'n02092002-Scottish_deerhound', 'n02092339-Weimaraner', 
    'n02093256-Staffordshire_bullterrier', 'n02093428-American_Staffordshire_terrier', 'n02093647-Bedlington_terrier', 'n02093754-Border_terrier', 
    'n02093859-Kerry_blue_terrier', 'n02093991-Irish_terrier', 'n02094114-Norfolk_terrier', 'n02094258-Norwich_terrier', 
    'n02094433-Yorkshire_terrier', 'n02095314-wire-haired_fox_terrier', 'n02095570-Lakeland_terrier', 'n02095889-Sealyham_terrier', 
    'n02096051-Airedale', 'n02096177-cairn', 'n02096294-Australian_terrier', 'n02096437-Dandie_Dinmont', 
    'n02096585-Boston_bull', 'n02097047-miniature_schnauzer', 'n02097130-giant_schnauzer', 'n02097209-standard_schnauzer', 
    'n02097298-Scotch_terrier', 'n02097474-Tibetan_terrier', 'n02097658-silky_terrier', 'n02098105-soft-coated_wheaten_terrier', 
    'n02098286-West_Highland_white_terrier', 'n02098413-Lhasa', 'n02099267-flat-coated_retriever', 'n02099429-curly-coated_retriever', 
    'n02099601-golden_retriever', 'n02099712-Labrador_retriever', 'n02099849-Chesapeake_Bay_retriever', 'n02100236-German_short-haired_pointer', 
    'n02100583-vizsla', 'n02100735-English_setter', 'n02100877-Irish_setter', 'n02101006-Gordon_setter', 
    'n02101388-Brittany_spaniel', 'n02101556-clumber', 'n02102040-English_springer', 'n02102177-Welsh_springer_spaniel', 
    'n02102318-cocker_spaniel', 'n02102480-Sussex_spaniel', 'n02102973-Irish_water_spaniel', 'n02104029-kuvasz', 
    'n02104365-schipperke', 'n02105056-groenendael', 'n02105162-malinois', 'n02105251-briard', 
    'n02105412-kelpie', 'n02105505-komondor', 'n02105641-Old_English_sheepdog', 'n02105855-Shetland_sheepdog', 
    'n02106030-collie', 'n02106166-Border_collie', 'n02106382-Bouvier_des_Flandres', 'n02106550-Rottweiler', 
    'n02106662-German_shepherd', 'n02107142-Doberman', 'n02107312-miniature_pinscher', 'n02107574-Greater_Swiss_Mountain_dog', 
    'n02107683-Bernese_mountain_dog', 'n02107908-Appenzeller', 'n02108000-EntleBucher', 'n02108089-boxer', 
    'n02108422-bull_mastiff', 'n02108551-Tibetan_mastiff', 'n02108915-French_bulldog', 'n02109047-Great_Dane', 
    'n02109525-Saint_Bernard', 'n02109961-Eskimo_dog', 'n02110063-malamute', 'n02110185-Siberian_husky', 
    'n02110627-affenpinscher', 'n02110806-basenji', 'n02110958-pug', 'n02111129-Leonberg', 
    'n02111277-Newfoundland', 'n02111500-Great_Pyrenees', 'n02111889-Samoyed', 'n02112018-P Pomeranian', 
    'n02112137-chow', 'n02112350-keeshond', 'n02112706-Brabancon_griffon', 'n02113023-Pembroke', 
    'n02113186-Cardigan', 'n02113624-toy_poodle', 'n02113712-miniature_poodle', 'n02113799-standard_poodle', 
    'n02113978-Mexican_hairless', 'n02115641-dingo', 'n02115913-dhole', 'n02116738-African_hunting_dog'
]

IMAGE_SIZE = 384
MODEL_WEIGHTS_PATH = "efficientnet_v2m_stanford_dogs_best_93.pth"

# Cloud hosting structures execute inference on the CPU layer by default
DEVICE = torch.device("cpu")

# =============================================================================
# 3. RESOURCE CACHED ARCHITECTURE GENERATOR
# =============================================================================
@st.cache_resource
def load_model():
    """Constructs model blueprint and injects weights once to preserve server memory."""
    model = models.efficientnet_v2_m(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    
    # Load and clean up distributed DDP prefixes (module.) automatically
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
    clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model

# Initialize runtime model checkpoint loading safely
try:
    model = load_model()
except FileNotFoundError:
    st.error(f"❌ Weights file `{MODEL_WEIGHTS_PATH}` not found in the local directory. Please verify file upload paths.")
    st.stop()

# =============================================================================
# 4. 5-WAY PURE GEOMETRIC TTA ENGINE
# =============================================================================
def run_5_way_tta(image, model):
    """Generates 5 complementary views of the image to ensure high-stability predictions."""
    # Base normalization pipeline matching the model training criteria
    base_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    hf_transform = transforms.RandomHorizontalFlip(p=1.0)
    scale_105 = transforms.Compose([transforms.Resize((int(IMAGE_SIZE*1.05), int(IMAGE_SIZE*1.05))), transforms.CenterCrop(IMAGE_SIZE)])
    scale_115 = transforms.Compose([transforms.Resize((int(IMAGE_SIZE*1.15), int(IMAGE_SIZE*1.15))), transforms.CenterCrop(IMAGE_SIZE)])

    # Transform raw image to initial base tensor
    orig_tensor = base_transform(image)

    # Generate the 5 distinct structural viewpoints
    v1 = orig_tensor
    v2 = hf_transform(orig_tensor)
    v3 = scale_105(orig_tensor)
    v4 = hf_transform(v3)
    v5 = scale_115(orig_tensor)

    # Stack variants into a combined batch execution shape: [5, 3, 384, 384]
    tta_batch = torch.stack([v1, v2, v3, v4, v5]).to(DEVICE)

    with torch.no_grad():
        outputs = model(tta_batch)
        # Average prediction logit maps to smooth out outlier errors
        averaged_logits = outputs.mean(dim=0, keepdim=True)
        probabilities = torch.nn.functional.softmax(averaged_logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    # Extract human-readable breed string by discarding alphanumeric folder prefixes
    raw_breed = CLASS_NAMES[predicted_idx.item()]
    clean_breed = raw_breed.split('-', 1)[-1].replace('_', ' ').title()
    
    return clean_breed, confidence.item() * 100

# =============================================================================
# 5. STREAMLIT INTERACTIVE USER PIPELINE
# =============================================================================
uploaded_file = st.file_uploader("📸 Choose a dog image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and render the loaded image asset cleanly
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📷 Uploaded Image Source", use_container_width=True)
    
    with st.spinner("🧠 Initializing 5-Way TTA framework and computing predictions..."):
        breed, confidence = run_5_way_tta(image, model)
        
    st.success("🎉 Computer vision analysis completed!")
    
    # Layout predictions in structured data blocks
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🐕 Predicted Breed", value=breed)
    with col2:
        st.metric(label="🔥 Model Prediction Confidence", value=f"{confidence:.2f}%")
        
    # Visual metric meter representation matching prediction confidence values
    st.progress(int(confidence))