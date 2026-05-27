import os
from datasets import load_dataset
from tqdm import tqdm

def download_stanford_dogs():
    # 1. Define the base storage path on your SSD
    BASE_DIR = "./data"
    splits = ["train", "test"]
    
    print("[*] Connecting to Hugging Face servers to download the Stanford Dogs dataset...")
    
    for split in splits:
        print(f"\n[*] Fetching data partition: {split.upper()}...")
        # Download the specific split from the official stable repository
        dataset = load_dataset("maurice-fp/stanford-dogs", split=split)
        
        # EXTRACT THE CLASS NAMES MAP FROM HUGGING FACE METADATA
        # This converts integer labels (0, 1, 2...) back to real breed names strings
        breed_names_list = dataset.features['label'].names
        
        print(f"[+] Fetching successful. Extracting and saving {len(dataset)} images to local storage...")
        
        # Create the specific directory for this partition (e.g., ./data/train or ./data/test)
        split_dir = os.path.join(BASE_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # 2. Iterate through images and save them into folders matching their breed names
        for item in tqdm(dataset, desc=f"Saving {split} images"):
            image = item['image'].convert('RGB')
            
            # CRITICAL FIX: Map the integer label to its actual breed name string
            label_int = item['label']
            breed_name = breed_names_list[label_int]
            
            # Sanitize the folder name by replacing spaces with underscores to prevent path issues
            breed_dir_name = breed_name.replace(" ", "_")
            breed_path = os.path.join(split_dir, breed_dir_name)
            os.makedirs(breed_path, exist_ok=True)
            
            # Generate a unique, structured filename inside the breed directory
            image_id = f"{len(os.listdir(breed_path)) + 1:04d}.jpg"
            image_save_path = os.path.join(breed_path, image_id)
            
            # Save the raw image file physically to the SSD
            image.save(image_save_path, "JPEG")
            
    print("\n" + "="*60)
    print("[+] Process completed successfully!")
    print(f"[+] Train dataset (12,000 images) saved at: {os.path.abspath(os.path.join(BASE_DIR, 'train'))}")
    print(f"[+] Test dataset (8,580 images) saved at: {os.path.abspath(os.path.join(BASE_DIR, 'test'))}")
    print("="*60)

if __name__ == "__main__":
    download_stanford_dogs()