import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import io

def run_dataset_download():
    # Target our local data folder we created earlier
    save_directory = Path("./data/Images")
    
    print("[*] Connecting to stable Parquet Stanford Dogs dataset on Hugging Face...")
    try:
        # Load the stable dataset split configuration
        hf_data = load_dataset("maurice-fp/stanford-dogs", split="train")
        print(f"[+] Connection secure. Total records found in train split: {len(hf_data)}")
    except Exception as error:
        print(f"[-] Failed to stream data repository: {error}")
        return

    print(f"[*] Reconstructing image files into: {save_directory.resolve()}...")
    
    # Extract structural label names features to map integers back to breed strings
    features = hf_data.features
    breed_labels = features["label"].names if "label" in features else None

    # Iterate through individual samples and serialize them to disk files
    for position, record in enumerate(hf_data):
        # Decode integer labels into string breed names if available
        if breed_labels and "label" in record:
            breed_name = breed_labels[record["label"]]
        else:
            breed_name = record.get("breed", "unknown_breed")
            
        # Format breed directories safely
        breed_folder = save_directory / str(breed_name).replace(" ", "_")
        breed_folder.mkdir(parents=True, exist_ok=True)
        
        # Build standard sequential image filename
        output_file_path = breed_folder / f"sample_{position}.jpg"
        
        # Save image matrix array to disk if not already processed
        if not output_file_path.exists():
            image_raw = record["image"]
            
            # Unpack byte wrappers if wrapped inside custom dict layers
            if isinstance(image_raw, dict) and "bytes" in image_raw:
                image_raw = Image.open(io.BytesIO(image_raw["bytes"]))
                
            if image_raw.mode != "RGB":
                image_raw = image_raw.convert("RGB")
            image_raw.save(output_file_path, "JPEG")
            
        # Display progress logs every 2,000 files processed
        if (position + 1) % 2000 == 0:
            print(f"[Progress Log] Successfully verified {position + 1}/{len(hf_data)} images on disk.")

    print(f"\n[+] SUCCESS! Images are fully structured inside your data/Images/ directory.")

if __name__ == "__main__":
    run_dataset_download()
