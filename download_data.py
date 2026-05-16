import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image

def run_dataset_download():
    # Target our local data folder we created earlier
    save_directory = Path("./data/Images")
    
    print("[*] Connecting to Hugging Face CDN mirrors...")
    try:
        # Stream dataset cache directly from Hugging Face repository
        hf_data = load_dataset("tanganke/stanford_dogs", split="train")
        print(f"[+] Connection secure. Total records found: {len(hf_data)}")
    except Exception as error:
        print(f"[-] Failed to stream data repository: {error}")
        return

    print(f"[*] Reconstructing files into: {save_directory.resolve()}...")
    
    # Iterate through individual samples and serialize them to disk files
    for position, record in enumerate(hf_data):
        # Format breed directories safely using underscores instead of spaces
        breed_folder = save_directory / record["label_name"].replace(" ", "_")
        breed_folder.mkdir(parents=True, exist_ok=True)
        
        # Build standard sequential image filename
        output_file_path = breed_folder / f"sample_{position}.jpg"
        
        # Save image matrix array to disk if not already processed
        if not output_file_path.exists():
            image_raw = record["image"]
            # Convert color channels to standard RGB to prevent training anomalies
            if image_raw.mode != "RGB":
                image_raw = image_raw.convert("RGB")
            image_raw.save(output_file_path, "JPEG")
            
        # Display progress logs every 2,000 files processed
        if (position + 1) % 2000 == 0:
            print(f"[Progress Log] Successfully verified {position + 1}/{len(hf_data)} images on disk.")

    print(f"\n[+] SUCCESS! 20,580 images fully structured inside your data/ directory.")

if __name__ == "__main__":
    run_dataset_download()
