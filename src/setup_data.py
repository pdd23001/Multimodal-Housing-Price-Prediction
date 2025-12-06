import kagglehub
import shutil
import os
import glob

def setup_dataset():
    print("Downloading dataset using kagglehub...")
    # Download latest version
    path = kagglehub.dataset_download("ted8080/house-prices-and-images-socal")
    print("Dataset downloaded to:", path)

    # Define target directory
    target_dir = 'data'
    os.makedirs(target_dir, exist_ok=True)

    print(f"Moving files to {target_dir}...")
    
    # Move all files from the download path to data/
    # The dataset likely contains a CSV and a folder of images or similar structure.
    # We'll move everything.
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(target_dir, item)
        if os.path.exists(d):
            print(f"Warning: {d} already exists. Skipping/Overwriting...")
            if os.path.isdir(d):
                shutil.rmtree(d)
            else:
                os.remove(d)
        
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
            
    print("Dataset setup complete.")
    
    # List files to confirm
    print("Files in data/:")
    print(os.listdir(target_dir))

if __name__ == "__main__":
    setup_dataset()
