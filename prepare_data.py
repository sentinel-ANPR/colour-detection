import os
import shutil
import random

# --- CONFIG ---
# Where your current folders are (e.g., "raw_data/red", "raw_data/blue")
SOURCE_DIR = "dataset/raw_enhanced" 
# Where you want the ready-to-train data
DEST_DIR = "dataset/yolo_ready"
# Split ratio
VAL_SPLIT = 0.2  # 20% for validation

def split_data():
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    print(f"Found classes: {classes}")

    for cls in classes:
        # Define paths
        src_path = os.path.join(SOURCE_DIR, cls)
        train_path = os.path.join(DEST_DIR, "train", cls)
        val_path = os.path.join(DEST_DIR, "val", cls)

        # Make directories
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        # Get all images
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        # Calculate split index
        split_idx = int(len(images) * (1 - VAL_SPLIT))
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        print(f"Processing {cls}: {len(train_imgs)} Train, {len(val_imgs)} Val")

        # Copy files
        for img in train_imgs:
            shutil.copy(os.path.join(src_path, img), os.path.join(train_path, img))
        for img in val_imgs:
            shutil.copy(os.path.join(src_path, img), os.path.join(val_path, img))

    print("Data preparation complete!")

if __name__ == "__main__":
    split_data()