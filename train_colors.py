from ultralytics import YOLO

def train_model():
    # 1. Load the Nano Classification Model (Pre-trained on ImageNet)
    model = YOLO('yolov8n-cls.pt') 

    # 2. Train with A40 Optimized Settings
    results = model.train(
        data='dataset/yolo_ready',  # Path to the folder created in Step 1
        epochs=50,                  # 50 is usually enough for color
        imgsz=224,                  # Standard classification size
        batch=256,                  # High batch size for A40
        device=0,                   # Use the first GPU
        workers=8,                  # High dataloader workers
        project='vehicle_color_project',
        name='A40_color_run_v1',
        augment=True,               # Enable default augmentations (flip/color/scale)
        dropout=0.1,                # Slight dropout to prevent overfitting on small data
        patience=10                 # Stop early if it stops learning
    )

if __name__ == "__main__":
    train_model()