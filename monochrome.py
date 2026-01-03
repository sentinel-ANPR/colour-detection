import cv2
import numpy as np
import os
from ultralytics import YOLO

def is_monochrome(image_bgr,
                  mean_thresh=0.02,
                  std_thresh=0.02):

    # Convert to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Normalize saturation channel to [0,1]
    S = hsv[:, :, 1].astype("float32") / 255.0

    mean_s = S.mean()
    std_s  = S.std()
    max_s  = S.max()

    print(f"Mean S = {mean_s:.4f}, Std S = {std_s:.4f}, Max S = {max_s:.4f}")

    # Grayscale / monochrome if saturation is uniformly tiny
    return (mean_s < mean_thresh) and (std_s < std_thresh)


def crop_vehicle(frame, box, frame_width, frame_height):
    """
    Returns a crop of the vehicle from the frame.
    - Prevents out-of-bounds cropping
    """
    x1, y1, x2, y2 = map(int, box)
    
    # No padding, just use the bounding box
    y1_padded = max(0, y1)
    x1_padded = max(0, x1)
    x2_padded = min(frame_width,  x2)
    y2_padded = min(frame_height, y2)
    
    # final safe crop
    vehicle_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
    return vehicle_crop


# Load YOLOv8 model
print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')

# Folder containing images
folder_path = "car"
night_mode_folder = "night mode"
normal_folder = "normal"

# Create output folders if they don't exist
os.makedirs(night_mode_folder, exist_ok=True)
os.makedirs(normal_folder, exist_ok=True)

# Vehicle classes in COCO dataset (excluding motorcycles)
# 2: car, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 5, 7]

# Check if source folder exists
if not os.path.exists(folder_path):
    print(f"Error: Folder '{folder_path}' not found!")
else:
    # Get all image files from the folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No images found in '{folder_path}' folder!")
    else:
        print(f"Found {len(image_files)} images. Processing...\n")
        
        grayscale_count = 0
        colored_count = 0
        no_detection_count = 0
        
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Warning: Could not read {image_file}\n")
                continue
            
            print(f"Processing: {image_file}")
            
            # Get frame dimensions
            frame_height, frame_width = img.shape[:2]
            
            # Run YOLO detection
            results = model(img, verbose=False)
            
            # Get detections
            detections = results[0].boxes
            
            # Filter for vehicle classes only (excluding motorcycles)
            vehicle_detected = False
            for detection in detections:
                class_id = int(detection.cls[0])
                confidence = float(detection.conf[0])
                
                # Only process if it's a vehicle (not motorcycle) and confidence > 0.5
                if class_id in VEHICLE_CLASSES and confidence > 0.5:
                    vehicle_detected = True
                    box = detection.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    
                    print(f"  Detected: class_id={class_id}, confidence={confidence:.2f}")
                    
                    # Crop the vehicle
                    cropped_img = crop_vehicle(img, box, frame_width, frame_height)
                    
                    # Check if crop is valid
                    if cropped_img.size == 0:
                        print(f"  Warning: Invalid crop\n")
                        continue
                    
                    # Classify the cropped image
                    if is_monochrome(cropped_img):
                        print(f"  → Classification: GRAYSCALE")
                        # Save to night mode folder with unique name
                        base_name = os.path.splitext(image_file)[0]
                        ext = os.path.splitext(image_file)[1]
                        destination = os.path.join(night_mode_folder, f"{base_name}_{grayscale_count}{ext}")
                        cv2.imwrite(destination, cropped_img)
                        print(f"  → Saved cropped image to '{night_mode_folder}' folder\n")
                        grayscale_count += 1
                    else:
                        print(f"  → Classification: COLORED")
                        # Save to normal folder with unique name
                        base_name = os.path.splitext(image_file)[0]
                        ext = os.path.splitext(image_file)[1]
                        destination = os.path.join(normal_folder, f"{base_name}_{colored_count}{ext}")
                        cv2.imwrite(destination, cropped_img)
                        print(f"  → Saved cropped image to '{normal_folder}' folder\n")
                        colored_count += 1
                    
                    # Only process the first vehicle detection per image
                    break
            
            if not vehicle_detected:
                print(f"  No vehicle detected in this image\n")
                no_detection_count += 1
        
        # Summary
        print("="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Total images processed: {len(image_files)}")
        print(f"Images with vehicle detections: {grayscale_count + colored_count}")
        print(f"Images with no vehicle detections: {no_detection_count}")
        print(f"Grayscale vehicles → '{night_mode_folder}' folder: {grayscale_count}")
        print(f"Colored vehicles → '{normal_folder}' folder: {colored_count}")