import cv2
from ultralytics import YOLO
import os
from pathlib import Path

def crop_vehicles_from_image(image_path, model, color_prefix=""):
    """
    Detect vehicles in an image using YOLO and create separate images for each vehicle.
    - If 1 vehicle: keeps original filename (replaces original)
    - If multiple vehicles: creates filename_DUPLICATE_1, filename_DUPLICATE_2, etc. (removes original)
    
    Args:
        image_path: Path to the input image
        model: Loaded YOLO model
        color_prefix: Prefix to add to output filenames (e.g., "gray_", "red_")
    
    Returns:
        Number of vehicles detected
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return 0
    
    # Run inference
    results = model(image)
    
    # Vehicle class IDs in COCO dataset
    # 2: car, 3: motorcycle, 5: bus, 7: truck
    vehicle_classes = [2]
    
    # Get directory and filename info
    image_dir = os.path.dirname(image_path)
    image_name = Path(image_path).stem
    image_ext = Path(image_path).suffix
    
    # Collect all detected vehicles
    detected_vehicles = []
    
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class ID
            class_id = int(box.cls[0])
            
            # Check if it's a vehicle
            if class_id in vehicle_classes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop the vehicle from the image
                cropped_vehicle = image[y1:y2, x1:x2]
                
                detected_vehicles.append({
                    'crop': cropped_vehicle,
                    'class': model.names[class_id],
                    'bbox': (x1, y1, x2, y2)
                })
    
    vehicle_count = len(detected_vehicles)
    
    if vehicle_count == 0:
        print(f"  No vehicles detected - original image kept")
        return 0
    
    elif vehicle_count == 1:
        # Single vehicle: keep original filename (overwrite original)
        output_filename = f"{color_prefix}{image_name}{image_ext}"
        output_path = os.path.join(image_dir, output_filename)
        
        cv2.imwrite(output_path, detected_vehicles[0]['crop'])
        print(f"  ✓ 1 vehicle detected - replaced original with cropped version")
        print(f"    File: {output_filename} (Class: {detected_vehicles[0]['class']})")
        
    else:
        # Multiple vehicles: create DUPLICATE files and remove original
        print(f"  ✓ {vehicle_count} vehicles detected - creating {vehicle_count} separate images")
        
        for i, vehicle in enumerate(detected_vehicles, start=1):
            output_filename = f"{color_prefix}{image_name}_DUPLICATE_{i}{image_ext}"
            output_path = os.path.join(image_dir, output_filename)
            
            cv2.imwrite(output_path, vehicle['crop'])
            print(f"    Saved: {output_filename} (Class: {vehicle['class']}, Size: {vehicle['bbox'][2]-vehicle['bbox'][0]}x{vehicle['bbox'][3]-vehicle['bbox'][1]})")
        
        # Remove the original image
        try:
            os.remove(image_path)
            print(f"    Removed original image: {Path(image_path).name}")
        except Exception as e:
            print(f"    Warning: Could not remove original image: {e}")
    
    return vehicle_count

def process_color_folders(base_dir, color_folders, model_path="yolov8n.pt"):
    """
    Process images from multiple color-coded folders.
    - Single vehicle images: replaced with cropped version (same filename)
    - Multiple vehicle images: split into filename_DUPLICATE_1, filename_DUPLICATE_2, etc. (original removed)
    
    Args:
        base_dir: Base directory containing color folders
        color_folders: List of color folder names (e.g., ["gray", "black", "white", "red", "blue"])
        model_path: Path to YOLO model weights
    """
    # Load YOLO model once
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Statistics
    total_images = 0
    total_vehicles = 0
    single_vehicle_images = 0
    multi_vehicle_images = 0
    no_vehicle_images = 0
    
    # Process each color folder
    for color in color_folders:
        color_dir = os.path.join(base_dir, color)
        
        # Check if the color directory exists
        if not os.path.exists(color_dir):
            print(f"\nWarning: Directory '{color_dir}' does not exist. Skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {color.upper()} folder: {color_dir}")
        print(f"{'='*60}")
        
        # Get all image files in the color directory
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(color_dir).glob(f"*{ext}"))
            image_files.extend(Path(color_dir).glob(f"*{ext.upper()}"))
        
        # Filter out any DUPLICATE files from previous runs
        image_files = [f for f in image_files if '_DUPLICATE_' not in f.name]
        
        if not image_files:
            print(f"No images found in {color_dir}")
            continue
        
        print(f"Found {len(image_files)} images in {color} folder")
        
        # Process each image in the color folder
        folder_vehicle_count = 0
        folder_single_vehicle = 0
        folder_multi_vehicle = 0
        folder_no_vehicle = 0
        
        for image_path in image_files:
            print(f"\nProcessing: {image_path.name}")
            vehicle_count = crop_vehicles_from_image(
                str(image_path), 
                model,
                color_prefix=""
            )
            
            if vehicle_count == 0:
                folder_no_vehicle += 1
                no_vehicle_images += 1
            elif vehicle_count == 1:
                folder_single_vehicle += 1
                single_vehicle_images += 1
            else:
                folder_multi_vehicle += 1
                multi_vehicle_images += 1
            
            folder_vehicle_count += vehicle_count
            total_images += 1
        
        print(f"\n{color.upper()} folder summary:")
        print(f"  - Images processed: {len(image_files)}")
        print(f"  - Single vehicle (replaced): {folder_single_vehicle}")
        print(f"  - Multiple vehicles (split): {folder_multi_vehicle}")
        print(f"  - No vehicles (kept): {folder_no_vehicle}")
        print(f"  - Total vehicles cropped: {folder_vehicle_count}")
        total_vehicles += folder_vehicle_count
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {total_images}")
    print(f"  - Single vehicle images (replaced): {single_vehicle_images}")
    print(f"  - Multiple vehicle images (split): {multi_vehicle_images}")
    print(f"  - No vehicle images (kept): {no_vehicle_images}")
    print(f"Total vehicles cropped: {total_vehicles}")
    print(f"Average vehicles per image: {total_vehicles/total_images:.2f}" if total_images > 0 else "N/A")
    print(f"\nProcessing complete! Check your color folders for results.")

def process_single_folder(folder_path, model_path="yolov8n.pt"):
    """
    Process images from a single folder.
    
    Args:
        folder_path: Path to folder containing images
        model_path: Path to YOLO model weights
    """
    if not os.path.exists(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist.")
        return
    
    # Load YOLO model
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    print(f"\n{'='*60}")
    print(f"Processing folder: {folder_path}")
    print(f"{'='*60}")
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    # Filter out DUPLICATE files
    image_files = [f for f in image_files if '_DUPLICATE_' not in f.name]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Statistics
    total_vehicles = 0
    single_vehicle_images = 0
    multi_vehicle_images = 0
    no_vehicle_images = 0
    
    # Process each image
    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")
        vehicle_count = crop_vehicles_from_image(
            str(image_path), 
            model,
            color_prefix=""
        )
        
        if vehicle_count == 0:
            no_vehicle_images += 1
        elif vehicle_count == 1:
            single_vehicle_images += 1
        else:
            multi_vehicle_images += 1
        
        total_vehicles += vehicle_count
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(image_files)}")
    print(f"  - Single vehicle images (replaced): {single_vehicle_images}")
    print(f"  - Multiple vehicle images (split): {multi_vehicle_images}")
    print(f"  - No vehicle images (kept): {no_vehicle_images}")
    print(f"Total vehicles cropped: {total_vehicles}")
    print(f"\nProcessing complete!")

if __name__ == "__main__":
    # Define your color folders
    color_folders = ["gray", "black", "white", "red", "blue"]
    
    # Option 1: Process multiple color folders
    print("="*60)
    print("STARTING VEHICLE DETECTION AND CROPPING")
    print("="*60)
    process_color_folders(
        base_dir="vehicle_images",  # Your base directory containing color folders
        color_folders=color_folders,
        model_path="yolov8n.pt"
    )