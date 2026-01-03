import cv2
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import os
import glob

class VehicleColorExtractor:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    # Crop from top of image
    @staticmethod
    def crop_top(img, crop_percentage=30):
        height = img.shape[0]
        crop_pixels = int(height * crop_percentage / 100)
        cropped = img[crop_pixels:, :]  # Remove top portion
        return cropped

    # --- Mask pixels that are likely paint (ignore very dark shadows and overexposed pixels) ---
    def mask_paint(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        # Keep all reasonably visible pixels
        mask = cv2.inRange(hsv, (0, 0, 40), (179, 255, 255))
        return cv2.bitwise_and(bgr_img, bgr_img, mask=mask)

    # --- Extract dominant RGB using KMeans ---
    def get_dominant_rgb(self, img):
        masked = self.mask_paint(img)
        pixels = masked.reshape(-1, 3)
        pixels = np.array([p for p in pixels if not np.all(p == 0)])

        if len(pixels) == 0:
            return [255, 255, 255]  # fallback white

        # Convert BGR → RGB
        pixels = pixels[:, ::-1]

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        kmeans.fit(pixels)
        counts = np.bincount(kmeans.labels_)
        dominant_rgb = kmeans.cluster_centers_[np.argmax(counts)]
        return dominant_rgb.astype(int).tolist()

    # --- Convert RGB → HEX ---
    @staticmethod
    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    # --- Classify color with improved detection for darker shades ---
    @staticmethod
    def classify_hsv_color(rgb):
        rgb_img = np.uint8([[rgb]])
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv

        # Very dark → black (lowered threshold to avoid misclassifying dark colors)
        if v < 60:
            return "black"
        
        # White detection with lenient saturation threshold
        if v > 180 and s < 70:
            return "white"
        
        # Additional white check for medium brightness
        if v > 160 and s < 45:
            return "white"
        
        # Check for colored hues FIRST, even with low saturation
        # This ensures darker shades of colors are detected correctly
        if s >= 20:  # Very low saturation threshold to catch desaturated darker shades
            # Expanded red hue range to catch maroon/dark red
            if 0 <= h < 15 or 165 <= h <= 179:
                return "red"
            elif 15 <= h < 28:
                return "orange"
            elif 28 <= h < 40:
                return "yellow"
            elif 40 <= h < 85:
                return "green"
            elif 85 <= h < 135:
                return "blue"
            elif 135 <= h < 165:
                return "purple"
        
        # Gray only for truly neutral colors (very restrictive)
        if s < 30 and 60 <= v <= 180:
            return "gray"

        # Final fallback: check hue even without saturation requirement
        # This catches very desaturated darker shades
        if 0 <= h < 15 or 165 <= h <= 179:
            return "red"
        elif 15 <= h < 28:
            return "orange"
        elif 28 <= h < 40:
            return "yellow"
        elif 40 <= h < 85:
            return "green"
        elif 85 <= h < 135:
            return "blue"
        elif 135 <= h < 165:
            return "purple"
        
        return "gray"

    # --- Full extraction: dominant color HEX + name ---
    def extract_color(self, img, crop_top_percent=30):
        # Crop the top portion
        cropped_img = self.crop_top(img, crop_top_percent)
        
        # Extract color from cropped image
        dominant_rgb = self.get_dominant_rgb(cropped_img)
        hex_color = self.rgb_to_hex(dominant_rgb)
        color_name = self.classify_hsv_color(dominant_rgb)
        return hex_color, color_name, dominant_rgb, cropped_img


def save_processed_image(image_path, color_name, rgb_values, hex_value, cropped_img, output_dir="output"):
    """Save processed image with color overlay"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        image = cv2.imread(image_path)
        if image is None:
            return
        
        overlay = image.copy()
        
        # Create color overlay with detected color
        rgb_bgr = (int(rgb_values[2]), int(rgb_values[1]), int(rgb_values[0]))  # Convert RGB to BGR
        cv2.rectangle(overlay, (10, 10), (400, 120), rgb_bgr, -1)
        
        # Add text with white color
        cv2.putText(overlay, f"Color: {color_name.upper()}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(overlay, f"RGB: {tuple(rgb_values)}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"HEX: {hex_value}", (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save images
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        # Save cropped image
        cv2.imwrite(f"{output_dir}/{name}_cropped{ext}", cropped_img)
        
        # Save result with overlay
        cv2.imwrite(f"{output_dir}/{name}_result{ext}", overlay)
        
    except Exception as e:
        print(f"  Error saving processed image: {e}")


def process_images_from_input_folder():
    """Process all images from 'in' folder and save to 'output' folder"""
    extractor = VehicleColorExtractor(n_clusters=3)
    
    input_dir = "in"
    output_dir = "output"
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ '{input_dir}' directory not found!")
        print(f"Create '{input_dir}' directory and put test images there")
        return
    
    # Find image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"❌ No images found in '{input_dir}' directory")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("📝 Note: Cropping 30% from top of each image before color detection")
    print("=" * 80)
    
    results = []
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print(f"\n{idx}. Processing: {filename}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"   ❌ Failed to read image")
            continue
        
        # Extract color with 30% top cropping
        hex_color, color_name, rgb, cropped_img = extractor.extract_color(img, crop_top_percent=30)
        
        print(f"   Detected Color: {color_name.upper()}")
        print(f"   HEX Code: {hex_color}")
        print(f"   RGB: {rgb}")
        
        # Save processed images with overlay
        save_processed_image(image_path, color_name, rgb, hex_color, cropped_img, output_dir)
        
        results.append((filename, color_name, rgb, hex_color))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    
    from collections import Counter
    color_counts = Counter([r[1] for r in results])
    for color, count in color_counts.most_common():
        print(f"{color:10} : {count:2} images")
    
    print(f"\n✅ Processing complete!")
    print(f"📁 Processed images saved to '{output_dir}' directory")
    print(f"   - *_cropped.jpg: Cropped vehicle body region")
    print(f"   - *_result.jpg: Original image with color overlay")


if __name__ == "__main__":
    process_images_from_input_folder()