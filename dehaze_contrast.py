import cv2
import numpy as np
import os

class ImageEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), gamma=1.2):
        """
        :param clip_limit: Threshold for contrast limiting (higher = more contrast but more noise). 2.0 is standard.
        :param tile_grid_size: Size of grid for histogram equalization.
        :param gamma: < 1.0 makes it darker, > 1.0 makes it brighter.
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.gamma = gamma
        # Create lookup table for fast gamma correction
        self.lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            self.lookUpTable[0, i] = np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)

    def adjust_gamma(self, image):
        """Fast Gamma correction using a lookup table."""
        return cv2.LUT(image, self.lookUpTable)

    def apply_clahe_color(self, image):
        """
        Applies CLAHE to the L-channel (Lightness) of the LAB color space.
        This cuts haze without messing up the colors (unlike RGB equalization).
        """
        # 1. Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 2. Apply CLAHE to L-channel
        l_enhanced = self.clahe.apply(l)

        # 3. Merge back and convert to BGR
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def sharpen(self, image):
        """Applies a subtle Unsharp Mask to pop edges."""
        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0, image)

    def process(self, image):
        """Run the full pipeline."""
        # Step 1: Gamma correction (bring up the darks)
        img = self.adjust_gamma(image)
        
        # Step 2: CLAHE (Dehaze / Local Contrast)
        img = self.apply_clahe_color(img)
        
        # Step 3: Sharpening (Optional - good for OCR/Plates)
        img = self.sharpen(img)
        
        return img

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Settings
    input_folder = "dataset/raw_images"
    output_folder = "dataset/enhanced_images"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize Enhancer
    enhancer = ImageEnhancer(clip_limit=2.5, gamma=1.1)  # Tweaked for Indian roads (usually bright/hazy)

    print(f"Processing images from {input_folder}...")

    # Loop through images
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                continue

            # ENHANCE!
            enhanced_img = enhancer.process(img)

            # Save
            cv2.imwrite(os.path.join(output_folder, filename), enhanced_img)

    print("Done! Check the enhanced_images folder.")