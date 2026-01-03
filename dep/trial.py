import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import color
import os, glob


class VehicleColorExtractor:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

        # Reference colors in RGB (approx auto paint tones)
        # (purple anchors remain — but map to RED)
        reference_rgb = {
            "white":  (245, 245, 245),
            "black":  (25, 25, 25),
            "gray":   (130, 130, 130),

            "red":    (185, 35, 35),
            "orange": (215, 125, 45),
            "yellow": (235, 205, 50),

            "green":  (45, 145, 65),

            "blue":   (40, 70, 170),

            # mapped to RED after ΔE selection
            "purple_soft": (130, 75, 165),
            "purple_dark": (80, 40, 110),
        }

        # Precompute LAB references
        self.reference_lab = {
            name: color.rgb2lab(
                np.uint8([[rgb]]) / 255.0
            )[0][0]
            for name, rgb in reference_rgb.items()
        }

    # ----- Crop top + sides -----
    @staticmethod
    def crop_top_and_sides(img, top_percent=35, side_percent=12):
        h, w = img.shape[:2]
        top = int(h * top_percent / 100)
        side = int(w * side_percent / 100)
        return img[top:, side:w - side]

    # ----- Mask visible paint (ignore deep shadows & highlights) -----
    def mask_paint(self, bgr_img):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            (0, 0, 55),
            (179, 255, 255)
        )

        return cv2.bitwise_and(bgr_img, bgr_img, mask=mask)

    # ----- Extract dominant RGB via KMeans -----
    def get_dominant_rgb(self, img):
        masked = self.mask_paint(img)

        pixels = masked.reshape(-1, 3)
        pixels = np.array([p for p in pixels if not np.all(p == 0)])

        if len(pixels) == 0:
            return [255, 255, 255]

        pixels = pixels[:, ::-1]  # BGR → RGB

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        kmeans.fit(pixels)

        counts = np.bincount(kmeans.labels_)
        dominant = kmeans.cluster_centers_[np.argmax(counts)]

        return dominant.astype(int).tolist()

    @staticmethod
    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    # ----- LAB + ΔE classifier (purple → red; tightened red weights) -----
    def classify_lab_delta_e(self, rgb):
        rgb_norm = np.array(rgb, dtype=np.float32) / 255.0
        lab = color.rgb2lab(rgb_norm.reshape(1, 1, 3))[0][0]

        best_color = None
        best_delta = 999

        for name, ref_lab in self.reference_lab.items():
            delta = color.deltaE_ciede2000(lab, ref_lab)

            # ---- tuning biases ----

            # black stricter
            if name == "black":
                delta *= 1.12

            # white more lenient
            elif name == "white":
                delta *= 0.90

            # 🔴 RED — tightened (requires closer perceptual match)
            elif name == "red":
                delta *= 1.12

            # purple anchors → mapped to red
            # slightly stricter than before to reduce false-red
            elif name in ("purple_soft", "purple_dark"):
                delta *= 1.04

            # -----------------------

            if delta < best_delta:
                best_delta = delta
                best_color = name

        # collapse purple anchors → final output RED
        if best_color in ("purple_soft", "purple_dark"):
            best_color = "red"

        return best_color, float(best_delta)

    # ----- Full Pipeline -----
    def extract_color(self, img, top_percent=35, side_percent=12):
        cropped = self.crop_top_and_sides(img, top_percent, side_percent)

        rgb = self.get_dominant_rgb(cropped)
        hex_color = self.rgb_to_hex(rgb)

        color_name, delta_e = self.classify_lab_delta_e(rgb)

        return hex_color, color_name, rgb, delta_e, cropped


def save_processed_image(image_path, color_name, rgb_values, hex_value, delta_e, cropped_img, output_dir="output"):
    try:
        os.makedirs(output_dir, exist_ok=True)

        image = cv2.imread(image_path)
        if image is None:
            return

        overlay = image.copy()

        rgb_bgr = (int(rgb_values[2]), int(rgb_values[1]), int(rgb_values[0]))

        cv2.rectangle(overlay, (10, 10), (450, 140), rgb_bgr, -1)

        cv2.putText(overlay, f"Color: {color_name.upper()}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(overlay, f"RGB: {tuple(rgb_values)}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.putText(overlay, f"HEX: {hex_value}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.putText(overlay, f"ΔE: {delta_e:.2f}", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        name, ext = os.path.splitext(os.path.basename(image_path))

        cv2.imwrite(f"{output_dir}/{name}_cropped{ext}", cropped_img)
        cv2.imwrite(f"{output_dir}/{name}_result{ext}", overlay)

    except Exception as e:
        print(f"  Error saving processed image: {e}")


def process_images_from_input_folder():
    extractor = VehicleColorExtractor(n_clusters=3)

    input_dir = "in"
    output_dir = "output"

    if not os.path.exists(input_dir):
        print(f"'{input_dir}' folder not found — create it and add images.")
        return

    files = []
    for ext in ['*.jpg','*.jpeg','*.png','*.bmp','*.JPG','*.JPEG','*.PNG']:
        files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not files:
        print(f"No images found in '{input_dir}'")
        return

    print(f"Found {len(files)} images to process")
    print("Model: LAB + ΔE (purple mapped to RED; tightened red weights)")
    print("Cropping: 35% top + 12% per side")
    print("=" * 70)

    from collections import Counter
    results = []

    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        print(f"\n{i}. Processing: {name}")

        img = cv2.imread(path)
        if img is None:
            print("   Failed to read image")
            continue

        hex_color, color_name, rgb, delta_e, cropped = extractor.extract_color(
            img, top_percent=35, side_percent=12
        )

        print(f"   Color : {color_name.upper()}")
        print(f"   HEX   : {hex_color}")
        print(f"   RGB   : {rgb}")
        print(f"   ΔE    : {delta_e:.2f}")

        save_processed_image(path, color_name, rgb, hex_color, delta_e, cropped, output_dir)

        results.append(color_name)

    print("\nSUMMARY")
    print("=" * 70)

    counts = Counter(results)
    for color, n in counts.most_common():
        print(f"{color:10} : {n} images")

    print("\nDone! Output saved in 'output/'")


if __name__ == "__main__":
    process_images_from_input_folder()
