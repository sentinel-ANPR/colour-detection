import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import Counter

INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_color_roi(img_rgb):
    """
    Extracts the region used for dominant color detection:
    - 5% margin on left/right
    - From 10% to 35% from bottom (with slight upward offset)
    """
    h, w, _ = img_rgb.shape

    # Horizontal crop (5% margins)
    x1 = int(0.05 * w)
    x2 = int(0.95 * w)

    # Vertical crop
    bottom_ignore = int(0.10 * h)
    sample_height = int(0.25 * h)

    # Small upward offset to avoid number plate (2–3%)
    offset_up = int(0.03 * h)

    y2 = h - bottom_ignore - offset_up
    y1 = max(0, y2 - sample_height)

    roi = img_rgb[y1:y2, x1:x2]

    return roi


def get_dominant_hex_and_rgb(image, k=3):
    # Resize for stability
    img = cv2.resize(image, (150, 150))

    # Flatten
    pixels = img.reshape(-1, 3)

    # Remove near-black pixels
    pixels = pixels[np.any(pixels > 30, axis=1)]

    if len(pixels) == 0:
        return "#000000", (0, 0, 0)

    # KMeans
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)

    dominant_label = Counter(labels).most_common(1)[0][0]
    dominant_rgb = kmeans.cluster_centers_[dominant_label].astype(int)

    hex_color = "#{:02x}{:02x}{:02x}".format(
        dominant_rgb[0], dominant_rgb[1], dominant_rgb[2]
    )

    return hex_color, tuple(dominant_rgb)


def overlay_color_info(image, hex_color, rgb_color):
    h, w, _ = image.shape

    square_size = 60
    padding = 10

    # Top-right corner
    x2 = w - padding
    x1 = x2 - square_size
    y1 = padding
    y2 = y1 + square_size

    # Ensure BGR tuple of Python ints
    bgr_color = tuple(int(c) for c in rgb_color[::-1])

    # Draw color square
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        bgr_color,
        -1
    )

    # Draw HEX text
    text_x = x1 - 150
    text_y = y1 + 40

    cv2.putText(
        image,
        hex_color,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return image


def process_images():
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file)

        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            print(f"Skipping {file} (cannot read)")
            continue

        # Convert BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # === If you already have a cropped car image, use it here ===
        cropped_car = extract_color_roi(img_rgb)

        hex_color, rgb_color = get_dominant_hex_and_rgb(cropped_car)

        # Overlay on original image
        result = overlay_color_info(img_bgr.copy(), hex_color, rgb_color)

        cv2.imwrite(output_path, result)
        print(f"Processed {file} → {hex_color}")


if __name__ == "__main__":
    process_images()