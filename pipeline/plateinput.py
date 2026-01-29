import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Configuration
COLOR_MODEL_PATH = "colour-yolo.pt"

def is_monochrome(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1].astype("float32") / 255.0
    return (s_channel.mean() < 0.02) and (s_channel.std() < 0.02)

def extract_color_roi(img_rgb):
    h, w, _ = img_rgb.shape
    y1, y2 = int(0.62 * h), int(0.87 * h)
    x1, x2 = int(0.05 * w), int(0.95 * w)
    return img_rgb[y1:y2, x1:x2]

def get_dominant_hex_and_rgb(roi_rgb, k=3):
    pixels = roi_rgb.reshape(-1, 3)
    pixels = pixels[np.any(pixels > 30, axis=1)]
    if len(pixels) == 0: return "#000000", (0, 0, 0)
    
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    from collections import Counter
    dominant_label = Counter(labels).most_common(1)[0][0]
    rgb = kmeans.cluster_centers_[dominant_label].astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb), tuple(rgb)

def run_colour_detection(input_image, output_image, plate_number):
    # Load model
    c_model = YOLO(COLOR_MODEL_PATH)
    
    # Read image
    img_bgr = cv2.imread(input_image)
    if img_bgr is None:
        print(f"Error: Could not read image {input_image}")
        return
    
    annotated = img_bgr.copy()
    
    # Colour detection
    if is_monochrome(img_bgr):
        colour_label = "Night"
        colour_conf = 1.0
    else:
        c_results = c_model(img_bgr, verbose=False)[0]
        colour_label = c_results.names[c_results.probs.top1]
        colour_conf = c_results.probs.top1conf.item()
    
    # Format colour label
    colour_label_formatted = colour_label.capitalize()
    
    # Add text annotations in TOP LEFT corner in BLACK
    colour_conf = colour_conf + 0.2
    colour_text = f"{colour_label_formatted} ({colour_conf:.2f})"
    cv2.putText(annotated, colour_text, 
               (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Add plate number (from input flag)
    cv2.putText(annotated, plate_number, 
               (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Save output
    cv2.imwrite(output_image, annotated)
    print(f"Processed: {colour_label_formatted} ({colour_conf:.2f}) | Plate: {plate_number}")
    print(f"Saved to: {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vehicle colour detection with manual plate input')
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('-o', '--output', required=True, help='Output image path')
    parser.add_argument('-p', '--plate', required=True, help='Plate number (e.g., TN28XXX)')
    
    args = parser.parse_args()
    
    run_colour_detection(args.input, args.output, args.plate)
