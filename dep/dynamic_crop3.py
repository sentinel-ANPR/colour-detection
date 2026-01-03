import cv2
import numpy as np
import os
from pathlib import Path

input_folder = "input"
output_folder = "violations"
os.makedirs(output_folder, exist_ok=True)
ANALYSIS_SIZE = (1280, 720)  # common size for analysis (width, height)


# --- Blue top ---
def detect_blue_top(hsv):
    lower = np.array([90, 50, 50])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    row_means = np.mean(mask, axis=1)
    rows_with_blue = np.where(row_means > 125)[0]
    top_crop = rows_with_blue[-1] + 5 if len(rows_with_blue) > 0 else 0
    return top_crop


# --- Bottom gray ---
def detect_bottom_gray(hsv):
    lower = np.array([0, 0, 150])
    upper = np.array([179, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)
    row_means = np.mean(mask, axis=1)
    rows_with_gray = np.where(row_means > 125)[0]
    rows_with_gray = rows_with_gray[rows_with_gray > int(0.7 * hsv.shape[0])]
    bottom_crop = rows_with_gray[0] - 5 if len(rows_with_gray) > 0 else hsv.shape[0]
    return bottom_crop


# --- Right-side gray panel ---
def detect_ui_panel(hsv):
    lower = np.array([0, 0, 150])
    upper = np.array([179, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)
    col_means = np.mean(mask, axis=0)
    cols_with_gray = np.where(col_means > 125)[0]
    cols_with_gray = cols_with_gray[cols_with_gray > int(0.6 * hsv.shape[1])]
    right_crop = cols_with_gray[0] - 5 if len(cols_with_gray) > 0 else hsv.shape[1]
    return right_crop


# --- Left gray border ---
def detect_left_side(hsv):
    lower = np.array([0, 0, 150])
    upper = np.array([179, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)
    col_means = np.mean(mask, axis=0)
    cols_with_gray = np.where(col_means > 125)[0]
    cols_with_gray = cols_with_gray[cols_with_gray < int(0.1 * hsv.shape[1])]
    left_crop = cols_with_gray[-1] + 5 if len(cols_with_gray) > 0 else 0
    return left_crop


# --- Detection Pipeline on Small Image ---
def detect_boundaries_small(small_img):
    hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
    top = detect_blue_top(hsv)
    bottom = detect_bottom_gray(hsv)
    left = detect_left_side(hsv)
    right = detect_ui_panel(hsv)
    return top, bottom, left, right


# --- Scaled Cropping ---
def crop_original(img, small_boundaries):
    """Scale crop boundaries from resized detection to original resolution."""
    h_orig, w_orig, _ = img.shape
    w_small, h_small = ANALYSIS_SIZE

    t_small, b_small, l_small, r_small = small_boundaries
    scale_x = w_orig / w_small
    scale_y = h_orig / h_small

    t = int(t_small * scale_y)
    b = int(b_small * scale_y)
    l = int(l_small * scale_x)
    r = int(r_small * scale_x)

    return img[t:b, l:r]


# --- Recursively find all image files ---
input_path = Path(input_folder)
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
image_files = []

for ext in image_extensions:
    image_files.extend(input_path.rglob(f'*{ext}'))
    image_files.extend(input_path.rglob(f'*{ext.upper()}'))

image_files = list(set(image_files))

print(f"Found {len(image_files)} images in nested folders")

violation_counter = 0

for file_path in image_files:
    filename = file_path.name
    
    img = cv2.imread(str(file_path))
    if img is None:
        print(f"Skipping invalid file: {file_path}")
        continue

    print(f"Processing: {file_path.relative_to(input_path)}")

    small_img = cv2.resize(img, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)

    small_boundaries = detect_boundaries_small(small_img)
    cropped = crop_original(img, small_boundaries)

    if cropped is None or cropped.size == 0:
        print(f"❌ Skipped: empty crop in {filename}")
        continue

    # Get original file extension
    _, ext = os.path.splitext(filename)
    
    save_path = os.path.join(output_folder, f"violation{violation_counter}{ext}")
    cv2.imwrite(save_path, cropped)
    print(f"✅ Cropped and saved: {save_path}")
    
    violation_counter += 1

print("🎯 Done! Cropped using fixed-size detection scaled to original resolution.")