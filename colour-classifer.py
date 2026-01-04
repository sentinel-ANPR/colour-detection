import cv2
import numpy as np
import os
import joblib
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import Counter

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = "input_cars"        # Folder containing your cropped car images
OUTPUT_BASE = "sorted_output"      # Where the 7 folders will be created

# Models
YOLO_MODEL_PATH = "colour-yolo.pt"
SVM_MODEL_PATH = "svm_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

# Classes
CLASSES = ['Black', 'Blue', 'Gray', 'White', 'Red', 'Night', 'Other']

# Ensemble Weights
W_CLS = 0.70  # YOLO Weight
W_SVM = 0.30  # SVM Weight
CONF_THRESH = 0.55  # If combined score is lower, it goes to "Other"
BOOST_VAL = 0.10    # Boost added if both models agree

# ==========================================
# 1. IMAGE PROCESSING HELPERS
# ==========================================

def adjust_gamma(image, gamma=1.2):
    """
    Simple gamma correction to brighten/dehaze without the heavy artifacts of CLAHE.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def is_monochrome(image_bgr, mean_thresh=0.03, std_thresh=0.03):
    """
    Detects if an image is Night Mode (grayscale/greenish IR).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Normalize saturation to [0,1]
    S = hsv[:, :, 1].astype("float32") / 255.0
    
    mean_s = S.mean()
    std_s = S.std()
    
    # If there is barely any color saturation variation, it's monochrome
    return (mean_s < mean_thresh) and (std_s < std_thresh)

def extract_color_roi(img_rgb):
    """
    Extracts the 'hood' area to detect dominant HEX color.
    (Avoids windshield and bumper)
    """
    h, w, _ = img_rgb.shape
    x1 = int(0.05 * w)
    x2 = int(0.95 * w)
    bottom_ignore = int(0.10 * h)
    sample_height = int(0.25 * h)
    offset_up = int(0.03 * h)
    y2 = h - bottom_ignore - offset_up
    y1 = max(0, y2 - sample_height)
    return img_rgb[y1:y2, x1:x2]

def get_hex_color(image_bgr, k=1):
    """
    Returns the dominant HEX color and RGB tuple using KMeans.
    """
    # Resize for speed
    img = cv2.resize(image_bgr, (100, 100))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)
    
    # Remove near-black pixels (shadows/tires)
    pixels = pixels[np.any(pixels > 20, axis=1)]
    
    if len(pixels) == 0: 
        return "#000000", (0,0,0)

    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
    kmeans.fit(pixels)
    
    # Get most common cluster
    counts = Counter(kmeans.labels_)
    center = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
    
    rgb = tuple(center.astype(int))
    hex_code = "#{:02x}{:02x}{:02x}".format(*rgb)
    
    return hex_code, rgb

# ==========================================
# 2. FEATURE EXTRACTION (FOR SVM)
# ==========================================
def extract_svm_features(image):
    """
    **CRITICAL**: This must match the logic used to train 'svm_model.pkl'.
    Extracts HSV histograms + Mean/Std from the hood area.
    """
    # 1. Crop to hood (bottom center) to avoid background noise
    h, w = image.shape[:2]
    crop = image[int(h*0.5):int(h*0.95), int(w*0.2):int(w*0.8)]
    if crop.size == 0: crop = image # Fallback

    img_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 2. HSV Means and Stds
    mean, std = cv2.meanStdDev(img_hsv)
    features = np.concatenate([mean, std]).flatten()

    # 3. Color Histograms
    # H (Hue): 12 bins, S (Sat): 4 bins, V (Val): 4 bins
    h_hist = cv2.calcHist([img_hsv], [0], None, [12], [0, 180])
    s_hist = cv2.calcHist([img_hsv], [1], None, [4], [0, 256])
    v_hist = cv2.calcHist([img_hsv], [2], None, [4], [0, 256])

    # Normalize histograms
    cv2.normalize(h_hist, h_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(s_hist, s_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(v_hist, v_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    features = np.concatenate([features, h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
    return features.reshape(1, -1)

# ==========================================
# 3. PIPELINE LOGIC
# ==========================================
def load_all_models():
    print("[INFO] Loading YOLO...")
    yolo = YOLO(YOLO_MODEL_PATH)
    
    print("[INFO] Loading SVM Bundle...")
    svm = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    
    return yolo, svm, scaler, encoder

def run_pipeline():
    # 1. Setup
    yolo_model, svm_model, scaler, encoder = load_all_models()
    
    # Ensure output dirs exist
    folders = ['Red', 'Blue', 'White', 'Gray', 'Black', 'Night', 'Other']
    for f in folders:
        os.makedirs(os.path.join(OUTPUT_BASE, f), exist_ok=True)

    # 2. Get Images
    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input folder {INPUT_FOLDER} not found.")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"[INFO] Found {len(files)} images. Starting classification...\n")

    for fname in files:
        fpath = os.path.join(INPUT_FOLDER, fname)
        original_img = cv2.imread(fpath)
        if original_img is None: continue

        # --- STEP A: Monochrome Check ---
        if is_monochrome(original_img):
            print(f"File: {fname} -> NIGHT MODE")
            # Save directly
            save_path = os.path.join(OUTPUT_BASE, 'Night', fname)
            cv2.imwrite(save_path, original_img)
            continue

        # --- STEP B: Enhance (Dehaze) ---
        # Only Gamma, no CLAHE (too heavy)
        enhanced_img = adjust_gamma(original_img, gamma=1.2)

        # --- STEP C: SVM Prediction ---
        # Extract features -> Scale -> Predict
        features = extract_svm_features(enhanced_img)
        features_scaled = scaler.transform(features)
        svm_probs_raw = svm_model.predict_proba(features_scaled)[0]
        
        # Map SVM probs to class names
        svm_classes = encoder.classes_
        svm_probs_dict = {cls: prob for cls, prob in zip(svm_classes, svm_probs_raw)}
        
        # Get SVM top prediction for boost logic
        svm_top_class = svm_classes[np.argmax(svm_probs_raw)]

        # --- STEP D: YOLO Prediction ---
        # Run inference
        results = yolo_model(enhanced_img, verbose=False)
        
        # Parse YOLO probs (assuming standard structure)
        # We need to map YOLO class indices to names. 
        # Check yolo_model.names to match keys with our target list.
        yolo_probs_dict = {k: 0.0 for k in folders if k != 'Night' and k != 'Other'}
        
        if results[0].probs is not None:
            # Classification model output
            for i, conf in enumerate(results[0].probs.data):
                class_name = results[0].names[i]
                # Normalize class names (Capitalize)
                class_name = class_name.capitalize() 
                if class_name in yolo_probs_dict:
                    yolo_probs_dict[class_name] = float(conf)
        else:
            # If YOLO fails/detects nothing, rely on SVM
            print(f"[WARN] YOLO gave no output for {fname}")

        # --- STEP E: Ensemble & Boost ---
        final_scores = {}
        
        # Calculate weighted average for shared classes
        for cls in yolo_probs_dict.keys():
            s_p = svm_probs_dict.get(cls, 0.0)
            y_p = yolo_probs_dict.get(cls, 0.0)
            
            score = (y_p * W_CLS) + (s_p * W_SVM)
            final_scores[cls] = score

        # Identify top class
        best_class = max(final_scores, key=final_scores.get)
        best_conf = final_scores[best_class]

        # Apply Boost if they agree
        yolo_top_class = max(yolo_probs_dict, key=yolo_probs_dict.get)
        
        if yolo_top_class == svm_top_class:
            best_conf += BOOST_VAL  # Give it a bump
            # Clamp to 1.0
            best_conf = min(best_conf, 1.0)

        # Threshold Check
        final_label = best_class
        if best_conf < CONF_THRESH:
            final_label = 'Other'

        # --- STEP F: Hex Extraction & Visualization ---
        # Extract ROI for consistent color picking
        roi = extract_color_roi(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        hex_code, rgb_val = get_hex_color(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

        # --- STEP G: Overlay & Save ---
        output_img = original_img.copy()
        
        # Draw Hex Square
        cv2.rectangle(output_img, (10, 10), (70, 70), rgb_val[::-1], -1) # RGB to BGR for cv2
        cv2.rectangle(output_img, (10, 10), (70, 70), (255, 255, 255), 2) # Border

        # Draw Text Info
        text = f"{final_label} ({best_conf*100:.1f}%)"
        text2 = f"Hex: {hex_code}"
        
        cv2.putText(output_img, text, (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output_img, text2, (80, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Save to correct folder
        save_dir = os.path.join(OUTPUT_BASE, final_label)
        cv2.imwrite(os.path.join(save_dir, fname), output_img)
        
        print(f"File: {fname} -> {final_label} (Conf: {best_conf:.2f}) [Hex: {hex_code}]")

    print("\n[DONE] Processing complete.")

if __name__ == "__main__":
    run_pipeline()