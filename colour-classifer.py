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
INPUT_FOLDER = "/opt/sentinel/extended/colour-test/input_cars"        # Folder containing your cropped car images
OUTPUT_BASE = "/opt/sentinel/extended/colour-test/sorted_output"      # Where the 7 folders will be created

# Models
YOLO_MODEL_PATH = "colour-yolo.pt"
SVM_MODEL_PATH = "svm_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

# Classes
CLASSES = ['Black', 'Blue', 'Gray', 'White', 'Red', 'Night', 'Other']

# Ensemble Weights (Used only when models agree)
W_CLS = 0.65  # YOLO Weight
W_SVM = 0.35  # SVM Weight
CONF_THRESH = 0.55  # If final score is lower, it goes to "Other"
BOOST_VAL = 0.05    # Boost added if both models agree

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
    Extracts 32 features to match the trained SVM model.
    Structure:
    - H Hist (12) + S Hist (4) + V Hist (8) = 24
    - Mean (3) + Std (3) = 6
    - P90 Sat + P90 Val = 2
    Total = 32
    """
    # 1. Hood Crop (Focus on the center-bottom to catch the hood)
    h, w, _ = image.shape
    crop = image[int(h*0.50):int(h*0.75), int(w*0.35):int(w*0.65)]
    if crop.size == 0: crop = image
    
    # 2. Internal Preprocess (Matches Training Logic)
    # We apply blur and slight saturation boost here to match training data
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Boost Saturation slightly (helps differentiate Grays vs Colors)
    s = cv2.multiply(s, 1.5)
    
    # CLAHE on Value channel (handles shadows)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)

    # 3. Features
    # Histograms
    hist_h = cv2.normalize(cv2.calcHist([h], [0], None, [12], [0, 180]), None).flatten()
    hist_s = cv2.normalize(cv2.calcHist([s], [0], None, [4], [0, 256]), None).flatten()
    hist_v = cv2.normalize(cv2.calcHist([v], [0], None, [8], [0, 256]), None).flatten() # NOTE: 8 bins here!
    
    # Statistics
    mean_h, std_h = cv2.meanStdDev(h)
    mean_s, std_s = cv2.meanStdDev(s)
    mean_v, std_v = cv2.meanStdDev(v)
    
    # 90th Percentiles (To detect highlights/peaks)
    p90_s = np.percentile(s, 90)
    p90_v = np.percentile(v, 90)

    # Concatenate all 32 features
    features = np.concatenate([
        hist_h, hist_s, hist_v, 
        mean_h.flatten(), std_h.flatten(), 
        mean_s.flatten(), std_s.flatten(), 
        mean_v.flatten(), std_v.flatten(), 
        [p90_s, p90_v]
    ])
    
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
        
        # Parse YOLO probs
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
            print(f"[WARN] YOLO gave no output for {fname}")

        # --- STEP E: Hybrid Ensemble Logic (UPDATED) ---
        
        # 1. Get Top Picks from each model
        svm_score = svm_probs_raw.max()
        # Find the single best class from YOLO
        yolo_top_class = max(yolo_probs_dict, key=yolo_probs_dict.get)
        yolo_score = yolo_probs_dict[yolo_top_class]

        final_label = "Other"
        best_conf = 0.0

        # 2. The Decision Tree
        if svm_top_class == yolo_top_class:
            # --- SCENARIO A: AGREEMENT ---
            # Both models see the same thing. Combine & Boost.
            raw_avg = (svm_score * W_SVM) + (yolo_score * W_CLS)
            best_conf = raw_avg + BOOST_VAL
            best_conf = min(best_conf, 1.0) # Cap at 1.0
            final_label = svm_top_class
            
        else:
            # --- SCENARIO B: CONFLICT ---
            diff = abs(svm_score - yolo_score)
            
            if diff < 0.10:
                # --- SUB-CASE: CLOSE CALL (Trust YOLO) ---
                # Example: SVM says Red (0.55), YOLO says Orange (0.52). 
                # Difference is small (< 10%), so we default to YOLO (usually more robust).
                final_label = yolo_top_class
                best_conf = yolo_score
                print(f"   [Conflict-Close] Trusting YOLO: {yolo_top_class} ({yolo_score:.2f}) over SVM {svm_top_class}")
            
            else:
                # --- SUB-CASE: CLEAR WINNER (Trust Highest) ---
                # Example: SVM says Blue (0.79), YOLO says Gray (0.50).
                # Difference is large (> 10%), so we trust the confident one.
                if svm_score > yolo_score:
                    final_label = svm_top_class
                    best_conf = svm_score
                    print(f"   [Conflict-Clear] SVM wins: {svm_top_class} ({svm_score:.2f})")
                else:
                    final_label = yolo_top_class
                    best_conf = yolo_score
                    print(f"   [Conflict-Clear] YOLO wins: {yolo_top_class} ({yolo_score:.2f})")

        # 3. Final Threshold Check
        if best_conf < CONF_THRESH:
            final_label = 'Other'

        # --- STEP F: Hex Extraction & Visualization ---
        roi = extract_color_roi(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        hex_code, rgb_val = get_hex_color(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

        # --- STEP G: Overlay & Save ---
        output_img = original_img.copy()
        
        # 1. Draw Hex Square (Fixed Integer Conversion)
        color_bgr = tuple(int(c) for c in rgb_val[::-1])
        cv2.rectangle(output_img, (10, 10), (70, 70), color_bgr, -1)
        cv2.rectangle(output_img, (10, 10), (70, 70), (255, 255, 255), 2) # Border

        # 2. Prepare Text Strings
        # Final Ensemble Result
        text_final = f"Final: {final_label} ({best_conf*100:.1f}%)"
        
        # Individual Model Predictions (Class + Confidence)
        text_svm   = f"SVM: {svm_top_class} ({svm_score:.2f})"
        text_yolo  = f"YOLO: {yolo_top_class} ({yolo_score:.2f})"
        text_hex   = f"Hex: {hex_code}"
        
        # 3. Draw Text (Stacked for readability)
        # Line 1: Final Decision (Green, Large)
        cv2.putText(output_img, text_final, (80, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Line 2: SVM (Cyan, Small)
        cv2.putText(output_img, text_svm, (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # Line 3: YOLO (Yellow, Small)
        cv2.putText(output_img, text_yolo, (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Line 4: Hex Code (Gray, Small)
        cv2.putText(output_img, text_hex, (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Save to correct folder
        save_dir = os.path.join(OUTPUT_BASE, final_label)
        cv2.imwrite(os.path.join(save_dir, fname), output_img)
        
        print(f"File: {fname} -> {final_label} | SVM: {svm_top_class} | YOLO: {yolo_top_class}")

    print("\n[DONE] Processing complete.")

if __name__ == "__main__":
    run_pipeline()