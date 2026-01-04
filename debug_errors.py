import os
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Artifacts
# -----------------------------
model = joblib.load("artifacts/hood_crop/svm_model.pkl")
scaler = joblib.load("artifacts/hood_crop/scaler.pkl")
le = joblib.load("artifacts/hood_crop/encoder.pkl")

# -----------------------------
# 2. Re-Define Feature Extractor (Must match training EXACTLY)
# -----------------------------
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return None

        h, w, _ = img.shape
        # HOOD CROP
        start_y = int(h * 0.50)
        end_y = int(h * 0.85)
        start_x = int(w * 0.30)
        end_x = int(w * 0.70)
        crop = img[start_y:end_y, start_x:end_x]
        if crop.size == 0: crop = img

        # PREPROCESSING
        crop = cv2.GaussianBlur(crop, (5, 5), 0)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        s_channel = cv2.multiply(s_channel, 1.5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v_channel = clahe.apply(v_channel)

        # FEATURES
        hist_h = cv2.calcHist([h_channel], [0], None, [12], [0, 180])
        hist_s = cv2.calcHist([s_channel], [0], None, [4], [0, 256])
        hist_v = cv2.calcHist([v_channel], [0], None, [8], [0, 256])
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        cv2.normalize(hist_v, hist_v)

        mean_h, std_h = cv2.meanStdDev(h_channel)
        mean_s, std_s = cv2.meanStdDev(s_channel)
        mean_v, std_v = cv2.meanStdDev(v_channel)
        p90_s = np.percentile(s_channel, 90)
        p90_v = np.percentile(v_channel, 90)

        features = np.concatenate([
            hist_h.flatten(), hist_s.flatten(), hist_v.flatten(),
            mean_h.flatten(), std_h.flatten(),
            mean_s.flatten(), std_s.flatten(),
            mean_v.flatten(), std_v.flatten(),
            [p90_s, p90_v]
        ])
        return features
    except Exception:
        return None

# -----------------------------
# 3. Check All Images & Print Errors
# -----------------------------
folders = {
    "colour_cleaning/black": "Black",
    "colour_cleaning/blue": "Blue",
    "colour_cleaning/gray": "Gray",
    "colour_cleaning/white": "White",
    "colour_cleaning/red": "Red"
}

print(f"{'FILENAME':<40} {'ACTUAL':<10} {'PREDICTED':<10} {'CONFIDENCE'}")
print("-" * 80)

mistakes = []

for folder, true_label in folders.items():
    if not os.path.exists(folder): continue
    
    for file in os.listdir(folder):
        if file.lower().endswith(('.png','.jpg','.jpeg')):
            img_path = os.path.join(folder, file)
            feat = extract_features(img_path)
            
            if feat is not None:
                # Scale & Predict
                feat_scaled = scaler.transform([feat])
                pred_idx = model.predict(feat_scaled)[0]
                pred_label = le.inverse_transform([pred_idx])[0]
                probs = model.predict_proba(feat_scaled)[0]
                confidence = np.max(probs) * 100

                # IF WRONG, PRINT IT
                if pred_label != true_label:
                    print(f"{file:<40} {true_label:<10} {pred_label:<10} {confidence:.1f}%")
                    mistakes.append([true_label, pred_label])

# -----------------------------
# 4. Save Confusion Matrix
# -----------------------------
if mistakes:
    print("\n--- CONFUSION SUMMARY ---")
    y_true = [m[0] for m in mistakes]
    y_pred = [m[1] for m in mistakes]
    labels = le.classes_
    
    # Simple Text Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"       {'  '.join([l[:5] for l in labels])}")
    for i, row in enumerate(cm):
        print(f"{labels[i][:5]:<5} {row}")
