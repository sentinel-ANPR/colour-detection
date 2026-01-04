import os
import cv2
import numpy as np
import joblib
import json
import logging
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# 1. Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -----------------------------
# 2. Feature Extractor (Hood Crop + Turbo Features)
# -----------------------------
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return None

        h, w, _ = img.shape
        
        # --- NEW: HOOD CROP LOGIC ---
        # Instead of the dead center, we look at the "Lower Center".
        # Vertical (Y): 50% to 85% (Skips windshield/roof, stops before the road)
        # Horizontal (X): 30% to 70% (Focuses strictly on the grille/hood)
        start_y = int(h * 0.50)
        end_y = int(h * 0.85)
        start_x = int(w * 0.30)
        end_x = int(w * 0.70)
        
        crop = img[start_y:end_y, start_x:end_x]
        
        # Safety: If crop failed (image too small), use original
        if crop.size == 0: crop = img

        # --- PREPROCESSING (Same as Turbo) ---
        # 1. Blur to remove grain
        crop = cv2.GaussianBlur(crop, (5, 5), 0)

        # 2. Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        # 3. Saturation Pump (Make colors pop)
        s_channel = cv2.multiply(s_channel, 1.5)
        
        # 4. CLAHE on Value (Fix lighting on dark cars)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v_channel = clahe.apply(v_channel)

        # --- FEATURES ---
        # Histograms (Decoupled)
        hist_h = cv2.calcHist([h_channel], [0], None, [12], [0, 180])
        hist_s = cv2.calcHist([s_channel], [0], None, [4], [0, 256])
        hist_v = cv2.calcHist([v_channel], [0], None, [8], [0, 256])

        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        cv2.normalize(hist_v, hist_v)

        # Stats + Percentiles (The Highlight Hunter)
        mean_h, std_h = cv2.meanStdDev(h_channel)
        mean_s, std_s = cv2.meanStdDev(s_channel)
        mean_v, std_v = cv2.meanStdDev(v_channel)
        
        p90_s = np.percentile(s_channel, 90)
        p90_v = np.percentile(v_channel, 90)

        # Combine
        features = np.concatenate([
            hist_h.flatten(), 
            hist_s.flatten(), 
            hist_v.flatten(),
            mean_h.flatten(), std_h.flatten(),
            mean_s.flatten(), std_s.flatten(),
            mean_v.flatten(), std_v.flatten(),
            [p90_s, p90_v]
        ])
        
        return features

    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
        return None

# -----------------------------
# 3. Load Data
# -----------------------------
folders = {
    "colour_cleaning/red": "Red",
    "colour_cleaning/white": "White",
    "colour_cleaning/blue": "Blue",
    "colour_cleaning/black": "Black",
    "colour_cleaning/gray": "Gray",
}

logger.info("Starting Data Loading (Hood Crop Strategy)...")
X, y = [], []

for folder, label in folders.items():
    if not os.path.exists(folder):
        logger.warning(f"Folder not found: {folder}")
        continue

    files = os.listdir(folder)
    count = 0
    for file in files:
        if file.lower().endswith(('.png','.jpg','.jpeg')):
            img_path = os.path.join(folder, file)
            features = extract_features(img_path)

            if features is not None:
                X.append(features)
                y.append(label)
                count += 1
    logger.info(f"Loaded {count} images for class '{label}'")

if not X:
    logger.error("No data loaded.")
    exit()

# -----------------------------
# 4. Encode, Split, Scale
# -----------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Grid Search & Training
# -----------------------------
logger.info("Starting Grid Search...")

param_grid = {
    'C': [1, 10, 50, 100, 200],        
    'gamma': [0.1, 0.01, 0.001], 
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2, cv=3)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
logger.info(f"Best Parameters found: {grid.best_params_}")

# Evaluate
test_preds = best_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_preds)

logger.info(f"Validation Accuracy: {test_acc:.4f}")

# -----------------------------
# 6. Save Artifacts
# -----------------------------
output_dir = "artifacts/hood_crop"
os.makedirs(output_dir, exist_ok=True)

logger.info(f"Saving model to {output_dir}...")
joblib.dump(best_model, f"{output_dir}/svm_model.pkl")
joblib.dump(scaler, f"{output_dir}/scaler.pkl")
joblib.dump(le, f"{output_dir}/encoder.pkl")

# Log detailed metrics
logger.info("\n" + classification_report(y_test, test_preds, target_names=le.classes_))

logger.info("Training Complete!")
