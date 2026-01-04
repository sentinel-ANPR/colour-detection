import os
import cv2
import numpy as np
import joblib
import json
import logging
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
# 2. Feature Extractor (HSV Histogram)
# -----------------------------
def extract_features(image_path):
    """
    Extracts a color histogram from the center of the image.
    Returns a 512-dimensional vector.
    """
    try:
        img = cv2.imread(image_path)
        if img is None: 
            logger.warning(f"Could not read: {image_path}")
            return None
        
        # B. Convert to HSV (Better for color perception)
        # Hue = Color, Saturation = Intensity, Value = Brightness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # C. Calculate 3D Color Histogram
        # 8 bins for H, 8 for S, 8 for V -> 8x8x8 = 512 features
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        
        # D. Normalize (So image resolution doesn't affect values)
        cv2.normalize(hist, hist)
        
        return hist.flatten() # Returns 512 numbers

    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
        return None

# -----------------------------
# 3. Load Data
# -----------------------------
# CHECK THESE PATHS MATCH YOUR ACTUAL FOLDERS
folders = {
    "colour_cleaning/red_cars": "Red",
    "colour_cleaning/white_cars": "White",
    "colour_cleaning/blue_cars": "Blue",
    "colour_cleaning/black_cars": "Black",
    "colour_cleaning/grey_cars": "Grey",
    "colour_cleaning/other_cars": "Others"
}

logger.info("Starting Data Loading...")
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
            
            # --- Use the histogram extractor ---
            features = extract_features(img_path)
            
            if features is not None:
                X.append(features)
                y.append(label)
                count += 1
    logger.info(f"Loaded {count} images for class '{label}'")

if not X:
    logger.error("No data loaded. Check your folder paths!")
    exit()

# -----------------------------
# 4. Encode, Split, Scale
# -----------------------------
logger.info("Encoding labels...")
le = LabelEncoder()
y_enc = le.fit_transform(y)

logger.info(f"Class Mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

logger.info("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Train SVM
# -----------------------------
logger.info("Training SVM (RBF Kernel)...")
model = SVC(kernel='rbf', C=10, gamma=0.5, probability=True)
model.fit(X_train_scaled, y_train)

# Evaluate
train_acc = model.score(X_train_scaled, y_train)
test_preds = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_preds)

logger.info(f"Training Accuracy: {train_acc:.4f}")
logger.info(f"Validation Accuracy: {test_acc:.4f}")

# -----------------------------
# 6. Save Artifacts
# -----------------------------
output_dir = "/histogram/artifacts"
os.makedirs(output_dir, exist_ok=True)

logger.info(f"Saving model to {output_dir}...")
joblib.dump(model, f"{output_dir}/svm_color_model.pkl")
joblib.dump(scaler, f"{output_dir}/svm_color_scaler.pkl")
joblib.dump(le, f"{output_dir}/label_encoder.pkl")

metadata = {
    "model_type": "SVM",
    "kernel": "RBF",
    "C": 10,
    "gamma": 0.5,
    "feature_description": "HSV Histogram (8x8x8 bins, flattened to 512 features)",
    "num_features": 512,
    "color_space": "HSV",
    "classes": list(le.classes_),
    "metrics": {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    }
}

with open(f"{output_dir}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

logger.info("Training Complete!")

# -----------------------------
# 7. Predict on New Images
# -----------------------------
# Only run this part if the folder exists
# new_folder = "new_data"
# if os.path.exists(new_folder):
#     logger.info(f"Predicting on {new_folder}...")
#     for file in os.listdir(new_folder):
#         if file.lower().endswith(('.png','.jpg','.jpeg')):
#             img_path = os.path.join(new_folder, file)
            
#             # Extract features using the SAME function
#             feat = extract_features(img_path)
            
#             if feat is not None:
#                 # Must be 2D array for scaler
#                 feat_reshaped = [feat] 
#                 feat_scaled = scaler.transform(feat_reshaped)
                
#                 pred_idx = model.predict(feat_scaled)[0]
#                 pred_prob = model.predict_proba(feat_scaled)[0][pred_idx]
#                 class_name = le.inverse_transform([pred_idx])[0]
                
#                 print(f"File: {file} -> {class_name} ({pred_prob:.2f})")