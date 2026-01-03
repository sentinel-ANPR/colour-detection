import os
import cv2
import numpy as np
import joblib
import json
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# Helper: Get top 3 LAB colors of an image
# -----------------------------
def top3_lab_colors(image_path):
    img = cv2.imread(image_path)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixels = img_lab.reshape(-1, 3)
    pixels_list = [tuple(p) for p in pixels]
    counter = Counter(pixels_list)
    top_colors = counter.most_common(3)
    
    feature = []
    for color, _ in top_colors:
        feature.extend(color)
    while len(feature) < 9:
        feature.extend([0, 0, 0])
    return feature

# -----------------------------
# Load all folders
# -----------------------------
folders = {
    "data/red_cars": "Red",
    "data/white_cars": "White",
    "data/blue_cars": "Blue",
    "data/black_cars": "Black",
    "data/grey_cars": "Grey",
    "data/other_cars": "Others"
}

X, y = [], []
for folder, label in folders.items():
    for file in os.listdir(folder):
        if file.lower().endswith(('.png','.jpg','.jpeg')):
            img_path = os.path.join(folder, file)
            X.append(top3_lab_colors(img_path))
            y.append(label)

# -----------------------------
# Encode labels, split, scale
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
# Train SVM
# -----------------------------
model = SVC(kernel='rbf', C=10, gamma=0.5, probability=True)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Save artifacts
# -----------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/svm_color_model.pkl")
joblib.dump(scaler, "artifacts/svm_color_scaler.pkl")
joblib.dump(le, "artifacts/label_encoder.pkl")

metadata = {
    "model_type": "SVM",
    "kernel": "RBF",
    "C": 10,
    "gamma": 0.5,
    "feature_description": "Top-3 most frequent LAB pixel values per image (flattened)",
    "num_features": 9,
    "color_space": "CIELAB",
    "classes": list(le.classes_),
    "train_test_split": {"test_size": 0.2, "random_state": 42, "stratified": True},
}

with open("artifacts/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# -----------------------------
# Predict on new images
# -----------------------------
new_folder = "new_data"
for file in os.listdir(new_folder):
    if file.lower().endswith(('.png','.jpg','.jpeg')):
        img_path = os.path.join(new_folder, file)
        feat = [top3_lab_colors(img_path)]
        feat_scaled = scaler.transform(feat)
        pred = model.predict(feat_scaled)
        print(f"{file}: {le.inverse_transform(pred)[0]}")
