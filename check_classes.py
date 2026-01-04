import joblib
from ultralytics import YOLO

# 1. Inspect SVM Classes
print("--- SVM CLASSES (from encoder.pkl) ---")
try:
    # Load the label encoder
    le = joblib.load("encoder.pkl")
    
    # .classes_ gives you the list of names in order [0, 1, 2, ...]
    svm_names = le.classes_
    print(f"Raw List: {svm_names}")
    print(f"Total: {len(svm_names)}")
    
    # Print mapping
    for i, name in enumerate(svm_names):
        print(f"  ID {i}: {name}")

except Exception as e:
    print(f"Error loading SVM encoder: {e}")

print("\n" + "="*40 + "\n")

# 2. Inspect YOLO Classes
print("--- YOLO CLASSES (from colour-yolo.pt) ---")
try:
    # Load the model
    model = YOLO("colour-yolo.pt")
    
    # .names gives a dictionary {0: 'classA', 1: 'classB'}
    yolo_names = model.names
    print(f"Raw Dict: {yolo_names}")
    
    # Print mapping
    for id, name in yolo_names.items():
        print(f"  ID {id}: {name}")

except Exception as e:
    print(f"Error loading YOLO model: {e}")