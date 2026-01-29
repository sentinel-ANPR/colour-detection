import os
import cv2
import json
import math
import re
import logging
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
from ultralytics import YOLO
os.environ['FLAGS_use_mkldnn'] = 'False'  
os.environ['PADDLE_USE_ONEDNN'] = '0'
from paddleocr import PaddleOCR
from sklearn.cluster import KMeans

# ==========================================
# CONFIGURATION & LOGGING
# ==========================================
INPUT_DIR = "input"
OUTPUT_DIR = "output"
VEHICLE_MODEL_PATH = "yolov8n.pt"  
COLOR_MODEL_PATH   = "colour-yolo.pt"
PLATE_MODEL_PATH   = "license_plate_detector.pt" 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Re-configure logging to suppress Paddle's internal logs manually
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('./logs/pipeline.log', mode='w'), logging.StreamHandler()],
    force=True
)
# Suppress heavy internal logging from Paddle
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddle').setLevel(logging.ERROR)

# ==========================================
# PHASE 1: COLOR & ILLUMINATION
# ==========================================

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
    dominant_label = Counter(labels).most_common(1)[0][0]
    rgb = kmeans.cluster_centers_[dominant_label].astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb), tuple(rgb)

# ==========================================
# PHASE 2: OCR PRE-PROCESSING & RANKING
# ==========================================

def preprocess_plate(plate_bgr):
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.5)
    sharpen = cv2.addWeighted(clahe, 1.6, blur, -0.6, 0)
    
    thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, otsu = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return {
        "Sharpen": cv2.cvtColor(sharpen, cv2.COLOR_GRAY2RGB),
        "Adaptive": cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB),
        "Otsu": cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)
    }

def correct_plate_text(text, ocr_score):
    if not text or ocr_score >= 0.98: return text
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    chars = list(text)
    if len(chars) < 6: return text

    d_to_l = {"0": "O", "1": "I", "4": "A", "5": "S", "8": "B"}
    l_to_d = {"O": "0", "I": "1", "A": "4", "S": "5", "B": "8"}

    for i in range(2): 
        if chars[i].isdigit(): chars[i] = d_to_l.get(chars[i], chars[i])
    for i in range(2, 4): 
        if chars[i].isalpha(): chars[i] = l_to_d.get(chars[i], chars[i])
        
    return "".join(chars)

def is_valid_indian_plate(text):
    formats = [r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', r'^[A-Z]{2}\d{2}EV\d{4}$', r'^\d{2}BH\d{4}[A-Z]{2}$']
    return any(re.match(p, text.replace(' ','')) for p in formats)

def rank_plate_candidates(candidates):
    grouped = defaultdict(list)
    for c in candidates: grouped[c["corrected_text"]].append(c)
    ranked = []
    for plate_text, items in grouped.items():
        avg_score = sum(i["ocr_score"] for i in items) / len(items)
        f_valid = is_valid_indian_plate(plate_text)
        final_score = (5.0 if f_valid else 0.0) + (2.0 * math.sqrt(len(items))) + (4.0 * avg_score)
        ranked.append({"plate": plate_text, "final_score": round(final_score, 4), "avg_ocr_score": avg_score})
    return sorted(ranked, key=lambda x: x["final_score"], reverse=True)

# ==========================================
# PHASE 3: MAIN EXECUTION ENGINE
# ==========================================

def run_pipeline():
    # Load Models
    v_model = YOLO(VEHICLE_MODEL_PATH)
    p_model = YOLO(PLATE_MODEL_PATH)
    c_model = YOLO(COLOR_MODEL_PATH)
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(
        use_textline_orientation=True, 
        lang='en'
    )

    for filename in os.listdir(INPUT_DIR):
        img_path = os.path.join(INPUT_DIR, filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        
        annotated = img_bgr.copy()
        
        # FIX: Initialize variable with a default value at the start of the image loop
        best_plate_text = "NO_VEHICLE_DETECTED" 
        colour_label_formatted = ""
        colour_conf = 0.0
        
        v_results = v_model(img_bgr, conf=0.2, verbose=False)[0]

        # Process ONLY the first vehicle detected
        if len(v_results.boxes) > 0:
            v_box = v_results.boxes[0]  # Take only first vehicle
            best_plate_text = "NO_PLATE_DETECTED" 
            
            vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
            vehicle_crop = img_bgr[vy1:vy2, vx1:vx2]
            
            if vehicle_crop.size > 0:
                # 1. Colour Logic
                if is_monochrome(vehicle_crop):
                    color_label, hex_c, rgb_c = "Night", "#4B4B4B", (75, 75, 75)
                    colour_conf = 1.0
                else:
                    c_results = c_model(vehicle_crop, verbose=False)[0]
                    color_label = c_results.names[c_results.probs.top1]
                    colour_conf = c_results.probs.top1conf.item()
                    roi_rgb = cv2.cvtColor(extract_color_roi(vehicle_crop), cv2.COLOR_BGR2RGB)
                    hex_c, rgb_c = get_dominant_hex_and_rgb(roi_rgb)
                
                colour_label_formatted = color_label.capitalize()
                    
                # 2. Plate Logic
                p_results = p_model(vehicle_crop, conf=0.4, verbose=False)[0]

                if len(p_results.boxes) > 0:
                    px1, py1, px2, py2 = map(int, p_results.boxes[0].xyxy[0])
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    
                    if plate_crop.size > 0:
                        variants = preprocess_plate(plate_crop)
                        candidates = []
                        for v_name, v_img in variants.items():
                            res = ocr.ocr(v_img)
                            if res and res[0]:
                                raw_text = "".join([line[1][0] for line in res[0]])
                                avg_conf = sum([line[1][1] for line in res[0]]) / len(res[0])
                                candidates.append({
                                    "corrected_text": correct_plate_text(raw_text, avg_conf),
                                    "ocr_score": avg_conf
                                })
                        
                        if candidates:
                            ranked = rank_plate_candidates(candidates)
                            best_plate_text = ranked[0]['plate']

                    # Annotate ONLY the plate with green rectangle
                    plate_x1_abs = vx1 + px1
                    plate_y1_abs = vy1 + py1
                    plate_x2_abs = vx1 + px2
                    plate_y2_abs = vy1 + py2
                    cv2.rectangle(annotated, (plate_x1_abs, plate_y1_abs), 
                                 (plate_x2_abs, plate_y2_abs), (0, 255, 0), 2)

        # Add text annotations in TOP LEFT corner in BLACK
        if colour_label_formatted:
            # Line 1: Colour with confidence (top left, bigger font)
            colour_conf = colour_conf - 0.03
            colour_text = f"{colour_label_formatted} ({colour_conf:.2f})"
            cv2.putText(annotated, colour_text, 
                       (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            
            # Line 2: Plate number (below colour text)
            cv2.putText(annotated, best_plate_text, 
                       (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Save annotated image
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), annotated)
        logging.info(f"Processed {filename} -> {best_plate_text}")

if __name__ == "__main__":
    run_pipeline()
