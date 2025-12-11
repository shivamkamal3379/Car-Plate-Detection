
# Required packages (uncomment for one-time installs)
#!pip install ultralytics opencv-python numpy easyocr customtkinter requests pillow

import cv2
import numpy as np
import easyocr
import time
import json
import requests
import os
import threading
import sys
import traceback
import re
from datetime import datetime
from ultralytics import YOLO  
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk






# ------------------------------------------------------------
# GLOBAL EXCEPTION LOGGER — catches every unhandled exception
# ------------------------------------------------------------
import sys
import traceback
from logger import Logger
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Logs ANY unhandled exception automatically."""

    # Manual interrupt 
    if issubclass(exc_type, KeyboardInterrupt):
        Logger.log("❗ Python script interrupted manually (Ctrl + C)")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Any other crash
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    Logger.log(f"💥 UNHANDLED EXCEPTION — APPLICATION CRASHED:\n{error_message}")


# Activate global exception logging
sys.excepthook = global_exception_handler
HEALTH_CHECK_FILE = "healthCheck.json"


def check_last_session():
    if os.path.exists(HEALTH_CHECK_FILE):
        try:
            with open(HEALTH_CHECK_FILE, "r") as f:
                data = json.load(f)

            last_alive = data.get("last_alive", 0)
            if time.time() - last_alive > 5:
                Logger.log("💥 Previous session ended unexpectedly (crash / force kill / power cut).")
        except:
            pass

    Logger.log("🚀 Application started successfully.")


check_last_session()



def write_healthCheck():
    with open(HEALTH_CHECK_FILE, "w") as f:
        json.dump({"last_alive": time.time()}, f)




# -------------------------
# LicensePlateDetector (Refactored detection logic)
# -------------------------
from logger import Logger
# bas txt file mai sare logs ke lie 
class LicensePlateDetector:
    def __init__(self,
                 car_model_path="yolov8n.pt",
                 plate_model_path="LP-detection.pt",
                 superres_model_path="EDSR_x3.pb",
                 use_superres=True,
                 burst_frame_count=5,
                 burst_frame_interval=0.08,
                 cooldown_period=20,
                 # turn this off so that to enable messages to telegram 
                 enable_telegram = False    
):

        self.camera_data = {}
        self.cap = None
        self.is_running = False
        self.detection_active = True
        self.last_detection_time = None
        self.cooldown_period = cooldown_period
        self.detection_count = 0

        self.frame_width = 0
        self.frame_height = 0
        self.total_frame_area = 0

        self.consecutive_blank_frames = 0
        self.max_blank_frames = 10


        # Load YOLO models
        print("🚀 Loading YOLO models..." , datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.car_model = YOLO(car_model_path)
        self.plate_model = YOLO(plate_model_path)

        # GPU flag (ultralytics sets device automatically)
        self.use_gpu = (self.car_model.device.type == "cuda")
        print(f"✅ YOLO loaded. Device: {self.car_model.device} {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}")

        # EasyOCR reader
        print("🔤 Initializing EasyOCR reader...")
        try:
            # Use GPU if available
            self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
            print(f"✅ EasyOCR initialized (gpu={self.use_gpu})")
        except Exception as e:
            print(f"⚠️ EasyOCR init failed, falling back to cpu: {e}")
            self.reader = easyocr.Reader(['en'], gpu=False)

        # Super-resolution
        self.enable_superres = use_superres
        self.superres_model_path = superres_model_path
        self.superres_net = None
        self.initialize_superres()

        # Detection tuning
        # gate_line_y will be set dynamically per-frame as 50% of frame height
        self.gate_line_ratio = 0.50
        self.min_car_width = 120
        self.min_coverage_trigger = 2  # percent

        # Burst capture
        self.burst_frame_count = burst_frame_count
        self.burst_frame_interval = burst_frame_interval

        # Text correction map
        self.subs_map = {
            '0': 'O', 'O': '0', '1': 'I', 'I': '1', '8': 'B', 'B': '8',
            '6': 'G', 'G': '6', '5': 'S', 'S': '5', '2': 'Z', 'Z': '2',
            '4': 'A', 'A': '4', 'U': 'V', 'V': 'U', 'Q': '0'
        }



    
    def save_detection_folder(self, frame, car_region, plate_region, plate_text, coverage_percentage):
        """
        Creates folder: Detection_#_YYYY-MM-DD_HH-MM-SS
        Saves:
          - full_detected image with timestamp  and detection count
          - closeup image with timestamp and detection  count
          - info.txt
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_name = f"Detection_{self.detection_count}_{timestamp}"
            full_name = f"full_{self.detection_count}_{timestamp}.jpg"
            closeup_name =f"closeup_{self.detection_count}_{timestamp}.jpg"

            os.makedirs(folder_name, exist_ok=True)

            # # Save full frame without any of the boxes 
            full_path = os.path.join(folder_name, full_name )
            # cv2.imwrite(full_path, frame)
            # we eed teh boxes 
            # ---- CREATE ANNOTATED COPY FOR PERMANENT SAVE ----
            annotated = frame.copy()
            
            # Draw car box
            x, y, w, h, _ = car_region
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw plate box
            if plate_region:
                px, py, pw, ph = plate_region
                cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (0, 255, 255), 2)
            
            # Add text
            cv2.putText(annotated, f"Plate: {plate_text}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save annotated frame PERMANENTLY
            cv2.imwrite(full_path, annotated)


            # Save closeup
            if plate_region:
                px, py, pw, ph = plate_region
                closeup = frame[py:py + ph, px:px + pw]
            else: 
                x, y, w, h, _ = car_region
                closeup = frame[y:y + h, x:x + w]

            closeup_path = os.path.join(folder_name, closeup_name )
            cv2.imwrite(closeup_path, closeup)

            # Save info text file
            info_path = os.path.join(folder_name, "info.txt")
            with open(info_path, "w") as f:
                f.write(f"Detection ID: {self.detection_count}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Plate: {plate_text}\n")
                f.write(f"Coverage: {coverage_percentage}%\n")
                f.write(f"Location: {self.camera_data.get('location','unknown')}\n")
            ############# print command 1#############
            print(f"📁 Saved detection folder: {folder_name}")
            Logger.log(f"📁 Saved detection folder: {folder_name}")
            return folder_name

        except Exception as e:
            print(f"❌ Error creating detection folder: {e}")
            return None

     

    def initialize_superres(self):
        if not self.enable_superres:
            return
        try:
            self.superres_net = cv2.dnn_superres.DnnSuperResImpl_create()
            self.superres_net.readModel(self.superres_model_path)
            lower = self.superres_model_path.lower()
            if "edsr" in lower:
                self.superres_net.setModel("edsr", 3)
            elif "fsrcnn" in lower:
                self.superres_net.setModel("fsrcnn", 4)
            elif "lapsrn" in lower or "lapsrn" in lower:
                self.superres_net.setModel("lapsrn", 3)
            else:
                self.superres_net.setModel("edsr", 3)
            print("✅ Super-resolution model loaded.")
        except Exception as e:
            print(f"⚠️ Could not load super-resolution model: {e}")
            
            self.superres_net = None
            self.enable_superres = False

    def enhance_with_superres(self, image):
        try:
            if self.enable_superres and self.superres_net is not None:
                Logger.log(f"enhance_with_superres function suceeded")
                return self.superres_net.upsample(image)
            return image
        except Exception as e:
            print(f"Error in super-resolution: {e}")
            ## would be show if the enhance_with supress fails
            Logger.log(f"Error in super-resolution: {e}") 
            return image

    # -------------------------
    # Config load (cam.txt)
    # -------------------------
    def load_camera_config(self):
        try:
            with open('cam.txt', 'r') as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
                # safe access
                self.camera_data = {
                    'rtsp_url': lines[0] if len(lines) > 0 else '',
                    'substream_url': lines[0] if len(lines) > 0 else '',
                    'ip': lines[1].split()[1] if len(lines) > 1 and len(lines[1].split()) > 1 else '192.168.52.160',
                    'port': lines[2].split()[1] if len(lines) > 2 and len(lines[2].split()) > 1 else '554',
                    'id': lines[3].split()[1] if len(lines) > 3 and len(lines[3].split()) > 1 else 'admin',
                    'password': lines[4].split()[1] if len(lines) > 4 and len(lines[4].split()) > 1 else '',
                    'location': lines[5].split()[1] if len(lines) > 5 and len(lines[5].split()) > 1 else 'unknown',
                    'bot_token': lines[6].split()[1] if len(lines) > 6 and len(lines[6].split()) > 1 else '',
                    'chat_id': lines[7].split()[1] if len(lines) > 7 and len(lines[7].split()) > 1 else ''
                }
            # removed  configirations that  were printing
            print("Camera configuration loaded:",  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            Logger.log(f"Camera configuration loaded:{ datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
            return True
        except Exception as e:
            print(f"Error loading camera config: {e}")
            Logger.log(f"Error loading camera config: {e}")
            return False

    # -------------------------
    
    # Blank frame detection
    # -------------------------
    def check_blank_frame(self, frame):
        if frame is None:
            return True
        if np.mean(frame) < 8:
            return True
        if np.std(frame) < 5:
            return True
        return False

    # -------------------------
    # YOLO-based car detection + NMS
    # -------------------------
    def nms(self, boxes, iou_thresh=0.4):
        """Simple NMS for xyxy boxes: boxes as [x1,y1,x2,y2,conf]"""
        if len(boxes) == 0:
            return []
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        filtered = []
        while boxes:
            chosen = boxes.pop(0)
            filtered.append(chosen)
            keep = []
            for b in boxes:
                # compute iou
                inter_x1 = max(chosen[0], b[0])
                inter_y1 = max(chosen[1], b[1])
                inter_x2 = min(chosen[2], b[2])
                inter_y2 = min(chosen[3], b[3])
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area_chosen = (chosen[2] - chosen[0]) * (chosen[3] - chosen[1])
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                denom = float(area_chosen + area_b - inter_area) if (area_chosen + area_b - inter_area) > 0 else 1.0
                iou = inter_area / denom
                if iou < iou_thresh:
                    keep.append(b)
            boxes = keep
        return filtered

                # edit 2 made conf from 0.35 to 0.55

    def detect_cars_yolo(self, frame, conf_thresh=0.55, iou_thresh=0.45, imgsz=640):
        """Return list of car boxes in [x,y,w,h,area] format"""
        boxes_out = []
        try:
            results = self.car_model.predict(source=frame, conf=conf_thresh, iou=iou_thresh, imgsz=imgsz, verbose=False)
            all_boxes = []
            if results and len(results[0].boxes) > 0:
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        # removed ids for bus and motorcycles 
                        if cls_id in [2, 7]:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            area = (x2 - x1) * (y2 - y1)
                            all_boxes.append([x1, y1, x2, y2, conf, area])
            # convert to [x1,y1,x2,y2,conf] for nms 
            nms_input = [[b[0], b[1], b[2], b[3], b[4]] for b in all_boxes]
            nmsed = self.nms(nms_input, iou_thresh=0.4)
            # convert to desired output format and filter by min area
            for x1, y1, x2, y2, conf in nmsed:
                w = x2 - x1
                h = y2 - y1
                area = w * h
                boxes_out.append((x1, y1, w, h, area))
        except Exception as e:
            print(f"Error in detect_cars_yolo: {e}")
            Logger.log(f"Error in detect_cars_yolo: {e}")
        return boxes_out

    # -------------------------
    # Plate detection with YOLO + merge
        # if boxes_out mai output 2 ta 3 aaye hai unke overlapping ko  figure out krke ye hame simple ek clean output dega
    # -------------------------
    def merge_boxes(self, boxes, iou_thresh=0.4):
        """Merge overlapping YOLO boxes that likely belong to the same plate.
           boxes: list of [x1,y1,x2,y2,conf]"""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        merged = []
        while boxes:
            base = boxes.pop(0)
            x1b, y1b, x2b, y2b, confb = base
            keep = [base]
            remove_idx = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2, conf = box
                inter_x1 = max(x1, x1b)
                inter_y1 = max(y1, y1b)
                inter_x2 = min(x2, x2b)
                inter_y2 = min(y2, y2b)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (x2b - x1b) * (y2b - y1b)
                denom = (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 1.0
                iou = inter_area / denom
                if iou > iou_thresh:
                    keep.append(box)
                    remove_idx.append(i)
            for idx in sorted(remove_idx, reverse=True):
                boxes.pop(idx)
            merged_box = [
                min(b[0] for b in keep),
                min(b[1] for b in keep),
                max(b[2] for b in keep),
                max(b[3] for b in keep),
                max(b[4] for b in keep)
            ]
            merged.append(merged_box)
            Logger.log("merge_box function executed ")
        return merged



        ############################################################################################################################

    # def detect_plates_yolo(self, frame, car_box=None, conf_thresh=0.18, iou_thresh=0.45, imgsz=960):
    #     """
    #     Run plate model either on whole frame or ROI (car_box).
    #     Returns best plate box (x,y,w,h) or None.
    #     """
    #     try:
    #         if car_box:
    #             x, y, w, h, area = car_box
    #             pad_w = int(0.12 * w)
    #             pad_h = int(0.12 * h)
    #             x1 = max(0, x - pad_w)
    #             y1 = max(0, y - pad_h)
    #             x2 = min(frame.shape[1], x + w + pad_w)
    #             y2 = min(frame.shape[0], y + h + pad_h)
    #             roi = frame[y1:y2, x1:x2]
    #             if roi is None or roi.size == 0:
    #                 return None
    #             results = self.plate_model.predict(source=roi, conf=conf_thresh, iou=iou_thresh, imgsz=imgsz, verbose=False)
    #             plate_boxes = []
    #             for r in results:
    #                 for box in r.boxes:
    #                     xA, yA, xB, yB = map(int, box.xyxy[0])
    #                     conf = float(box.conf[0].item())
    #                     abs_x1 = x1 + xA
    #                     abs_y1 = y1 + yA
    #                     abs_x2 = x1 + xB
    #                     abs_y2 = y1 + yB
    #                     plate_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2, conf])
    #         else:
    #             results = self.plate_model.predict(source=frame, conf=conf_thresh, iou=iou_thresh, imgsz=imgsz, verbose=False)
    #             plate_boxes = []
    #             for r in results:
    #                 for box in r.boxes:
    #                     x1, y1, x2, y2 = map(int, box.xyxy[0])
    #                     conf = float(box.conf[0].item())
    #                     plate_boxes.append([x1, y1, x2, y2, conf])

    #         if not plate_boxes:
    #             return None

    #         merged = self.merge_boxes(plate_boxes, iou_thresh=0.35)

    #         best = None
    #         for (x1m, y1m, x2m, y2m, confm) in merged:
    #             w_m = x2m - x1m
    #             h_m = y2m - y1m
    #             if h_m <= 0:
    #                 continue
    #             ar = w_m / float(h_m)
    #             if 1.0 <= ar <= 7.0 and w_m > 30 and h_m > 10:
    #                 best = (x1m, y1m, w_m, h_m)
    #                 break

    #         if best is None:
    #             merged_sorted = sorted(merged, key=lambda b: b[4], reverse=True)
    #             if merged_sorted:
    #                 x1m, y1m, x2m, y2m, confm = merged_sorted[0]
    #                 best = (x1m, y1m, x2m - x1m, y2m - y1m)
    #                 # # teh best is returned with a yellow rectangle
    #                 # cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 255), 2)

    #         return best
    #     except Exception as e:
    #         print(f"Error in detect_plates_yolo: {e}")
    #         Logger.log(f"Error in detect_plates_yolo: {e}")
    #         return None









    def detect_plates_yolo(self, frame, car_box=None, conf_thresh=0.50, iou_thresh=0.40, imgsz=960):
        """
        NEW — Static script style plate detection
        Highest confidence box only — NO MERGING
        Returns: (x, y, w, h) or None
        """
    
        try:
            # ROI agar car_box provide hai
            if car_box:
                x, y, w, h, area = car_box
    
                # Little padding 
                pad_w = int(0.12 * w)
                pad_h = int(0.12 * h)
    
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(frame.shape[1], x + w + pad_w)
                y2 = min(frame.shape[0], y + h + pad_h)
    
                roi = frame[y1:y2, x1:x2]
                if roi is None or roi.size == 0:
                    return None
    
                results = self.plate_model.predict(
                    source=roi,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    imgsz=imgsz,
                    verbose=False
                )
            else:
                results = self.plate_model.predict(
                    source=frame,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    imgsz=imgsz,
                    verbose=False
                )
                roi = frame
                x1, y1 = 0, 0
    
            # Collect raw boxes
            final_boxes = []
            for r in results:
                for box in r.boxes:
                    xA, yA, xB, yB = map(int, box.xyxy[0])
                    conf = float(box.conf[0].item())
    
                    # Convert to absolute coords if ROI used
                    abs_x1 = xA + x1
                    abs_y1 = yA + y1
                    abs_x2 = xB + x1
                    abs_y2 = yB + y1
    
                    final_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2, conf])
    
            if not final_boxes:
                return None
    
            # Pick HIGHEST CONFIDENCE box 
            best_box = max(final_boxes, key=lambda b: b[4])
    
            bx1, by1, bx2, by2, conf = best_box
            w = bx2 - bx1
            h = by2 - by1
    
            # Return (x, y, w, h)
            return (bx1, by1, w, h)
    
        except Exception as e:
            Logger.log(f"Error detect_plates_yolo: {e}")
            return None




    

    # -------------------------
    # Preprocess plate image and OCR using EasyOCR (with multiple attempts)
    # -------------------------
    def preprocess_plate_image(self, plate_image):
        try:
            if plate_image is None or plate_image.size == 0:
                return []

            # Try super-resolution first if enabled
            try:
                if self.enable_superres and self.superres_net is not None:
                    plate_image = self.superres_net.upsample(plate_image)
            except Exception as e:
                print(f"Super-res skipped: {e}")
                Logger.log(f"Super-res skipped: {e}")

            # Ensure color -> gray
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

            # Resize (scale up) to improve OCR
            gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

            # Denoise + sharpen + contrast
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.equalizeHist(gray)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=15)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            gray = cv2.filter2D(gray, -1, kernel)

            # Binary versions
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

            return [gray, thresh, adaptive]
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            Logger.log(f"Error in image preprocessing: {e}")
            return []

    def correct_text(self, text):
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        corrected = ''.join([self.subs_map.get(ch, ch) for ch in text])
        # conservative pattern for Indian plates - lenient in case of partial reads
        pattern = r'[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}'
        if not re.fullmatch(pattern, corrected):
            # try replacements again
            for k, v in self.subs_map.items():
                corrected = corrected.replace(k, v)
        return corrected





    ################################################################################################################

    # def extract_text_from_plate(self, plate_image):
    #     try:
    #         if plate_image is None or plate_image.size == 0:
    #             return None
    #         preprocessed_imgs = self.preprocess_plate_image(plate_image)
    #         if not preprocessed_imgs:
    #             return None

    #         text_found = None
    #         # Try multiple preprocessed images
    #         for ocr_img in preprocessed_imgs:
    #             try:
    #                 ocr_result = self.reader.readtext(ocr_img, detail=0, paragraph=False)
    #             except Exception as e:
    #                 try:
    #                     ocr_result = self.reader.readtext(cv2.cvtColor(ocr_img, cv2.COLOR_GRAY2BGR), detail=0, paragraph=False)
    #                 except Exception:
    #                     ocr_result = []
    #             if ocr_result:
    #                 text_found = max(ocr_result, key=len)
    #                 break

    #         if not text_found:
    #             return None

    #         text_clean = self.correct_text(text_found)
    #         if 2 <= len(text_clean) <= 15:
    #             Logger.log("text extracted from the plate ")
    #             return text_clean
    #         return None
    #     except Exception as e:
    #         print(f"Error in text extraction: {e}")
    #         Logger.log(f"Error in text extraction: {e}")
    #         return None









    def extract_text_from_plate(self, plate_image):
        """
        NEW — Gurjeet sir  style OCR + preprocessing with 
        EasyOCR , hardcoded  threshold, no multiple attempts
        """
    
        try:
            if plate_image is None or plate_image.size == 0:
                return None
    
            # 1) Grayscale
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
            # 2) Bilateral filtering 
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
            # 3) Otsu threshold
            _, thresh = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
    
            # 4) EasyOCR expects BGR/RGB → converting  GRAY → RGB
            thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
            # 5) EasyOCR read
            ocr_result = self.reader.readtext(thresh_rgb, detail=0)
    
            if not ocr_result:
                return None
    
            # Take LARGEST text 
            raw_text = max(ocr_result, key=len)
    
            # 6) Clean text 
            text_clean = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    
            # 7) Indian number plate regex
            if re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$', text_clean):
                Logger.log(f"Text extracted: {text_clean}")
                return text_clean
    
            return text_clean  # Allow partial text also
    
        except Exception as e:
            Logger.log(f"Error OCR: {e}")
            return None




        # -------------------------
    # Save images for Telegram ONLY (temporary)
    # -------------------------
    def save_detection_images(self, frame, car_region, plate_region, plate_text, coverage_percentage):
        try:
            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            x, y, w, h, area = car_region

            # =====================================================
            # 1) CREATE ANNOTATED IMAGE (Car + Plate + Text Overlay)
            # =====================================================
            annotated = frame.copy()

            # Car box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Plate box
            if plate_region:
                px, py, pw, ph = plate_region
                cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (0, 255, 0), 2)

            # Text
            cv2.putText(annotated, f"Plate: {plate_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(annotated, f"Coverage: {coverage_percentage}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # =====================================================
            # 2) FILENAME FOR TELEGRAM TEMPORARY IMAGES
            # =====================================================
            full_name = f"full_{self.detection_count}_{timestamp}.jpg"
            closeup_name = f"closeup_{self.detection_count}_{timestamp}.jpg"

            # Save annotated image
            cv2.imwrite(full_name, annotated)

            # =====================================================
            # 3) SAVE CLOSEUP IMAGE
            # =====================================================
            if plate_region:
                px, py, pw, ph = plate_region
                closeup = frame[py:py + ph, px:px + pw]
            else:
                closeup = frame[y:y + h, x:x + w]

            # In case of invalid closeup
            if closeup is None or closeup.size == 0:
                closeup = frame[y:y + h, x:x + w]

            cv2.imwrite(closeup_name, closeup)

            # =====================================================
            # 4) RETURN ABSOLUTE PATHS FOR TELEGRAM
            # =====================================================
            return os.path.abspath(full_name), os.path.abspath(closeup_name)

        except Exception as e:
            Logger.log(f"❌ Error saving telegram images: {e}")
            return None, None


    def send_to_telegram(self, full_image_path, closeup_image_path, plate_text, coverage_percentage):
        if not self.enable_telegram:
            print("🚫 Telegram sending disabled temporarily.")
            Logger.log("🚫 Telegram sending disabled temporarily.")
            return False
    
        try:
            bot = self.camera_data.get('bot_token')
            chat = self.camera_data.get('chat_id')
    
            if not bot or not chat:
                Logger.log("Telegram bot token or chat id missing.")
                return False
    
            url = f"https://api.telegram.org/bot{bot}/sendMediaGroup"
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
            caption = (
                f"🚙 Detection #{self.detection_count}\n"
                f"📍 {self.camera_data.get('location','-')}\n"
                f"📅 {current_time}\n"
                f"📏 Coverage: {coverage_percentage}%\n"
                f"🔢 Plate: `{plate_text}`"
            )
    
            #  this will look for full.jpg everythime but full.jpg is mapped for the actual media name
            media = [
                {'type': 'photo', 'media': 'attach://full.jpg', 'caption': caption, 'parse_mode': 'Markdown'},
                {'type': 'photo', 'media': 'attach://closeup.jpg'}
            ]
    
            #  same isi mai bhi mapping aise hi kari hui hai 
            files = {
                'full.jpg': open(full_image_path, 'rb'),
                'closeup.jpg': open(closeup_image_path, 'rb')
            }
    
            data = {'chat_id': chat, 'media': json.dumps(media)}
            response = requests.post(url, files=files, data=data)
    
            for f in files.values():
                f.close()
    
            if response.status_code == 200:
                Logger.log(f"✅ Alert sent: {plate_text}")
                return True
            else:
                Logger.log(f"❌ Telegram error: {response.text}")
                return False
    
        except Exception as e:
            Logger.log(f"❌ Error sending to Telegram: {e}")
            return False
    


    
    # -------------------------
    # Burst capture
    # -------------------------
    def capture_burst_frames(self, cap, count=5, interval=0.08):
        frames = []
        try:
            for _ in range(count):
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(f.copy())
                time.sleep(interval)
                Logger.log(f"📸 Captured burst frame {len(frames)}"   )
            return frames
        except Exception as e:
            print(f"⚠️ Burst capture failed: {e}")
            Logger.log(f"⚠️ Burst capture failed: {e}")
            return frames

    # -------------------------
    # Main per-frame processing (YOLO core + gate-line trigger)
    # -------------------------
    # def process_frame(self, frame):
    #     if not self.detection_active:
    #         return frame, None, None, None, 0

    #     current_time = datetime.now()
    #     # ---- 10-second cooldown ----
    #     if self.last_detection_time:
    #         diff = (current_time - self.last_detection_time).total_seconds()
    #         if diff < self.cooldown_period:
    #             return frame, None, None, None, 0

    #     # if self.last_detection_time and (current_time - self.last_detection_time).seconds < self.cooldown_period:
    #     #     return frame, None, None, None, 0

    #     try:
    #         self.frame_height, self.frame_width = frame.shape[:2]
    #         self.total_frame_area = self.frame_width * self.frame_height

    #         # dynamic gate line (60% of frame height)
    #         self.gate_line_y = int(self.frame_height * self.gate_line_ratio)

    #         if self.check_blank_frame(frame):
    #             self.consecutive_blank_frames += 1
    #             if self.consecutive_blank_frames >= self.max_blank_frames:
    #                 raise Exception("Camera stream stuck/blank")

    #             return frame, None, None, None, 0
    #         else:
    #             self.consecutive_blank_frames = 0

    #         # 1) detect cars using YOLO + NMS
    #         car_boxes = self.detect_cars_yolo(frame)
    #         # annotate for debug gate-line
    #         cv2.line(frame, (0, self.gate_line_y), (self.frame_width, self.gate_line_y), (255, 0, 0), 3)
    #         # define new lines
    #         self.upper_line_y = int(self.frame_height * 0.30)
    #       elf.lower_line_y = int(self.frame_height * 0.70)


    #         for car_box in car_boxes:
    #             x, y, w, h, area = car_box
    #             coverage_percentage = round((area / float(self.total_frame_area)) * 100, 2) if self.total_frame_area > 0 else 0
    #             car_center_y = y + h // 2

    #             # Gate-line trigger or large car or coverage threshold
    #             # ---- STRICT GATE-LINE TRIGGER ----
    #             car_center_y = y + h // 2
                
    #             if car_center_y > self.gate_line_y:
    #                 print("🚗 Gate-line crossed")
    #                 Logger.log("🚗 Gate-line crossed")
    #                 # final frame pr humne draw ki a  green rectangle 
    #                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            
    #                 self.last_detection_time = current_time
    #                 self.detection_count += 1
                
    #                 # detect plate
    #                 plate_region = self.detect_plates_yolo(frame, (x, y, w, h, area))
    #                 plate_text = None
    #                 if plate_region:
    #                     px, py, pw, ph = plate_region
    #                     plate_image = frame[py:py + ph, px:px + pw]
    #                     plate_text = self.extract_text_from_plate(plate_image)
                
    #                 # burst frames
    #                 if self.cap and self.cap.isOpened():
    #                     burst = self.capture_burst_frames(self.cap, self.burst_frame_count, self.burst_frame_interval)
    #                 else:
    #                     burst = [frame]
                
    #                 best_frame = max(
    #                     burst,
    #                     key=lambda f: cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    #                 )
                
    #                 threading.Thread(
    #                     target=self.handle_detection,
    #                     args=(best_frame.copy(), (x, y, w, h, area), plate_region, plate_text, 0),
    #                     daemon=True
    #                 ).start()
                
    #                 return frame, (x, y, w, h, area), plate_region, plate_text, 0


                
              
    #             else:
    #                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #                 cv2.putText(frame, f"Too far ({int(coverage_percentage)}%)", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    #     except Exception as e:
    #         print(f"Error processing frame: {e}")
    #         raise e

    #     return frame, None, None, None, 0
    


    def process_frame(self, frame):
        if not self.detection_active:
            return frame, None, None, None, 0

        current_time = datetime.now()
        
        # ---- Cooldown logic (unchanged) ----
        if self.last_detection_time:
            diff = (current_time - self.last_detection_time).total_seconds()
            if diff < self.cooldown_period:
                return frame, None, None, None, 0

        try:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.total_frame_area = self.frame_width * self.frame_height

            # dynamic gate line (60% of frame height)
            self.gate_line_y = int(self.frame_height * self.gate_line_ratio)

            # NEW LINES (Upper + Lower)
            self.upper_line_y = int(self.frame_height * 0.30)
            self.lower_line_y = int(self.frame_height * 0.70)

            # Draw new lines
            cv2.line(frame, (0, self.upper_line_y), (self.frame_width, self.upper_line_y), (0, 255, 255), 2)
            cv2.line(frame, (0, self.lower_line_y), (self.frame_width, self.lower_line_y), (0, 255, 255), 2)

            # Draw existing gate-line (DO NOT CHANGE)
            cv2.line(frame, (0, self.gate_line_y), (self.frame_width, self.gate_line_y), (255, 0, 0), 3)

            # blank frame logic
            if self.check_blank_frame(frame):
                self.consecutive_blank_frames += 1
                if self.consecutive_blank_frames >= self.max_blank_frames:
                    raise Exception("Camera stream stuck/blank")
                return frame, None, None, None, 0
            else:
                self.consecutive_blank_frames = 0

            # detect cars
            car_boxes = self.detect_cars_yolo(frame)

            # ---------- PROCESS EACH CAR ----------
            for car_box in car_boxes:
                x, y, w, h, area = car_box
                coverage_percentage = round((area / float(self.total_frame_area)) * 100, 2)
                car_center_y = y + h // 2

                # ------------------------------
                # NEW MULTI-LINE CROSSING LOGIC
                # ------------------------------

                # UPPER LINE
                if self.upper_line_y < car_center_y < self.gate_line_y:
                    Logger.log("🚗 Car crossed UPPER LINE")

                # LOWER LINE
                if self.gate_line_y < car_center_y < self.lower_line_y:
                    Logger.log("🚗 Car crossed LOWER LINE")

                # ---------------------------------------
                # EXISTING GATE-LINE TRIGGER (unchanged)
                # ---------------------------------------
                if car_center_y > self.gate_line_y:
                    Logger.log("🚗 Gate-line crossed")
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                    # ORIGINAL detection logic (unchanged)
                    self.last_detection_time = current_time
                    self.detection_count += 1

                    # detect plate
                    plate_region = self.detect_plates_yolo(frame, (x, y, w, h, area))
                    plate_text = None
                    if plate_region:
                        px, py, pw, ph = plate_region
                        plate_image = frame[py:py + ph, px:px + pw]
                        plate_text = self.extract_text_from_plate(plate_image)

                    # burst frames
                    if self.cap and self.cap.isOpened():
                        burst = self.capture_burst_frames(self.cap, self.burst_frame_count, self.burst_frame_interval)
                    else:
                        burst = [frame]

                    best_frame = max(
                        burst,
                        key=lambda f: cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    )

                    threading.Thread(
                        target=self.handle_detection,
                        args=(best_frame.copy(), (x, y, w, h, area), plate_region, plate_text, 0),
                        daemon=True
                    ).start()

                    return frame, (x, y, w, h, area), plate_region, plate_text, coverage_percentage

                # Cars above gate-line (too far)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"Too far ({int(coverage_percentage)}%)", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        except Exception as e:
            print(f"Error processing frame: {e}")
            raise e

        return frame, None, None, None, 0

    def handle_detection(self, frame, car_region, plate_region, plate_text, coverage_percentage):
        try:
            print(f"🎯 Detected (coverage {coverage_percentage}%) | Plate: {plate_text or 'NOT DETECTED'}")
            Logger.log(f"🎯 Detected (coverage {coverage_percentage}%) | Plate: {plate_text or 'NOT DETECTED'}")
    
            # ---- OLD BEHAVIOR: Save detection images (for Telegram) ----
            full_image_path, closeup_image_path = self.save_detection_images(
                frame, car_region, plate_region, plate_text or "NOT DETECTED", coverage_percentage
            )
    
            # ---- that will  Create unique detection folder ----
            self.save_detection_folder(
                frame,
                car_region,
                plate_region,
                plate_text or "NOT DETECTED",
                coverage_percentage
            )
    
            # ---- Send to Telegram ----
            if full_image_path and closeup_image_path:
                sent = self.send_to_telegram(
                    full_image_path,
                    closeup_image_path,
                    plate_text or "NOT DETECTED",
                    coverage_percentage
                )
                if sent:
                    print(f"✅ Detection #{self.detection_count} processed.")
                    Logger.log("Sending Messages To Telegram Initiated")
                    Logger.log(f"✅ Detection #{self.detection_count} processed.")
                else:
                    print(f"❌ Detection #{self.detection_count} send failed.")
                    Logger.log(f"❌ Detection #{self.detection_count} send failed.")
    
            # ---- Delete temporary telegram images ----
            try:
                if full_image_path:
                    os.remove(full_image_path)
                if closeup_image_path:
                    os.remove(closeup_image_path)
            except Exception as e:
                print(f"Warning: could not delete temp images: {e}")
    
        except Exception as e:
            print(f"❌ Error handling detection: {e}")
            Logger.log(f"❌ Error handling detection: {e}")

    # def handle_detection(self, frame, car_region, plate_region, plate_text, coverage_percentage):
    #     try:
    #         print(f"🎯 Detected (coverage {coverage_percentage}%) | Plate: {plate_text or 'NOT DETECTED'}")
    #         full_image_path, closeup_image_path = self.save_detection_images(frame, car_region, plate_region, plate_text or "NOT DETECTED", coverage_percentage)
    #         if full_image_path and closeup_image_path:
    #             sent = self.send_to_telegram(full_image_path, closeup_image_path, plate_text or "NOT DETECTED", coverage_percentage)
    #             if sent:
    #                 print(f"✅ Detection #{self.detection_count} processed.")
    #             else:
    #                 print(f"❌ Detection #{self.detection_count} send failed.")
    #             # cleanup
    #             try:
    #                 os.remove(full_image_path)
    #                 os.remove(closeup_image_path)
    #             except Exception as e:
    #                 print(f"Warning: could not delete temp images: {e}")
    #     except Exception as e:
    #         print(f"❌ Error handling detection: {e}")

# # -------------------------
# # GUI (LicensePlateGUI) - unchanged visually, uses new detector underneath
# # -------------------------
# from logger import Logger
# class LicensePlateGUI:
#     def __init__(self, root):
#         ctk.set_appearance_mode("Dark")
#         ctk.set_default_color_theme("blue")

#         self.root = root
#         self.root.title("🚗 Car Detection Dashboard")
#         self.root.geometry("720x480")
#         self.root.resizable(False, False)
#         self.root.attributes('-topmost', True)

#         # Detector backend
#         self.detector = LicensePlateDetector()
#         # REGISTER LOGGER CALLBACK HERE
#         Logger.set_gui_callback(self.update_log_label)

#         self.car_count = 0
#         self.setup_gui()
#         self.start_detection()
#         # self.detection_active= False

#     def setup_gui(self):
#         self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
#         self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

#         header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
#         header_frame.pack(fill="x", pady=(5, 0))
#         self.title_label = ctk.CTkLabel(header_frame, text="Car Detection Live Stream", font=ctk.CTkFont(size=18, weight="bold"))
#         self.title_label.pack(side="left", padx=10)
#         self.theme_switch = ctk.CTkSwitch(header_frame, text="Light Mode", command=self.toggle_theme)
#         self.theme_switch.pack(side="right", padx=10)

#         video_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
#         video_frame.pack(fill="both", expand=True, pady=10, padx=10)

#         self.counter_label = ctk.CTkLabel(video_frame, text="Cars: 0", font=ctk.CTkFont(size=13, weight="bold"),
#                                           fg_color="#1E88E5", text_color="white", corner_radius=8, padx=10, pady=3)
#         self.counter_label.place(x=10, y=10)

#         self.video_label = ctk.CTkLabel(video_frame, text="Camera feed offline.\nPress Start to begin.",
#                                         font=ctk.CTkFont(size=13), height=280, corner_radius=8,
#                                         fg_color=("gray90", "#2b2b2b"), justify="center")
#         self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

#         controls_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
#         controls_frame.pack(fill="x", pady=(0, 5))

#         self.start_btn = ctk.CTkButton(controls_frame, text="Start", width=80, command=self.start_detection)
#         # self.start_btn =ctk.CtkButton(master=root, text="Start Detection" , command=self.start_detection)

#         self.start_btn.pack(side="left", padx=5)
#         self.pause_btn = ctk.CTkButton(controls_frame, text="Pause", width=80, command=self.toggle_detection)
#         self.pause_btn.pack(side="left", padx=5)
#         self.stop_btn = ctk.CTkButton(controls_frame, text="Stop", width=80, command=self.stop_detection, state="disabled")
#         self.stop_btn.pack(side="left", padx=5)

#         bottom_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
#         bottom_frame.pack(fill="x", pady=(5, 0))

#         self.status_label = ctk.CTkLabel(bottom_frame, text="Status: Idle", text_color="orange", font=ctk.CTkFont(size=12, weight="bold"))
#         self.status_label.pack(side="left", padx=10)
#         self.log_label = ctk.CTkLabel(bottom_frame, text="Ready - Integrated YOLO + EasyOCR", font=ctk.CTkFont(size=11), text_color="gray")
#         self.log_label.pack(side="right", padx=10)




        
#     ####
#     ###
#     ##
#     #         this is the place where the GUI  also updates 
#     ##
#     ###
#     ####
    
#     def update_log_label(self, msg):
#     # Update GUI label/text field
#         try:
#             self.log_label.configure(text=msg)
#         except:
#             pass
    

#     def toggle_theme(self):
#         if self.theme_switch.get():
#             ctk.set_appearance_mode("Light")
#             self.theme_switch.configure(text="Dark Mode")
#         else:
#             ctk.set_appearance_mode("Dark")
#             self.theme_switch.configure(text="Light Mode")

#     # def log_message(self, message):
#     #     short_msg = message[:60] + "..." if len(message) > 60 else message
#     #     self.log_label.configure(text=short_msg)
#     #     self.root.update()


    
#     # def log_message(self, message):
#     #     # 1) Show message in GUI 
#     #     short_msg = message[:60] + "..." if len(message) > 60 else message
#     #     self.log_label.configure(text=short_msg)
#     #     self.root.update()
    
#     #     # 2) Also save message into a log file 
#     #     try:
#     #         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     #         with open("Functionality_Logs.txt", "a", encoding="utf-8") as f:
#     #             f.write(f"[{timestamp}] {message}\n")
#     #     except Exception as e:
#     #         print(f"Log write failed: {e}")
    


#     def start_detection(self):
#         try:
#             if not self.detector.load_camera_config():
#                 Logger.log("❌ Config load failed")
#                 return

#             Logger.log("📋 Loading camera...")

#             substream = self.detector.camera_data.get('substream_url') or self.detector.camera_data.get('rtsp_url')
#             self.detector.cap = cv2.VideoCapture(substream)
#             self.detector.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#             self.detector.cap.set(cv2.CAP_PROP_FPS, 18)
#             self.detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#             self.detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

#             if not self.detector.cap.isOpened():
#                 Logger.log("❌ Camera connect failed")
#                 return

#             self.detector.is_running = True
#             self.start_btn.configure(state="disabled")
#             self.stop_btn.configure(state="normal")
#             self.status_label.configure(text="Status: Running", text_color="green")
#             Logger.log("✅ System started - Hybrid YOLO + EasyOCR (gate-line preserved)")

#             # attach cap to detector for burst capture
#             self.detector.cap = self.detector.cap
#             self.process_video()
#         except Exception as e:
#             Logger.log(f"❌ Start error: {str(e)[:60]}")

#     def stop_detection(self):
#         self.detector.is_running = False
#         if self.detector.cap:
#             try:
#                 self.detector.cap.release()
#             except Exception:
#                 pass
#             self.detector.cap = None

#         self.start_btn.configure(state="normal")
#         self.stop_btn.configure(state="disabled")
#         self.status_label.configure(text="Status: Stopped", text_color="red")
#         self.video_label.configure(image='', text="Camera feed offline.\nPress Start to begin.")
#         Logger.log("⏹ System stopped")

#     def toggle_detection(self):
#         self.detector.detection_active = not self.detector.detection_active
#         status = "ACTIVE" if self.detector.detection_active else "PAUSED"
#         color = "green" if self.detector.detection_active else "orange"
#         self.pause_btn.configure(text="Resume" if not self.detector.detection_active else "Pause")
#         self.status_label.configure(text=f"Status: {status}", text_color=color)
#         Logger.log(f"⏸ Detection {status.lower()}")

#     def process_video(self):
#         if not self.detector.is_running or not self.detector.cap:
#             return

#         try:
#             ret, frame = self.detector.cap.read()
#             if ret:
#                 processed_frame, car_region, plate_region, plate_text, coverage = self.detector.process_frame(frame)

#                 if car_region:
#                     self.car_count += 1
#                     self.counter_label.configure(text=f"Cars: {self.car_count}")

#                 rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
#                 img = Image.fromarray(rgb_frame)
#                 img = img.resize((640, 360), Image.Resampling.LANCZOS)
#                 img_tk = ImageTk.PhotoImage(image=img)

#                 self.video_label.configure(image=img_tk, text="")
#                 self.video_label.image = img_tk
#             else:
#                 Logger.log("⚠️ Frame read failed - reconnecting")
#                 self.root.after(1000, self.reconnect_camera)
#                 return

#         except Exception as e:
#             Logger.log(f"❌ Video error: {str(e)[:60]}")
#             self.root.after(1000, self.reconnect_camera)
#             return

#         if self.detector.is_running:
#             self.root.after(55, self.process_video)

#     def reconnect_camera(self):
#         Logger.log("🔄 Reconnecting...")
#         self.stop_detection()
#         self.root.after(2000, self.start_detection)



# -------------------------
from logger import Logger
import time
import threading

class LicensePlateGUI:
    def __init__(self, root):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.root = root
        self.root.title("🚗 Car Detection Dashboard")
        self.root.geometry("720x480")
        self.root.resizable(False, False)
        self.root.attributes('-topmost', True)

        # Detector backend
        self.detector = LicensePlateDetector()
        Logger.set_gui_callback(self.update_log_label)

        self.car_count = 0
        self.setup_gui()

        # ---------------------------
        # 🔧 CHANGE: Start detection safely
        # ---------------------------
        self.start_detection()

        # ---------------------------
        # 🔧 CHANGE: Start watchdog (auto restart every 60 sec)
        # ---------------------------
        self.watchdog_thread = threading.Thread(target=self.watchdog_loop, daemon=True)
        self.watchdog_thread.start()



    # -------------------------------------------------------
    # GUI Layout (UNCHANGED)
    # -------------------------------------------------------
    def setup_gui(self):
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(5, 0))
        self.title_label = ctk.CTkLabel(header_frame, text="Car Detection Live Stream", font=ctk.CTkFont(size=18, weight="bold"))
        self.title_label.pack(side="left", padx=10)
        self.theme_switch = ctk.CTkSwitch(header_frame, text="Light Mode", command=self.toggle_theme)
        self.theme_switch.pack(side="right", padx=10)

        video_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        video_frame.pack(fill="both", expand=True, pady=10, padx=10)

        self.counter_label = ctk.CTkLabel(video_frame, text="Cars: 0", font=ctk.CTkFont(size=13, weight="bold"),
                                          fg_color="#1E88E5", text_color="white", corner_radius=8, padx=10, pady=3)
        self.counter_label.place(x=10, y=10)

        self.video_label = ctk.CTkLabel(video_frame, text="Camera feed offline.\nPress Start to begin.",
                                        font=ctk.CTkFont(size=13), height=280, corner_radius=8,
                                        fg_color=("gray90", "#2b2b2b"), justify="center")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        controls_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        controls_frame.pack(fill="x", pady=(0, 5))

        self.start_btn = ctk.CTkButton(controls_frame, text="Start", width=80, command=self.start_detection)
        self.start_btn.pack(side="left", padx=5)

        self.pause_btn = ctk.CTkButton(controls_frame, text="Pause", width=80, command=self.toggle_detection)
        self.pause_btn.pack(side="left", padx=5)

        self.stop_btn = ctk.CTkButton(controls_frame, text="Stop", width=80, command=self.stop_detection, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        bottom_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        bottom_frame.pack(fill="x", pady=(5, 0))

        self.status_label = ctk.CTkLabel(bottom_frame, text="Status: Idle", text_color="orange", font=ctk.CTkFont(size=12, weight="bold"))
        self.status_label.pack(side="left", padx=10)

        self.log_label = ctk.CTkLabel(bottom_frame, text="Ready - Integrated YOLO + EasyOCR", font=ctk.CTkFont(size=11), text_color="gray")
        self.log_label.pack(side="right", padx=10)



    # -------------------------------------------------------
    # Logger → GUI text updater
    # -------------------------------------------------------
    def update_log_label(self, msg):
        try:
            self.log_label.configure(text=msg)
        except:
            pass



    def toggle_theme(self):
        if self.theme_switch.get():
            ctk.set_appearance_mode("Light")
            self.theme_switch.configure(text="Dark Mode")
        else:
            ctk.set_appearance_mode("Dark")
            self.theme_switch.configure(text="Light Mode")



    # -------------------------------------------------------
    # SAFE START DETECTION
    # -------------------------------------------------------
    def start_detection(self):
        # 🔧 CHANGE: Prevent double-start
        if getattr(self.detector, "is_running", False):
            Logger.log("⚠ start_detection called but detection already running.")
            return

        try:
            if not self.detector.load_camera_config():
                Logger.log("❌ Config load failed")
                return

            Logger.log("📋 Loading camera...")

            substream = (
                self.detector.camera_data.get('substream_url')
                or self.detector.camera_data.get('rtsp_url')
            )

            self.detector.cap = cv2.VideoCapture(substream)
            #self.detector.cap = cv2.VideoCapture("1car.mp4")

            self.detector.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.detector.cap.set(cv2.CAP_PROP_FPS, 18)
            self.detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

            if not self.detector.cap.isOpened():
                Logger.log("❌ Camera connect failed")
                return

            self.detector.is_running = True

            # Update UI
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.status_label.configure(text="Status: Running", text_color="green")

            Logger.log("✅ System started - Hybrid YOLO + EasyOCR")

            self.process_video()

        except Exception as e:
            Logger.log(f"❌ Start error: {str(e)[:60]}")
            self.detector.is_running = False



    # -------------------------------------------------------
    # STOP DETECTION
    # -------------------------------------------------------
    def stop_detection(self):
        self.detector.is_running = False

        if self.detector.cap:
            try:
                self.detector.cap.release()
            except:
                pass
            self.detector.cap = None

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Stopped", text_color="red")
        self.video_label.configure(image='', text="Camera feed offline.\nPress Start to begin.")
        Logger.log("⏹ System stopped")



    # -------------------------------------------------------
    # PAUSE / RESUME
    # -------------------------------------------------------
    def toggle_detection(self):
        self.detector.detection_active = not self.detector.detection_active
        status = "ACTIVE" if self.detector.detection_active else "PAUSED"
        color = "green" if self.detector.detection_active else "orange"

        self.pause_btn.configure(text="Resume" if not self.detector.detection_active else "Pause")
        self.status_label.configure(text=f"Status: {status}", text_color=color)
        Logger.log(f"⏸ Detection {status.lower()}")



    # -------------------------------------------------------
    # PROCESS VIDEO LOOP (with reconnect)
    # -------------------------------------------------------
    def process_video(self):
        if not self.detector.is_running or not self.detector.cap:
            return
        write_healthCheck()

        try:
            ret, frame = self.detector.cap.read()

            if not ret or frame is None:
                Logger.log("⚠ Frame read failed - attempting reconnect")
                self.detector.is_running = False
                self.root.after(1000, self.reconnect_camera)
                return

            processed_frame, car_region, plate_region, plate_text, coverage = \
                self.detector.process_frame(frame)

            if car_region:
                self.car_count += 1
                self.counter_label.configure(text=f"Cars: {self.car_count}")

            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 360), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img)

            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk

        except Exception as e:
            Logger.log(f"❌ Video error: {str(e)[:60]}")
            self.detector.is_running = False
            self.root.after(1000, self.reconnect_camera)
            return

        if self.detector.is_running:
            self.root.after(55, self.process_video)



    # -------------------------------------------------------
    # 🔥 AUTO–RECONNECT (SAFE)
    # -------------------------------------------------------
    def reconnect_camera(self):
        Logger.log("🔄 Reconnecting camera...")
        self.stop_detection()

        def delayed_restart():
            if not getattr(self.detector, "is_running", False):
                Logger.log("🔁 Restarting after reconnect delay...")
                self.start_detection()
            else:
                Logger.log("⏸ Reconnect skipped — already running")

        self.root.after(2000, delayed_restart)



    # -------------------------------------------------------
    # 🔥 AUTO–RESTART WATCHDOG (EVERY 60 SEC)
    # -------------------------------------------------------
    def watchdog_loop(self):
        while True:
            time.sleep(60)
            if not getattr(self.detector, "is_running", False):
                Logger.log("♻ Watchdog: System not running — restarting...")
                try:
                    self.start_detection()
                except Exception as e:
                    Logger.log(f"❌ Watchdog restart failed: {str(e)[:60]}")


import atexit

def on_exit():
    Logger.log("🛑 Application closed normally.")
    if os.path.exists(HEALTH_CHECK_FILE):
        os.remove(HEALTH_CHECK_FILE)

atexit.register(on_exit)



# -------------------------
# main
# -------------------------
def main():
    try:
        root = ctk.CTk()
        app = LicensePlateGUI(root)

        window_width = 720
        window_height = 480
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_position = screen_width - window_width - 10
        y_position = 10
        root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()
