
# ANPR System Architecture Overview

This document provides a structured Markdown flowchart-style explanation of your ANPR (Automatic Number Plate Recognition) System. It covers all major modules, their interactions, inputs, outputs, and logical flow.

---

## ## 1. System Startup Flow

```text
Application Start → Initialize GUI → Initialize Detector → Load Config → Start Camera → Start Watchdog
```

---

## ## 2. High-Level Flowchart (Markdown Representation)

```text
[Start]
   ↓
[Load cam.txt Configuration]
   ↓
[Initialize YOLO Models + EasyOCR + Super-Resolution]
   ↓
[Open RTSP Stream]
   ↓
[Video Loop Starts]
   ↓
[Read Frame]
   ↓
 ┌───────────────────────────────┐
 │ Check Blank/Low-Quality Frame │
 └───────────────────────────────┘
          ↓           ↘
       [Good]        [Blank]
          ↓             ↘
[Detect Cars via YOLO]   [Skip & Wait]
          ↓
[Gate-Line / Upper-Line / Lower-Line Logic]
          ↓
     ┌──────────────┐
     │ Cross Gate?   │──No──► [Too Far → Draw Red Box]
     └──────────────┘
          ↓Yes
[Plate Detection YOLO]
          ↓
[Crop Plate Region]
          ↓
[OCR Processing (EasyOCR)]
          ↓
[Correct & Clean Plate Text]
          ↓
[Capture Burst Frames (Optional)]
          ↓
[Pick Best Frame by Sharpness]
          ↓
[Save Detection Folder]
          ↓
[Prepare Telegram Images]
          ↓
[Send to Telegram]
          ↓
[Delete Temporary Images]
          ↓
[Cooldown Timer Starts]
          ↓
[Loop Back to Video]
```

---

## ## 3. Functional Modules & Their Inputs/Outputs

### **3.1 YOLO Car Detection**

**Input:** Full frame
**Output:** List of car boxes → `(x, y, w, h, area)`
**Used By:** `process_frame()`

---

### **3.2 Gate-Line Logic**

**Input:** Car bounding box
**Output:** Trigger boolean
**Used By:** `process_frame()` to decide whether to start detection pipeline

---

### **3.3 Plate Detection YOLO**

**Input:** Full frame + cropped ROI
**Output:** Plate bounding box `(x, y, w, h)`
**Used By:** `process_frame()` → passed to OCR

---

### **3.4 OCR Module**

**Input:** Plate cropped image
**Output:** Raw text + Clean corrected text
**Used By:** `handle_detection()`

---

### **3.5 save_detection_folder()**

**Input:** frame + car box + plate box + plate text
**Output:** Folder with images + info file
**Used By:** `handle_detection()`

---

### **3.6 Telegram Sender**

**Input:** image paths + plate text + metadata
**Output:** Telegram message (SUCCESS/FAIL)
**Used By:** `handle_detection()`

---

### **3.7 GUI Module**

**Input:** processed frame
**Output:** Live updated window, logs, counters
**Used By:** `process_video()`

---

### **3.8 Watchdog**

**Function:** Automatically restarts detection every 60 seconds if stopped

---

## ## 4. End-to-End Flow Summary (Human-Friendly)

```text
Camera → Read Frame → Detect Cars → Gate Trigger → Detect Plate → Crop Plate → OCR → Save Folder → Telegram → Loop
```

---

