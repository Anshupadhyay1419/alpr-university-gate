# ALPR University Gate — Pipeline Documentation

## Overview

The ALPR (Automatic License Plate Recognition) University Gate system is a real-time pipeline that detects, tracks, and logs vehicles entering and exiting a university campus using CCTV footage or RTSP streams.

## Pipeline Flow

```
Video Source
    │
    ▼
Frame Capture (src/capture/frame_capture.py)
    │  Reads frames from RTSP stream or video file with frame-skip support
    ▼
Vehicle Detection (src/detection/vehicle_detector.py)
    │  YOLOv8n pretrained on COCO — detects car, truck, bus, motorcycle
    ▼
Vehicle Tracking (src/tracking/vehicle_tracker.py)
    │  ByteTrack — assigns consistent integer IDs across frames
    ▼
Plate Detection (src/detection/plate_detector.py)
    │  Custom YOLOv8m trained on Indian license plates
    ▼
Preprocessing (src/preprocessing/plate_preprocessor.py)
    │  Grayscale → CLAHE → Denoising → Unsharp mask
    ▼
OCR (src/ocr/)
    │  PaddleOCR (default) or TrOCR — reads plate text
    ▼
Post-processing (src/ocr/ocr_postprocessor.py)
    │  Removes noise characters, applies position-aware corrections
    ▼
Validation (src/validation/plate_validator.py)
    │  Validates against Indian plate regex (XX00XX0000 / BH00XX0000)
    ▼
OCR Fusion (src/validation/ocr_fusion.py)
    │  Majority voting over sliding window of N frames per track
    ▼
Super Resolution (src/enhancement/super_resolution.py)
    │  Real-ESRGAN applied once at fusion time for final OCR pass
    ▼
Classification (src/classification/)
    │  Color classifier (HSV) → Vehicle type (White=Private, Yellow=Commercial, etc.)
    ▼
Duplicate Filter (src/database/duplicate_filter.py)
    │  Suppresses repeated events within configurable time window
    ▼
Direction Detection (src/utils/direction_detector.py)
    │  Virtual line crossing → IN / OUT
    ▼
Database Storage (src/database/db.py)
    │  SQLite via SQLAlchemy — stores plate, type, color, direction, timestamp
    ▼
API / Dashboard (src/api/, src/dashboard/)
   FastAPI REST endpoints + Streamlit live monitor
```

## Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| Frame Capture | `src/capture/` | RTSP/video reader with reconnect |
| Vehicle Detector | `src/detection/vehicle_detector.py` | YOLOv8n COCO |
| Plate Detector | `src/detection/plate_detector.py` | Custom YOLOv8m |
| Tracker | `src/tracking/vehicle_tracker.py` | ByteTrack |
| Preprocessor | `src/preprocessing/` | CLAHE + denoise + sharpen |
| OCR | `src/ocr/` | PaddleOCR / TrOCR |
| Validator | `src/validation/plate_validator.py` | Indian plate regex |
| OCR Fusion | `src/validation/ocr_fusion.py` | Majority voting |
| Super Resolution | `src/enhancement/` | Real-ESRGAN x4 |
| Color Classifier | `src/classification/color_classifier.py` | HSV-based |
| Vehicle Classifier | `src/classification/vehicle_classifier.py` | Color → type |
| Duplicate Filter | `src/database/duplicate_filter.py` | Time-window dedup |
| Direction Detector | `src/utils/direction_detector.py` | Virtual line crossing |
| Database | `src/database/db.py` | SQLite + SQLAlchemy |
| REST API | `src/api/server.py` | FastAPI |
| Dashboard | `src/dashboard/app.py` | Streamlit |

## Configuration

All settings are in `config/config.yaml`. Key sections:

- `video`: source path, frame skip, retry settings
- `detection`: model paths and confidence thresholds
- `ocr`: backend selection (paddleocr/trocr) and parameters
- `fusion`: majority voting window size and minimum confidence
- `database`: SQLite path and plate image save directory
- `enhancement`: Real-ESRGAN model path and threshold

## Running the Pipeline

```bash
# Full pipeline (main entry point)
python main.py

# With custom source
python main.py --source rtsp://camera-ip/stream

# Simple CSV output (no tracking)
python scripts/run_alpr.py --source ALPR.mp4

# Annotated video output
python scripts/video_output.py --source ALPR.mp4

# Test on static images
python scripts/test_model.py --images test_2_images/test/images

# Start REST API
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# Start dashboard
streamlit run src/dashboard/app.py
```

## Training

```bash
# Convert dataset to YOLO format
python -m training.dataset_converter

# Train plate detector
python -m training.trainer

# Evaluate model
python -m training.evaluator
```
