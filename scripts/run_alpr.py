"""
Simple ALPR runner — processes a video and saves detected plates to CSV.

No tracking required — detects plates directly from each frame,
deduplicates by plate number, and saves to CSV.

Usage:
  python scripts/run_alpr.py --source ALPR.mp4
  python scripts/run_alpr.py --source mycarplate.mp4 --output output/results.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import cv2
import numpy as np


def _classify_by_plate_number(plate_number: str) -> tuple[str, str]:
    """Classify vehicle type from plate number. Returns (vehicle_type, plate_color)."""
    if plate_number.startswith("BH"):
        return ("Private", "White")
    return ("Private", "White")


def main():
    parser = argparse.ArgumentParser(description="ALPR — detect plates and save to CSV")
    parser.add_argument("--source", required=True, help="Video file path")
    parser.add_argument("--output", default="output/results.csv", help="Output CSV path")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--min-reads", type=int, default=2,
                        help="Minimum consistent reads before storing a plate (default: 2)")
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.utils.logger import get_logger
    from src.detection.vehicle_detector import VehicleDetector
    from src.detection.plate_detector import PlateDetector
    from src.ocr.paddle_ocr_engine import PaddleOCREngine
    from src.validation.plate_validator import PlateValidator
    from src.ocr.ocr_postprocessor import remove_noise_characters

    config = load_config(args.config)
    log = get_logger("run_alpr", config=config)
    det_cfg = config["detection"]

    log.info("Loading models...")
    vehicle_detector = VehicleDetector(det_cfg["vehicle_model_path"], 0.2)
    plate_detector   = PlateDetector(det_cfg["plate_model_path"], 0.15)
    ocr_engine       = PaddleOCREngine(use_angle_cls=True, lang="en")
    plate_validator  = PlateValidator()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        log.error("Cannot open video: %s", args.source)
        sys.exit(1)

    log.info("Processing video: %s | min_reads=%d", args.source, args.min_reads)

    # plate_number -> list of (confidence, frame_idx, plate_crop)
    plate_reads: dict[str, list] = defaultdict(list)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Detect vehicles
        detections = vehicle_detector.detect(frame)
        if not detections:
            continue

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            h, w = frame.shape[:2]
            vehicle_crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if vehicle_crop.size == 0:
                continue

            # Detect plates in vehicle crop
            plate_crops = plate_detector.detect(vehicle_crop)

            # If plate detector finds nothing, use multiple crops of vehicle
            if not plate_crops:
                h_v, w_v = vehicle_crop.shape[:2]
                fallbacks = [
                    vehicle_crop[int(h_v * 0.60):, :],   # bottom 40%
                    vehicle_crop[int(h_v * 0.75):, :],   # bottom 25%
                    vehicle_crop,                          # full crop
                ]
                plate_crops = [c for c in fallbacks if c.size > 0]

            for plate_crop in plate_crops:
                ph, pw = plate_crop.shape[:2]

                # Upscale small plates
                if pw < 150:
                    scale = max(150 / pw, 3.0)
                    ocr_input = cv2.resize(plate_crop, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_CUBIC)
                elif pw < 200:
                    scale = max(200 / pw, 2.0)
                    ocr_input = cv2.resize(plate_crop, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_CUBIC)
                else:
                    ocr_input = plate_crop

                # OCR
                raw_text, confidence = ocr_engine.recognize(ocr_input)
                if not raw_text or confidence < 0.40:
                    continue

                raw_text = remove_noise_characters(raw_text)
                plate_number, series_type = plate_validator.validate(raw_text)
                if plate_number is None:
                    continue

                plate_reads[plate_number].append((confidence, frame_idx, plate_crop))
                log.info("Frame %d: OCR='%s' conf=%.2f", frame_idx, plate_number, confidence)

    cap.release()

    # Build final results — only plates with enough consistent reads
    results = []
    for plate_number, reads in plate_reads.items():
        if len(reads) < args.min_reads:
            continue

        # Use highest confidence read
        best_conf, best_frame, best_crop = max(reads, key=lambda x: x[0])
        _, series_type = plate_validator.validate(plate_number)
        vehicle_type, color = _classify_by_plate_number(plate_number)

        log.info("✅ FINAL: %s | reads=%d | best_conf=%.2f", plate_number, len(reads), best_conf)
        results.append({
            "plate_number": plate_number,
            "vehicle_type": vehicle_type,
            "plate_color":  color,
            "series_type":  series_type or "normal",
            "confidence":   f"{best_conf:.2f}",
            "total_reads":  len(reads),
            "best_frame":   best_frame,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        })

    # Sort by best_frame (order of appearance)
    results.sort(key=lambda x: x["best_frame"])

    # Write CSV
    if results:
        fieldnames = ["plate_number", "vehicle_type", "plate_color", "series_type",
                      "confidence", "total_reads", "best_frame", "timestamp"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        log.info("✅ Results saved to: %s (%d plates)", output_path, len(results))
    else:
        log.warning("No plates detected with enough reads.")

    print(f"\n{'='*50}")
    print(f"RESULTS: {len(results)} unique plates detected")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['plate_number']} | reads={r['total_reads']} | conf={r['confidence']}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
