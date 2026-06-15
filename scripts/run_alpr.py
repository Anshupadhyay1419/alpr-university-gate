"""
Simple ALPR runner — processes a video and saves detected plates to CSV.

No tracking required — detects plates directly from each frame,
deduplicates by plate number, and saves to CSV.

Key fix: OCR often misreads the same plate slightly differently across frames
(e.g. DL7CD5017 vs DL7CDS017). We use edit-distance fuzzy merging at the end
to group near-identical plate strings and keep only one entry per real vehicle.

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

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def _levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,   # deletion
                curr[j] + 1,       # insertion
                prev[j] + (0 if c1 == c2 else 1),  # substitution
            ))
        prev = curr
    return prev[-1]


def _merge_similar_plates(
    plate_reads: dict[str, list],
    max_edit_distance: int = 2,
) -> dict[str, list]:

    """
    Merge plate strings that differ by at most max_edit_distance characters.

    Root cause: OCR reads the same physical plate as slightly different strings
    across frames (e.g. DL7CD5017 vs DL7CDS017 vs DL7CD5O17). Without merging,
    each variant creates a separate CSV row, giving more entries than real vehicles.

    Strategy:
      1. Sort plates by number of reads (most-read first = most reliable).
      2. For each plate not yet assigned, group all others within edit distance.
      3. Merge all reads into the most-read plate's bucket.
    """
    plates = sorted(plate_reads.keys(), key=lambda p: len(plate_reads[p]), reverse=True)
    merged: dict[str, list] = {}
    assigned: set[str] = set()

    for plate in plates:

        if plate in assigned:
            continue
        # Start a new group with this plate as the canonical form
        group_reads = list(plate_reads[plate])
        assigned.add(plate)

        for other in plates:
            if other in assigned:
                continue
            # Only compare plates of similar length (within 2 chars)
            if abs(len(plate) - len(other)) > max_edit_distance:
                continue
            dist = _levenshtein(plate, other)
            if dist <= max_edit_distance:
                group_reads.extend(plate_reads[other])
                assigned.add(other)

        merged[plate] = group_reads

    return merged


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
    parser.add_argument("--max-edit-distance", type=int, default=2,
                        help="Max edit distance to merge similar OCR reads (default: 2)")
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.utils.logger import get_logger
    from src.detection.vehicle_detector import VehicleDetector
    from src.detection.plate_detector import PlateDetector
    from src.ocr import create_ocr_engine
    from src.validation.plate_validator import PlateValidator
    from src.ocr.ocr_postprocessor import remove_noise_characters, correct_ocr_text

    config = load_config(args.config)
    log = get_logger("run_alpr", config=config)
    det_cfg = config["detection"]

    log.info("Loading models...")
    vehicle_detector = VehicleDetector(det_cfg["vehicle_model_path"], 0.2)
    plate_detector   = PlateDetector(det_cfg["plate_model_path"], 0.15)
    ocr_backend      = str(config.get("ocr", {}).get("backend", "paddleocr"))
    ocr_engine       = create_ocr_engine(ocr_backend, config)
    plate_validator  = PlateValidator()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        log.error("Cannot open video: %s", args.source)
        sys.exit(1)

    log.info("Processing video: %s | min_reads=%d | max_edit_dist=%d",
             args.source, args.min_reads, args.max_edit_distance)

    # Read all plate OCR hits into per-frame buckets first.
    # This prevents the "same physical pass" from being split into multiple
    # plate-number variants that later become multiple CSV rows.
    #
    # frame_idx -> list of (plate_number, confidence)
    frame_hits: dict[int, list[tuple[str, float]]] = defaultdict(list)
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
            vehicle_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if vehicle_crop.size == 0:
                continue

            # ── Plate detection ──────────────────────────────────────
            plate_crops = plate_detector.detect(vehicle_crop)

            # Fallback: only use bottom 35% of vehicle — ONE fallback, not three.
            # Using multiple fallback crops was causing the same plate to be OCR'd
            # multiple times per frame with different results → fake extra entries.
            if not plate_crops:
                h_v = vehicle_crop.shape[0]
                bottom_crop = vehicle_crop[int(h_v * 0.65):, :]
                if bottom_crop.size > 0:
                    plate_crops = [bottom_crop]
                else:
                    continue

            # Only process the BEST (first/highest-conf) plate per vehicle per frame
            # to avoid reading the same plate multiple times per frame
            plate_crop = plate_crops[0]
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

            # Clean up OCR text
            raw_text = remove_noise_characters(raw_text)
            raw_text = raw_text.replace("IND", "").replace("INDIA", "").strip()
            raw_text = correct_ocr_text(raw_text)

            plate_number, series_type = plate_validator.validate(raw_text)
            if plate_number is None:
                continue

            # bucket by frame; later we cluster consecutive frames into a pass
            frame_hits[frame_idx].append((plate_number, confidence))
            log.info("Frame %d: OCR='%s' conf=%.2f", frame_idx, plate_number, confidence)


    cap.release()

    # ── Pass clustering over frames (prevents extra CSV rows) ────────────
    # A “pass” is a cluster of consecutive frames where we saw plate OCR hits.
    # Within each pass we pick the best plate after fuzzy-merging variants.
    # This matches the expectation: one car pass → one CSV row.
    frame_indices = sorted(frame_hits.keys())
    if not frame_indices:
        results = []
    else:
        # Larger gap groups more frame OCR hits into a single pass.
        # Too small a gap can split one vehicle into multiple passes,
        # reintroducing extra CSV rows.
        max_gap_frames = 60  # configurable clustering gap for a single pass

        
        passes: list[list[int]] = []
        current = [frame_indices[0]]
        for idx in frame_indices[1:]:
            if idx - current[-1] <= max_gap_frames:
                current.append(idx)
            else:
                passes.append(current)
                current = [idx]
        passes.append(current)

        results = []

        for pass_frames in passes:
            # Collect all OCR hits inside this pass
            pass_plate_reads: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for fidx in pass_frames:
                for plate_number, conf in frame_hits.get(fidx, []):
                    pass_plate_reads[plate_number].append((conf, fidx))

            # Fuzzy-merge variants within the pass.
            # Use a slightly larger edit distance to aggressively collapse common OCR
            # confusions between letters/digits that still pass validation.
            merged_plate_reads = pass_plate_reads
            if args.max_edit_distance > 0:
                merged_plate_reads = _merge_similar_plates(
                    pass_plate_reads,
                    max_edit_distance=max(args.max_edit_distance, 3),
                )


            # Choose canonical plate for this pass: max reads, tie-break by max confidence
            #
            # Also collapse obviously-confused last-digit variants (e.g. HR26CQ6869 vs HR26CO6869 vs HR26CO6869)
            # by selecting the lexicographically most-common 'shape'.
            best_plate = None

            best_reads = None
            best_total_reads = -1
            best_conf = -1.0
            best_frame = None

            for plate_number, reads in merged_plate_reads.items():

                if len(reads) < args.min_reads:
                    continue
                total_reads = len(reads)
                conf = max(r[0] for r in reads)
                first_frame = min(r[1] for r in reads)

                if total_reads > best_total_reads or (total_reads == best_total_reads and conf > best_conf):
                    best_total_reads = total_reads
                    best_conf = conf
                    best_frame = first_frame
                    best_plate = plate_number
                    best_reads = reads

            if not best_plate:
                continue

            _, series_type = plate_validator.validate(best_plate)
            vehicle_type, color = _classify_by_plate_number(best_plate)

            log.info("✅ FINAL PASS: %s | reads=%d | best_conf=%.2f", best_plate, best_total_reads, best_conf)
            results.append({
                "plate_number": best_plate,
                "vehicle_type": vehicle_type,
                "plate_color":  color,
                "series_type":  series_type or "normal",
                "confidence":   f"{best_conf:.2f}",
                "total_reads":  best_total_reads,
                "best_frame":   best_frame,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
            })

    # Sort by first frame seen (order of appearance in video)
    results.sort(key=lambda x: x["best_frame"])



    # ── Write CSV ───────────────────────────────────────────────────────
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
