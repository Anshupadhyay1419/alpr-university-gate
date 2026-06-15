"""
Vehicle detector for the ALPR University Gate system.

Uses a YOLOv8 model pretrained on COCO to detect vehicles (car, truck,
bus, motorcycle) in each frame. Returns Detection dataclasses filtered
by confidence threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("detection.vehicle_detector")

# COCO class names for vehicle types we care about
_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class Detection:
    """A single object detection result."""
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 (pixel coords)
    class_label: str
    confidence: float


class VehicleDetector:
    """Detect vehicles in a frame using YOLOv8 COCO pretrained weights.

    Args:
        model_path:           Path to YOLOv8 weights (.pt file).
        confidence_threshold: Minimum confidence score to keep a detection.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the YOLO model on first use."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            _logger.info("Vehicle detector loaded from '%s'", self.model_path)
        except Exception as exc:
            _logger.error("Failed to load vehicle detector from '%s': %s", self.model_path, exc)
            raise

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run vehicle detection on a single frame.

        Args:
            frame: BGR image as NumPy array (H, W, 3).

        Returns:
            List of Detection objects for detected vehicles above the
            confidence threshold. Returns empty list if none found.
        """
        self._load_model()

        try:
            results = self._model(frame, verbose=False)
        except Exception as exc:
            _logger.warning("Vehicle detection inference failed: %s", exc)
            return []

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < self.confidence_threshold:
                    continue

                cls_id = int(boxes.cls[i])
                class_name = result.names.get(cls_id, "").lower()

                if class_name not in _VEHICLE_CLASSES:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_label=class_name,
                    confidence=conf,
                ))

        return self.non_max_suppression(detections, iou_threshold=0.5)

    @staticmethod
    def non_max_suppression(
        detections: list[Detection],
        iou_threshold: float = 0.5,
    ) -> list[Detection]:
        """Remove overlapping duplicate detections using per-class NMS.

        This keeps the highest-confidence box for the same vehicle when the
        detector returns multiple highly overlapping boxes in a single frame.
        """
        if len(detections) <= 1:
            return detections

        kept: list[Detection] = []

        for class_label in {d.class_label for d in detections}:
            class_detections = [d for d in detections if d.class_label == class_label]
            class_detections.sort(key=lambda d: d.confidence, reverse=True)

            while class_detections:
                best = class_detections.pop(0)
                kept.append(best)
                class_detections = [
                    det
                    for det in class_detections
                    if VehicleDetector._iou(best.bbox, det.bbox) < iou_threshold
                ]

        return kept

    @staticmethod
    def _iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """Compute intersection-over-union for two boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0

        return inter_area / union

    @staticmethod
    def filter_by_confidence(
        detections: list[Detection], threshold: float
    ) -> list[Detection]:
        """Filter a list of detections to those at or above threshold.

        This static method is used by property-based tests to verify the
        filtering invariant independently of model inference.
        """
        return [d for d in detections if d.confidence >= threshold]
