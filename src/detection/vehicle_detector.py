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

        return detections

    @staticmethod
    def filter_by_confidence(
        detections: list[Detection], threshold: float
    ) -> list[Detection]:
        """Filter a list of detections to those at or above threshold.

        This static method is used by property-based tests to verify the
        filtering invariant independently of model inference.
        """
        return [d for d in detections if d.confidence >= threshold]
