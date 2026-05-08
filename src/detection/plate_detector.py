"""
License plate detector for the ALPR University Gate system.

Uses the custom-trained YOLOv8m model to detect license plate regions
within vehicle crops. Returns cropped plate images as NumPy arrays.
"""

from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("detection.plate_detector")


class PlateDetector:
    """Detect license plates within a vehicle crop using custom YOLOv8m.

    Args:
        model_path:           Path to the trained plate detection weights.
        confidence_threshold: Minimum confidence to keep a plate detection.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.4) -> None:
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
            _logger.info("Plate detector loaded from '%s'", self.model_path)
        except Exception as exc:
            _logger.error(
                "Failed to load plate detector from '%s': %s", self.model_path, exc
            )
            raise

    def detect(self, vehicle_crop: np.ndarray) -> list[np.ndarray]:
        """Detect license plates within a vehicle crop.

        Args:
            vehicle_crop: BGR image crop of a detected vehicle (H, W, 3).

        Returns:
            List of plate crop images (NumPy arrays). Empty list if no plate
            detected above the confidence threshold.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        self._load_model()

        try:
            results = self._model(vehicle_crop, verbose=False)
        except Exception as exc:
            _logger.warning("Plate detection inference failed: %s", exc)
            return []

        plate_crops: list[np.ndarray] = []
        h, w = vehicle_crop.shape[:2]

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < self.confidence_threshold:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy()
                x1 = max(0, int(xyxy[0]))
                y1 = max(0, int(xyxy[1]))
                x2 = min(w, int(xyxy[2]))
                y2 = min(h, int(xyxy[3]))

                if x2 <= x1 or y2 <= y1:
                    continue

                plate_crop = vehicle_crop[y1:y2, x1:x2]
                if plate_crop.size > 0:
                    plate_crops.append(plate_crop)

        return plate_crops
