"""
HSV-based plate color classifier for the ALPR University Gate system.

Detects the dominant background color of a license plate crop and maps
it to one of the defined color categories used for vehicle type classification.

Color → Vehicle Type mapping:
  White      → Private
  Yellow     → Commercial
  Green      → EV
  Red        → Govt/Temp
  Blue       → Diplomatic
  Black      → Rental
  Army_Green → Military
  Unknown    → Unknown
"""

from __future__ import annotations

import numpy as np
import cv2

from src.utils.logger import get_logger

_logger = get_logger("classification.color_classifier")

# Valid output labels
VALID_COLORS = frozenset({
    "White", "Yellow", "Green", "Red", "Blue", "Black", "Army_Green", "Unknown"
})

# Default HSV ranges: [[H_min, H_max], [S_min, S_max], [V_min, V_max]]
DEFAULT_HSV_RANGES: dict[str, list[list[int]]] = {
    "White":      [[0, 180],   [0, 30],    [200, 255]],
    "Yellow":     [[20, 35],   [100, 255], [100, 255]],
    "Green":      [[40, 80],   [50, 255],  [50, 255]],
    "Red":        [[0, 10],    [100, 255], [100, 255]],
    "Red2":       [[160, 180], [100, 255], [100, 255]],  # red wraps in HSV
    "Blue":       [[100, 130], [100, 255], [100, 255]],
    "Black":      [[0, 180],   [0, 255],   [0, 50]],
    "Army_Green": [[35, 75],   [30, 120],  [30, 120]],
}


class ColorClassifier:
    """Classify the background color of a license plate crop using HSV.

    Args:
        hsv_ranges: Dict mapping color name → [[H_min,H_max],[S_min,S_max],[V_min,V_max]].
                    Defaults to DEFAULT_HSV_RANGES if not provided.
    """

    def __init__(
        self,
        hsv_ranges: dict[str, list[list[int]]] | None = None,
    ) -> None:
        self._ranges = hsv_ranges if hsv_ranges is not None else DEFAULT_HSV_RANGES

    @classmethod
    def from_config(cls, config: dict) -> "ColorClassifier":
        """Construct from the full config dict."""
        cc_cfg = config.get("color_classifier", {})
        hsv_ranges = cc_cfg.get("hsv_ranges", None)
        return cls(hsv_ranges=hsv_ranges)

    def classify(self, plate_crop: np.ndarray) -> str:
        """Classify the dominant background color of a plate crop.

        Args:
            plate_crop: BGR or grayscale plate image as NumPy array.

        Returns:
            One of: "White", "Yellow", "Green", "Red", "Blue", "Black",
            "Army_Green", or "Unknown".
        """
        if plate_crop is None or plate_crop.size == 0:
            return "Unknown"

        try:
            # Convert to BGR if grayscale
            if len(plate_crop.shape) == 2:
                bgr = cv2.cvtColor(plate_crop, cv2.COLOR_GRAY2BGR)
            else:
                bgr = plate_crop.copy()

            if bgr.dtype != np.uint8:
                bgr = np.clip(bgr, 0, 255).astype(np.uint8)

            # Convert to HSV
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            # Count pixels matching each color range
            best_color = "Unknown"
            best_count = 0

            for color_name, ranges in self._ranges.items():
                h_min, h_max = ranges[0]
                s_min, s_max = ranges[1]
                v_min, v_max = ranges[2]

                lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
                upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

                mask = cv2.inRange(hsv, lower, upper)
                count = int(np.sum(mask > 0))

                # Map "Red2" back to "Red"
                canonical = "Red" if color_name == "Red2" else color_name

                if count > best_count:
                    best_count = count
                    best_color = canonical

            # Require at least 5% of pixels to match
            total_pixels = plate_crop.shape[0] * plate_crop.shape[1]
            if best_count < max(1, total_pixels * 0.05):
                return "Unknown"

            return best_color

        except Exception as exc:
            _logger.warning("Color classification failed: %s", exc)
            return "Unknown"
