"""
PaddleOCR backend for the ALPR University Gate OCR engine.

PaddleOCR is initialized once at startup. On inference failure, returns
("", 0.0) and logs a WARNING — never raises to the pipeline.

Installation:
    pip install paddlepaddle paddleocr
"""

from __future__ import annotations

import numpy as np

from src.ocr.base import OCREngine
from src.utils.logger import get_logger

_logger = get_logger("ocr.paddle_ocr_engine")


class PaddleOCREngine(OCREngine):
    """OCR engine backed by PaddleOCR.

    Args:
        use_angle_cls: Enable angle classification for rotated text.
        lang:          Language model to use (default 'en').
    """

    def __init__(self, use_angle_cls: bool = True, lang: str = "en") -> None:
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self._ocr = None
        self._init_failed = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize PaddleOCR. Logs ERROR and sets flag on failure."""
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False,
            )
            _logger.info(
                "PaddleOCR initialized (use_angle_cls=%s, lang=%s)",
                self.use_angle_cls, self.lang,
            )
        except ImportError as exc:
            _logger.error(
                "PaddleOCR is not installed: %s. "
                "Install with: pip install paddlepaddle paddleocr", exc
            )
            self._init_failed = True
        except Exception as exc:
            _logger.error("PaddleOCR initialization failed: %s", exc)
            self._init_failed = True

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Run PaddleOCR on a plate image.

        Args:
            image: Grayscale or BGR NumPy array.

        Returns:
            (text, confidence) — ("", 0.0) on any failure.
        """
        if self._init_failed or self._ocr is None:
            return ("", 0.0)

        if image is None or image.size == 0:
            return ("", 0.0)

        try:
            import cv2

            # PaddleOCR works best with BGR uint8
            if len(image.shape) == 2:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = image.copy()

            if img_bgr.dtype != np.uint8:
                img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

            result = self._ocr.ocr(img_bgr, cls=self.use_angle_cls)

            if not result or result[0] is None:
                return ("", 0.0)

            # Collect all text lines and their confidences
            texts: list[str] = []
            confidences: list[float] = []

            for line in result[0]:
                if line is None:
                    continue
                # line format: [bbox, (text, confidence)]
                text_info = line[1]
                if text_info and len(text_info) == 2:
                    text = str(text_info[0]).strip()
                    conf = float(text_info[1])
                    if text:
                        texts.append(text)
                        confidences.append(conf)

            if not texts:
                return ("", 0.0)

            # Concatenate all text regions (plate may span multiple detections)
            combined_text = "".join(texts).upper().replace(" ", "")
            avg_confidence = float(np.mean(confidences))
            avg_confidence = float(np.clip(avg_confidence, 0.0, 1.0))

            return (combined_text, avg_confidence)

        except Exception as exc:
            _logger.warning("PaddleOCR inference failed: %s", exc)
            return ("", 0.0)
