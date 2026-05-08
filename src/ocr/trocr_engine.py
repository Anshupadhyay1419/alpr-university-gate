"""
TrOCR backend for the ALPR University Gate OCR engine.

This is a functional stub that satisfies the OCREngine interface.
It can be fully implemented by loading microsoft/trocr-base-printed
from HuggingFace Transformers.

To enable: set config['ocr']['backend'] = 'trocr'
"""

from __future__ import annotations

import numpy as np

from src.ocr.base import OCREngine
from src.utils.logger import get_logger

_logger = get_logger("ocr.trocr_engine")


class TrOCREngine(OCREngine):
    """OCR engine backed by Microsoft TrOCR (HuggingFace Transformers).

    Args:
        model_name: HuggingFace model identifier.
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-printed") -> None:
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._init_failed = False
        self._initialize()

    def _initialize(self) -> None:
        """Load TrOCR processor and model."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            _logger.info("Loading TrOCR model '%s'...", self.model_name)
            self._processor = TrOCRProcessor.from_pretrained(self.model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self._model.eval()
            _logger.info("TrOCR model '%s' loaded.", self.model_name)

        except ImportError as exc:
            _logger.error(
                "transformers/torch not installed for TrOCR: %s. "
                "Install with: pip install transformers torch", exc
            )
            self._init_failed = True
        except Exception as exc:
            _logger.error("TrOCR initialization failed: %s", exc)
            self._init_failed = True

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Run TrOCR on a plate image.

        Args:
            image: Grayscale or BGR NumPy array.

        Returns:
            (text, confidence) — ("", 0.0) on any failure.
        """
        if self._init_failed or self._processor is None or self._model is None:
            return ("", 0.0)

        if image is None or image.size == 0:
            return ("", 0.0)

        try:
            import torch
            from PIL import Image
            import cv2

            # Convert to RGB PIL image
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(rgb)

            pixel_values = self._processor(
                images=pil_image, return_tensors="pt"
            ).pixel_values

            with torch.no_grad():
                generated_ids = self._model.generate(pixel_values)

            text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip().upper().replace(" ", "")

            # TrOCR doesn't provide per-token confidence easily;
            # return 0.8 as a placeholder when text is found
            confidence = 0.8 if text else 0.0
            return (text, confidence)

        except Exception as exc:
            _logger.warning("TrOCR inference failed: %s", exc)
            return ("", 0.0)
