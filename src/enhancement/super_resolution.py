"""
Super-resolution enhancer for the ALPR University Gate system.

Uses Real-ESRGAN to upscale low-resolution plate crops before OCR.
Enhancement is conditional: only applied when plate width < sr_threshold_px.

Real-ESRGAN must be cloned from https://github.com/xinntao/Real-ESRGAN
and weights placed at the path specified in config.yaml.

Setup:
    git clone https://github.com/xinntao/Real-ESRGAN.git
    pip install basicsr realesrgan
    # Download RealESRGAN_x4plus.pth to models/realesrgan/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("enhancement.super_resolution")


class SuperResolutionEnhancer:
    """Conditionally upscale plate crops using Real-ESRGAN.

    Args:
        model_path:      Path to RealESRGAN_x4plus.pth weights.
        sr_threshold_px: Upscale only when plate width < this value (pixels).
    """

    def __init__(self, model_path: str, sr_threshold_px: int = 80) -> None:
        self.model_path = model_path
        self.sr_threshold_px = sr_threshold_px
        self._upsampler = None
        self._load_failed = False

    def _load_model(self) -> bool:
        """Lazy-load Real-ESRGAN. Returns True on success, False on failure."""
        if self._upsampler is not None:
            return True
        if self._load_failed:
            return False

        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_file = Path(self.model_path)
            if not model_file.exists():
                _logger.error(
                    "Real-ESRGAN weights not found at '%s'. "
                    "Download from https://github.com/xinntao/Real-ESRGAN/releases",
                    self.model_path,
                )
                self._load_failed = True
                return False

            arch = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32, scale=4,
            )
            self._upsampler = RealESRGANer(
                scale=4,
                model_path=str(model_file),
                model=arch,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
            )
            _logger.info("Real-ESRGAN loaded from '%s'", self.model_path)
            return True

        except ImportError as exc:
            _logger.error(
                "Real-ESRGAN dependencies not installed (%s). "
                "Run: pip install basicsr realesrgan", exc
            )
            self._load_failed = True
            return False
        except Exception as exc:
            _logger.error("Failed to load Real-ESRGAN: %s", exc)
            self._load_failed = True
            return False

    def enhance(self, plate_crop: np.ndarray) -> np.ndarray:
        """Upscale plate crop if its width is below the threshold.

        Args:
            plate_crop: Grayscale (H, W) or BGR (H, W, 3) plate image.

        Returns:
            Enhanced image if width < sr_threshold_px and model loaded
            successfully. Otherwise returns the original crop unchanged.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        # Determine width
        width = plate_crop.shape[1] if len(plate_crop.shape) >= 2 else 0

        # Pass-through: width at or above threshold
        if width >= self.sr_threshold_px:
            return plate_crop

        # Try to load model
        if not self._load_model():
            _logger.warning(
                "Real-ESRGAN unavailable; returning unenhanced crop (width=%d).", width
            )
            return plate_crop

        try:
            import cv2

            # Real-ESRGAN expects BGR uint8
            if len(plate_crop.shape) == 2:
                bgr = cv2.cvtColor(plate_crop, cv2.COLOR_GRAY2BGR)
            else:
                bgr = plate_crop.copy()

            if bgr.dtype != np.uint8:
                bgr = np.clip(bgr, 0, 255).astype(np.uint8)

            enhanced_bgr, _ = self._upsampler.enhance(bgr, outscale=4)

            # Return in same format as input
            if len(plate_crop.shape) == 2:
                return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
            return enhanced_bgr

        except Exception as exc:
            _logger.warning(
                "Real-ESRGAN inference failed (%s); returning unenhanced crop.", exc
            )
            return plate_crop
