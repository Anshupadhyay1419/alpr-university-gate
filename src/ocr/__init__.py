"""
OCR engine factory for the ALPR University Gate system.

Usage:
    from src.ocr import create_ocr_engine
    engine = create_ocr_engine(backend="paddleocr", config=cfg)
    text, confidence = engine.recognize(plate_image)
"""

from __future__ import annotations

from src.ocr.base import OCREngine
from src.utils.logger import get_logger

_logger = get_logger("ocr")


def create_ocr_engine(backend: str, config: dict) -> OCREngine:
    """Instantiate the OCR engine specified by *backend*.

    Args:
        backend: "paddleocr" or "trocr".
        config:  Full config dict (backend-specific settings read from
                 config['ocr'][backend]).

    Returns:
        An OCREngine instance.

    Raises:
        ValueError: If *backend* is not a supported value.
    """
    ocr_cfg = config.get("ocr", {})

    if backend == "paddleocr":
        from src.ocr.paddle_ocr_engine import PaddleOCREngine
        paddle_cfg = ocr_cfg.get("paddleocr", {})
        return PaddleOCREngine(
            use_angle_cls=bool(paddle_cfg.get("use_angle_cls", True)),
            lang=str(paddle_cfg.get("lang", "en")),
        )

    elif backend == "trocr":
        from src.ocr.trocr_engine import TrOCREngine
        trocr_cfg = ocr_cfg.get("trocr", {})
        return TrOCREngine(
            model_name=str(trocr_cfg.get("model_name", "microsoft/trocr-base-printed")),
        )

    elif backend == "ensemble":
        from src.ocr.ensemble_engine import EnsembleOCREngine

        ensemble_cfg = ocr_cfg.get("ensemble", {})
        backends = ensemble_cfg.get("backends", ["paddleocr", "trocr"])
        return EnsembleOCREngine.from_config(
            config=config,
            backends=[str(name) for name in backends],
            min_vote_count=int(ensemble_cfg.get("min_vote_count", 1)),
            use_variants=bool(ensemble_cfg.get("use_variants", True)),
        )

    else:
        msg = (
            f"Unsupported OCR backend: '{backend}'. "
            f"Valid options are: 'paddleocr', 'trocr', 'ensemble'."
        )
        _logger.error(msg)
        raise ValueError(msg)
