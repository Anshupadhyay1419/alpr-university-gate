"""
YOLOv8m trainer for the ALPR University Gate plate detection model.

Usage:
    python -m training.trainer
    python -m training.trainer --config config/config.yaml --dataset dataset/yolo/dataset.yaml
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

_logger = get_logger("training.trainer")


class Trainer:
    """Train a YOLOv8m model on the custom plate detection dataset."""

    def __init__(self, config: dict) -> None:
        train_cfg = config.get("training", {})

        self.pretrained_weights: str = train_cfg.get("pretrained_weights", "yolov8m.pt")
        self.epochs: int = int(train_cfg.get("epochs", 100))
        self.image_size: int = int(train_cfg.get("image_size", 640))
        self.batch_size: int = int(train_cfg.get("batch_size", 8))
        self.workers: int = int(train_cfg.get("workers", 4))
        self.device: str = str(train_cfg.get("device", "0"))
        self.cache: bool = bool(train_cfg.get("cache", False))
        self.amp: bool = bool(train_cfg.get("amp", True))

        self.best_pt_path = Path("models/plate_detector/best.pt")
        self.logs_dir = Path("models/plate_detector/logs")

    def train(
        self,
        dataset_yaml: str = "dataset/yolo/dataset.yaml",
        project: str = "models/plate_detector",
        name: str = "train",
    ) -> dict:
        """Run YOLOv8m training."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            _logger.error("ultralytics is not installed: %s", exc)
            raise

        _logger.info(
            "Starting YOLOv8m training — weights=%s epochs=%d imgsz=%d batch=%d",
            self.pretrained_weights, self.epochs, self.image_size, self.batch_size,
        )

        model = YOLO(self.pretrained_weights)

        results = model.train(
            data=dataset_yaml,
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            workers=self.workers,
            device=self.device,
            cache=self.cache,
            amp=self.amp,
            project=project,
            name=name,
            exist_ok=True,
            verbose=True,
        )

        # Copy best.pt to canonical location
        run_dir = Path(project) / name
        source_best = run_dir / "weights" / "best.pt"
        if source_best.exists():
            self.best_pt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_best, self.best_pt_path)
            _logger.info("Best checkpoint saved to '%s'", self.best_pt_path)
        else:
            _logger.warning("best.pt not found at '%s' after training.", source_best)

        # Copy logs
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            shutil.copy2(results_csv, self.logs_dir / "results.csv")

        metrics = self._extract_metrics(results)
        _logger.info(
            "Training complete — mAP50=%.4f precision=%.4f recall=%.4f",
            metrics.get("mAP50", 0.0),
            metrics.get("precision", 0.0),
            metrics.get("recall", 0.0),
        )

        return {
            "best_model_path": str(self.best_pt_path),
            "metrics": metrics,
        }

    @staticmethod
    def _extract_metrics(results) -> dict:
        """Extract mAP50, precision, recall from Ultralytics results object."""
        try:
            box = results.results_dict
            return {
                "mAP50": float(box.get("metrics/mAP50(B)", 0.0)),
                "precision": float(box.get("metrics/precision(B)", 0.0)),
                "recall": float(box.get("metrics/recall(B)", 0.0)),
            }
        except Exception as exc:
            _logger.warning("Could not extract metrics from results: %s", exc)
            return {"mAP50": 0.0, "precision": 0.0, "recall": 0.0}


if __name__ == "__main__":
    import argparse
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description="Train YOLOv8m plate detector.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default="dataset/yolo/dataset.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = Trainer(config=cfg)
    result = trainer.train(dataset_yaml=args.dataset)
    print(f"Done — best model: {result['best_model_path']}")
    print(f"Metrics: {result['metrics']}")
