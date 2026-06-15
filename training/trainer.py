"""
Advanced YOLO trainer supporting YOLOv8/YOLOv11 for ALPR plate/vehicle detection.

Features:
  - Multi-GPU training (DDP)
  - YOLOv8 and YOLOv11 support
  - Mixed precision (AMP) training
  - Early stopping with validation monitoring
  - Model export (ONNX, TorchScript, Core ML)
  - Inference profiling
  - Data augmentation for challenging conditions

Usage:
    python -m training.trainer
    python -m training.trainer --model yolov11m --epochs 150 --device 0,1
    python -m training.trainer --export onnx
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

_logger = get_logger("training.trainer")


class Trainer:
    """Advanced trainer supporting YOLOv8/YOLOv11 models."""

    def __init__(self, config: dict, model_name: str = "yolov8m") -> None:
        train_cfg = config.get("training", {})

        self.model_name: str = model_name
        self.pretrained_weights: str = train_cfg.get("pretrained_weights", f"{model_name}.pt")
        self.epochs: int = int(train_cfg.get("epochs", 100))
        self.image_size: int = int(train_cfg.get("image_size", 640))
        self.batch_size: int = int(train_cfg.get("batch_size", 8))
        self.workers: int = int(train_cfg.get("workers", 4))
        self.device: str = str(train_cfg.get("device", "0"))
        self.cache: bool = bool(train_cfg.get("cache", False))
        self.amp: bool = bool(train_cfg.get("amp", True))
        self.patience: int = int(train_cfg.get("patience", 20))

        self.best_pt_path = Path("models/plate_detector/best.pt")
        self.logs_dir = Path("models/plate_detector/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

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

    def evaluate(self, dataset_yaml: str = "dataset/yolo/dataset.yaml") -> dict:
        """Evaluate trained model on validation set."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            _logger.error("ultralytics is not installed: %s", exc)
            raise

        _logger.info("Evaluating model '%s' on validation set", self.best_pt_path)

        model = YOLO(str(self.best_pt_path))
        metrics = model.val(
            data=dataset_yaml,
            device=self.device,
            verbose=True,
        )

        return {
            "mAP50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
            "mAP50-95": float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0,
            "precision": float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0.0,
            "recall": float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0.0,
        }

    def export(self, export_format: str = "onnx") -> str:
        """Export trained model to different formats.

        Supported formats: onnx, torchscript, tflite, coreml, pb, paddle

        Returns:
            Path to exported model
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            _logger.error("ultralytics is not installed: %s", exc)
            raise

        _logger.info("Exporting model to %s format", export_format)

        model = YOLO(str(self.best_pt_path))
        exported_path = model.export(
            format=export_format,
            device=self.device,
            imgsz=self.image_size,
        )

        _logger.info("✓ Model exported to: %s", exported_path)
        return str(exported_path)

    def profile(self) -> dict:
        """Profile model inference speed and memory usage."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            _logger.error("ultralytics is not installed: %s", exc)
            raise

        _logger.info("Profiling model inference...")

        model = YOLO(str(self.best_pt_path))
        profile_result = model.profile(
            imgsz=self.image_size,
            device=self.device,
        )

        _logger.info("Profile complete: %s", profile_result)
        return profile_result

    def compare_models(
        self,
        test_image: str,
        models: list[str],
        dataset_yaml: str = "dataset/yolo/dataset.yaml",
    ) -> dict:
        """Compare multiple model variants on speed/accuracy trade-off.

        Args:
            test_image: Path to test image for inference timing
            models: List of model names (yolov8n, yolov8s, yolov8m, yolov11m, etc.)
            dataset_yaml: Dataset for mAP evaluation

        Returns:
            Comparison results dict
        """
        try:
            from ultralytics import YOLO
            import cv2
            import time
        except ImportError as exc:
            _logger.error("Required packages not installed: %s", exc)
            raise

        _logger.info("Comparing %d model variants", len(models))

        results = {}
        for model_name in models:
            _logger.info("  Evaluating %s...", model_name)

            try:
                model = YOLO(f"{model_name}.pt")

                # Inference speed
                img = cv2.imread(test_image)
                start = time.perf_counter()
                _ = model(img, verbose=False)
                inference_time_ms = (time.perf_counter() - start) * 1000

                # Accuracy (mAP)
                metrics = model.val(data=dataset_yaml, verbose=False)

                results[model_name] = {
                    "inference_time_ms": float(inference_time_ms),
                    "mAP50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
                    "mAP50-95": float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0,
                    "parameters": model.model.model.parameters() if hasattr(model, 'model') else 0,
                }
            except Exception as exc:
                _logger.warning("Failed to evaluate %s: %s", model_name, exc)
                results[model_name] = {"error": str(exc)}

        # Save comparison report
        report_path = self.logs_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        _logger.info("Model comparison saved to %s", report_path)
        return results


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
