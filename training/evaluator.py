"""
Model evaluator for the ALPR University Gate plate detection model.

Usage:
    python -m training.evaluator
    python -m training.evaluator --config config/config.yaml --dataset dataset/yolo/dataset.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.logger import get_logger

_logger = get_logger("training.evaluator")


class Evaluator:
    """Evaluate the trained YOLOv8m plate detector on the held-out test split."""

    def __init__(self, config: dict) -> None:
        train_cfg = config.get("training", {})
        self.image_size: int = int(train_cfg.get("image_size", 640))
        self.compute_flops: bool = bool(train_cfg.get("compute_flops", True))

        self.model_path = Path("models/plate_detector/best.pt")
        self.report_path = Path("models/plate_detector/eval_report.json")

    def evaluate(
        self,
        dataset_yaml: str = "dataset/yolo/dataset.yaml",
        split: str = "test",
    ) -> dict:
        """Run evaluation on the test split."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            _logger.error("ultralytics is not installed: %s", exc)
            raise

        if not self.model_path.exists():
            msg = f"Model not found at '{self.model_path}'. Run training first."
            _logger.error(msg)
            raise FileNotFoundError(msg)

        _logger.info("Evaluating model '%s' on split='%s'", self.model_path, split)

        model = YOLO(str(self.model_path))

        metrics = model.val(
            data=dataset_yaml,
            split=split,
            imgsz=self.image_size,
            verbose=False,
        )

        report = self._build_report(metrics)

        if self.compute_flops:
            try:
                flops = self._compute_flops(model)
                report["flops_gflops"] = flops
                _logger.info("Model FLOPs: %.2f GFLOPs", flops)
            except Exception as exc:
                _logger.warning("Could not compute FLOPs: %s", exc)

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        _logger.info(
            "Evaluation complete — mAP50=%.4f precision=%.4f recall=%.4f",
            report["mAP50"], report["precision"], report["recall"],
        )
        _logger.info("Report saved to '%s'", self.report_path)

        return report

    @staticmethod
    def _build_report(metrics) -> dict:
        """Extract standard metrics from Ultralytics val results."""
        try:
            box = metrics.results_dict
            return {
                "mAP50": float(box.get("metrics/mAP50(B)", 0.0)),
                "mAP50_95": float(box.get("metrics/mAP50-95(B)", 0.0)),
                "precision": float(box.get("metrics/precision(B)", 0.0)),
                "recall": float(box.get("metrics/recall(B)", 0.0)),
            }
        except Exception as exc:
            _logger.warning("Could not extract metrics: %s", exc)
            return {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}

    @staticmethod
    def _compute_flops(model) -> float:
        """Compute model GFLOPs using Ultralytics info()."""
        try:
            result = model.info(verbose=False)
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                return float(result[1])
            import torch
            from thop import profile as thop_profile
            dummy = torch.zeros(1, 3, 640, 640)
            macs, _ = thop_profile(model.model, inputs=(dummy,), verbose=False)
            return float(macs) / 1e9
        except Exception:
            return 0.0


if __name__ == "__main__":
    import argparse
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description="Evaluate trained plate detector.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default="dataset/yolo/dataset.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluator = Evaluator(config=cfg)
    report = evaluator.evaluate(dataset_yaml=args.dataset, split=args.split)
    print(json.dumps(report, indent=2))
