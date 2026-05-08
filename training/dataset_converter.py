"""
Dataset converter for the ALPR University Gate system.

Converts JSON label files to YOLO-format .txt files and organises images
into train/test splits under dataset/yolo/.

Usage:
    python -m training.dataset_converter
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import yaml

from src.utils.logger import get_logger

_logger = get_logger("training.dataset_converter")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DatasetConverter:
    """Convert JSON-labelled plate images to YOLO format with train/test split."""

    def __init__(
        self,
        labels_dir: str = "dataset/train/labels",
        images_dir: str = "dataset/train/images",
        output_dir: str = "dataset",
        test_split_count: int = 300,
        random_seed: int = 42,
    ) -> None:
        self.labels_dir = Path(labels_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.test_split_count = test_split_count
        self.random_seed = random_seed

        self._yolo_root = self.output_dir / "yolo"
        self._train_images = self._yolo_root / "train" / "images"
        self._train_labels = self._yolo_root / "train" / "labels"
        self._test_images = self._yolo_root / "test" / "images"
        self._test_labels = self._yolo_root / "test" / "labels"

    def convert(self) -> dict:
        """Run the full conversion pipeline."""
        json_files = sorted(self.labels_dir.glob("*.json"))
        total = len(json_files)
        _logger.info("Found %d JSON label files in '%s'", total, self.labels_dir)

        if total == 0:
            _logger.warning("No JSON label files found in '%s'", self.labels_dir)
            self._write_dataset_yaml(
                str(self._train_images),
                str(self._train_images),
                str(self._test_images),
            )
            return {"total": 0, "train": 0, "test": 0, "skipped": 0}

        rng = random.Random(self.random_seed)
        actual_test_count = min(self.test_split_count, total)
        test_stems = set(p.stem for p in rng.sample(json_files, actual_test_count))

        for d in (self._train_images, self._train_labels, self._test_images, self._test_labels):
            d.mkdir(parents=True, exist_ok=True)

        train_count = 0
        test_count = 0
        skipped_count = 0

        for json_path in json_files:
            stem = json_path.stem

            yolo_lines = self._json_to_yolo(json_path)
            if yolo_lines is None:
                skipped_count += 1
                continue

            image_path = self._find_image(stem)
            if image_path is None:
                _logger.warning("No image found for label '%s' in '%s'; skipping.", stem, self.images_dir)
                skipped_count += 1
                continue

            if stem in test_stems:
                dest_images = self._test_images
                dest_labels = self._test_labels
                test_count += 1
            else:
                dest_images = self._train_images
                dest_labels = self._train_labels
                train_count += 1

            shutil.copy2(image_path, dest_images / image_path.name)

            label_dest = dest_labels / f"{stem}.txt"
            label_dest.write_text("\n".join(yolo_lines), encoding="utf-8")

        _logger.info(
            "Conversion complete — total=%d train=%d test=%d skipped=%d",
            total, train_count, test_count, skipped_count,
        )

        self._write_dataset_yaml(
            str(self._train_images),
            str(self._train_images),
            str(self._test_images),
        )

        return {"total": total, "train": train_count, "test": test_count, "skipped": skipped_count}

    def _json_to_yolo(self, json_path: Path) -> list[str] | None:
        """Convert a single JSON label file to YOLO-format lines."""
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            _logger.warning("Label file not found: '%s'; skipping.", json_path)
            return None
        except json.JSONDecodeError as exc:
            _logger.warning("Malformed JSON in '%s' (%s); skipping.", json_path, exc)
            return None

        if not isinstance(data, dict):
            _logger.warning("Unexpected JSON structure in '%s'; skipping.", json_path)
            return None

        annotations = data.get("annotations")
        if annotations is None:
            _logger.warning("Missing 'annotations' key in '%s'; skipping.", json_path)
            return None

        if not isinstance(annotations, list):
            _logger.warning("'annotations' in '%s' is not a list; skipping.", json_path)
            return None

        lines: list[str] = []
        for idx, ann in enumerate(annotations):
            if not isinstance(ann, dict):
                _logger.warning("Annotation %d in '%s' is not a dict; skipping file.", idx, json_path)
                return None

            required = ("class_id", "x_center", "y_center", "width", "height")
            missing = [k for k in required if k not in ann]
            if missing:
                _logger.warning("Annotation %d in '%s' is missing fields %s; skipping file.", idx, json_path, missing)
                return None

            try:
                class_id = int(ann["class_id"])
                x_center = float(ann["x_center"])
                y_center = float(ann["y_center"])
                width = float(ann["width"])
                height = float(ann["height"])
            except (TypeError, ValueError) as exc:
                _logger.warning("Annotation %d in '%s' has non-numeric fields (%s); skipping file.", idx, json_path, exc)
                return None

            lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

        return lines

    def _find_image(self, stem: str) -> Path | None:
        """Return the image path for the given stem, or None if not found."""
        for ext in _IMAGE_EXTENSIONS:
            candidate = self.images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
            candidate_upper = self.images_dir / f"{stem}{ext.upper()}"
            if candidate_upper.exists():
                return candidate_upper
        return None

    def _write_dataset_yaml(self, train_dir: str, val_dir: str, test_dir: str) -> None:
        """Write the YOLO dataset.yaml file with absolute paths."""
        yaml_path = self._yolo_root / "dataset.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_cfg = {
            "train": str(Path(train_dir).resolve()),
            "val":   str(Path(val_dir).resolve()),
            "test":  str(Path(test_dir).resolve()),
            "nc": 1,
            "names": ["license_plate"],
        }

        with yaml_path.open("w", encoding="utf-8") as fh:
            yaml.dump(dataset_cfg, fh, default_flow_style=False, sort_keys=False)

        _logger.info("Wrote dataset YAML to '%s'", yaml_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSON labels to YOLO format.")
    parser.add_argument("--labels-dir", default="dataset/train/labels")
    parser.add_argument("--images-dir", default="dataset/train/images")
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--test-split-count", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    converter = DatasetConverter(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        test_split_count=args.test_split_count,
        random_seed=args.random_seed,
    )
    stats = converter.convert()
    print(f"Done — total={stats['total']} train={stats['train']} test={stats['test']} skipped={stats['skipped']}")
