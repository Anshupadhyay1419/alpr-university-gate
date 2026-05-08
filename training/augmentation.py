"""
Augmentation pipeline for the ALPR University Gate system.

Usage:
    python -m training.augmentation --config config/config.yaml
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("training.augmentation")


class AugmentationPipeline:
    """Apply image augmentations to training data with bounding box preservation."""

    def __init__(self, config: dict, output_dir: Optional[str] = None) -> None:
        aug_cfg = config.get("training", {}).get("augmentation", {})

        self.brightness_range: list = aug_cfg.get("brightness_range", [-0.2, 0.2])
        self.contrast_range: list = aug_cfg.get("contrast_range", [0.8, 1.2])
        self.blur_kernel_max: int = int(aug_cfg.get("blur_kernel_max", 5))
        self.blur_probability: float = float(aug_cfg.get("blur_probability", 0.3))
        self.noise_std_range: list = aug_cfg.get("noise_std_range", [5, 25])
        self.fog_intensity: float = float(aug_cfg.get("fog_intensity", 0.3))
        self.fog_probability: float = float(aug_cfg.get("fog_probability", 0.2))
        self.rotation_degrees: float = float(aug_cfg.get("rotation_degrees", 10))

        self.output_dir = Path(output_dir) if output_dir else None

    def augment_image(
        self,
        image: np.ndarray,
        bboxes: list[list[float]],
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, list[list[float]]]:
        """Apply all augmentations to a single image and its bounding boxes."""
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        img = image.copy().astype(np.float32)

        img = self._apply_brightness_contrast(img, rng)

        if rng.random() < self.blur_probability:
            img = self._apply_blur(img, rng)

        img = self._apply_noise(img, np_rng)

        if rng.random() < self.fog_probability:
            img = self._apply_fog(img, rng)

        img, bboxes = self._apply_rotation(img, bboxes, rng)

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, bboxes

    def augment_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        output_images_dir: str,
        output_labels_dir: str,
        augmentations_per_image: int = 1,
    ) -> int:
        """Augment all images in a directory and save results."""
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        out_images = Path(output_images_dir)
        out_labels = Path(output_labels_dir)

        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png"}
        image_files = [
            f for f in images_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        written = 0
        for img_path in image_files:
            label_path = labels_path / f"{img_path.stem}.txt"
            if not label_path.exists():
                _logger.warning("No label file for '%s'; skipping.", img_path.name)
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                _logger.warning("Could not read image '%s'; skipping.", img_path.name)
                continue

            bboxes = self._read_yolo_labels(label_path)

            for i in range(augmentations_per_image):
                aug_img, aug_bboxes = self.augment_image(image, bboxes, seed=None)

                out_stem = f"{img_path.stem}_aug{i}"
                out_img_path = out_images / f"{out_stem}{img_path.suffix}"
                out_lbl_path = out_labels / f"{out_stem}.txt"

                cv2.imwrite(str(out_img_path), aug_img)
                self._write_yolo_labels(out_lbl_path, aug_bboxes)
                written += 1

        _logger.info("Augmentation complete — %d augmented images written to '%s'", written, out_images)
        return written

    def _apply_brightness_contrast(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        brightness = rng.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast = rng.uniform(self.contrast_range[0], self.contrast_range[1])
        img = img * contrast + brightness * 255.0
        return img

    def _apply_blur(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        max_k = max(3, self.blur_kernel_max)
        k = rng.choice([k for k in range(3, max_k + 1, 2)])
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, (k, k), 0)
        return blurred.astype(np.float32)

    def _apply_noise(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        std = rng.uniform(self.noise_std_range[0], self.noise_std_range[1])
        noise = rng.normal(0, std, img.shape).astype(np.float32)
        return img + noise

    def _apply_fog(self, img: np.ndarray, rng: random.Random) -> np.ndarray:
        intensity = rng.uniform(0, self.fog_intensity)
        fog_layer = np.full_like(img, 255.0)
        return img * (1 - intensity) + fog_layer * intensity

    def _apply_rotation(
        self,
        img: np.ndarray,
        bboxes: list[list[float]],
        rng: random.Random,
    ) -> tuple[np.ndarray, list[list[float]]]:
        angle = rng.uniform(-self.rotation_degrees, self.rotation_degrees)
        if abs(angle) < 1e-6:
            return img, bboxes

        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(img_uint8, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)

        new_bboxes: list[list[float]] = []
        for bbox in bboxes:
            class_id = bbox[0]
            x_c, y_c, bw, bh = bbox[1], bbox[2], bbox[3], bbox[4]

            x1 = (x_c - bw / 2) * w
            y1 = (y_c - bh / 2) * h
            x2 = (x_c + bw / 2) * w
            y2 = (y_c + bh / 2) * h

            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            ones = np.ones((4, 1), dtype=np.float32)
            corners_h = np.hstack([corners, ones])
            rotated_corners = (M @ corners_h.T).T

            rx1 = float(np.clip(rotated_corners[:, 0].min(), 0, w))
            ry1 = float(np.clip(rotated_corners[:, 1].min(), 0, h))
            rx2 = float(np.clip(rotated_corners[:, 0].max(), 0, w))
            ry2 = float(np.clip(rotated_corners[:, 1].max(), 0, h))

            new_w = rx2 - rx1
            new_h = ry2 - ry1

            if new_w <= 0 or new_h <= 0:
                continue

            new_x_c = float(np.clip((rx1 + rx2) / 2.0 / w, 0.0, 1.0))
            new_y_c = float(np.clip((ry1 + ry2) / 2.0 / h, 0.0, 1.0))
            new_bw = float(np.clip(new_w / w, 0.0, 1.0))
            new_bh = float(np.clip(new_h / h, 0.0, 1.0))

            new_bboxes.append([class_id, new_x_c, new_y_c, new_bw, new_bh])

        return rotated.astype(np.float32), new_bboxes

    @staticmethod
    def _read_yolo_labels(label_path: Path) -> list[list[float]]:
        bboxes: list[list[float]] = []
        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                bboxes.append([float(p) for p in parts])
        return bboxes

    @staticmethod
    def _write_yolo_labels(label_path: Path, bboxes: list[list[float]]) -> None:
        lines = []
        for bbox in bboxes:
            class_id = int(bbox[0])
            lines.append(f"{class_id} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}")
        label_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description="Augment training dataset.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--images-dir", default="dataset/yolo/train/images")
    parser.add_argument("--labels-dir", default="dataset/yolo/train/labels")
    parser.add_argument("--output-images-dir", default="dataset/yolo/train/images")
    parser.add_argument("--output-labels-dir", default="dataset/yolo/train/labels")
    parser.add_argument("--augmentations-per-image", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = AugmentationPipeline(config=cfg)
    count = pipeline.augment_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_images_dir=args.output_images_dir,
        output_labels_dir=args.output_labels_dir,
        augmentations_per_image=args.augmentations_per_image,
    )
    print(f"Done — {count} augmented images written.")
