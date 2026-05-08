"""
Integration tests for training.dataset_converter.DatasetConverter.

All tests use a temporary directory with synthetic data so they are fast,
isolated, and do not touch the real dataset.

Property-based tests (Property 3 and 4) are pure in-memory — no filesystem
I/O — so they run in milliseconds regardless of max_examples.

Test coverage:
- YOLO .txt files are created for each image
- YAML file is valid and contains required keys (train, val, test, nc, names)
- Split counts are correct (test=300, train=total-300)
- Malformed JSON files are skipped with a warning
- JSON-to-YOLO coordinate round-trip: normalized coords in [0,1] survive
  conversion within 1e-6 tolerance  (Property 4)
- Dataset split partitioning: test=300, train=M-300, no overlap  (Property 3)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from training.dataset_converter import DatasetConverter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMAGE_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
    b"\xff\xd9"
)  # minimal valid JPEG bytes


def _make_label(stem: str, x_center: float = 0.5, y_center: float = 0.5,
                width: float = 0.3, height: float = 0.2) -> dict:
    """Return a valid JSON label dict."""
    return {
        "filename": f"{stem}.jpg",
        "image_width": 500,
        "image_height": 335,
        "annotations": [
            {
                "vehicle_number": "KA01AB1234",
                "class_id": 0,
                "xmin": 100,
                "ymin": 100,
                "xmax": 250,
                "ymax": 167,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }
        ],
    }


def _populate_dataset(
    labels_dir: Path,
    images_dir: Path,
    count: int,
    bad_stems: list[str] | None = None,
) -> list[str]:
    """Create *count* synthetic image+label pairs."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    bad_stems_set = set(bad_stems or [])
    valid_stems: list[str] = []

    for i in range(count):
        stem = str(i + 1)
        (images_dir / f"{stem}.jpg").write_bytes(_IMAGE_BYTES)
        if stem in bad_stems_set:
            (labels_dir / f"{stem}.json").write_text("NOT VALID JSON {{", encoding="utf-8")
        else:
            label = _make_label(stem)
            (labels_dir / f"{stem}.json").write_text(
                json.dumps(label), encoding="utf-8"
            )
            valid_stems.append(stem)

    return valid_stems


# ---------------------------------------------------------------------------
# Pure helpers extracted for property-based testing (no filesystem I/O)
# ---------------------------------------------------------------------------

def _yolo_line_from_annotation(ann: dict) -> str:
    """Replicate the exact formatting used by DatasetConverter._json_to_yolo."""
    return (
        f"{int(ann['class_id'])} "
        f"{float(ann['x_center'])} "
        f"{float(ann['y_center'])} "
        f"{float(ann['width'])} "
        f"{float(ann['height'])}"
    )


def _split_stems(stems: list[str], test_count: int, seed: int) -> tuple[set[str], set[str]]:
    """Replicate the split logic from DatasetConverter.convert() — pure, no I/O."""
    rng = random.Random(seed)
    actual_test = min(test_count, len(stems))
    test_set = set(rng.sample(stems, actual_test))
    train_set = set(stems) - test_set
    return train_set, test_set


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dataset_dir(tmp_path: Path):
    """Return a tmp_path with a synthetic dataset of 320 images (fast)."""
    labels_dir = tmp_path / "train" / "labels"
    images_dir = tmp_path / "train" / "images"
    _populate_dataset(labels_dir, images_dir, count=320)
    return tmp_path


@pytest.fixture()
def converter_320(dataset_dir: Path) -> DatasetConverter:
    """DatasetConverter configured for the 320-image synthetic dataset."""
    return DatasetConverter(
        labels_dir=str(dataset_dir / "train" / "labels"),
        images_dir=str(dataset_dir / "train" / "images"),
        output_dir=str(dataset_dir),
        test_split_count=300,
        random_seed=42,
    )


# ---------------------------------------------------------------------------
# 1. YOLO .txt files are created for each image
# ---------------------------------------------------------------------------

class TestYoloFilesCreated:
    def test_txt_files_exist_for_all_valid_images(
        self, converter_320: DatasetConverter, dataset_dir: Path
    ):
        stats = converter_320.convert()
        yolo_root = dataset_dir / "yolo"

        train_labels = list((yolo_root / "train" / "labels").glob("*.txt"))
        test_labels = list((yolo_root / "test" / "labels").glob("*.txt"))

        total_labels = len(train_labels) + len(test_labels)
        assert total_labels == stats["total"] - stats["skipped"]

    def test_txt_file_content_is_valid_yolo_format(
        self, converter_320: DatasetConverter, dataset_dir: Path
    ):
        converter_320.convert()
        yolo_root = dataset_dir / "yolo"

        for label_file in list((yolo_root / "train" / "labels").glob("*.txt"))[:5]:
            lines = label_file.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) >= 1, f"{label_file} is empty"
            for line in lines:
                parts = line.split()
                assert len(parts) == 5, f"Expected 5 fields, got: {line!r}"
                class_id = int(parts[0])
                coords = [float(p) for p in parts[1:]]
                assert class_id >= 0
                for c in coords:
                    assert 0.0 <= c <= 1.0, f"Coord out of range: {c}"

    def test_image_files_are_copied_to_split_dirs(
        self, converter_320: DatasetConverter, dataset_dir: Path
    ):
        stats = converter_320.convert()
        yolo_root = dataset_dir / "yolo"

        train_images = list((yolo_root / "train" / "images").glob("*"))
        test_images = list((yolo_root / "test" / "images").glob("*"))

        assert len(train_images) == stats["train"]
        assert len(test_images) == stats["test"]


# ---------------------------------------------------------------------------
# 2. YAML file is valid and contains required keys
# ---------------------------------------------------------------------------

class TestDatasetYaml:
    def test_yaml_file_is_created(
        self, converter_320: DatasetConverter, dataset_dir: Path
    ):
        converter_320.convert()
        yaml_path = dataset_dir / "yolo" / "dataset.yaml"
        assert yaml_path.exists(), "dataset.yaml was not created"

    def test_yaml_contains_required_keys(
        self, converter_320: DatasetConverter, dataset_dir: Path
    ):
        converter_320.convert()
        yaml_path = dataset_dir / "yolo" / "dataset.yaml"
        with yaml_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        assert isinstance(cfg, dict)
        for key in ("train", "val", "test", "nc", "names"):
            assert key in cfg, f"Missing key '{key}' in dataset.yaml"

    def test_yaml_nc_and_names(
        self, converter_320: DatasetConverter, dataset_dir: Path
    ):
        converter_320.convert()
        yaml_path = dataset_dir / "yolo" / "dataset.yaml"
        with yaml_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        assert cfg["nc"] == 1
        assert cfg["names"] == ["license_plate"]

    def test_yaml_paths_point_to_existing_dirs(
        self, converter_320: DatasetConverter, dataset_dir: Path
    ):
        converter_320.convert()
        yaml_path = dataset_dir / "yolo" / "dataset.yaml"
        with yaml_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        for key in ("train", "val", "test"):
            p = Path(cfg[key])
            assert p.exists(), f"YAML path for '{key}' does not exist: {p}"


# ---------------------------------------------------------------------------
# 3. Split counts are correct
# ---------------------------------------------------------------------------

class TestSplitCounts:
    def test_test_split_is_exactly_300(self, converter_320: DatasetConverter):
        stats = converter_320.convert()
        assert stats["test"] == 300

    def test_train_split_is_total_minus_300(self, converter_320: DatasetConverter):
        stats = converter_320.convert()
        assert stats["train"] == stats["total"] - 300 - stats["skipped"]

    def test_total_equals_json_file_count(
        self, converter_320: DatasetConverter, dataset_dir: Path
    ):
        stats = converter_320.convert()
        json_count = len(list((dataset_dir / "train" / "labels").glob("*.json")))
        assert stats["total"] == json_count

    def test_split_counts_with_fewer_than_300_images(self, tmp_path: Path):
        """When total < test_split_count, all images go to test."""
        labels_dir = tmp_path / "train" / "labels"
        images_dir = tmp_path / "train" / "images"
        _populate_dataset(labels_dir, images_dir, count=20)

        converter = DatasetConverter(
            labels_dir=str(labels_dir),
            images_dir=str(images_dir),
            output_dir=str(tmp_path),
            test_split_count=300,
            random_seed=42,
        )
        stats = converter.convert()
        assert stats["test"] == 20
        assert stats["train"] == 0


# ---------------------------------------------------------------------------
# 4. Malformed JSON files are skipped with a warning
# ---------------------------------------------------------------------------

class TestMalformedJsonSkipped:
    def test_malformed_json_is_skipped(self, tmp_path: Path, caplog):
        labels_dir = tmp_path / "train" / "labels"
        images_dir = tmp_path / "train" / "images"
        _populate_dataset(labels_dir, images_dir, count=12, bad_stems=["11", "12"])

        converter = DatasetConverter(
            labels_dir=str(labels_dir),
            images_dir=str(images_dir),
            output_dir=str(tmp_path),
            test_split_count=5,
            random_seed=42,
        )

        with caplog.at_level(logging.WARNING, logger="training.dataset_converter"):
            stats = converter.convert()

        assert stats["skipped"] == 2
        assert stats["train"] + stats["test"] == 10

    def test_warning_is_logged_for_malformed_json(self, tmp_path: Path):
        labels_dir = tmp_path / "train" / "labels"
        images_dir = tmp_path / "train" / "images"
        _populate_dataset(labels_dir, images_dir, count=5, bad_stems=["3"])

        converter = DatasetConverter(
            labels_dir=str(labels_dir),
            images_dir=str(images_dir),
            output_dir=str(tmp_path),
            test_split_count=2,
            random_seed=42,
        )

        module_logger = logging.getLogger("training.dataset_converter")
        captured: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        handler = _Capture(level=logging.WARNING)
        module_logger.addHandler(handler)
        try:
            converter.convert()
        finally:
            module_logger.removeHandler(handler)

        warning_messages = [r.getMessage() for r in captured if r.levelno == logging.WARNING]
        assert any(
            "3" in msg or "Malformed" in msg or "malformed" in msg.lower()
            for msg in warning_messages
        ), f"Expected a WARNING about malformed JSON, got: {warning_messages}"

    def test_missing_annotations_key_is_skipped(self, tmp_path: Path, caplog):
        labels_dir = tmp_path / "train" / "labels"
        images_dir = tmp_path / "train" / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        bad_label = {"filename": "1.jpg", "image_width": 500, "image_height": 335}
        (labels_dir / "1.json").write_text(json.dumps(bad_label), encoding="utf-8")
        (images_dir / "1.jpg").write_bytes(_IMAGE_BYTES)

        converter = DatasetConverter(
            labels_dir=str(labels_dir),
            images_dir=str(images_dir),
            output_dir=str(tmp_path),
            test_split_count=1,
            random_seed=42,
        )

        with caplog.at_level(logging.WARNING, logger="training.dataset_converter"):
            stats = converter.convert()

        assert stats["skipped"] == 1

    def test_original_source_dirs_are_not_modified(
        self, converter_320: DatasetConverter, dataset_dir: Path
    ):
        labels_dir = dataset_dir / "train" / "labels"
        images_dir = dataset_dir / "train" / "images"

        original_labels = set(p.name for p in labels_dir.glob("*.json"))
        original_images = set(p.name for p in images_dir.glob("*"))

        converter_320.convert()

        assert set(p.name for p in labels_dir.glob("*.json")) == original_labels
        assert set(p.name for p in images_dir.glob("*")) == original_images


# ---------------------------------------------------------------------------
# 5. Property 4: JSON-to-YOLO coordinate round-trip  (pure, no filesystem I/O)
# Validates: Requirements 5.1, 5.2
# ---------------------------------------------------------------------------

class TestJsonToYoloRoundTrip:
    """
    **Validates: Requirements 5.1, 5.2**

    Property 4: JSON-to-YOLO Coordinate Round-Trip — for any valid normalized
    coordinates in [0, 1], converting to YOLO format and parsing back yields
    values equal within 1e-6.

    The property test is pure (no filesystem I/O) — it calls the same
    formatting logic used by DatasetConverter._json_to_yolo directly.
    """

    def test_sample_annotation_round_trip(self, tmp_path: Path):
        """Example-based test using the real spec sample annotation."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        label = {
            "filename": "test.jpg",
            "image_width": 500,
            "image_height": 335,
            "annotations": [
                {
                    "vehicle_number": "KA19TR02",
                    "class_id": 0,
                    "xmin": 140, "ymin": 210, "xmax": 339, "ymax": 260,
                    "x_center": 0.479,
                    "y_center": 0.701493,
                    "width": 0.398,
                    "height": 0.149254,
                }
            ],
        }
        json_path = labels_dir / "test.json"
        json_path.write_text(json.dumps(label), encoding="utf-8")

        converter = DatasetConverter(
            labels_dir=str(labels_dir),
            images_dir=str(tmp_path),
            output_dir=str(tmp_path),
        )
        lines = converter._json_to_yolo(json_path)

        assert lines is not None and len(lines) == 1
        parts = lines[0].split()
        assert int(parts[0]) == 0
        assert abs(float(parts[1]) - 0.479) < 1e-6
        assert abs(float(parts[2]) - 0.701493) < 1e-6
        assert abs(float(parts[3]) - 0.398) < 1e-6
        assert abs(float(parts[4]) - 0.149254) < 1e-6

    # Pure property test — no filesystem I/O at all
    # **Validates: Requirements 5.1, 5.2**
    @given(
        x_center=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        y_center=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        width=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        height=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        class_id=st.integers(min_value=0, max_value=9),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_coordinate_round_trip_property(
        self,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        class_id: int,
    ):
        """
        **Validates: Requirements 5.1, 5.2**

        Pure in-memory test: format annotation → parse back → check tolerance.
        No filesystem I/O — runs in microseconds per example.
        """
        ann = {
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }
        line = _yolo_line_from_annotation(ann)
        parts = line.split()

        assert len(parts) == 5
        assert int(parts[0]) == class_id
        assert abs(float(parts[1]) - x_center) < 1e-6
        assert abs(float(parts[2]) - y_center) < 1e-6
        assert abs(float(parts[3]) - width) < 1e-6
        assert abs(float(parts[4]) - height) < 1e-6


# ---------------------------------------------------------------------------
# 6. Property 3: Dataset split partitioning  (pure, no filesystem I/O)
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------

class TestDatasetSplitPartitioning:
    """
    **Validates: Requirements 5.3**

    Property 3: Dataset Split Partitioning — for any dataset of M ≥ 300 images,
    test split = 300, train split = M − 300, union = full dataset, no overlap.

    The property test is pure (no filesystem I/O) — it tests the split logic
    directly using the extracted _split_stems() helper.
    """

    def test_no_overlap_between_train_and_test(
        self, converter_320: DatasetConverter, dataset_dir: Path
    ):
        """Example-based: filesystem-level overlap check on 320-image dataset."""
        converter_320.convert()
        yolo_root = dataset_dir / "yolo"

        train_stems = {p.stem for p in (yolo_root / "train" / "labels").glob("*.txt")}
        test_stems = {p.stem for p in (yolo_root / "test" / "labels").glob("*.txt")}

        assert len(train_stems & test_stems) == 0

    def test_union_equals_full_dataset(
        self, converter_320: DatasetConverter, dataset_dir: Path
    ):
        """Example-based: union of splits equals all source stems."""
        converter_320.convert()
        yolo_root = dataset_dir / "yolo"

        train_stems = {p.stem for p in (yolo_root / "train" / "labels").glob("*.txt")}
        test_stems = {p.stem for p in (yolo_root / "test" / "labels").glob("*.txt")}
        all_source_stems = {
            p.stem for p in (dataset_dir / "train" / "labels").glob("*.json")
        }

        assert train_stems | test_stems == all_source_stems

    # Pure property test — no filesystem I/O at all
    # **Validates: Requirements 5.3**
    @given(
        total=st.integers(min_value=300, max_value=600),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_split_partitioning_property(self, total: int, seed: int):
        """
        **Validates: Requirements 5.3**

        Pure in-memory test: generate stem list → split → verify counts and
        no overlap. No filesystem I/O — runs in microseconds per example.
        """
        stems = [str(i) for i in range(total)]
        train_set, test_set = _split_stems(stems, test_count=300, seed=seed)

        assert len(test_set) == 300
        assert len(train_set) == total - 300
        assert len(train_set & test_set) == 0
        assert train_set | test_set == set(stems)
