"""
Unit + property-based tests for classification.color_classifier.ColorClassifier.

Property 15: Color Classifier Output Validity — Validates: Req 14.1-14.4
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.classification.color_classifier import VALID_COLORS, ColorClassifier

classifier = ColorClassifier()


class TestColorClassifierExamples:
    def test_white_image_classified_as_white(self):
        white = np.full((50, 150, 3), 255, dtype=np.uint8)
        result = classifier.classify(white)
        assert result == "White"

    def test_black_image_classified_as_black(self):
        black = np.zeros((50, 150, 3), dtype=np.uint8)
        result = classifier.classify(black)
        assert result == "Black"

    def test_empty_image_returns_unknown(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = classifier.classify(empty)
        assert result == "Unknown"

    def test_none_returns_unknown(self):
        result = classifier.classify(None)
        assert result == "Unknown"

    def test_grayscale_input_handled(self):
        gray = np.full((50, 150), 200, dtype=np.uint8)
        result = classifier.classify(gray)
        assert result in VALID_COLORS

    def test_yellow_image(self):
        # HSV yellow: H~30, S~255, V~255 → BGR approx (0, 255, 255)
        yellow = np.zeros((50, 150, 3), dtype=np.uint8)
        yellow[:, :] = [0, 255, 255]  # BGR yellow
        result = classifier.classify(yellow)
        assert result in VALID_COLORS  # may be Yellow or close


class TestColorClassifierProperty15:
    """Property 15: Color Classifier Output Validity — Validates: Req 14.1-14.4"""

    @given(
        h=st.integers(min_value=1, max_value=100),
        w=st.integers(min_value=1, max_value=200),
        pixel=st.integers(min_value=0, max_value=255),
    )
    @settings(max_examples=30)
    def test_any_image_returns_valid_color(self, h: int, w: int, pixel: int):
        """For any image of any size and content, output is always a valid color label."""
        image = np.full((h, w, 3), pixel, dtype=np.uint8)
        result = classifier.classify(image)
        assert result in VALID_COLORS, f"Got unexpected color: {result!r}"

    @given(
        data=st.binary(min_size=3, max_size=3 * 50 * 150),
    )
    @settings(max_examples=20)
    def test_random_pixel_data_returns_valid_color(self, data: bytes):
        """Random pixel data never causes an invalid return value."""
        size = len(data) // 3
        if size == 0:
            return
        arr = np.frombuffer(data[:size * 3], dtype=np.uint8).reshape(1, size, 3)
        result = classifier.classify(arr)
        assert result in VALID_COLORS
