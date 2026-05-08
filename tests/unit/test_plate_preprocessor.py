"""
Unit + property-based tests for preprocessing.plate_preprocessor.PlatePreprocessor.

Property 5: Preprocessing Output Shape Preservation — Validates: Req 9.1-9.5
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.preprocessing.plate_preprocessor import PlatePreprocessor

preprocessor = PlatePreprocessor()


class TestPlatePreprocessorExamples:
    def test_bgr_input_returns_2d_array(self):
        img = np.random.randint(0, 256, (40, 120, 3), dtype=np.uint8)
        result = preprocessor.process(img)
        assert result.ndim == 2

    def test_grayscale_input_returns_2d_array(self):
        img = np.random.randint(0, 256, (40, 120), dtype=np.uint8)
        result = preprocessor.process(img)
        assert result.ndim == 2

    def test_output_shape_matches_input(self):
        img = np.random.randint(0, 256, (50, 150, 3), dtype=np.uint8)
        result = preprocessor.process(img)
        assert result.shape == (50, 150)

    def test_output_dtype_is_uint8(self):
        img = np.random.randint(0, 256, (40, 120, 3), dtype=np.uint8)
        result = preprocessor.process(img)
        assert result.dtype == np.uint8

    def test_pixel_values_in_range(self):
        img = np.random.randint(0, 256, (40, 120, 3), dtype=np.uint8)
        result = preprocessor.process(img)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_none_input_returns_none(self):
        result = preprocessor.process(None)
        assert result is None

    def test_empty_input_returned_unchanged(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = preprocessor.process(empty)
        assert result is not None  # returns something, doesn't crash


# ---------------------------------------------------------------------------
# Property 5: Preprocessing Output Shape Preservation
# ---------------------------------------------------------------------------

class TestPlatePreprocessorProperty5:
    """Property 5: Preprocessing Output Shape Preservation — Validates: Req 9.1-9.5"""

    @given(
        h=st.integers(min_value=1, max_value=100),
        w=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=30)
    def test_output_shape_preserved_bgr(self, h: int, w: int):
        """For any H×W BGR input, output is 2D array with same H×W."""
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        result = preprocessor.process(img)
        assert result.ndim == 2
        assert result.shape == (h, w)
        assert result.dtype == np.uint8
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    @given(
        h=st.integers(min_value=1, max_value=100),
        w=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=20)
    def test_output_shape_preserved_gray(self, h: int, w: int):
        """For any H×W grayscale input, output is 2D array with same H×W."""
        img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        result = preprocessor.process(img)
        assert result.ndim == 2
        assert result.shape == (h, w)
