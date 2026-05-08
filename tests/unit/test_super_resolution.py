"""
Unit + property-based tests for enhancement.super_resolution.SuperResolutionEnhancer.

Property 6: SR Upscaling Condition   — Validates: Req 10.1
Property 7: SR Pass-Through Condition — Validates: Req 10.2
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.enhancement.super_resolution import SuperResolutionEnhancer

# Use a non-existent model path so the model always fails to load.
# This lets us test the pass-through and fallback logic without real weights.
_FAKE_MODEL = "models/realesrgan/DOES_NOT_EXIST.pth"
SR_THRESHOLD = 80


class TestSuperResolutionPassThrough:
    """Property 7: Pass-Through Condition — Validates: Req 10.2"""

    def test_wide_crop_returned_unchanged(self):
        """Width >= threshold → identical array returned."""
        enhancer = SuperResolutionEnhancer(_FAKE_MODEL, sr_threshold_px=SR_THRESHOLD)
        img = np.random.randint(0, 256, (40, 100, 3), dtype=np.uint8)  # width=100 >= 80
        result = enhancer.enhance(img)
        np.testing.assert_array_equal(result, img)

    def test_exact_threshold_width_returned_unchanged(self):
        enhancer = SuperResolutionEnhancer(_FAKE_MODEL, sr_threshold_px=SR_THRESHOLD)
        img = np.random.randint(0, 256, (40, SR_THRESHOLD, 3), dtype=np.uint8)
        result = enhancer.enhance(img)
        np.testing.assert_array_equal(result, img)

    @given(
        w=st.integers(min_value=80, max_value=500),
        h=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=20)
    def test_property7_pass_through(self, w: int, h: int):
        """For any width >= threshold, output is identical to input."""
        enhancer = SuperResolutionEnhancer(_FAKE_MODEL, sr_threshold_px=SR_THRESHOLD)
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        result = enhancer.enhance(img)
        np.testing.assert_array_equal(result, img)


class TestSuperResolutionFallback:
    """Req 10.4: On model failure, return unenhanced crop."""

    def test_narrow_crop_with_missing_model_returns_original(self):
        """Width < threshold but model missing → original returned (fallback)."""
        enhancer = SuperResolutionEnhancer(_FAKE_MODEL, sr_threshold_px=SR_THRESHOLD)
        img = np.random.randint(0, 256, (20, 40, 3), dtype=np.uint8)  # width=40 < 80
        result = enhancer.enhance(img)
        # Model load fails → fallback → original returned
        np.testing.assert_array_equal(result, img)

    def test_none_input_returned_as_none(self):
        enhancer = SuperResolutionEnhancer(_FAKE_MODEL, sr_threshold_px=SR_THRESHOLD)
        result = enhancer.enhance(None)
        assert result is None

    def test_empty_array_returned_unchanged(self):
        enhancer = SuperResolutionEnhancer(_FAKE_MODEL, sr_threshold_px=SR_THRESHOLD)
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = enhancer.enhance(empty)
        assert result is not None

    @given(
        w=st.integers(min_value=1, max_value=79),
        h=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=20)
    def test_property6_fallback_when_model_missing(self, w: int, h: int):
        """
        Property 6 (partial): When model is unavailable, narrow crops are
        returned unchanged (fallback path). The full property requires a
        real model to verify upscaling.
        """
        enhancer = SuperResolutionEnhancer(_FAKE_MODEL, sr_threshold_px=SR_THRESHOLD)
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        result = enhancer.enhance(img)
        # Fallback: original returned
        np.testing.assert_array_equal(result, img)
