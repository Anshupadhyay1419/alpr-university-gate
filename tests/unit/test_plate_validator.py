"""
Unit + property-based tests for validation.plate_validator.PlateValidator.

Properties tested:
  Property 9:  Plate Validation Regex Correctness
  Property 10: Plate Series Type Classification
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.validation.plate_validator import PLATE_PATTERN, PlateValidator

validator = PlateValidator()

# ---------------------------------------------------------------------------
# Valid plate generators
# ---------------------------------------------------------------------------

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS  = "0123456789"

def _normal_plate(state: str, num: str, series: str, reg: str) -> str:
    return f"{state}{num}{series}{reg}"

VALID_NORMAL = [
    "KA19TR0234", "MH12AB1234", "DL01CD5678", "TN99ZZ9999",
]
VALID_BH = [
    "BH01AB1234", "BH99ZZ9999", "BH23CD5678",
]
INVALID = [
    "", "ABC", "KA19TR023",  # too short
    "KA19TR02345",           # too long
    "1A19TR0234",            # starts with digit
    "KA19T10234",            # digit in series letters
    "HELLO WORLD",
    "BH0AB1234",             # BH missing digit
]


# ---------------------------------------------------------------------------
# Example-based tests
# ---------------------------------------------------------------------------

class TestValidPlates:
    @pytest.mark.parametrize("plate", VALID_NORMAL)
    def test_valid_normal_plate_accepted(self, plate):
        result, series = validator.validate(plate)
        assert result == plate
        assert series == "normal"

    @pytest.mark.parametrize("plate", VALID_BH)
    def test_valid_bh_plate_accepted(self, plate):
        result, series = validator.validate(plate)
        assert result == plate
        assert series == "BH"

    def test_lowercase_input_normalized(self):
        result, series = validator.validate("ka19tr0234")
        assert result == "KA19TR0234"
        assert series == "normal"

    def test_whitespace_stripped(self):
        result, series = validator.validate("  KA19TR0234  ")
        assert result == "KA19TR0234"

    def test_hyphen_removed(self):
        result, series = validator.validate("KA-19-TR-0234")
        assert result == "KA19TR0234"


class TestInvalidPlates:
    @pytest.mark.parametrize("plate", INVALID)
    def test_invalid_plate_rejected(self, plate):
        result, series = validator.validate(plate)
        assert result is None
        assert series is None

    def test_none_like_empty_string(self):
        result, series = validator.validate("")
        assert result is None


# ---------------------------------------------------------------------------
# Property 9: Plate Validation Regex Correctness
# ---------------------------------------------------------------------------

# Strategy: generate strings that match the pattern
_valid_plate_st = st.one_of(
    # Normal: XX00XX0000
    st.builds(
        lambda s, n, l, d: f"{s}{n:02d}{l}{d:04d}",
        s=st.text(alphabet=LETTERS, min_size=2, max_size=2),
        n=st.integers(min_value=0, max_value=99),
        l=st.text(alphabet=LETTERS, min_size=2, max_size=2),
        d=st.integers(min_value=0, max_value=9999),
    ),
    # BH: BH00XX0000
    st.builds(
        lambda n, l, d: f"BH{n:02d}{l}{d:04d}",
        n=st.integers(min_value=0, max_value=99),
        l=st.text(alphabet=LETTERS, min_size=2, max_size=2),
        d=st.integers(min_value=0, max_value=9999),
    ),
)


class TestPlateValidatorProperty9:
    """Property 9: Plate Validation Regex Correctness — Validates: Req 12.1, 12.2, 12.3"""

    @given(plate=_valid_plate_st)
    @settings(max_examples=50)
    def test_valid_pattern_always_accepted(self, plate: str):
        """Any string matching the regex must be returned as valid."""
        assert PLATE_PATTERN.match(plate), f"Test plate doesn't match regex: {plate}"
        result, _ = validator.validate(plate)
        assert result == plate

    @given(text=st.text(min_size=0, max_size=20))
    @settings(max_examples=50)
    def test_invalid_pattern_always_rejected(self, text: str):
        """Any string NOT matching the regex must return (None, None)."""
        normalized = text.strip().upper().replace(" ", "").replace("-", "")
        if PLATE_PATTERN.match(normalized):
            return  # skip valid plates
        result, series = validator.validate(text)
        assert result is None
        assert series is None


# ---------------------------------------------------------------------------
# Property 10: Plate Series Type Classification
# ---------------------------------------------------------------------------

class TestPlateValidatorProperty10:
    """Property 10: Plate Series Type Classification — Validates: Req 12.4"""

    @given(plate=_valid_plate_st)
    @settings(max_examples=50)
    def test_bh_plates_get_bh_series(self, plate: str):
        if not plate.startswith("BH"):
            return
        _, series = validator.validate(plate)
        assert series == "BH"

    @given(plate=_valid_plate_st)
    @settings(max_examples=50)
    def test_non_bh_plates_get_normal_series(self, plate: str):
        if plate.startswith("BH"):
            return
        _, series = validator.validate(plate)
        assert series == "normal"
