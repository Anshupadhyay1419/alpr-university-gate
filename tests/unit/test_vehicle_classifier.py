"""
Unit + property-based tests for classification.vehicle_classifier.

Property 16: Vehicle Type Mapping Correctness — Validates: Req 15.1, 15.2, 15.3
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.classification.vehicle_classifier import (
    COLOR_TO_TYPE,
    VALID_VEHICLE_TYPES,
    VehicleClassifier,
)

classifier = VehicleClassifier()


class TestVehicleClassifierExamples:
    @pytest.mark.parametrize("color,expected", [
        ("White",      "Private"),
        ("Yellow",     "Commercial"),
        ("Green",      "EV"),
        ("Red",        "Govt/Temp"),
        ("Blue",       "Diplomatic"),
        ("Black",      "Rental"),
        ("Army_Green", "Military"),
        ("Unknown",    "Unknown"),
    ])
    def test_known_color_mapping(self, color, expected):
        assert classifier.classify(color) == expected

    def test_unknown_color_returns_unknown(self):
        assert classifier.classify("Purple") == "Unknown"
        assert classifier.classify("") == "Unknown"
        assert classifier.classify("INVALID") == "Unknown"


class TestVehicleClassifierProperty16:
    """Property 16: Vehicle Type Mapping Correctness — Validates: Req 15.1, 15.2, 15.3"""

    @given(color=st.sampled_from(list(COLOR_TO_TYPE.keys())))
    @settings(max_examples=20)
    def test_known_color_always_returns_valid_type(self, color: str):
        result = classifier.classify(color)
        assert result in VALID_VEHICLE_TYPES
        assert result == COLOR_TO_TYPE[color]

    @given(color=st.text(min_size=0, max_size=20))
    @settings(max_examples=30)
    def test_any_input_returns_valid_type(self, color: str):
        """For any input, the result is always a valid vehicle type."""
        result = classifier.classify(color)
        assert result in VALID_VEHICLE_TYPES
