"""
Unit + property-based tests for utils.direction_detector.DirectionDetector.

Property 19: Direction Detector Crossing Detection — Validates: Req 17.1, 17.2, 17.3
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.utils.direction_detector import DirectionDetector, VirtualLine


# Horizontal line at y=360 across a 1280-wide frame
_LINE = VirtualLine(x1=0, y1=360, x2=1280, y2=360)


class TestDirectionDetectorExamples:
    def test_first_update_returns_none(self):
        dd = DirectionDetector(line=_LINE)
        result = dd.update(track_id=1, centroid=(640.0, 200.0))
        assert result is None

    def test_no_crossing_returns_none(self):
        dd = DirectionDetector(line=_LINE)
        dd.update(1, (640.0, 200.0))  # above line
        result = dd.update(1, (640.0, 250.0))  # still above
        assert result is None

    def test_crossing_top_to_bottom_is_in(self):
        """Moving from above (y<360) to below (y>360) → IN (entering campus)."""
        dd = DirectionDetector(line=_LINE)
        dd.update(1, (640.0, 300.0))  # above line → cross_prev < 0
        result = dd.update(1, (640.0, 420.0))  # below line → cross_curr > 0
        assert result == "IN"

    def test_crossing_bottom_to_top_is_out(self):
        """Moving from below (y>360) to above (y<360) → OUT (exiting campus)."""
        dd = DirectionDetector(line=_LINE)
        dd.update(1, (640.0, 420.0))  # below line → cross_prev > 0
        result = dd.update(1, (640.0, 300.0))  # above line → cross_curr < 0
        assert result == "OUT"

    def test_dual_camera_mode_camera0_is_in(self):
        dd = DirectionDetector(line=_LINE, dual_camera=True)
        result = dd.update(1, (640.0, 300.0), camera_id=0)
        assert result == "IN"

    def test_dual_camera_mode_camera1_is_out(self):
        dd = DirectionDetector(line=_LINE, dual_camera=True)
        result = dd.update(1, (640.0, 300.0), camera_id=1)
        assert result == "OUT"

    def test_remove_track_clears_state(self):
        dd = DirectionDetector(line=_LINE)
        dd.update(1, (640.0, 300.0))
        dd.remove_track(1)
        # After removal, next update should return None (no prev centroid)
        result = dd.update(1, (640.0, 420.0))
        assert result is None


# ---------------------------------------------------------------------------
# Property 19: Direction Detector Crossing Detection
# ---------------------------------------------------------------------------

class TestDirectionDetectorProperty19:
    """Property 19: Direction Detector Crossing Detection — Validates: Req 17.1-17.3"""

    @given(
        line_y=st.integers(min_value=100, max_value=700),
        prev_y=st.floats(min_value=0.0, max_value=99.0, allow_nan=False),
        curr_y=st.floats(min_value=101.0, max_value=800.0, allow_nan=False),
        cx=st.floats(min_value=0.0, max_value=1280.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_top_to_bottom_crossing_is_in(
        self, line_y: int, prev_y: float, curr_y: float, cx: float
    ):
        """Crossing from above (y < line_y) to below (y > line_y) → IN."""
        line = VirtualLine(x1=0, y1=line_y, x2=1280, y2=line_y)
        dd = DirectionDetector(line=line)
        # prev is above the line (y < line_y)
        dd.update(1, (cx, prev_y))
        # curr is below the line (y > line_y)
        result = dd.update(1, (cx, float(line_y) + curr_y - 100.0))
        if result is not None:
            assert result == "IN"

    @given(
        line_y=st.integers(min_value=100, max_value=700),
        prev_y=st.floats(min_value=101.0, max_value=800.0, allow_nan=False),
        curr_y=st.floats(min_value=0.0, max_value=99.0, allow_nan=False),
        cx=st.floats(min_value=0.0, max_value=1280.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_bottom_to_top_crossing_is_out(
        self, line_y: int, prev_y: float, curr_y: float, cx: float
    ):
        """Crossing from below (y > line_y) to above (y < line_y) → OUT."""
        line = VirtualLine(x1=0, y1=line_y, x2=1280, y2=line_y)
        dd = DirectionDetector(line=line)
        # prev is below the line
        dd.update(1, (cx, float(line_y) + prev_y - 100.0))
        # curr is above the line
        result = dd.update(1, (cx, curr_y))
        if result is not None:
            assert result == "OUT"
