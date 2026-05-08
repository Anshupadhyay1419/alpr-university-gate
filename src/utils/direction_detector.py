"""
Entry/exit direction detector for the ALPR University Gate system.

Uses virtual line crossing logic: tracks the centroid of each vehicle
across frames and detects when it crosses a configured virtual line.
The sign of the cross-product determines IN vs OUT direction.

Also supports dual-camera mode where direction is assigned by camera source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VirtualLine:
    """A virtual line defined by two points in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_config(cls, config: dict) -> "VirtualLine":
        """Construct from the full config dict."""
        line_cfg = config.get("direction", {}).get("virtual_line", {})
        return cls(
            x1=int(line_cfg.get("x1", 0)),
            y1=int(line_cfg.get("y1", 360)),
            x2=int(line_cfg.get("x2", 1280)),
            y2=int(line_cfg.get("y2", 360)),
        )


class DirectionDetector:
    """Detect vehicle crossing direction using virtual line logic.

    Args:
        line:         The virtual line to monitor.
        dual_camera:  If True, direction is assigned by camera source ID
                      rather than line crossing.
    """

    def __init__(self, line: VirtualLine, dual_camera: bool = False) -> None:
        self.line = line
        self.dual_camera = dual_camera
        # track_id → previous centroid (cx, cy)
        self._prev_centroids: dict[int, tuple[float, float]] = {}

    @classmethod
    def from_config(cls, config: dict) -> "DirectionDetector":
        """Construct from the full config dict."""
        dir_cfg = config.get("direction", {})
        line = VirtualLine.from_config(config)
        dual_camera = bool(dir_cfg.get("dual_camera", False))
        return cls(line=line, dual_camera=dual_camera)

    def update(
        self,
        track_id: int,
        centroid: tuple[float, float],
        camera_id: int = 0,
    ) -> Optional[str]:
        """Update tracker with new centroid and detect line crossing.

        Args:
            track_id:  Integer tracking ID.
            centroid:  Current (cx, cy) centroid in pixel coordinates.
            camera_id: Camera source ID (used in dual_camera mode).

        Returns:
            "IN" if vehicle crossed into campus,
            "OUT" if vehicle crossed out,
            None if no crossing detected yet.
        """
        if self.dual_camera:
            return self._dual_camera_direction(camera_id)

        prev = self._prev_centroids.get(track_id)
        self._prev_centroids[track_id] = centroid

        if prev is None:
            return None  # First frame for this track — no crossing yet

        return self._check_crossing(prev, centroid)

    def remove_track(self, track_id: int) -> None:
        """Clean up state for a retired track."""
        self._prev_centroids.pop(track_id, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_crossing(
        self,
        prev: tuple[float, float],
        curr: tuple[float, float],
    ) -> Optional[str]:
        """Determine if the centroid crossed the virtual line."""
        # Line direction vector
        lx = self.line.x2 - self.line.x1
        ly = self.line.y2 - self.line.y1

        # Vectors from line start to prev and curr centroids
        px = prev[0] - self.line.x1
        py = prev[1] - self.line.y1
        cx = curr[0] - self.line.x1
        cy = curr[1] - self.line.y1

        # Cross products (z-component)
        cross_prev = lx * py - ly * px
        cross_curr = lx * cy - ly * cx

        # No crossing if same side or on the line
        if cross_prev * cross_curr >= 0:
            return None

        if cross_prev < 0:
            return "IN"
        else:
            return "OUT"

    @staticmethod
    def _dual_camera_direction(camera_id: int) -> str:
        """Assign direction based on camera source."""
        return "IN" if camera_id == 0 else "OUT"
