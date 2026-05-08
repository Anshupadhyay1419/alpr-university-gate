"""
Duplicate event filter for the ALPR University Gate system.

Suppresses repeated vehicle events for the same plate number or tracking ID
within a configurable time window (default 30 seconds).
"""

from __future__ import annotations

import time
from typing import Optional


class DuplicateFilter:
    """Suppress duplicate vehicle events within a time window.

    Args:
        window_seconds: Events for the same plate/track within this window
                        are considered duplicates (default 30).
    """

    def __init__(self, window_seconds: int = 30) -> None:
        self.window_seconds = window_seconds
        # plate_number → last recorded timestamp
        self._plate_times: dict[str, float] = {}
        # track_id → last recorded timestamp
        self._track_times: dict[int, float] = {}

    @classmethod
    def from_config(cls, config: dict) -> "DuplicateFilter":
        """Construct from the full config dict."""
        dedup_cfg = config.get("deduplication", {})
        return cls(window_seconds=int(dedup_cfg.get("window_seconds", 30)))

    def is_duplicate(
        self,
        plate_number: str,
        track_id: int,
        now: Optional[float] = None,
    ) -> bool:
        """Check whether this event is a duplicate.

        Args:
            plate_number: Validated plate string.
            track_id:     Integer tracking ID.
            now:          Current timestamp (defaults to time.time()).

        Returns:
            True if the same plate or track_id was recorded within the window.
        """
        t = now if now is not None else time.time()

        # Check plate
        last_plate = self._plate_times.get(plate_number)
        if last_plate is not None and (t - last_plate) < self.window_seconds:
            return True

        # Check track_id
        last_track = self._track_times.get(track_id)
        if last_track is not None and (t - last_track) < self.window_seconds:
            return True

        return False

    def record(
        self,
        plate_number: str,
        track_id: int,
        now: Optional[float] = None,
    ) -> None:
        """Record a vehicle event to enable future duplicate detection.

        Args:
            plate_number: Validated plate string.
            track_id:     Integer tracking ID.
            now:          Current timestamp (defaults to time.time()).
        """
        t = now if now is not None else time.time()
        self._plate_times[plate_number] = t
        self._track_times[track_id] = t

    def cleanup(self, now: Optional[float] = None) -> None:
        """Remove expired entries to prevent unbounded memory growth."""
        t = now if now is not None else time.time()
        cutoff = t - self.window_seconds

        self._plate_times = {
            k: v for k, v in self._plate_times.items() if v >= cutoff
        }
        self._track_times = {
            k: v for k, v in self._track_times.items() if v >= cutoff
        }
