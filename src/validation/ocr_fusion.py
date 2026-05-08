"""
Multi-frame OCR fusion for the ALPR University Gate system.

Collects validated plate strings across a sliding window of frames for
each tracked vehicle and applies majority voting to produce a final result.

If no majority exists, falls back to the result with the highest confidence.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional


class OCRFusion:
    """Fuse OCR results across multiple frames using majority voting.

    Args:
        window_size: Number of frames to collect before emitting a result.
                     Also the maximum buffer size per track_id.
    """

    def __init__(self, window_size: int = 7) -> None:
        self.window_size = window_size
        # track_id → deque of (plate_string, confidence)
        self._buffers: dict[int, deque[tuple[str, float]]] = {}

    def add_result(
        self, track_id: int, plate: str, confidence: float
    ) -> None:
        """Add a validated OCR result for a tracked vehicle.

        Args:
            track_id:   Integer tracking ID from VehicleTracker.
            plate:      Validated plate string.
            confidence: OCR confidence score in [0.0, 1.0].
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.window_size)
        self._buffers[track_id].append((plate, confidence))

    def get_result(self, track_id: int) -> tuple[str, float] | None:
        """Get the current fused result for a track without clearing the buffer.

        Returns:
            (plate_string, confidence) from majority voting, or None if the
            buffer is empty.
        """
        buf = self._buffers.get(track_id)
        if not buf:
            return None
        return self._majority_vote(list(buf))

    def flush(self, track_id: int) -> tuple[str, float] | None:
        """Emit the final fused result and clear the buffer for a track.

        Called when a track is retired by VehicleTracker.

        Returns:
            (plate_string, confidence) from majority voting, or None if the
            buffer was empty.
        """
        buf = self._buffers.pop(track_id, None)
        if not buf:
            return None
        return self._majority_vote(list(buf))

    def active_track_ids(self) -> list[int]:
        """Return list of track IDs with non-empty buffers."""
        return [tid for tid, buf in self._buffers.items() if buf]

    def flush_all(self) -> dict[int, tuple[str, float]]:
        """Flush all active track buffers. Used during graceful shutdown.

        Returns:
            Dict mapping track_id → (plate_string, confidence).
        """
        results: dict[int, tuple[str, float]] = {}
        for track_id in list(self._buffers.keys()):
            result = self.flush(track_id)
            if result is not None:
                results[track_id] = result
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _majority_vote(
        entries: list[tuple[str, float]]
    ) -> tuple[str, float]:
        """Apply majority voting to a list of (plate, confidence) pairs.

        Returns the most frequent plate string. If no majority (tie),
        returns the entry with the highest confidence score.
        """
        if not entries:
            return ("", 0.0)

        if len(entries) == 1:
            return entries[0]

        # Count occurrences
        plates = [e[0] for e in entries]
        counter = Counter(plates)
        most_common_plate, most_common_count = counter.most_common(1)[0]
        total = len(entries)

        # Majority: appears more than half the time
        if most_common_count > total / 2:
            # Return with average confidence for that plate
            confs = [e[1] for e in entries if e[0] == most_common_plate]
            avg_conf = sum(confs) / len(confs)
            return (most_common_plate, avg_conf)

        # No majority — return highest confidence entry
        best = max(entries, key=lambda e: e[1])
        return best
