"""
Frame capture module for the ALPR University Gate system.

Wraps cv2.VideoCapture to support RTSP streams and local video files.
Implements frame-skip for performance and automatic reconnection on failure.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("capture.frame_capture")


class FrameCapture:
    """Capture frames from an RTSP stream or local video file.

    Args:
        source:      RTSP URL or local video file path.
        frame_skip:  Process every Nth frame (1 = every frame).
        max_retries: Number of reconnect attempts before halting.
    """

    def __init__(
        self,
        source: str,
        frame_skip: int = 2,
        max_retries: int = 5,
    ) -> None:
        self.source = source
        self.frame_skip = max(1, frame_skip)
        self.max_retries = max_retries

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0

    @classmethod
    def from_config(cls, config: dict) -> "FrameCapture":
        """Construct from the full config dict."""
        video_cfg = config.get("video", {})
        return cls(
            source=str(video_cfg.get("source", "")),
            frame_skip=int(video_cfg.get("frame_skip", 2)),
            max_retries=int(video_cfg.get("max_retries", 5)),
        )

    def open(self) -> bool:
        """Open the video source with retry logic.

        Returns:
            True if opened successfully.

        Raises:
            RuntimeError: If the source cannot be opened after all retries.
        """
        for attempt in range(1, self.max_retries + 1):
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                self._cap = cap
                self._frame_count = 0
                _logger.info("Video source opened: '%s'", self.source)
                return True

            _logger.warning(
                "Failed to open video source '%s' (attempt %d/%d). "
                "Retrying in 5s...",
                self.source, attempt, self.max_retries,
            )
            cap.release()
            time.sleep(5)

        msg = (
            f"Could not open video source '{self.source}' "
            f"after {self.max_retries} attempts."
        )
        _logger.critical(msg)
        raise RuntimeError(msg)

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the next frame to be processed, respecting frame_skip.

        Returns:
            (success, frame) — frame is None when success is False.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        # Read and discard frames to implement frame_skip
        frame = None
        for _ in range(self.frame_skip):
            ret, frame = self._cap.read()
            if not ret:
                # Check if this is a video file (not RTSP) — if so, stop cleanly
                if self._is_video_file():
                    return False, None

                # RTSP stream lost — attempt reconnect
                _logger.warning(
                    "Frame read failed on source '%s'. Attempting reconnect...",
                    self.source,
                )
                reconnected = self._reconnect()
                if not reconnected:
                    return False, None
                ret, frame = self._cap.read()
                if not ret:
                    return False, None

        self._frame_count += 1
        return True, frame

    def _is_video_file(self) -> bool:
        """Return True if the source is a local video file (not RTSP/stream)."""
        src = str(self.source).lower()
        return not (src.startswith("rtsp://") or
                    src.startswith("rtmp://") or
                    src.startswith("http://") or
                    src.startswith("https://") or
                    src.isdigit())  # webcam index

    def release(self) -> None:
        """Release the video capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            _logger.info("Video source released.")

    def __enter__(self) -> "FrameCapture":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reconnect(self) -> bool:
        """Attempt to reconnect to the video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        for attempt in range(1, self.max_retries + 1):
            _logger.warning(
                "Reconnect attempt %d/%d for '%s'...",
                attempt, self.max_retries, self.source,
            )
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                self._cap = cap
                _logger.info("Reconnected to '%s'", self.source)
                return True
            cap.release()
            time.sleep(5)

        _logger.critical(
            "Could not reconnect to '%s' after %d attempts. Halting.",
            self.source, self.max_retries,
        )
        return False
