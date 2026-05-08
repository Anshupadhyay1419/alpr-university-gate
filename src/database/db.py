"""
Database session management and CRUD operations for the ALPR system.

Uses SQLAlchemy with SQLite. Schema is auto-created on first run.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base, VehicleEvent
from src.utils.logger import get_logger

_logger = get_logger("database.db")

_engine = None
_SessionFactory = None


def init_db(db_path: str = "data/alpr.db") -> None:
    """Initialize the SQLite database and create schema if needed.

    Args:
        db_path: Path to the SQLite database file.
    """
    global _engine, _SessionFactory

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(_engine)
    _SessionFactory = sessionmaker(bind=_engine)
    _logger.info("Database initialized at '%s'", db_path)


def get_engine():
    """Return the current SQLAlchemy engine."""
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager that yields a database session."""
    if _SessionFactory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def insert_event(session: Session, event_data: dict) -> Optional[VehicleEvent]:
    """Insert a vehicle event record with one retry on failure.

    Args:
        session:    Active SQLAlchemy session.
        event_data: Dict with keys matching VehicleEvent columns.

    Returns:
        The inserted VehicleEvent, or None if both attempts failed.
    """
    for attempt in range(2):
        try:
            event = VehicleEvent(
                plate_number=event_data["plate_number"],
                vehicle_type=event_data["vehicle_type"],
                plate_color=event_data["plate_color"],
                series_type=event_data["series_type"],
                timestamp=event_data.get(
                    "timestamp",
                    datetime.now(timezone.utc).isoformat()
                ),
                direction=event_data["direction"],
                image_path=event_data.get("image_path", ""),
            )
            session.add(event)
            session.flush()
            return event
        except Exception as exc:
            if attempt == 0:
                _logger.error(
                    "DB insert failed (attempt 1), retrying: %s", exc
                )
                session.rollback()
            else:
                _logger.error(
                    "DB insert failed (attempt 2), discarding event: %s", exc
                )
    return None


def get_all_events(session: Session) -> list[dict]:
    """Return all vehicle events ordered by timestamp descending."""
    events = (
        session.query(VehicleEvent)
        .order_by(VehicleEvent.timestamp.desc())
        .all()
    )
    return [e.to_dict() for e in events]


def search_events(session: Session, plate_number: str) -> list[dict]:
    """Return all events matching the given plate number."""
    events = (
        session.query(VehicleEvent)
        .filter(VehicleEvent.plate_number == plate_number)
        .order_by(VehicleEvent.timestamp.desc())
        .all()
    )
    return [e.to_dict() for e in events]


def save_plate_image(
    plate_crop: np.ndarray,
    plate_number: str,
    save_dir: str = "data/plate_crops/",
) -> str:
    """Save a plate crop image and return its relative path.

    Args:
        plate_crop:   NumPy array (grayscale or BGR).
        plate_number: Used to build the filename.
        save_dir:     Directory to save images.

    Returns:
        Relative path to the saved image, or "" on failure.
    """
    try:
        import cv2

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{plate_number}_{ts}.jpg"
        filepath = Path(save_dir) / filename

        cv2.imwrite(str(filepath), plate_crop)
        return str(filepath)
    except Exception as exc:
        _logger.warning("Failed to save plate image: %s", exc)
        return ""
