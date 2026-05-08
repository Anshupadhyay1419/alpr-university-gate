"""
SQLAlchemy ORM models for the ALPR University Gate database.
"""

from __future__ import annotations

from sqlalchemy import Column, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class VehicleEvent(Base):
    """Persisted record of a vehicle entry/exit event."""

    __tablename__ = "vehicle_events"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    plate_number = Column(String, nullable=False)
    vehicle_type = Column(String, nullable=False)
    plate_color  = Column(String, nullable=False)
    series_type  = Column(String, nullable=False)   # "BH" or "normal"
    timestamp    = Column(String, nullable=False)   # ISO 8601
    direction    = Column(String, nullable=False)   # "IN" or "OUT"
    image_path   = Column(String, nullable=False)   # relative path to plate crop

    __table_args__ = (
        Index("idx_plate_number", "plate_number"),
        Index("idx_timestamp",    "timestamp"),
    )

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "plate_number": self.plate_number,
            "vehicle_type": self.vehicle_type,
            "plate_color":  self.plate_color,
            "series_type":  self.series_type,
            "timestamp":    self.timestamp,
            "direction":    self.direction,
            "image_path":   self.image_path,
        }
