"""
Pydantic request/response schemas for the ALPR FastAPI server.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class EntryRequest(BaseModel):
    plate_number: str
    vehicle_type: str
    plate_color:  str
    series_type:  str
    direction:    Literal["IN", "OUT"]
    image_path:   str = ""


class EventResponse(BaseModel):
    id:           int
    plate_number: str
    vehicle_type: str
    plate_color:  str
    series_type:  str
    timestamp:    str
    direction:    str
    image_path:   str
