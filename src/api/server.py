"""
FastAPI REST server for the ALPR University Gate system.

Endpoints:
  POST /entry          — Record a vehicle event
  GET  /logs           — Retrieve all events (newest first)
  GET  /search?plate=  — Search events by plate number

Run with:
  uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.schemas import EntryRequest, EventResponse
from src.database import db as database
from src.utils.logger import get_logger

_logger = get_logger("api.server")

app = FastAPI(
    title="ALPR University Gate API",
    description="License plate recognition entry/exit log API",
    version="1.0.0",
)


def _get_db_session():
    """Helper to get a DB session, raising 503 if DB is unavailable."""
    try:
        return database.get_session()
    except RuntimeError as exc:
        _logger.error("Database unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.post("/entry", response_model=EventResponse, status_code=201)
def create_entry(request: EntryRequest):
    """Record a new vehicle entry/exit event."""
    try:
        with database.get_session() as session:
            event_data = {
                "plate_number": request.plate_number,
                "vehicle_type": request.vehicle_type,
                "plate_color":  request.plate_color,
                "series_type":  request.series_type,
                "direction":    request.direction,
                "image_path":   request.image_path,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
            }
            event = database.insert_event(session, event_data)
            if event is None:
                raise HTTPException(status_code=500, detail="Failed to insert event")
            return EventResponse(**event.to_dict())
    except HTTPException:
        raise
    except RuntimeError as exc:
        _logger.error("Database unavailable on POST /entry: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        _logger.error("Unexpected error on POST /entry: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/logs", response_model=list[EventResponse])
def get_logs():
    """Return all vehicle events ordered by timestamp descending."""
    try:
        with database.get_session() as session:
            events = database.get_all_events(session)
            return [EventResponse(**e) for e in events]
    except RuntimeError as exc:
        _logger.error("Database unavailable on GET /logs: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        _logger.error("Unexpected error on GET /logs: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/search", response_model=list[EventResponse])
def search_by_plate(plate: str = Query(..., description="Plate number to search for")):
    """Search vehicle events by plate number."""
    try:
        with database.get_session() as session:
            events = database.search_events(session, plate_number=plate)
            return [EventResponse(**e) for e in events]
    except RuntimeError as exc:
        _logger.error("Database unavailable on GET /search: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        _logger.error("Unexpected error on GET /search: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}
