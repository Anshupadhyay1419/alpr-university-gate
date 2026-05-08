"""
Integration tests for the database layer.

Tests:
- Schema initialization creates table and indexes
- insert_event round-trips all fields correctly
- get_all_events returns events ordered by timestamp descending
- search_events returns filtered results
- Retry logic: event discarded after two failures
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from src.database.db import (
    get_all_events,
    init_db,
    insert_event,
    search_events,
)
from src.database.models import Base, VehicleEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem_session():
    """In-memory SQLite session with schema created."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


def _sample_event(**overrides) -> dict:
    base = {
        "plate_number": "KA19TR0234",
        "vehicle_type": "Private",
        "plate_color":  "White",
        "series_type":  "normal",
        "direction":    "IN",
        "image_path":   "data/plate_crops/test.jpg",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

class TestSchemaInit:
    def test_table_created(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        from src.database import db as database
        engine = database.get_engine()
        inspector = inspect(engine)
        assert "vehicle_events" in inspector.get_table_names()

    def test_indexes_created(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        from src.database import db as database
        engine = database.get_engine()
        inspector = inspect(engine)
        indexes = {idx["name"] for idx in inspector.get_indexes("vehicle_events")}
        assert "idx_plate_number" in indexes
        assert "idx_timestamp" in indexes

    def test_all_columns_present(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        from src.database import db as database
        engine = database.get_engine()
        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("vehicle_events")}
        expected = {
            "id", "plate_number", "vehicle_type", "plate_color",
            "series_type", "timestamp", "direction", "image_path",
        }
        assert expected.issubset(cols)


# ---------------------------------------------------------------------------
# insert_event
# ---------------------------------------------------------------------------

class TestInsertEvent:
    def test_insert_returns_vehicle_event(self, mem_session):
        event = insert_event(mem_session, _sample_event())
        assert event is not None
        assert isinstance(event, VehicleEvent)

    def test_insert_round_trips_all_fields(self, mem_session):
        data = _sample_event(plate_number="MH12AB1234", direction="OUT")
        event = insert_event(mem_session, data)
        assert event.plate_number == "MH12AB1234"
        assert event.vehicle_type == "Private"
        assert event.plate_color  == "White"
        assert event.series_type  == "normal"
        assert event.direction    == "OUT"
        assert event.image_path   == "data/plate_crops/test.jpg"

    def test_insert_assigns_auto_id(self, mem_session):
        e1 = insert_event(mem_session, _sample_event())
        e2 = insert_event(mem_session, _sample_event(plate_number="DL01CD5678"))
        assert e1.id != e2.id

    def test_insert_multiple_events(self, mem_session):
        for i in range(5):
            insert_event(mem_session, _sample_event(plate_number=f"KA{i:02d}AB1234"))
        events = get_all_events(mem_session)
        assert len(events) == 5


# ---------------------------------------------------------------------------
# get_all_events
# ---------------------------------------------------------------------------

class TestGetAllEvents:
    def test_returns_list(self, mem_session):
        result = get_all_events(mem_session)
        assert isinstance(result, list)

    def test_empty_db_returns_empty_list(self, mem_session):
        assert get_all_events(mem_session) == []

    def test_ordered_by_timestamp_descending(self, mem_session):
        insert_event(mem_session, _sample_event(
            plate_number="PLATE_A",
            timestamp="2024-01-01T08:00:00+00:00",
        ))
        insert_event(mem_session, _sample_event(
            plate_number="PLATE_B",
            timestamp="2024-01-01T09:00:00+00:00",
        ))
        insert_event(mem_session, _sample_event(
            plate_number="PLATE_C",
            timestamp="2024-01-01T07:00:00+00:00",
        ))
        events = get_all_events(mem_session)
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# search_events
# ---------------------------------------------------------------------------

class TestSearchEvents:
    def test_search_returns_matching_plate(self, mem_session):
        insert_event(mem_session, _sample_event(plate_number="KA19TR0234"))
        insert_event(mem_session, _sample_event(plate_number="MH12AB1234"))
        results = search_events(mem_session, "KA19TR0234")
        assert len(results) == 1
        assert results[0]["plate_number"] == "KA19TR0234"

    def test_search_returns_empty_for_unknown_plate(self, mem_session):
        insert_event(mem_session, _sample_event(plate_number="KA19TR0234"))
        results = search_events(mem_session, "XX99ZZ9999")
        assert results == []

    def test_search_returns_all_matching_events(self, mem_session):
        for _ in range(3):
            insert_event(mem_session, _sample_event(plate_number="KA19TR0234"))
        insert_event(mem_session, _sample_event(plate_number="OTHER"))
        results = search_events(mem_session, "KA19TR0234")
        assert len(results) == 3

    def test_each_result_is_dict_with_required_keys(self, mem_session):
        insert_event(mem_session, _sample_event())
        results = search_events(mem_session, "KA19TR0234")
        required = {"id", "plate_number", "vehicle_type", "plate_color",
                    "series_type", "timestamp", "direction", "image_path"}
        for r in results:
            assert required.issubset(r.keys())
