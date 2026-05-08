"""
Integration tests for the FastAPI server endpoints.

Tests:
- POST /entry with valid body returns 201
- POST /entry with missing fields returns 422
- GET /logs returns array ordered by timestamp
- GET /search returns filtered results
- GET /health returns 200
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.database import db as database


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Initialize an in-memory test database before each test."""
    db_path = str(tmp_path / "test_api.db")
    database.init_db(db_path)
    yield
    database._engine = None
    database._SessionFactory = None


client = TestClient(app)


# ---------------------------------------------------------------------------
# POST /entry
# ---------------------------------------------------------------------------

class TestPostEntry:
    def _valid_payload(self, **overrides) -> dict:
        base = {
            "plate_number": "KA19TR0234",
            "vehicle_type": "Private",
            "plate_color":  "White",
            "series_type":  "normal",
            "direction":    "IN",
            "image_path":   "",
        }
        base.update(overrides)
        return base

    def test_valid_entry_returns_201(self):
        response = client.post("/entry", json=self._valid_payload())
        assert response.status_code == 201

    def test_valid_entry_returns_event_with_id(self):
        response = client.post("/entry", json=self._valid_payload())
        data = response.json()
        assert "id" in data
        assert data["plate_number"] == "KA19TR0234"

    def test_missing_plate_number_returns_422(self):
        payload = self._valid_payload()
        del payload["plate_number"]
        response = client.post("/entry", json=payload)
        assert response.status_code == 422

    def test_missing_direction_returns_422(self):
        payload = self._valid_payload()
        del payload["direction"]
        response = client.post("/entry", json=payload)
        assert response.status_code == 422

    def test_invalid_direction_returns_422(self):
        payload = self._valid_payload(direction="SIDEWAYS")
        response = client.post("/entry", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        response = client.post("/entry", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /logs
# ---------------------------------------------------------------------------

class TestGetLogs:
    def test_empty_db_returns_empty_list(self):
        response = client.get("/logs")
        assert response.status_code == 200
        assert response.json() == []

    def test_logs_returns_inserted_events(self):
        client.post("/entry", json={
            "plate_number": "KA19TR0234", "vehicle_type": "Private",
            "plate_color": "White", "series_type": "normal",
            "direction": "IN", "image_path": "",
        })
        response = client.get("/logs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["plate_number"] == "KA19TR0234"

    def test_logs_ordered_newest_first(self):
        for plate in ["PLATE_A", "PLATE_B", "PLATE_C"]:
            client.post("/entry", json={
                "plate_number": plate, "vehicle_type": "Private",
                "plate_color": "White", "series_type": "normal",
                "direction": "IN", "image_path": "",
            })
        response = client.get("/logs")
        data = response.json()
        timestamps = [e["timestamp"] for e in data]
        assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------

class TestGetSearch:
    def test_search_returns_matching_plate(self):
        client.post("/entry", json={
            "plate_number": "KA19TR0234", "vehicle_type": "Private",
            "plate_color": "White", "series_type": "normal",
            "direction": "IN", "image_path": "",
        })
        client.post("/entry", json={
            "plate_number": "MH12AB1234", "vehicle_type": "Commercial",
            "plate_color": "Yellow", "series_type": "normal",
            "direction": "OUT", "image_path": "",
        })
        response = client.get("/search?plate=KA19TR0234")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["plate_number"] == "KA19TR0234"

    def test_search_unknown_plate_returns_empty(self):
        response = client.get("/search?plate=XX99ZZ9999")
        assert response.status_code == 200
        assert response.json() == []

    def test_search_missing_plate_param_returns_422(self):
        response = client.get("/search")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
