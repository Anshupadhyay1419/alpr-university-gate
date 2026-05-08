"""
Shared pytest fixtures and Hypothesis configuration for the ALPR test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------

settings.register_profile(
    "fast",
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("fast")


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------

@pytest.fixture()
def random_bgr_image():
    """Return a factory that creates random BGR images of given size."""
    def _make(height: int = 64, width: int = 128) -> np.ndarray:
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return _make


@pytest.fixture()
def random_gray_image():
    """Return a factory that creates random grayscale images of given size."""
    def _make(height: int = 32, width: int = 80) -> np.ndarray:
        return np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return _make


@pytest.fixture()
def small_plate_crop():
    """A small (30×80) BGR plate crop — below SR threshold."""
    return np.random.randint(0, 256, (30, 80, 3), dtype=np.uint8)


@pytest.fixture()
def large_plate_crop():
    """A large (60×200) BGR plate crop — above SR threshold."""
    return np.random.randint(0, 256, (60, 200, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# In-memory SQLite session
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_session():
    """Provide an in-memory SQLite session for database tests."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture()
def initialized_db(tmp_path):
    """Initialize the database module with a temp SQLite file."""
    from src.database import db as database
    db_path = str(tmp_path / "test_alpr.db")
    database.init_db(db_path)
    yield database
    # Cleanup: reset module-level engine
    database._engine = None
    database._SessionFactory = None
