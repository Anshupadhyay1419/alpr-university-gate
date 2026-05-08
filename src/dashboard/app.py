"""
Streamlit dashboard for the ALPR University Gate system.
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path

# Add project root to path so imports work when run from any directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.utils.config import load_config
from src.database import db as database


def _init_db(config: dict) -> None:
    """Initialize DB connection if not already done."""
    db_path = config.get("database", {}).get("path", "data/alpr.db")
    try:
        database.init_db(db_path)
    except Exception:
        pass  # Already initialized


def _load_events(limit: int = 100) -> list[dict]:
    """Load recent events from the database."""
    try:
        with database.get_session() as session:
            return database.get_all_events(session)[:limit]
    except Exception as exc:
        st.error(f"Database error: {exc}")
        return []


def _search_events(plate: str) -> list[dict]:
    """Search events by plate number."""
    try:
        with database.get_session() as session:
            return database.search_events(session, plate_number=plate.upper().strip())
    except Exception as exc:
        st.error(f"Database error: {exc}")
        return []


def _render_events_table(events: list[dict]) -> None:
    """Render events as a Streamlit table with image previews."""
    if not events:
        st.info("No events found.")
        return

    for event in events:
        with st.container():
            col1, col2 = st.columns([1, 3])

            # Image preview
            with col1:
                img_path = event.get("image_path", "")
                if img_path and Path(img_path).exists():
                    st.image(img_path, caption=event.get("plate_number", ""), width=150)
                else:
                    st.markdown("📷 *No image*")

            # Event details
            with col2:
                st.markdown(f"**Plate:** `{event.get('plate_number', 'N/A')}`")
                st.markdown(
                    f"**Type:** {event.get('vehicle_type', 'N/A')} | "
                    f"**Color:** {event.get('plate_color', 'N/A')} | "
                    f"**Series:** {event.get('series_type', 'N/A')}"
                )
                st.markdown(
                    f"**Direction:** {event.get('direction', 'N/A')} | "
                    f"**Time:** {event.get('timestamp', 'N/A')}"
                )
            st.divider()


def main() -> None:
    st.set_page_config(
        page_title="ALPR University Gate",
        page_icon="🚗",
        layout="wide",
    )

    # Load config
    try:
        config = load_config("config/config.yaml")
    except Exception as exc:
        st.error(f"Failed to load config: {exc}")
        return

    dashboard_cfg = config.get("dashboard", {})
    if not dashboard_cfg.get("enabled", True):
        st.warning("Dashboard is disabled in config/config.yaml.")
        return

    refresh_interval = int(dashboard_cfg.get("refresh_interval_seconds", 3))

    _init_db(config)

    st.title("🚗 ALPR University Gate — Live Monitor")

    # Sidebar
    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    search_plate = st.sidebar.text_input("Search by plate number", "").strip()

    # Main content
    tab_live, tab_search = st.tabs(["📋 Live Logs", "🔍 Search"])

    with tab_live:
        st.subheader("Recent Vehicle Events")
        events = _load_events(limit=50)
        _render_events_table(events)

        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

    with tab_search:
        st.subheader("Search by Plate Number")
        if search_plate:
            results = _search_events(search_plate)
            st.markdown(f"**{len(results)} result(s) for `{search_plate}`**")
            _render_events_table(results)
        else:
            st.info("Enter a plate number in the sidebar to search.")


if __name__ == "__main__":
    main()
