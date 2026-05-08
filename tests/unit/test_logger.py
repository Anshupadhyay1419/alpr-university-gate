"""
Unit tests for utils/logger.py — get_logger() factory.

Tests cover:
- Logger returns a logging.Logger instance
- Logger has exactly two handlers (StreamHandler + RotatingFileHandler)
- Log format includes timestamp, level, component name, and message
- Calling get_logger() twice with the same name does NOT add duplicate handlers
- Logger supports all required severity levels (INFO, WARNING, ERROR, CRITICAL)
- logs/ directory is created automatically if it does not exist
- Optional config dict is respected (custom log_file path)
"""

import logging
import os
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from src.utils.logger import get_logger, LOG_FORMAT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(log_file: str) -> dict:
    """Return a minimal config dict pointing to a custom log file."""
    return {
        "logging": {
            "log_file": log_file,
            "max_bytes": 1_048_576,  # 1 MB
            "backup_count": 3,
        }
    }


def _clear_logger(name: str) -> None:
    """Remove all handlers from a named logger so tests start clean."""
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetLoggerReturnsLogger:
    def test_returns_logging_logger_instance(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        _clear_logger("test.returns_logger")
        logger = get_logger("test.returns_logger", config=_make_config(log_file))
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches_argument(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        _clear_logger("test.name_match")
        logger = get_logger("test.name_match", config=_make_config(log_file))
        assert logger.name == "test.name_match"


class TestHandlers:
    def test_has_exactly_two_handlers(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        _clear_logger("test.two_handlers")
        logger = get_logger("test.two_handlers", config=_make_config(log_file))
        assert len(logger.handlers) == 2

    def test_has_stream_handler(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        _clear_logger("test.stream_handler")
        logger = get_logger("test.stream_handler", config=_make_config(log_file))
        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types

    def test_has_rotating_file_handler(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        _clear_logger("test.rotating_handler")
        logger = get_logger("test.rotating_handler", config=_make_config(log_file))
        handler_types = [type(h) for h in logger.handlers]
        assert RotatingFileHandler in handler_types

    def test_no_duplicate_handlers_on_repeated_calls(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        _clear_logger("test.no_duplicates")
        get_logger("test.no_duplicates", config=_make_config(log_file))
        get_logger("test.no_duplicates", config=_make_config(log_file))
        logger = logging.getLogger("test.no_duplicates")
        assert len(logger.handlers) == 2  # still exactly 2, not 4


class TestLogFormat:
    def test_formatter_uses_expected_format(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        _clear_logger("test.format")
        logger = get_logger("test.format", config=_make_config(log_file))
        for handler in logger.handlers:
            assert handler.formatter is not None
            assert handler.formatter._fmt == LOG_FORMAT

    def test_log_file_contains_expected_fields(self, tmp_path):
        """Verify a written log line contains timestamp, level, name, and message."""
        log_file = str(tmp_path / "test.log")
        _clear_logger("test.fields")
        logger = get_logger("test.fields", config=_make_config(log_file))
        logger.info("hello world")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        content = Path(log_file).read_text(encoding="utf-8")
        assert "INFO" in content
        assert "test.fields" in content
        assert "hello world" in content
        # Timestamp pattern: YYYY-MM-DD HH:MM:SS
        import re
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", content)


class TestSeverityLevels:
    def test_supports_info_level(self, tmp_path):
        log_file = str(tmp_path / "levels.log")
        _clear_logger("test.levels.info")
        logger = get_logger("test.levels.info", config=_make_config(log_file))
        # Should not raise
        logger.info("info message")

    def test_supports_warning_level(self, tmp_path):
        log_file = str(tmp_path / "levels.log")
        _clear_logger("test.levels.warning")
        logger = get_logger("test.levels.warning", config=_make_config(log_file))
        logger.warning("warning message")

    def test_supports_error_level(self, tmp_path):
        log_file = str(tmp_path / "levels.log")
        _clear_logger("test.levels.error")
        logger = get_logger("test.levels.error", config=_make_config(log_file))
        logger.error("error message")

    def test_supports_critical_level(self, tmp_path):
        log_file = str(tmp_path / "levels.log")
        _clear_logger("test.levels.critical")
        logger = get_logger("test.levels.critical", config=_make_config(log_file))
        logger.critical("critical message")

    def test_all_levels_written_to_file(self, tmp_path):
        log_file = str(tmp_path / "all_levels.log")
        _clear_logger("test.all_levels")
        logger = get_logger("test.all_levels", config=_make_config(log_file))
        logger.info("info")
        logger.warning("warning")
        logger.error("error")
        logger.critical("critical")

        for handler in logger.handlers:
            handler.flush()

        content = Path(log_file).read_text(encoding="utf-8")
        assert "INFO" in content
        assert "WARNING" in content
        assert "ERROR" in content
        assert "CRITICAL" in content


class TestDirectoryCreation:
    def test_creates_logs_directory_if_missing(self, tmp_path):
        nested_log = str(tmp_path / "nested" / "deep" / "app.log")
        _clear_logger("test.dir_creation")
        logger = get_logger("test.dir_creation", config=_make_config(nested_log))
        assert Path(nested_log).parent.exists()

    def test_log_file_is_created(self, tmp_path):
        log_file = str(tmp_path / "created.log")
        _clear_logger("test.file_created")
        logger = get_logger("test.file_created", config=_make_config(log_file))
        logger.info("trigger file creation")
        for handler in logger.handlers:
            handler.flush()
        assert Path(log_file).exists()


class TestRotatingFileHandlerConfig:
    def test_max_bytes_is_applied(self, tmp_path):
        log_file = str(tmp_path / "rotating.log")
        config = {
            "logging": {
                "log_file": log_file,
                "max_bytes": 512,
                "backup_count": 2,
            }
        }
        _clear_logger("test.rotating_config")
        logger = get_logger("test.rotating_config", config=config)
        rfh = next(h for h in logger.handlers if isinstance(h, RotatingFileHandler))
        assert rfh.maxBytes == 512
        assert rfh.backupCount == 2


class TestNoPropagation:
    def test_logger_does_not_propagate(self, tmp_path):
        log_file = str(tmp_path / "propagate.log")
        _clear_logger("test.propagate")
        logger = get_logger("test.propagate", config=_make_config(log_file))
        assert logger.propagate is False
