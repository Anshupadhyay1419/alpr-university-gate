"""
Logger utility for the ALPR University Gate system.

Provides a get_logger() factory that creates loggers writing to both
console (StreamHandler) and a rotating file (RotatingFileHandler),
configured from config/config.yaml.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# Default log format: timestamp | level | component | message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default logging config (used if config.yaml is unavailable)
_DEFAULT_LOG_FILE = "logs/alpr.log"
_DEFAULT_MAX_BYTES = 10_485_760  # 10 MB
_DEFAULT_BACKUP_COUNT = 5


def get_logger(name: str, config: Optional[dict] = None) -> logging.Logger:
    """
    Return a named logger configured with a StreamHandler and a RotatingFileHandler.

    If the logger already has handlers attached (i.e. get_logger was called
    previously with the same name), the existing logger is returned as-is to
    avoid duplicate handlers.

    Args:
        name:   Component name used as the logger name (appears in every log line).
        config: Optional pre-loaded config dict.  If omitted, config/config.yaml is
                read from the project root.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls with the same name
    if logger.handlers:
        return logger

    # Resolve logging settings
    log_file, max_bytes, backup_count = _resolve_logging_config(config)

    # Ensure the logs directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler — INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Rotating file handler — all levels
    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Prevent log records from propagating to the root logger
    logger.propagate = False

    return logger


def _resolve_logging_config(config: Optional[dict]) -> tuple[str, int, int]:
    """
    Extract logging parameters from the provided config dict or from config/config.yaml.

    Returns:
        (log_file, max_bytes, backup_count)
    """
    if config is not None:
        logging_cfg = config.get("logging", {})
    else:
        logging_cfg = _load_logging_section_from_yaml()

    log_file = logging_cfg.get("log_file", _DEFAULT_LOG_FILE)
    max_bytes = int(logging_cfg.get("max_bytes", _DEFAULT_MAX_BYTES))
    backup_count = int(logging_cfg.get("backup_count", _DEFAULT_BACKUP_COUNT))

    return log_file, max_bytes, backup_count


def _load_logging_section_from_yaml() -> dict:
    """
    Read config/config.yaml from the project root and return the 'logging' section.

    Falls back to an empty dict (triggering defaults) if the file is missing
    or cannot be parsed, so that the logger itself never raises during setup.
    """
    try:
        import yaml  # PyYAML — listed in requirements.txt

        # Try config/config.yaml first, fall back to config.yaml for compatibility
        for config_path in [Path("config/config.yaml"), Path("config.yaml")]:
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                return data.get("logging", {})

        return {}

    except Exception:
        # Logger setup must never crash the application
        return {}
