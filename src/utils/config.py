"""
Config loader utility for the ALPR University Gate system.

Provides:
- ConfigError: custom exception raised on any configuration problem
- load_config(path): reads and validates config.yaml, returning the full dict
"""

import yaml

from src.utils.logger import get_logger

# Logger for this module (uses defaults since config may not be loaded yet)
_logger = get_logger("utils.config")

# All top-level keys that must be present and must map to a dict
REQUIRED_KEYS = [
    "video",
    "detection",
    "tracking",
    "preprocessing",
    "enhancement",
    "ocr",
    "fusion",
    "deduplication",
    "color_classifier",
    "direction",
    "database",
    "api",
    "dashboard",
    "logging",
    "training",
]


class ConfigError(Exception):
    """Raised when the configuration file is missing, unparseable, or invalid."""


def load_config(path: str = "config/config.yaml") -> dict:
    """
    Read and validate the YAML configuration file at *path*.

    Validation rules:
    - The file must exist and be readable.
    - The file must be valid YAML that parses to a dict.
    - Every key in REQUIRED_KEYS must be present at the top level.
    - Each required key must map to a dict (not None, not a scalar, not a list).

    Args:
        path: Filesystem path to the YAML config file.
              Defaults to "config/config.yaml".

    Returns:
        The full configuration as a plain Python dict.

    Raises:
        ConfigError: If the file is missing, cannot be parsed, or fails validation.
    """
    # --- 1. Read the file ---
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        msg = f"Configuration file not found: '{path}'"
        _logger.critical(msg)
        raise ConfigError(msg) from None
    except yaml.YAMLError as exc:
        msg = f"Failed to parse configuration file '{path}': {exc}"
        _logger.critical(msg)
        raise ConfigError(msg) from exc

    # --- 2. Top-level must be a dict ---
    if not isinstance(raw, dict):
        msg = (
            f"Configuration file '{path}' must contain a YAML mapping at the top "
            f"level, got {type(raw).__name__}."
        )
        _logger.critical(msg)
        raise ConfigError(msg)

    # --- 3. Validate required keys ---
    for key in REQUIRED_KEYS:
        if key not in raw:
            msg = (
                f"Required configuration key '{key}' is missing from '{path}'."
            )
            _logger.critical(msg)
            raise ConfigError(msg)

        value = raw[key]
        if value is None:
            msg = (
                f"Required configuration key '{key}' in '{path}' must be a "
                f"mapping (dict), but its value is None."
            )
            _logger.critical(msg)
            raise ConfigError(msg)

        if not isinstance(value, dict):
            msg = (
                f"Required configuration key '{key}' in '{path}' must be a "
                f"mapping (dict), but got {type(value).__name__}."
            )
            _logger.critical(msg)
            raise ConfigError(msg)

    return raw
