"""
Unit tests for utils/config.py — load_config() and ConfigError.

Tests cover:
- Valid config.yaml loads successfully and returns a dict
- Missing file raises ConfigError with a descriptive message
- Missing required top-level key raises ConfigError naming the missing key
- Key present but with None value raises ConfigError
- Key present but with wrong type (e.g. string instead of dict) raises ConfigError
- All 15 required keys are validated individually
"""

import textwrap
from pathlib import Path

import pytest
import yaml

from src.utils.config import ConfigError, load_config, REQUIRED_KEYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: dict | str, filename: str = "config.yaml") -> str:
    """Write *content* (dict or raw YAML string) to a temp file and return its path."""
    config_file = tmp_path / filename
    if isinstance(content, dict):
        config_file.write_text(yaml.dump(content), encoding="utf-8")
    else:
        config_file.write_text(content, encoding="utf-8")
    return str(config_file)


def _full_valid_config() -> dict:
    """Return a minimal but fully valid config dict with all 15 required keys."""
    return {key: {"_placeholder": True} for key in REQUIRED_KEYS}


# ---------------------------------------------------------------------------
# REQUIRED_KEYS list
# ---------------------------------------------------------------------------

class TestRequiredKeysList:
    def test_required_keys_contains_all_15(self):
        expected = [
            "video", "detection", "tracking", "preprocessing", "enhancement",
            "ocr", "fusion", "deduplication", "color_classifier", "direction",
            "database", "api", "dashboard", "logging", "training",
        ]
        assert REQUIRED_KEYS == expected

    def test_required_keys_has_15_entries(self):
        assert len(REQUIRED_KEYS) == 15


# ---------------------------------------------------------------------------
# Valid config
# ---------------------------------------------------------------------------

class TestValidConfig:
    def test_valid_config_returns_dict(self, tmp_path):
        path = _write_yaml(tmp_path, _full_valid_config())
        result = load_config(path)
        assert isinstance(result, dict)

    def test_valid_config_contains_all_required_keys(self, tmp_path):
        path = _write_yaml(tmp_path, _full_valid_config())
        result = load_config(path)
        for key in REQUIRED_KEYS:
            assert key in result

    def test_valid_config_returns_full_content(self, tmp_path):
        """Extra keys beyond the required set are preserved in the returned dict."""
        cfg = _full_valid_config()
        cfg["extra_section"] = {"foo": "bar"}
        path = _write_yaml(tmp_path, cfg)
        result = load_config(path)
        assert "extra_section" in result
        assert result["extra_section"] == {"foo": "bar"}

    def test_real_config_yaml_loads_successfully(self):
        """The project's own config.yaml must pass validation."""
        result = load_config("config/config.yaml")
        assert isinstance(result, dict)
        for key in REQUIRED_KEYS:
            assert key in result


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------

class TestMissingFile:
    def test_missing_file_raises_config_error(self, tmp_path):
        missing = str(tmp_path / "nonexistent.yaml")
        with pytest.raises(ConfigError):
            load_config(missing)

    def test_missing_file_error_message_contains_path(self, tmp_path):
        missing = str(tmp_path / "nonexistent.yaml")
        with pytest.raises(ConfigError, match="nonexistent.yaml"):
            load_config(missing)

    def test_missing_file_error_is_descriptive(self, tmp_path):
        missing = str(tmp_path / "missing_config.yaml")
        with pytest.raises(ConfigError) as exc_info:
            load_config(missing)
        assert len(str(exc_info.value)) > 10  # not an empty message


# ---------------------------------------------------------------------------
# Invalid YAML
# ---------------------------------------------------------------------------

class TestInvalidYaml:
    def test_invalid_yaml_raises_config_error(self, tmp_path):
        path = _write_yaml(tmp_path, "key: [unclosed bracket", filename="bad.yaml")
        with pytest.raises(ConfigError):
            load_config(path)

    def test_non_mapping_yaml_raises_config_error(self, tmp_path):
        """A YAML file that parses to a list (not a dict) should raise ConfigError."""
        path = _write_yaml(tmp_path, "- item1\n- item2\n", filename="list.yaml")
        with pytest.raises(ConfigError):
            load_config(path)


# ---------------------------------------------------------------------------
# Missing required keys
# ---------------------------------------------------------------------------

class TestMissingRequiredKey:
    @pytest.mark.parametrize("missing_key", REQUIRED_KEYS)
    def test_missing_key_raises_config_error(self, tmp_path, missing_key):
        cfg = _full_valid_config()
        del cfg[missing_key]
        path = _write_yaml(tmp_path, cfg)
        with pytest.raises(ConfigError):
            load_config(path)

    @pytest.mark.parametrize("missing_key", REQUIRED_KEYS)
    def test_missing_key_error_names_the_key(self, tmp_path, missing_key):
        cfg = _full_valid_config()
        del cfg[missing_key]
        path = _write_yaml(tmp_path, cfg)
        with pytest.raises(ConfigError, match=missing_key):
            load_config(path)

    def test_empty_config_raises_config_error(self, tmp_path):
        path = _write_yaml(tmp_path, {})
        with pytest.raises(ConfigError):
            load_config(path)


# ---------------------------------------------------------------------------
# Key present but None value
# ---------------------------------------------------------------------------

class TestNoneValue:
    @pytest.mark.parametrize("null_key", REQUIRED_KEYS)
    def test_none_value_raises_config_error(self, tmp_path, null_key):
        cfg = _full_valid_config()
        cfg[null_key] = None
        path = _write_yaml(tmp_path, cfg)
        with pytest.raises(ConfigError):
            load_config(path)

    @pytest.mark.parametrize("null_key", REQUIRED_KEYS)
    def test_none_value_error_names_the_key(self, tmp_path, null_key):
        cfg = _full_valid_config()
        cfg[null_key] = None
        path = _write_yaml(tmp_path, cfg)
        with pytest.raises(ConfigError, match=null_key):
            load_config(path)

    def test_yaml_null_value_raises_config_error(self, tmp_path):
        """Explicit YAML null (~) should also raise ConfigError."""
        cfg = _full_valid_config()
        # Overwrite one key with explicit null in raw YAML
        raw_yaml = yaml.dump(cfg)
        raw_yaml = raw_yaml.replace(
            "video:\n  _placeholder: true",
            "video: ~",
        )
        path = _write_yaml(tmp_path, raw_yaml, filename="null_val.yaml")
        with pytest.raises(ConfigError, match="video"):
            load_config(path)


# ---------------------------------------------------------------------------
# Key present but wrong type (not a dict)
# ---------------------------------------------------------------------------

class TestWrongType:
    @pytest.mark.parametrize("bad_value", ["a string", 42, 3.14, True, [1, 2, 3]])
    def test_string_value_raises_config_error(self, tmp_path, bad_value):
        cfg = _full_valid_config()
        cfg["video"] = bad_value
        path = _write_yaml(tmp_path, cfg)
        with pytest.raises(ConfigError):
            load_config(path)

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_string_value_for_each_key_raises_config_error(self, tmp_path, key):
        cfg = _full_valid_config()
        cfg[key] = "not_a_dict"
        path = _write_yaml(tmp_path, cfg)
        with pytest.raises(ConfigError, match=key):
            load_config(path)

    def test_list_value_raises_config_error(self, tmp_path):
        cfg = _full_valid_config()
        cfg["detection"] = ["item1", "item2"]
        path = _write_yaml(tmp_path, cfg)
        with pytest.raises(ConfigError, match="detection"):
            load_config(path)

    def test_integer_value_raises_config_error(self, tmp_path):
        cfg = _full_valid_config()
        cfg["tracking"] = 99
        path = _write_yaml(tmp_path, cfg)
        with pytest.raises(ConfigError, match="tracking"):
            load_config(path)

    def test_wrong_type_error_names_the_key(self, tmp_path):
        cfg = _full_valid_config()
        cfg["ocr"] = "paddleocr"
        path = _write_yaml(tmp_path, cfg)
        with pytest.raises(ConfigError, match="ocr"):
            load_config(path)


# ---------------------------------------------------------------------------
# ConfigError is a proper Exception subclass
# ---------------------------------------------------------------------------

class TestConfigErrorClass:
    def test_config_error_is_exception_subclass(self):
        assert issubclass(ConfigError, Exception)

    def test_config_error_can_be_raised_and_caught(self):
        with pytest.raises(ConfigError):
            raise ConfigError("test error")

    def test_config_error_message_is_preserved(self):
        msg = "something went wrong"
        with pytest.raises(ConfigError, match=msg):
            raise ConfigError(msg)
