"""Tests for configuration loading.

All tests use fake environment variable mappings and temporary files. They
never read real local settings and require no Telegram credentials.
"""

from pathlib import Path

import pytest

from referral_finder.config import ConfigError, load_config, parse_chats_file


def make_env(**overrides: str) -> dict:
    env = {
        "TG_API_ID": "123456",
        "TG_API_HASH": "fake_api_hash_value",
        "TG_SESSION_PATH": "C:/secrets/referral.session",
    }
    env.update(overrides)
    return env


def test_load_config_valid_values(tmp_path):
    # Pass an explicit (non-existent) chats file so the test never reads the
    # developer's real config/chats.txt.
    config = load_config(make_env(), chats_file=tmp_path / "chats.txt")
    assert config.api_id == 123456
    assert config.api_hash == "fake_api_hash_value"
    assert config.session_path == Path("C:/secrets/referral.session")
    assert config.log_level == "INFO"
    assert config.chats == ()


def test_load_config_missing_required_raises():
    with pytest.raises(ConfigError):
        load_config({})
    with pytest.raises(ConfigError):
        load_config({"TG_API_ID": "123456"})


def test_api_id_must_be_integer():
    with pytest.raises(ConfigError):
        load_config(make_env(TG_API_ID="not-a-number"))


def test_api_id_must_be_positive():
    with pytest.raises(ConfigError):
        load_config(make_env(TG_API_ID="0"))


def test_optional_defaults_and_overrides():
    config = load_config(make_env(DB_PATH="C:/tmp/other.db", LOG_LEVEL="debug"))
    assert config.db_path == Path("C:/tmp/other.db")
    assert config.log_level == "DEBUG"


def test_safe_repr_hides_api_hash():
    config = load_config(make_env(TG_API_HASH="SUPERSECRET"))
    text = config.safe_repr()
    assert "SUPERSECRET" not in text
    assert "***" in text


def test_parse_chats_file_ignores_blanks_and_comments(tmp_path):
    chats_file = tmp_path / "chats.txt"
    chats_file.write_text(
        "# a comment\n\n   -1001234567890   \n-1009876543210\n@example_public_group\n",
        encoding="utf-8",
    )
    config = load_config(make_env(), chats_file=chats_file)
    assert config.chats == ("-1001234567890", "-1009876543210", "@example_public_group")


def test_missing_chats_file_yields_empty(tmp_path):
    config = load_config(make_env(), chats_file=tmp_path / "missing.txt")
    assert config.chats == ()


def test_parse_chats_file_direct(tmp_path):
    chats_file = tmp_path / "chats.txt"
    chats_file.write_text("1\n2\n", encoding="utf-8")
    assert parse_chats_file(chats_file) == ("1", "2")
    assert parse_chats_file(tmp_path / "nope.txt") == ()
