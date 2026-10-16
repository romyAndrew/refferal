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


def test_classify_on_ingest_defaults_false(tmp_path):
    config = load_config(make_env(), chats_file=tmp_path / "chats.txt")
    assert config.classify_on_ingest is False


def test_classify_on_ingest_truthy_and_falsy(tmp_path):
    chats = tmp_path / "chats.txt"
    assert load_config(make_env(CLASSIFY_ON_INGEST="true"), chats_file=chats).classify_on_ingest is True
    assert load_config(make_env(CLASSIFY_ON_INGEST="ON"), chats_file=chats).classify_on_ingest is True
    assert load_config(make_env(CLASSIFY_ON_INGEST="0"), chats_file=chats).classify_on_ingest is False


def test_classify_on_ingest_invalid_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(make_env(CLASSIFY_ON_INGEST="maybe"), chats_file=tmp_path / "chats.txt")


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


def test_llm_defaults_are_off_but_configured(tmp_path):
    config = load_config(make_env(), chats_file=tmp_path / "chats.txt")
    assert config.llm_enabled is False
    assert config.llm_provider == "deepseek"
    assert config.llm_model == "deepseek-flash"
    assert config.llm_api_key is None


def test_llm_values_can_be_overridden(tmp_path):
    config = load_config(
        make_env(
            LLM_ENABLED="true",
            LLM_MODEL="deepseek-chat",
            LLM_API_KEY="secret-key-value",
            LLM_MAX_RETRIES="0",
        ),
        chats_file=tmp_path / "chats.txt",
    )
    assert config.llm_enabled is True
    assert config.llm_model == "deepseek-chat"
    assert config.llm_api_key == "secret-key-value"
    assert config.llm_max_retries == 0


def test_llm_invalid_numbers_raise(tmp_path):
    chats = tmp_path / "chats.txt"
    with pytest.raises(ConfigError):
        load_config(make_env(LLM_TIMEOUT="nope"), chats_file=chats)
    with pytest.raises(ConfigError):
        load_config(make_env(LLM_MAX_RETRIES="-1"), chats_file=chats)


def test_safe_repr_hides_llm_key(tmp_path):
    config = load_config(
        make_env(LLM_API_KEY="SUPERSECRETLLMKEY"), chats_file=tmp_path / "chats.txt"
    )
    text = config.safe_repr()
    assert "SUPERSECRETLLMKEY" not in text
    assert "llm_model='deepseek-flash'" in text


def test_llm_worker_defaults(tmp_path):
    config = load_config(make_env(), chats_file=tmp_path / "chats.txt")
    assert config.llm_criteria == ""
    assert config.llm_poll_interval == 5.0
    assert config.llm_batch_size == 25
    assert config.llm_max_attempts == 3


def test_llm_worker_settings_from_env(tmp_path):
    config = load_config(
        make_env(
            LLM_CRITERIA="курьер, доставка",
            LLM_POLL_INTERVAL="1.5",
            LLM_BATCH_SIZE="10",
            LLM_MAX_ATTEMPTS="5",
        ),
        chats_file=tmp_path / "chats.txt",
    )
    assert config.llm_criteria == "курьер, доставка"
    assert config.llm_poll_interval == 1.5
    assert config.llm_batch_size == 10
    assert config.llm_max_attempts == 5


def test_llm_worker_invalid_values_raise(tmp_path):
    chats = tmp_path / "chats.txt"
    with pytest.raises(ConfigError):
        load_config(make_env(LLM_POLL_INTERVAL="0"), chats_file=chats)
    with pytest.raises(ConfigError):
        load_config(make_env(LLM_POLL_INTERVAL="-1"), chats_file=chats)
    with pytest.raises(ConfigError):
        load_config(make_env(LLM_POLL_INTERVAL="nope"), chats_file=chats)
    with pytest.raises(ConfigError):
        load_config(make_env(LLM_BATCH_SIZE="0"), chats_file=chats)
    with pytest.raises(ConfigError):
        load_config(make_env(LLM_MAX_ATTEMPTS="0"), chats_file=chats)
    with pytest.raises(ConfigError):
        load_config(make_env(LLM_BATCH_SIZE="many"), chats_file=chats)


def test_missing_worker_vars_keep_old_config(tmp_path):
    # An environment without any of the new variables still loads fine.
    env = {
        "TG_API_ID": "123456",
        "TG_API_HASH": "fake_api_hash_value",
        "TG_SESSION_PATH": "C:/secrets/referral.session",
    }
    config = load_config(env, chats_file=tmp_path / "chats.txt")
    assert config.llm_criteria == ""
    assert config.llm_poll_interval == 5.0
    assert config.llm_batch_size == 25
    assert config.llm_max_attempts == 3
