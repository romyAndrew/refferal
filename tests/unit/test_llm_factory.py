"""Tests for the provider factory (no network)."""

import contextlib
import io
import logging

import pytest

from referral_finder.config import load_config
from referral_finder.llm import (
    DEEPSEEK_API_KEY_ENV,
    DeepSeekProvider,
    FakeProvider,
    LLMConfigError,
    provider_from_config,
)
from referral_finder.logging_setup import setup_logging
from referral_finder.storage import MessageStore


def make_config(tmp_path, **overrides):
    env = {
        "TG_API_ID": "123456",
        "TG_API_HASH": "fake_hash",
        "TG_SESSION_PATH": "C:/secrets/s.session",
    }
    env.update(overrides)
    return load_config(env, chats_file=tmp_path / "chats.txt")


def test_disabled_returns_fake_provider(tmp_path):
    config = make_config(tmp_path, LLM_ENABLED="false")
    assert isinstance(provider_from_config(config, environ={}), FakeProvider)


def test_enabled_fake_provider_stays_offline(tmp_path):
    config = make_config(tmp_path, LLM_ENABLED="true", LLM_PROVIDER="fake")
    assert isinstance(provider_from_config(config, environ={}), FakeProvider)


def test_enabled_deepseek_with_key_builds_real_provider(tmp_path):
    config = make_config(
        tmp_path, LLM_ENABLED="true", LLM_PROVIDER="deepseek", LLM_MODEL="deepseek-flash"
    )
    with MessageStore(tmp_path / "m.db") as store:
        provider = provider_from_config(
            config,
            store=store,
            environ={DEEPSEEK_API_KEY_ENV: "env-key"},
            transport=object(),
        )
        assert isinstance(provider, DeepSeekProvider)
        assert provider.model == "deepseek-flash"
        assert provider.prompt_version  # template version present
        assert provider._budget.limit == config.llm_max_api_requests_per_day


def test_enabled_deepseek_without_store_raises(tmp_path):
    # A live provider must never be built without a persistent budget store.
    config = make_config(tmp_path, LLM_ENABLED="true", LLM_PROVIDER="deepseek")
    with pytest.raises(LLMConfigError):
        provider_from_config(config, environ={DEEPSEEK_API_KEY_ENV: "k"})


def test_enabled_without_key_raises(tmp_path):
    config = make_config(tmp_path, LLM_ENABLED="true")
    with pytest.raises(LLMConfigError):
        provider_from_config(config, environ={})


def test_key_is_read_only_from_environment_not_config(tmp_path):
    # A key present in config (e.g. from .env) must be ignored; only the
    # environment variable counts.
    config = make_config(tmp_path, LLM_ENABLED="true", LLM_API_KEY="dotenv-key")
    with pytest.raises(LLMConfigError):
        provider_from_config(config, environ={})


def test_unsupported_provider_raises(tmp_path):
    config = make_config(tmp_path, LLM_ENABLED="true", LLM_PROVIDER="other")
    with pytest.raises(LLMConfigError):
        provider_from_config(config, environ={DEEPSEEK_API_KEY_ENV: "k"})


def test_factory_registers_key_in_secret_filter(tmp_path):
    config = make_config(tmp_path, LLM_ENABLED="true")
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(buffer):
            setup_logging(level="INFO", secrets=[])
            with MessageStore(tmp_path / "m.db") as store:
                provider = provider_from_config(
                    config, store=store, environ={DEEPSEEK_API_KEY_ENV: "S3CRETKEY"}
                )
            assert isinstance(provider, DeepSeekProvider)
            logging.getLogger("unit.factory").info("using key S3CRETKEY now")
    finally:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)

    output = buffer.getvalue()
    assert "S3CRETKEY" not in output
    assert "***" in output
