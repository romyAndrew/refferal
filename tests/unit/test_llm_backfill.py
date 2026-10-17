"""Tests for the opt-in historical backfill CLI (offline, temporary SQLite).

No real provider is constructed here: the CLI is exercised with an injected
offline provider, so no Telegram or DeepSeek connection is ever made.
"""

import pytest

from referral_finder.config import ConfigError
from referral_finder.llm import (
    AnalysisResult,
    ApiRequestBudget,
    FakeProvider,
    LLMBudgetExceededError,
    LLMConfigError,
)
from referral_finder.llm import cli, smoke
from referral_finder.llm.prompt import PROMPT_HASH, PROMPT_VERSION
from referral_finder.llm.versioning import effective_analysis_version
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore

CHAT = -1001
MATCHING = "Ищу подработку курьером, подскажите про реферальную программу"


def seed(db, count):
    with MessageStore(db) as store:
        for message_id in range(1, count + 1):
            store.save(
                IncomingMessage(
                    chat_id=CHAT,
                    message_id=message_id,
                    date_utc=message_id,
                    received_at_utc=message_id,
                    text=MATCHING,
                    sender_id=7,
                )
            )


def provider_ok():
    return FakeProvider(
        handler=lambda request: AnalysisResult("job_interest", True, 0.9, "ok")
    )


class BudgetProvider:
    provider_name = "budget"
    model = "budget-model"
    prompt_version = PROMPT_VERSION
    prompt_hash = PROMPT_HASH

    def __init__(self, budget):
        self._budget = budget
        self.calls = 0

    async def analyze(self, request) -> AnalysisResult:
        if not self._budget.reserve():
            raise LLMBudgetExceededError("daily budget exhausted")
        self.calls += 1
        return AnalysisResult("job_interest", True, 0.9, "ok")


# --- argument gating --------------------------------------------------------


def test_include_history_requires_live(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 1)
    assert cli.run(
        ["--db", str(db), "--include-history", "--limit", "1"],
        enabled=True,
        provider=provider_ok(),
    ) == 2
    assert "requires --live" in capsys.readouterr().err


def test_live_requires_include_history(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 1)
    assert cli.run(
        ["--db", str(db), "--live"], enabled=True, provider=provider_ok()
    ) == 2
    assert "together with --include-history" in capsys.readouterr().err


def test_include_history_requires_positive_limit(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 1)
    assert cli.run(
        ["--db", str(db), "--live", "--include-history"],
        enabled=True,
        provider=provider_ok(),
    ) == 2
    assert "positive integer" in capsys.readouterr().err

    assert cli.run(
        ["--db", str(db), "--live", "--include-history", "--limit", "0"],
        enabled=True,
        provider=provider_ok(),
    ) == 2


def test_live_missing_env_is_refused(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 1)
    assert cli.run(
        ["--db", str(db), "--live", "--include-history", "--limit", "1"],
        env={},
    ) == 2
    assert "LLM_ENABLED is not true" in capsys.readouterr().err


def test_live_missing_key_is_refused(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 1)
    assert cli.run(
        ["--db", str(db), "--live", "--include-history", "--limit", "1"],
        env={"LLM_ENABLED": "true"},
    ) == 2
    assert "is not set" in capsys.readouterr().err


@pytest.mark.parametrize("provider_name", ["fake", "none"])
def test_live_rejects_offline_provider_without_processing(
    tmp_path, capsys, monkeypatch, provider_name
):
    db = tmp_path / "m.db"
    seed(db, 2)

    def must_not_build(*args, **kwargs):
        raise AssertionError("provider_from_config must not be called")

    monkeypatch.setattr(cli, "provider_from_config", must_not_build)
    assert cli.run(
        ["--db", str(db), "--live", "--include-history", "--limit", "1"],
        env={
            "LLM_ENABLED": "true",
            "DEEPSEEK_API_KEY": "fake-key",
            "LLM_PROVIDER": provider_name,
        },
    ) == 2
    assert "real provider" in capsys.readouterr().err

    with MessageStore(db) as store:
        count = store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis"
        ).fetchone()[0]
        assert count == 0


def test_live_invalid_limit_is_refused_without_processing(
    tmp_path, capsys, monkeypatch
):
    db = tmp_path / "m.db"
    seed(db, 2)

    def must_not_build(*args, **kwargs):
        raise AssertionError("provider_from_config must not be called")

    monkeypatch.setattr(cli, "provider_from_config", must_not_build)
    assert cli.run(
        ["--db", str(db), "--live", "--include-history", "--limit", "1"],
        env={
            "LLM_ENABLED": "true",
            "DEEPSEEK_API_KEY": "fake-key",
            "LLM_PROVIDER": "deepseek",
            "LLM_MAX_API_REQUESTS_PER_DAY": "0",
        },
    ) == 2
    assert "LLM_MAX_API_REQUESTS_PER_DAY" in capsys.readouterr().err

    with MessageStore(db) as store:
        count = store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis"
        ).fetchone()[0]
        assert count == 0


def test_live_rejects_injected_fake_provider(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 1)

    assert cli.run(
        ["--db", str(db), "--live", "--include-history", "--limit", "1"],
        env={"LLM_ENABLED": "true", "DEEPSEEK_API_KEY": "fake-key"},
        provider=provider_ok(),
    ) == 2
    assert "FakeProvider" in capsys.readouterr().err

    with MessageStore(db) as store:
        count = store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis"
        ).fetchone()[0]
        assert count == 0


def test_live_missing_db_creates_nothing(tmp_path, capsys):
    db = tmp_path / "missing.db"
    assert cli.run(
        ["--db", str(db), "--live", "--include-history", "--limit", "1"],
        env={"LLM_ENABLED": "true", "DEEPSEEK_API_KEY": "fake-key"},
    ) == 2
    assert not db.exists()
    assert "not found" in capsys.readouterr().err


def test_missing_company_catalog_is_refused_without_processing(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 2)
    missing_catalog = tmp_path / "missing-companies.yaml"

    assert cli.run(
        ["--db", str(db), "--companies", str(missing_catalog)],
        enabled=True,
        provider=provider_ok(),
    ) == 2
    assert "catalog" in capsys.readouterr().err.lower()

    with MessageStore(db) as store:
        count = store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis"
        ).fetchone()[0]
        assert count == 0


def test_batch_size_must_be_positive(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 1)
    assert cli.run(
        ["--db", str(db), "--batch-size", "0"],
        enabled=True,
        provider=provider_ok(),
    ) == 2
    assert "batch-size" in capsys.readouterr().err


# --- shared daily limit -----------------------------------------------------


def test_live_paths_share_budget_limit_resolution():
    env = {
        "LLM_ENABLED": "true",
        "DEEPSEEK_API_KEY": "fake-key",
        "LLM_PROVIDER": "deepseek",
        "LLM_MAX_API_REQUESTS_PER_DAY": "4",
    }
    config = cli._llm_config_from_env(env)
    assert config.llm_max_api_requests_per_day == 4
    assert smoke._limit_from_env(env) == 4


def test_live_paths_reject_invalid_budget_limit():
    env = {
        "LLM_ENABLED": "true",
        "DEEPSEEK_API_KEY": "fake-key",
        "LLM_PROVIDER": "deepseek",
        "LLM_MAX_API_REQUESTS_PER_DAY": "0",
    }
    with pytest.raises(ConfigError):
        cli._llm_config_from_env(env)
    with pytest.raises(LLMConfigError):
        smoke._limit_from_env(env)


# --- budget / queue behaviour ----------------------------------------------


def test_backfill_stops_at_budget_and_leaves_queue(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 5)
    with MessageStore(db) as store:
        budget = ApiRequestBudget(store, limit=2, clock=lambda: 0)
        provider = BudgetProvider(budget)

        assert cli.run(
            ["--db", str(db), "--limit", "5"], enabled=True, provider=provider
        ) == 0

        assert provider.calls == 2
        assert budget.used() == 2
        version = effective_analysis_version(provider.prompt_version, "")
        # Three messages remain queued (nothing deleted, no attempts recorded).
        assert store.count_ai_unprocessed(
            "budget", "budget-model", version
        ) == 3
        rows = store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis WHERE status = 'ok'"
        ).fetchone()[0]
        assert rows == 2

    out = capsys.readouterr().out
    assert "budget_exhausted=1" in out
    assert "saved=2" in out


def test_backfill_is_idempotent_for_success(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 3)
    provider = provider_ok()

    cli.run(["--db", str(db)], enabled=True, provider=provider)
    assert "saved=3" in capsys.readouterr().out

    cli.run(["--db", str(db)], enabled=True, provider=provider_ok())
    assert "checked=0" in capsys.readouterr().out
