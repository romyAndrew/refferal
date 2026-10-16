"""Tests for the offline pipeline CLI (temporary DB, FakeProvider, no network)."""

import logging

import pytest

from referral_finder.classifier import JOB_INTEREST
from referral_finder.llm import AnalysisResult, FakeProvider
from referral_finder.llm import cli
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore

CHAT = -1001234567890
MATCHING = "Ищу работу курьером, пишите @ivan_petrov или ivan@test.ru"
UNRELATED = "Сегодня отличная погода, идём гулять"


def seed(db_path, items):
    with MessageStore(db_path) as store:
        for message_id, text in items:
            store.save(
                IncomingMessage(
                    chat_id=CHAT,
                    message_id=message_id,
                    date_utc=message_id,
                    received_at_utc=message_id,
                    text=text,
                    sender_id=7,
                )
            )


def provider_ok():
    return FakeProvider(handler=lambda request: AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))


def test_run_is_cwd_independent(tmp_path, monkeypatch, capsys):
    db = tmp_path / "m.db"
    seed(db, [(1, MATCHING)])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    # The default companies catalog is resolved from the repository, not CWD.
    assert cli.run(["--db", str(db)], enabled=True, provider=provider_ok()) == 0
    out = capsys.readouterr().out
    assert "saved=1" in out


def test_dry_run_counts_and_writes_nothing(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, [(1, MATCHING), (2, MATCHING), (3, UNRELATED)])

    assert cli.run(["--db", str(db), "--dry-run"], provider=provider_ok()) == 0
    out = capsys.readouterr().out
    assert "dry_run=1" in out
    assert "pending=3" in out

    with MessageStore(db) as store:
        assert store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis"
        ).fetchone()[0] == 0


def test_disabled_does_not_process_or_store(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, [(1, MATCHING)])
    provider = provider_ok()

    assert cli.run(["--db", str(db)], enabled=False, provider=provider) == 0
    out = capsys.readouterr().out
    assert "enabled=0" in out
    assert provider.calls == []
    with MessageStore(db) as store:
        assert store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis"
        ).fetchone()[0] == 0


def test_enabled_processes_and_reports_aggregates_only(tmp_path, capsys, caplog):
    db = tmp_path / "m.db"
    seed(db, [(1, MATCHING), (2, MATCHING), (3, UNRELATED)], )

    with caplog.at_level(logging.DEBUG, logger="referral_finder.llm"):
        assert cli.run(["--db", str(db)], enabled=True, provider=provider_ok()) == 0
    out = capsys.readouterr().out

    assert "enabled=1" in out
    assert "checked=3" in out
    assert "prefilter_passed=2" in out
    assert "saved=2" in out
    assert "prefilter_skipped=1" in out
    # Aggregate-only: no message text or PII anywhere.
    for leaked in ("@ivan_petrov", "ivan@test.ru", MATCHING, str(CHAT)):
        assert leaked not in out
        assert leaked not in caplog.text

    with MessageStore(db) as store:
        assert store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis"
        ).fetchone()[0] == 2


def test_limit_is_respected(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, [(1, MATCHING), (2, MATCHING), (3, MATCHING)])

    assert cli.run(
        ["--db", str(db), "--limit", "1"], enabled=True, provider=provider_ok()
    ) == 0
    out = capsys.readouterr().out
    assert "checked=1" in out


def test_second_run_is_idempotent(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, [(1, MATCHING), (2, MATCHING)])
    cli.run(["--db", str(db)], enabled=True, provider=provider_ok())
    capsys.readouterr()

    cli.run(["--db", str(db)], enabled=True, provider=provider_ok())
    out = capsys.readouterr().out
    assert "checked=0" in out  # nothing left to process for this version

    with MessageStore(db) as store:
        assert store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis"
        ).fetchone()[0] == 2


def test_enabled_read_from_environment(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, [(1, MATCHING)])

    cli.run(["--db", str(db)], env={}, provider=provider_ok())
    assert "enabled=0" in capsys.readouterr().out

    cli.run(
        ["--db", str(db)], env={"LLM_ENABLED": "true"}, provider=provider_ok()
    )
    assert "enabled=1" in capsys.readouterr().out


def test_db_argument_is_required():
    with pytest.raises(SystemExit):
        cli.run([], enabled=True, provider=provider_ok())


def test_iter_ai_unprocessed_skips_done_rows(tmp_path):
    db = tmp_path / "m.db"
    seed(db, [(1, MATCHING), (2, UNRELATED)])
    provider = provider_ok()
    with MessageStore(db) as store:
        batches = list(
            store.iter_ai_unprocessed(
                provider.provider_name, provider.model, provider.prompt_version
            )
        )
        ids = sorted(mid for batch in batches for _, mid, _ in batch)
        assert ids == [1, 2]
