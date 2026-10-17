"""Tests for the aggregate report and its read-only CLI (temporary SQLite)."""

import os
import sqlite3
import stat

import pytest

from referral_finder.classifier import (
    COMPANY_CONTACT_SEARCH,
    JOB_INTEREST,
    REFERRAL_REQUEST,
    UNCERTAIN,
)
from referral_finder.llm import AnalysisResult, ai_analysis_row, build_report, format_report
from referral_finder.llm import report as report_module
from referral_finder.llm.prompt import PROMPT_VERSION
from referral_finder.llm.versioning import effective_analysis_version
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore, ReadOnlyMessageStore

CHAT_A = -1001
CHAT_B = -1002
MATCHING = "Ищу подработку курьером, подскажите про реферальную программу"
PROVIDER = "fake"
MODEL = "fake-model"


def seed(store, chat_id, message_id, text=MATCHING):
    store.save(
        IncomingMessage(
            chat_id=chat_id,
            message_id=message_id,
            date_utc=message_id,
            received_at_utc=message_id,
            text=text,
            sender_id=7,
        )
    )


def version(criteria: str = "") -> str:
    return effective_analysis_version(PROMPT_VERSION, criteria)


def analysis_row(chat_id, message_id, *, category=None, status="ok",
                 criteria="", provider=PROVIDER, model=MODEL):
    result = None
    if category is not None:
        relevant = None if category == UNCERTAIN else True
        result = AnalysisResult(category, relevant, 0.9, "ok")
    return ai_analysis_row(
        chat_id,
        message_id,
        result,
        provider=provider,
        model=model,
        prompt_version=version(criteria),
        prompt_hash="h",
        analyzed_at_utc=1,
        status=status,
        error=None if status == "ok" else "boom",
    )


def test_report_counts_and_categories(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        for message_id in range(1, 8):
            seed(store, CHAT_A, message_id)
        store.snapshot_bootstrap([CHAT_A], limit=200)
        store.upsert_ai_analysis(
            [
                analysis_row(CHAT_A, 1, category=REFERRAL_REQUEST),
                analysis_row(CHAT_A, 2, category=JOB_INTEREST),
                analysis_row(CHAT_A, 3, category=COMPANY_CONTACT_SEARCH),
                analysis_row(CHAT_A, 4, category=UNCERTAIN),
                analysis_row(CHAT_A, 5, status="error"),
                analysis_row(CHAT_A, 6, status="invalid"),
                # message 7 has no AI row -> pending
            ]
        )

        report = build_report(
            store,
            chat_ids=[CHAT_A],
            provider=PROVIDER,
            model=MODEL,
            prompt_version=PROMPT_VERSION,
        )

        chat = report["chats"][0]
        assert chat["chat_id"] == "chat_1"
        assert chat["available_count"] == 7
        assert chat["selected_count"] == 7
        assert chat["success"] == 4
        assert chat["error"] == 1
        assert chat["invalid"] == 1
        # Still queued: the un-analysed message plus the error/invalid rows whose
        # attempts are below max_attempts (retryable). Overlap with the failure
        # counters is expected (they answer different questions).
        assert chat["pending"] == 3
        assert chat["categories"][REFERRAL_REQUEST] == 1
        assert chat["categories"][JOB_INTEREST] == 1
        assert chat["categories"][COMPANY_CONTACT_SEARCH] == 1
        assert chat["categories"][UNCERTAIN] == 1
        # The four explicitly required categories are always present as keys.
        for name in (COMPANY_CONTACT_SEARCH, REFERRAL_REQUEST, JOB_INTEREST, UNCERTAIN):
            assert name in chat["categories"]

        assert report["totals"]["success"] == 4
        assert report["totals"]["pending"] == 3
        assert report["budget"]["used"] == 0
        assert report["budget"]["limit"] == 10000
        assert report["budget"]["remaining"] == 10000
        assert report["prompt_version"] == version()


def test_report_ignores_other_versions_and_models(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        for message_id in (1, 2):
            seed(store, CHAT_A, message_id)
        store.snapshot_bootstrap([CHAT_A], limit=200)
        store.upsert_ai_analysis(
            [
                analysis_row(CHAT_A, 1, category=JOB_INTEREST),  # current version
                analysis_row(CHAT_A, 2, category=JOB_INTEREST, criteria="other"),
                analysis_row(CHAT_A, 2, category=JOB_INTEREST, model="other-model"),
            ]
        )

        report = build_report(
            store,
            chat_ids=[CHAT_A],
            provider=PROVIDER,
            model=MODEL,
            prompt_version=PROMPT_VERSION,
        )

        assert report["totals"]["success"] == 1
        assert report["chats"][0]["categories"][JOB_INTEREST] == 1


def test_report_empty_data_is_zero(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        report = build_report(
            store,
            chat_ids=[CHAT_A, CHAT_B],
            provider=PROVIDER,
            model=MODEL,
            prompt_version=PROMPT_VERSION,
        )

        assert [chat["chat_id"] for chat in report["chats"]] == ["chat_1", "chat_2"]
        for chat in report["chats"]:
            assert chat["selected_count"] == 0
            assert chat["success"] == 0
            assert chat["pending"] == 0
        assert report["totals"]["success"] == 0
        assert report["budget"]["used"] == 0


def test_report_handles_new_categories(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, CHAT_A, 1)
        store.snapshot_bootstrap([CHAT_A], limit=200)
        store.upsert_ai_analysis(
            [analysis_row(CHAT_A, 1, category="brand_new_category")]
        )

        report = build_report(
            store,
            chat_ids=[CHAT_A],
            provider=PROVIDER,
            model=MODEL,
            prompt_version=PROMPT_VERSION,
        )

        categories = report["chats"][0]["categories"]
        assert categories["brand_new_category"] == 1
        # The always-visible key categories are still present (as zeros).
        assert categories[REFERRAL_REQUEST] == 0


def test_report_uses_explicit_version_components(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, CHAT_A, 1)
        store.snapshot_bootstrap([CHAT_A], limit=200)
        store.upsert_ai_analysis([analysis_row(CHAT_A, 1, category=JOB_INTEREST)])

        report = build_report(
            store,
            chat_ids=[CHAT_A],
            provider=PROVIDER,
            model=MODEL,
            prompt_version="some-other-prompt",
        )

        # A different template version does not see the current results.
        assert report["totals"]["success"] == 0


def test_format_report_is_aggregate_only(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, CHAT_A, 1)
        store.snapshot_bootstrap([CHAT_A], limit=200)
        store.upsert_ai_analysis([analysis_row(CHAT_A, 1, category=JOB_INTEREST)])
        text = format_report(
            build_report(
                store,
                chat_ids=[CHAT_A],
                provider=PROVIDER,
                model=MODEL,
                prompt_version=PROMPT_VERSION,
            )
        )

        assert "chat_id=chat_1" in text
        assert "success=1" in text
        assert JOB_INTEREST in text
        assert MATCHING not in text
        assert "report_scope=bootstrap" in text
        assert "steady-only messages are excluded" in text
        assert "pending_note=" in text
        assert "overlap" in text
        assert "not disjoint" in text


def test_report_output_uses_temporary_chat_labels_not_real_ids(tmp_path):
    import json

    with MessageStore(tmp_path / "m.db") as store:
        seed(store, CHAT_A, 1)
        seed(store, CHAT_B, 1)
        store.snapshot_bootstrap([CHAT_A, CHAT_B], limit=200)
        report = build_report(
            store,
            chat_ids=[CHAT_A, CHAT_B],
            provider=PROVIDER,
            model=MODEL,
            prompt_version=PROMPT_VERSION,
        )

    text = format_report(report)
    payload = json.dumps(report)
    for leaked in (str(CHAT_A), str(CHAT_B)):
        assert leaked not in text
        assert leaked not in payload
    assert "message_id" not in payload
    assert "steady_after_message_id" not in payload
    assert [chat["chat_id"] for chat in report["chats"]] == ["chat_1", "chat_2"]


def test_cli_text_explains_pending_overlap_and_bootstrap_scope(tmp_path, capsys):
    db = tmp_path / "m.db"
    with MessageStore(db) as store:
        seed(store, CHAT_A, 1)
        store.snapshot_bootstrap([CHAT_A], limit=200)

    assert report_module.run(
        [
            "--db",
            str(db),
            "--provider",
            "fake",
            "--model",
            "fake-model",
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "report_scope=bootstrap" in out
    assert "pending_note=pending includes unprocessed items" in out
    assert "not disjoint" in out


def test_cli_help_marks_bootstrap_only_scope(capsys):
    with pytest.raises(SystemExit):
        report_module._build_parser().parse_args(["--help"])
    help_text = capsys.readouterr().out.lower()
    assert "bootstrap-only" in help_text
    assert "steady" in help_text


def test_cli_reports_and_writes_nothing(tmp_path, capsys):
    db = tmp_path / "m.db"
    with MessageStore(db) as store:
        seed(store, CHAT_A, 1)
        store.snapshot_bootstrap([CHAT_A], limit=200)
        store.upsert_ai_analysis([analysis_row(CHAT_A, 1, category=JOB_INTEREST)])

    assert report_module.run(
        [
            "--db",
            str(db),
            "--provider",
            "fake",
            "--model",
            "fake-model",
            "--json",
        ]
    ) == 0
    out = capsys.readouterr().out
    import json

    payload = json.loads(out)
    assert payload["totals"]["success"] == 1
    assert payload["provider"] == "fake"  # explicit CLI selection, printed
    assert MATCHING not in out

    # Read-only: no analysis rows were added and the budget was untouched.
    with MessageStore(db) as store:
        assert store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis"
        ).fetchone()[0] == 1
        assert store.api_requests_used("1970-01-01") == 0


def test_cli_missing_db_creates_nothing(tmp_path, capsys):
    db = tmp_path / "nope.db"
    assert report_module.run(["--db", str(db)]) == 2
    assert not db.exists()
    assert "not found" in capsys.readouterr().err


def test_cli_non_numeric_chats_is_refused(tmp_path, capsys):
    db = tmp_path / "nope.db"
    assert report_module.run(["--db", str(db), "--chats", "not-an-int"]) == 2
    assert not db.exists()
    assert "--chats" in capsys.readouterr().err


def test_cli_zero_budget_limit_is_refused(tmp_path, capsys):
    db = tmp_path / "nope.db"
    assert report_module.run(["--db", str(db), "--budget-limit", "0"]) == 2
    assert not db.exists()
    assert "--budget-limit" in capsys.readouterr().err


def test_cli_chat_filter(tmp_path, capsys):
    db = tmp_path / "m.db"
    with MessageStore(db) as store:
        seed(store, CHAT_A, 1)
        seed(store, CHAT_B, 1)
        store.snapshot_bootstrap([CHAT_A, CHAT_B], limit=200)

    assert report_module.run(["--db", str(db), "--chats", str(CHAT_A)]) == 0
    out = capsys.readouterr().out
    assert "chat_id=chat_1" in out
    assert "chat_id=chat_2" not in out


# --- read-only guarantees ---------------------------------------------------


def _db_snapshot(path):
    """Full logical snapshot: objects, columns, indexes and row data."""
    conn = sqlite3.connect(str(path))
    try:
        objects = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'index') ORDER BY type, name"
        ).fetchall()
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        columns = {
            table: conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            for table in tables
        }
        indexes = {
            table: conn.execute(f'PRAGMA index_list("{table}")').fetchall()
            for table in tables
        }
        data = {
            table: conn.execute(f'SELECT * FROM "{table}"').fetchall()
            for table in tables
        }
        return objects, columns, indexes, data
    finally:
        conn.close()


def _create_old_schema(path):
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE messages ("
            "chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL, "
            "date_utc INTEGER NOT NULL, received_at_utc INTEGER NOT NULL, "
            "text TEXT, sender_id INTEGER, PRIMARY KEY (chat_id, message_id))"
        )
        conn.execute("INSERT INTO messages VALUES (-1, 5, 1, 1, 'old text', 9)")
        conn.commit()
    finally:
        conn.close()


def test_cli_old_schema_is_refused_without_migration(tmp_path, capsys):
    db = tmp_path / "old.db"
    _create_old_schema(db)
    before = _db_snapshot(db)

    assert report_module.run(["--db", str(db)]) == 2
    err = capsys.readouterr().err
    assert "incompatible schema" in err

    # No tables/columns/indexes were created and no data was changed.
    assert _db_snapshot(db) == before
    conn = sqlite3.connect(str(db))
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()
    assert "ai_bootstrap_state" not in names
    assert "message_ai_analysis" not in names


def test_cli_empty_existing_file_is_refused_without_writing(tmp_path, capsys):
    db = tmp_path / "empty.db"
    db.write_bytes(b"")

    assert report_module.run(["--db", str(db)]) == 2
    assert capsys.readouterr().err.startswith("error=")
    assert db.read_bytes() == b""


def test_cli_foreign_sqlite_database_is_refused_without_changes(tmp_path, capsys):
    db = tmp_path / "foreign.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO notes(body) VALUES ('synthetic')")
        conn.commit()
    finally:
        conn.close()
    before = _db_snapshot(db)

    assert report_module.run(["--db", str(db)]) == 2
    assert "incompatible schema" in capsys.readouterr().err
    assert _db_snapshot(db) == before


def test_read_only_store_cannot_write(tmp_path):
    db = tmp_path / "m.db"
    with MessageStore(db) as store:
        seed(store, CHAT_A, 1)
        store.snapshot_bootstrap([CHAT_A], limit=200)

    with ReadOnlyMessageStore(db) as store:
        with pytest.raises(sqlite3.Error):
            store._conn.execute(
                "INSERT INTO messages "
                "(chat_id, message_id, date_utc, received_at_utc, text, sender_id) "
                "VALUES (-1, 99, 1, 1, 'x', 1)"
            )


def test_cli_works_when_database_file_is_not_writable(tmp_path, capsys):
    db = tmp_path / "m.db"
    with MessageStore(db) as store:
        seed(store, CHAT_A, 1)
        store.snapshot_bootstrap([CHAT_A], limit=200)
        store.upsert_ai_analysis([analysis_row(CHAT_A, 1, category=JOB_INTEREST)])
    before = _db_snapshot(db)

    os.chmod(db, stat.S_IREAD)
    try:
        assert report_module.run(
            [
                "--db",
                str(db),
                "--provider",
                "fake",
                "--model",
                "fake-model",
                "--json",
            ]
        ) == 0
        out = capsys.readouterr().out
        assert '"success": 1' in out
    finally:
        os.chmod(db, stat.S_IWRITE | stat.S_IREAD)

    assert _db_snapshot(db) == before
