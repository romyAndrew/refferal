"""Tests for the offline backfill CLI (no Telegram, temporary DB)."""

import referral_finder.classify as cli
from referral_finder.classifier import CLASSIFIER_VERSION
from referral_finder.classify import ClassifierService
from referral_finder.ingest import MessageWriter
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore


def seed(db_path, count, text="ищу работу курьером"):
    store = MessageStore(db_path)
    for message_id in range(1, count + 1):
        store.save(
            IncomingMessage(
                chat_id=-1,
                message_id=message_id,
                date_utc=message_id,
                received_at_utc=message_id,
                text=text,
            )
        )
    store.close()


def test_classifier_service_classifies_saved_messages(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    message = IncomingMessage(
        chat_id=-1,
        message_id=1,
        date_utc=1,
        received_at_utc=1,
        text="дайте реферальную ссылку",
    )
    store.save(message)

    assert ClassifierService(store).classify_saved([message]) == 1
    stored = store.get_classification(-1, 1, CLASSIFIER_VERSION)
    assert stored["category"] == "referral_request"
    store.close()


def test_classifier_service_skips_textless_messages(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    message = IncomingMessage(
        chat_id=-1, message_id=1, date_utc=1, received_at_utc=1, text=None
    )
    assert ClassifierService(store).classify_saved([message]) == 0
    store.close()


async def test_online_hook_classifies_saved_message(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    service = ClassifierService(store)
    writer = MessageWriter(store, on_saved=service.classify_saved)
    await writer.start()
    try:
        await writer.submit(
            IncomingMessage(
                chat_id=-1,
                message_id=1,
                date_utc=1,
                received_at_utc=1,
                text="Набираю курьеров, оплата ежедневно",
            )
        )
        await writer.flush()
    finally:
        await writer.stop()

    stored = store.get_classification(-1, 1, CLASSIFIER_VERSION)
    assert stored["category"] == "recruitment_offer"
    store.close()


def test_backfill_classifies_and_reports(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 3)

    assert cli.run(["--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert f"version={CLASSIFIER_VERSION}" in out
    assert "processed=3" in out
    assert "stored=3" in out
    assert "errors=0" in out
    assert "job_interest=3" in out
    assert "курьером" not in out  # never print message text

    store = MessageStore(db)
    assert store.count_unclassified(CLASSIFIER_VERSION) == 0
    stored = store.get_classification(-1, 1, CLASSIFIER_VERSION)
    assert stored is not None and stored["category"] == "job_interest"
    store.close()


def test_dry_run_writes_nothing(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 2)

    assert cli.run(["--db", str(db), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "unclassified=2" in out
    assert "dry_run=1" in out

    store = MessageStore(db)
    assert store.count_unclassified(CLASSIFIER_VERSION) == 2
    store.close()


def test_limit_processes_only_requested_count(tmp_path):
    db = tmp_path / "m.db"
    seed(db, 5)

    assert cli.run(["--db", str(db), "--limit", "2"]) == 0

    store = MessageStore(db)
    assert store.count_unclassified(CLASSIFIER_VERSION) == 3
    store.close()


def test_second_run_is_idempotent(tmp_path, capsys):
    db = tmp_path / "m.db"
    seed(db, 3)

    cli.run(["--db", str(db)])
    capsys.readouterr()
    cli.run(["--db", str(db)])
    out = capsys.readouterr().out
    assert "processed=0" in out
    assert "stored=0" in out

    store = MessageStore(db)
    count = store._conn.execute(
        "SELECT COUNT(*) FROM message_classifications"
    ).fetchone()[0]
    assert count == 3
    store.close()


def test_classification_error_is_counted_and_does_not_stop(tmp_path, capsys, monkeypatch):
    db = tmp_path / "m.db"
    seed(db, 3)

    def boom(text):
        raise RuntimeError("classifier down")

    monkeypatch.setattr(cli, "classify", boom)
    assert cli.run(["--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "processed=3" in out
    assert "errors=3" in out
    assert "stored=0" in out

    store = MessageStore(db)
    assert store.count_unclassified(CLASSIFIER_VERSION) == 3
    store.close()


def test_batches_are_resumable(tmp_path):
    db = tmp_path / "m.db"
    seed(db, 7)

    assert cli.run(["--db", str(db), "--batch-size", "2", "--limit", "4"]) == 0
    store = MessageStore(db)
    remaining = store.count_unclassified(CLASSIFIER_VERSION)
    store.close()
    assert remaining == 3

    assert cli.run(["--db", str(db), "--batch-size", "2"]) == 0
    store = MessageStore(db)
    assert store.count_unclassified(CLASSIFIER_VERSION) == 0
    store.close()
