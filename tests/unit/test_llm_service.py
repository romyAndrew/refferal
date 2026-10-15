"""Tests for the offline analysis service (temporary DBs, no network)."""

import logging

from referral_finder.classifier import IRRELEVANT, JOB_INTEREST, UNCERTAIN
from referral_finder.llm import (
    STATUS_DISABLED,
    STATUS_DUPLICATE,
    STATUS_PREFILTER_SKIPPED,
    STATUS_SAVED,
    STATUS_STORAGE_ERROR,
    AnalysisResult,
    FakeProvider,
    LLMRequest,
    LLMResponseError,
    MessageAnalysisService,
)
from referral_finder.llm.deepseek import DeepSeekProvider, HttpResponse
from referral_finder.llm.versioning import effective_analysis_version
from referral_finder.models import IncomingMessage
from referral_finder.scrubbing import scrub
from referral_finder.storage import MessageStore


def eff(provider, criteria: str = "") -> str:
    return effective_analysis_version(provider.prompt_version, criteria)


CHAT = -1001234567890
MATCHING = "Ищу работу курьером, пишите @ivan_petrov или ivan@test.ru, https://t.me/ivan/5"
UNRELATED = "Сегодня отличная погода, идём гулять"


class FakeTransport:
    def __init__(self):
        self.calls = 0

    async def send(self, request):
        self.calls += 1
        return HttpResponse(200, b"{}")


def seed(store, message_id, text):
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


def provider_returning(result):
    return FakeProvider(handler=lambda request: result)


async def test_message_not_passing_prefilter_never_calls_provider(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, UNRELATED)
        provider = provider_returning(AnalysisResult(JOB_INTEREST, True, 0.9, "x"))
        service = MessageAnalysisService(store, provider, enabled=True)

        outcome = await service.analyze(CHAT, 1, UNRELATED)

        assert outcome.status == STATUS_PREFILTER_SKIPPED
        assert provider.calls == []
        assert store.get_ai_analysis(CHAT, 1, "fake", "fake-model", eff(provider)) is None


async def test_provider_receives_scrubbed_text_only(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)
        provider = provider_returning(AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))
        service = MessageAnalysisService(store, provider, enabled=True, criteria="курьер")

        await service.analyze(CHAT, 1, MATCHING)

        assert len(provider.calls) == 1
        sent = provider.calls[0]
        assert sent.text == scrub(MATCHING).text
        assert sent.criteria == "курьер"
        for leaked in ("@ivan_petrov", "ivan@test.ru", "t.me", "ivan"):
            assert leaked not in sent.text
        # The request object carries no Telegram identifiers at all.
        assert not hasattr(sent, "chat_id")
        assert not hasattr(sent, "message_id")


async def test_valid_result_is_stored_separately_from_messages(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)
        provider = provider_returning(AnalysisResult(JOB_INTEREST, True, 0.9, "reason"))
        service = MessageAnalysisService(store, provider, enabled=True)

        outcome = await service.analyze(CHAT, 1, MATCHING)

        assert outcome.status == STATUS_SAVED
        assert store.fetch(CHAT, 1)[4] == MATCHING  # source row untouched
        stored = store.get_ai_analysis(CHAT, 1, "fake", "fake-model", eff(provider))
        assert stored["category"] == JOB_INTEREST
        assert stored["relevant"] is True


async def test_reprocessing_same_version_is_idempotent(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)
        provider = provider_returning(AnalysisResult(JOB_INTEREST, True, 0.9, "reason"))
        service = MessageAnalysisService(store, provider, enabled=True)

        first = await service.analyze(CHAT, 1, MATCHING)
        second = await service.analyze(CHAT, 1, MATCHING)

        assert first.status == STATUS_SAVED
        assert second.status == STATUS_DUPLICATE
        count = store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis"
        ).fetchone()[0]
        assert count == 1


async def test_uncertain_is_stored_as_uncertain_not_irrelevant(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)
        provider = provider_returning(AnalysisResult(UNCERTAIN, None, 0.2, "?"))
        service = MessageAnalysisService(store, provider, enabled=True)

        await service.analyze(CHAT, 1, MATCHING)

        stored = store.get_ai_analysis(CHAT, 1, "fake", "fake-model", eff(provider))
        assert stored["category"] == UNCERTAIN
        assert stored["category"] != IRRELEVANT
        assert stored["relevant"] is None


async def test_provider_error_does_not_stop_other_messages(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)
        seed(store, 2, MATCHING + " срочно")

        good = AnalysisResult(JOB_INTEREST, True, 0.9, "ok")

        def handler(request: LLMRequest):
            if "срочно" in request.text:
                raise RuntimeError("boom")
            return good

        service = MessageAnalysisService(store, FakeProvider(handler), enabled=True)

        first = await service.analyze(CHAT, 1, MATCHING)
        second = await service.analyze(CHAT, 2, MATCHING + " срочно")

        assert first.status == STATUS_SAVED
        assert second.status == "error"
        # Both messages were recorded; the failure did not abort the batch.
        rows = store._conn.execute(
            "SELECT COUNT(*) FROM message_ai_analysis"
        ).fetchone()[0]
        assert rows == 2


async def test_provider_response_error_is_stored_as_invalid(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)

        def handler(request: LLMRequest):
            raise LLMResponseError("bad json")

        provider = FakeProvider(handler)
        service = MessageAnalysisService(store, provider, enabled=True)

        outcome = await service.analyze(CHAT, 1, MATCHING)

        assert outcome.status == "invalid"
        stored = store.get_ai_analysis(CHAT, 1, "fake", "fake-model", eff(provider))
        assert stored["status"] == "invalid"
        assert stored["category"] is None  # not stored as a successful label


async def test_non_result_object_is_rejected(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)

        def handler(request: LLMRequest):
            return {"category": JOB_INTEREST}  # not an AnalysisResult

        provider = FakeProvider(handler)
        service = MessageAnalysisService(store, provider, enabled=True)

        outcome = await service.analyze(CHAT, 1, MATCHING)

        assert outcome.status == "invalid"
        stored = store.get_ai_analysis(CHAT, 1, "fake", "fake-model", eff(provider))
        assert stored["status"] == "invalid"


async def test_disabled_service_never_calls_network_provider(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)
        transport = FakeTransport()
        provider = DeepSeekProvider(api_key="placeholder", transport=transport)
        service = MessageAnalysisService(store, provider, enabled=False)

        outcome = await service.analyze(CHAT, 1, MATCHING)

        assert outcome.status == STATUS_DISABLED
        assert transport.calls == 0
        assert store.get_ai_analysis(
            CHAT, 1, provider.provider_name, provider.model, eff(provider)
        ) is None


async def test_storage_error_is_reported_without_leaking_data(tmp_path, caplog):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)
        provider = provider_returning(AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))
        service = MessageAnalysisService(store, provider, enabled=True)

        def boom(rows):
            raise RuntimeError("db down")

        store.upsert_ai_analysis = boom  # type: ignore[assignment]

        with caplog.at_level(logging.DEBUG, logger="referral_finder.llm"):
            outcome = await service.analyze(CHAT, 1, MATCHING)

        assert outcome.status == STATUS_STORAGE_ERROR
        assert outcome.error == "storage_error"
        for leaked in ("@ivan_petrov", "ivan@test.ru", "t.me", MATCHING, str(CHAT)):
            assert leaked not in caplog.text


async def test_logs_contain_no_text_or_pii_on_success(tmp_path, caplog):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)
        provider = provider_returning(AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))
        service = MessageAnalysisService(store, provider, enabled=True)

        with caplog.at_level(logging.DEBUG, logger="referral_finder.llm"):
            await service.analyze(CHAT, 1, MATCHING)

        for leaked in ("@ivan_petrov", "ivan@test.ru", "t.me", MATCHING, str(CHAT)):
            assert leaked not in caplog.text


async def test_same_criteria_yields_one_result(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)
        provider = provider_returning(AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))
        service = MessageAnalysisService(store, provider, enabled=True, criteria="курьер")

        first = await service.analyze(CHAT, 1, MATCHING)
        second = await service.analyze(CHAT, 1, MATCHING)

        assert first.status == STATUS_SAVED
        assert second.status == STATUS_DUPLICATE
        count = store._conn.execute("SELECT COUNT(*) FROM message_ai_analysis").fetchone()[0]
        assert count == 1


async def test_changed_criteria_stores_a_separate_result(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)
        provider = provider_returning(AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))
        narrow = MessageAnalysisService(store, provider, enabled=True, criteria="курьер")
        wide = MessageAnalysisService(store, provider, enabled=True, criteria="курьер, доставка")

        assert (await narrow.analyze(CHAT, 1, MATCHING)).status == STATUS_SAVED
        # The old criteria result must not satisfy the new criteria identity.
        assert store.get_ai_analysis(
            CHAT, 1, "fake", "fake-model", eff(provider, "курьер, доставка")
        ) is None
        assert (await wide.analyze(CHAT, 1, MATCHING)).status == STATUS_SAVED

        count = store._conn.execute("SELECT COUNT(*) FROM message_ai_analysis").fetchone()[0]
        assert count == 2
        assert store.get_ai_analysis(
            CHAT, 1, "fake", "fake-model", eff(provider, "курьер")
        ) is not None
        assert store.get_ai_analysis(
            CHAT, 1, "fake", "fake-model", eff(provider, "курьер, доставка")
        ) is not None


async def test_success_is_not_reanalyzed(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)
        provider = provider_returning(AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))
        service = MessageAnalysisService(store, provider, enabled=True)

        await service.analyze(CHAT, 1, MATCHING)
        second = await service.analyze(CHAT, 1, MATCHING)

        assert second.status == STATUS_DUPLICATE
        assert len(provider.calls) == 1  # provider not called again


async def test_temporary_failure_is_retried(tmp_path):
    with MessageStore(tmp_path / "m.db") as store:
        seed(store, 1, MATCHING)

        def failing(request):
            raise LLMResponseError("transient")

        bad_provider = FakeProvider(failing)
        first = MessageAnalysisService(store, bad_provider, enabled=True)
        assert (await first.analyze(CHAT, 1, MATCHING)).status == "invalid"
        stored = store.get_ai_analysis(CHAT, 1, "fake", "fake-model", eff(bad_provider))
        assert stored["status"] == "invalid"
        assert stored["attempts"] == 1

        # A later attempt with a working provider succeeds and replaces the
        # failure (a temporary error must not block the message forever).
        good = FakeProvider(handler=lambda request: AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))
        second = MessageAnalysisService(store, good, enabled=True)
        assert (await second.analyze(CHAT, 1, MATCHING)).status == STATUS_SAVED
        stored = store.get_ai_analysis(CHAT, 1, "fake", "fake-model", eff(good))
        assert stored["status"] == "ok"
        assert stored["attempts"] == 2
