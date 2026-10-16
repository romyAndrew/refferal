"""Tests for the background AI worker (offline, temporary SQLite only)."""

import asyncio

import pytest

from referral_finder.classifier import JOB_INTEREST
from referral_finder.llm import AiWorker, AnalysisResult, FakeProvider
from referral_finder.llm.base import LLMResponseError
from referral_finder.llm.prompt import PROMPT_HASH, PROMPT_VERSION
from referral_finder.llm.versioning import effective_analysis_version
from referral_finder.models import IncomingMessage
from referral_finder.prefilter import PREFILTER_VERSION
from referral_finder.storage import MessageStore

CHAT = -1001234567890
MATCHING = "Ищу подработку курьером, подскажите про реферальную программу"
UNRELATED = "Сегодня отличная погода, идём гулять"


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


def ai_version(provider, criteria: str = "") -> str:
    return effective_analysis_version(provider.prompt_version, criteria)


def ai_count(store) -> int:
    return store._conn.execute(
        "SELECT COUNT(*) FROM message_ai_analysis"
    ).fetchone()[0]


async def wait_until(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition was not met before the timeout")


def provider_ok():
    return FakeProvider(handler=lambda request: AnalysisResult(JOB_INTEREST, True, 0.9, "ok"))


def provider_failing():
    def handler(request):
        raise LLMResponseError("transient")

    return FakeProvider(handler)


class CycleWait:
    """Deterministic poll replacement: yields and counts cycles."""

    def __init__(self):
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1
        await asyncio.sleep(0)


class ConcurrencyProbe:
    provider_name = "probe"
    model = "probe-model"
    prompt_version = PROMPT_VERSION
    prompt_hash = PROMPT_HASH

    def __init__(self):
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def analyze(self, request) -> AnalysisResult:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return AnalysisResult(JOB_INTEREST, True, 0.9, "ok")


class BlockingProbe(ConcurrencyProbe):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def analyze(self, request) -> AnalysisResult:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return AnalysisResult(JOB_INTEREST, True, 0.9, "ok")


# --- basic behaviour -------------------------------------------------------


async def test_disabled_worker_does_not_call_provider(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, 1, MATCHING)
        provider = provider_ok()
        worker = AiWorker(store, provider, enabled=False)

        await worker.start()
        await worker.stop()

        assert provider.calls == []
        assert ai_count(store) == 0
        assert worker.running is False
    finally:
        store.close()


async def test_enabled_worker_analyzes_candidate(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, 1, MATCHING)
        provider = provider_ok()
        worker = AiWorker(store, provider, enabled=True, poll_interval=0.01)
        await worker.start()
        try:
            await wait_until(
                lambda: store.get_ai_analysis(
                    CHAT, 1, "fake", "fake-model", ai_version(provider)
                )
                is not None
            )
        finally:
            await worker.stop()

        stored = store.get_ai_analysis(CHAT, 1, "fake", "fake-model", ai_version(provider))
        assert stored["category"] == JOB_INTEREST
        assert stored["status"] == "ok"
    finally:
        store.close()


async def test_message_not_passing_prefilter_is_not_sent(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, 1, UNRELATED)
        provider = provider_ok()
        worker = AiWorker(store, provider, enabled=True, poll_interval=0.01)
        await worker.start()
        try:
            await wait_until(lambda: worker.processed >= 1)
        finally:
            await worker.stop()

        assert provider.calls == []
        assert ai_count(store) == 0
    finally:
        store.close()


async def test_provider_error_does_not_damage_source_message(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, 1, MATCHING)
        provider = provider_failing()
        worker = AiWorker(store, provider, enabled=True, poll_interval=0.01)
        await worker.start()
        try:
            await wait_until(lambda: ai_count(store) >= 1)
        finally:
            await worker.stop()

        stored = store.get_ai_analysis(CHAT, 1, "fake", "fake-model", ai_version(provider))
        assert stored["status"] == "invalid"  # recorded failure, not success
        assert stored["category"] is None
        assert store.fetch(CHAT, 1)[4] == MATCHING  # source text untouched
    finally:
        store.close()


async def test_success_is_not_analyzed_again(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, 1, MATCHING)
        provider = provider_ok()
        wait = CycleWait()
        worker = AiWorker(store, provider, enabled=True, wait=wait)
        await worker.start()
        try:
            await wait_until(lambda: len(provider.calls) == 1)
            await wait_until(lambda: wait.calls >= 6)  # several more cycles
        finally:
            await worker.stop()

        assert len(provider.calls) == 1
        assert ai_count(store) == 1
    finally:
        store.close()


async def test_temporary_error_retried_only_to_limit(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, 1, MATCHING)
        provider = provider_failing()
        wait = CycleWait()
        worker = AiWorker(store, provider, enabled=True, max_attempts=2, wait=wait)
        await worker.start()
        try:
            await wait_until(lambda: len(provider.calls) == 2)
            await wait_until(lambda: wait.calls >= 8)  # keep cycling, no more calls
        finally:
            await worker.stop()

        assert len(provider.calls) == 2
        stored = store.get_ai_analysis(CHAT, 1, "fake", "fake-model", ai_version(provider))
        assert stored["attempts"] == 2
    finally:
        store.close()


async def test_provider_calls_are_sequential(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in range(1, 4):
            seed(store, message_id, MATCHING)
        probe = ConcurrencyProbe()
        worker = AiWorker(store, probe, enabled=True, wait=CycleWait())
        await worker.start()
        try:
            await wait_until(lambda: probe.calls >= 3)
        finally:
            await worker.stop()

        assert probe.max_active == 1
    finally:
        store.close()


# --- batch bounding --------------------------------------------------------


def test_next_batch_is_bounded(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in range(1, 6):
            seed(store, message_id, MATCHING)

        small = AiWorker(store, provider_ok(), enabled=True, batch_size=2)
        assert len(small._next_batch()) == 2

        large = AiWorker(store, provider_ok(), enabled=True, batch_size=25)
        assert len(large._next_batch()) == 5  # never more than available
    finally:
        store.close()


def test_worker_rejects_invalid_parameters(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        with pytest.raises(ValueError):
            AiWorker(store, provider_ok(), enabled=True, poll_interval=0)
        with pytest.raises(ValueError):
            AiWorker(store, provider_ok(), enabled=True, batch_size=0)
        with pytest.raises(ValueError):
            AiWorker(store, provider_ok(), enabled=True, max_attempts=0)
    finally:
        store.close()


# --- lifecycle -------------------------------------------------------------


async def test_start_is_idempotent(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        worker = AiWorker(store, provider_ok(), enabled=True, wait=CycleWait())
        await worker.start()
        try:
            await worker.start()  # must not create a second task
            tasks = [t for t in asyncio.all_tasks() if t.get_name() == "ai-worker"]
            assert len(tasks) == 1
            assert worker.running is True
        finally:
            await worker.stop()
    finally:
        store.close()


async def test_stop_is_reentrant_and_leaves_no_tasks(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        worker = AiWorker(store, provider_ok(), enabled=True, wait=CycleWait())
        await worker.start()
        await worker.stop()
        await worker.stop()  # safe to call again

        assert worker.running is False
        leaked = [t for t in asyncio.all_tasks() if t.get_name() == "ai-worker"]
        assert leaked == []
    finally:
        store.close()


async def test_stop_before_start_is_safe(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        worker = AiWorker(store, provider_ok(), enabled=True)
        await worker.stop()
        assert worker.running is False
    finally:
        store.close()


async def test_cancellation_mid_analysis_writes_nothing(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, 1, MATCHING)
        probe = BlockingProbe()
        worker = AiWorker(store, probe, enabled=True, wait=CycleWait())
        await worker.start()
        await asyncio.wait_for(probe.entered.wait(), timeout=2)

        # Cancel while the provider is in flight: this must not become a
        # success and must not increment attempts.
        worker._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker.stop()

        assert ai_count(store) == 0
        assert store.fetch(CHAT, 1)[4] == MATCHING
        leaked = [t for t in asyncio.all_tasks() if t.get_name() == "ai-worker"]
        assert leaked == []
    finally:
        store.close()


async def test_worker_does_not_close_the_store(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, 1, MATCHING)
        provider = provider_ok()
        worker = AiWorker(store, provider, enabled=True, poll_interval=0.01)
        await worker.start()
        try:
            await wait_until(lambda: ai_count(store) >= 1)
        finally:
            await worker.stop()

        # The store is still usable: the lifecycle owner closes it, not us.
        assert store.count() == 1
        assert store.save(
            IncomingMessage(
                chat_id=CHAT,
                message_id=2,
                date_utc=2,
                received_at_utc=2,
                text="another",
            )
        ) is True
    finally:
        store.close()


async def test_relevant_messages_beyond_first_batch_are_processed(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        # The first batch (ordered by chat_id, message_id) is unrelated and never
        # passes prefilter; the matching messages come later. Without persisting
        # the prefilter decision, the first batch would be re-selected forever and
        # the later messages would starve.
        for message_id in range(1, 4):
            seed(store, message_id, UNRELATED)
        for message_id in range(4, 6):
            seed(store, message_id, MATCHING)

        provider = provider_ok()
        worker = AiWorker(store, provider, enabled=True, batch_size=2, wait=CycleWait())
        await worker.start()
        try:
            await wait_until(lambda: len(provider.calls) == 2, timeout=2.0)
        finally:
            await worker.stop()

        assert len(provider.calls) == 2
        assert ai_count(store) == 2
    finally:
        store.close()


async def test_prefilter_decisions_are_persisted(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, 1, UNRELATED)
        seed(store, 2, MATCHING)
        provider = provider_ok()
        worker = AiWorker(store, provider, enabled=True, wait=CycleWait())
        await worker.start()
        try:
            await wait_until(lambda: len(provider.calls) == 1)
        finally:
            await worker.stop()

        assert store.get_prefilter_result(CHAT, 1, PREFILTER_VERSION)["matched"] is False
        assert store.get_prefilter_result(CHAT, 2, PREFILTER_VERSION)["matched"] is True
    finally:
        store.close()


async def test_restart_does_not_reanalyze_success(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, 1, MATCHING)
        first = provider_ok()
        worker = AiWorker(store, first, enabled=True, wait=CycleWait())
        await worker.start()
        try:
            await wait_until(lambda: ai_count(store) == 1)
        finally:
            await worker.stop()
        assert len(first.calls) == 1

        # A brand-new worker on the same store must not re-analyze a success,
        # and must not be starved by persisted prefilter decisions.
        second = provider_ok()
        restarted = AiWorker(store, second, enabled=True, wait=CycleWait())
        await restarted.start()
        try:
            await wait_until(lambda: restarted.cycles >= 5)
        finally:
            await restarted.stop()

        assert len(second.calls) == 0
        assert ai_count(store) == 1
    finally:
        store.close()


async def test_bumping_prefilter_version_rechecks_rejected(tmp_path, monkeypatch):
    import referral_finder.llm.service as service_module
    import referral_finder.llm.worker as worker_module

    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, 1, UNRELATED)
        provider = provider_ok()
        worker = AiWorker(store, provider, enabled=True, wait=CycleWait())
        await worker.start()
        try:
            await wait_until(
                lambda: store.get_prefilter_result(CHAT, 1, PREFILTER_VERSION) is not None
            )
        finally:
            await worker.stop()

        # Rejection recorded under the current version.
        assert store.get_prefilter_result(CHAT, 1, PREFILTER_VERSION)["matched"] is False

        # Bump the prefilter version: the old decision no longer applies, so the
        # message is checked again and a fresh decision is recorded.
        bumped = "prefilter-test-bump"
        monkeypatch.setattr(worker_module, "PREFILTER_VERSION", bumped)
        monkeypatch.setattr(service_module, "PREFILTER_VERSION", bumped)

        worker2 = AiWorker(store, provider, enabled=True, wait=CycleWait())
        await worker2.start()
        try:
            await wait_until(
                lambda: store.get_prefilter_result(CHAT, 1, bumped) is not None
            )
        finally:
            await worker2.stop()

        assert store.get_prefilter_result(CHAT, 1, bumped) is not None
        assert store.get_prefilter_result(CHAT, 1, PREFILTER_VERSION)["matched"] is False
    finally:
        store.close()
