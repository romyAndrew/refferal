"""Lifecycle integration tests: Telegram runner + optional AI worker (offline).

Everything here uses a fake runner, a temporary SQLite database and the
deterministic :class:`FakeProvider` (or a small budget-backed fake). No Telegram,
no DeepSeek and no real database/session are touched.
"""

import asyncio
from types import SimpleNamespace

import pytest

from referral_finder.app import run_application
from referral_finder.classifier import JOB_INTEREST
from referral_finder.companies import CompanyCatalogError, parse_companies
from referral_finder.config import Config
from referral_finder.llm import (
    AnalysisResult,
    ApiRequestBudget,
    BootstrapSteadyWorker,
    FakeProvider,
    LLMBudgetExceededError,
    LLMConfigError,
)
from referral_finder.llm.base import LLMResponseError
from referral_finder.llm.prompt import PROMPT_HASH, PROMPT_VERSION
from referral_finder.llm.versioning import effective_analysis_version
from referral_finder.models import IncomingMessage
from referral_finder.storage import MessageStore

CHAT_A = -1001
CHAT_B = -1002
MATCHING = "Ищу подработку курьером, подскажите про реферальную программу"

CATALOG = parse_companies(
    "companies:\n  - id: foo\n    canonical: Foo\n    aliases:\n      - Foo\n"
)


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


def provider_ok():
    return FakeProvider(
        handler=lambda request: AnalysisResult(JOB_INTEREST, True, 0.9, "ok")
    )


class BudgetProvider:
    """Offline provider that reserves from the shared daily budget."""

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
        return AnalysisResult(JOB_INTEREST, True, 0.9, "ok")


class FakeRunner:
    """Minimal structural stand-in for :class:`TelegramRunner`."""

    def __init__(
        self,
        chat_ids,
        *,
        start_error=None,
        history_sync_enabled=False,
        history_results=(),
    ):
        self._chat_ids = [int(chat_id) for chat_id in chat_ids]
        self._start_error = start_error
        self._history_sync_enabled = bool(history_sync_enabled)
        self._history_results = tuple(history_results)
        self.events = []
        self._release = asyncio.Event()

    @property
    def history_sync_enabled(self):
        return self._history_sync_enabled

    @property
    def history_results(self):
        return self._history_results

    async def start(self):
        self.events.append("runner.start")
        if self._start_error is not None:
            raise self._start_error

    async def wait(self):
        self.events.append("runner.wait")
        await self._release.wait()

    async def shutdown(self):
        self.events.append("runner.shutdown")

    def configured_chat_ids(self):
        return tuple(self._chat_ids)

    def release(self):
        self._release.set()


class HistoryFailRunner(FakeRunner):
    """FakeRunner that reports selected chats as history-sync failures."""

    def __init__(self, chat_ids, failed_chat_ids):
        ids = [int(chat_id) for chat_id in chat_ids]
        failed = {int(chat_id) for chat_id in failed_chat_ids}
        super().__init__(
            ids,
            history_sync_enabled=True,
            history_results=[
                SimpleNamespace(chat_id=chat_id, ok=chat_id not in failed)
                for chat_id in ids
            ],
        )


def make_config(tmp_path, **overrides) -> Config:
    values = dict(
        api_id=1,
        api_hash="fake",
        session_path=tmp_path / "s.session",
        db_path=tmp_path / "m.db",
        log_level="INFO",
        chats_file=tmp_path / "chats.txt",
        chats=(str(CHAT_A),),
        llm_enabled=True,
        llm_criteria="",
        llm_bootstrap_limit=200,
        llm_poll_interval=0.01,
        llm_batch_size=25,
        llm_max_attempts=3,
        llm_max_api_requests_per_day=10000,
    )
    values.update(overrides)
    return Config(**values)


async def wait_until(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition was not met before the timeout")


def recording_factory(created):
    def factory(store, provider, chat_ids, **kwargs):
        worker = BootstrapSteadyWorker(store, provider, chat_ids, **kwargs)
        created["worker"] = worker
        return worker

    return factory


async def run_until_bootstrap_done(
    cfg, store, provider, chat_ids=(CHAT_A,), catalog=CATALOG, runner=None
):
    """Run the app until the AI worker has finished bootstrap, then stop."""
    created = {}
    runner = runner or FakeRunner(chat_ids)
    task = asyncio.create_task(
        run_application(
            cfg,
            store=store,
            runner=runner,
            provider=provider,
            catalog=catalog,
            worker_factory=recording_factory(created),
        )
    )
    await wait_until(lambda: "worker" in created)
    worker = created["worker"]
    await wait_until(lambda: worker.bootstrap_done)
    runner.release()
    await task
    return runner, worker


# --- enable/disable and failure isolation ----------------------------------


async def test_disabled_llm_skips_worker(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        provider = provider_ok()
        runner = FakeRunner([CHAT_A])
        runner.release()
        cfg = make_config(tmp_path, llm_enabled=False)

        await run_application(
            cfg, store=store, runner=runner, provider=provider, catalog=CATALOG
        )

        assert runner.events == ["runner.start", "runner.wait", "runner.shutdown"]
        assert provider.calls == []
        assert store.get_bootstrap_state(CHAT_A) is None
    finally:
        store.close()


async def test_enabled_llm_snapshots_and_processes(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in (1, 2, 3):
            seed(store, CHAT_A, message_id)
        provider = provider_ok()
        cfg = make_config(tmp_path)

        runner, worker = await run_until_bootstrap_done(cfg, store, provider)

        assert worker.bootstrap_done is True
        assert len(provider.calls) == 3
        assert store.get_bootstrap_state(CHAT_A)["status"] == "done"
        assert runner.events[0] == "runner.start"
        assert runner.events[-1] == "runner.shutdown"
    finally:
        store.close()


async def test_lifecycle_order_worker_between_start_and_shutdown(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        provider = provider_ok()
        cfg = make_config(tmp_path)
        runner = FakeRunner([CHAT_A])
        events = runner.events
        created = {}

        def factory(store_, provider_, chat_ids_, **kwargs):
            worker = BootstrapSteadyWorker(store_, provider_, chat_ids_, **kwargs)
            created["worker"] = worker
            original_start, original_stop = worker.start, worker.stop

            async def start():
                events.append("worker.start")
                await original_start()

            async def stop():
                events.append("worker.stop")
                await original_stop()

            worker.start = start
            worker.stop = stop
            return worker

        task = asyncio.create_task(
            run_application(
                cfg,
                store=store,
                runner=runner,
                provider=provider,
                catalog=CATALOG,
                worker_factory=factory,
            )
        )
        await wait_until(lambda: "worker" in created)
        await wait_until(lambda: created["worker"].bootstrap_done)
        runner.release()
        await task

        assert events == [
            "runner.start",
            "worker.start",
            "runner.wait",
            "worker.stop",
            "runner.shutdown",
        ]
    finally:
        store.close()


async def test_provider_config_error_keeps_running(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        runner = FakeRunner([CHAT_A])
        runner.release()
        cfg = make_config(tmp_path)

        def factory(config, *, store=None, **kwargs):
            raise LLMConfigError("DEEPSEEK_API_KEY is not set")

        await run_application(
            cfg,
            store=store,
            runner=runner,
            provider=None,
            provider_factory=factory,
            catalog=CATALOG,
        )

        assert runner.events == ["runner.start", "runner.wait", "runner.shutdown"]
        assert store.get_bootstrap_state(CHAT_A) is None
    finally:
        store.close()


async def test_catalog_error_keeps_running(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        runner = FakeRunner([CHAT_A])
        runner.release()
        cfg = make_config(tmp_path)

        def loader():
            raise CompanyCatalogError("missing companies.yaml")

        await run_application(
            cfg,
            store=store,
            runner=runner,
            provider=provider_ok(),
            catalog=None,
            catalog_loader=loader,
        )

        assert runner.events == ["runner.start", "runner.wait", "runner.shutdown"]
        assert store.get_bootstrap_state(CHAT_A) is None
    finally:
        store.close()


async def test_start_error_skips_worker_and_shuts_down(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        provider = provider_ok()
        runner = FakeRunner([CHAT_A], start_error=RuntimeError("no network"))
        cfg = make_config(tmp_path)

        with pytest.raises(RuntimeError):
            await run_application(
                cfg, store=store, runner=runner, provider=provider, catalog=CATALOG
            )

        assert runner.events == ["runner.start", "runner.shutdown"]
        assert provider.calls == []
        assert store.get_bootstrap_state(CHAT_A) is None
    finally:
        store.close()


async def test_ai_provider_failure_does_not_lose_messages(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        provider = FakeProvider(handler=lambda request: LLMResponseError("transient"))
        cfg = make_config(tmp_path)

        await run_until_bootstrap_done(cfg, store, provider)

        assert store.count() == 1  # source message preserved
        version = effective_analysis_version(provider.prompt_version, "")
        stored = store.get_ai_analysis(CHAT_A, 1, "fake", "fake-model", version)
        assert stored["status"] == "invalid"  # recorded failure, not success
        assert stored["category"] is None
    finally:
        store.close()


# --- restart / persistence --------------------------------------------------


async def test_failed_history_chat_gets_no_snapshot_and_recovers(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        seed(store, CHAT_B, 1)
        cfg = make_config(tmp_path)
        first_provider = provider_ok()
        first_created = {}
        first_runner = HistoryFailRunner([CHAT_A, CHAT_B], {CHAT_B})
        first_task = asyncio.create_task(
            run_application(
                cfg,
                store=store,
                runner=first_runner,
                provider=first_provider,
                catalog=CATALOG,
                worker_factory=recording_factory(first_created),
            )
        )
        await wait_until(lambda: "worker" in first_created)
        await wait_until(lambda: first_created["worker"].bootstrap_done)
        first_runner.release()
        await first_task

        version = effective_analysis_version(first_provider.prompt_version, "")
        # A synced normally; B must have no snapshot and no analysis.
        assert store.get_bootstrap_state(CHAT_A)["status"] == "done"
        assert store.get_bootstrap_state(CHAT_B) is None
        assert (
            store.get_ai_analysis(CHAT_B, 1, "fake", "fake-model", version) is None
        )
        # Without a bootstrap state, B's old message cannot enter steady.
        assert list(
            store.iter_ai_unprocessed(
                "fake", "fake-model", version, steady_only=True, chat_ids=[CHAT_B]
            )
        ) == []

        # A later startup with a successful sync can create B's snapshot and
        # process the previously skipped history as bootstrap, not steady.
        second_provider = provider_ok()
        second_created = {}
        second_runner = HistoryFailRunner([CHAT_A, CHAT_B], set())
        second_task = asyncio.create_task(
            run_application(
                cfg,
                store=store,
                runner=second_runner,
                provider=second_provider,
                catalog=CATALOG,
                worker_factory=recording_factory(second_created),
            )
        )
        await wait_until(lambda: "worker" in second_created)
        await wait_until(lambda: second_created["worker"].bootstrap_done)
        second_runner.release()
        await second_task

        assert store.get_bootstrap_state(CHAT_B) is not None
        assert (
            store.get_ai_analysis(CHAT_B, 1, "fake", "fake-model", version)
            is not None
        )
        assert len(second_provider.calls) == 1
    finally:
        store.close()


async def test_empty_history_results_skip_all_snapshots(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        seed(store, CHAT_B, 1)
        cfg = make_config(tmp_path)
        provider = provider_ok()
        runner = FakeRunner(
            [CHAT_A, CHAT_B], history_sync_enabled=True, history_results=()
        )
        runner.release()

        await run_application(
            cfg,
            store=store,
            runner=runner,
            provider=provider,
            catalog=CATALOG,
        )

        assert provider.calls == []
        assert store.get_bootstrap_state(CHAT_A) is None
        assert store.get_bootstrap_state(CHAT_B) is None
    finally:
        store.close()


async def test_missing_history_result_skips_only_that_chat(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        seed(store, CHAT_B, 1)
        cfg = make_config(tmp_path)
        provider = provider_ok()
        runner = FakeRunner(
            [CHAT_A, CHAT_B],
            history_sync_enabled=True,
            history_results=[SimpleNamespace(chat_id=CHAT_A, ok=True)],
        )

        await run_until_bootstrap_done(
            cfg, store, provider, chat_ids=(CHAT_A, CHAT_B), runner=runner
        )

        version = effective_analysis_version(provider.prompt_version, "")
        assert store.get_bootstrap_state(CHAT_A)["status"] == "done"
        assert store.get_bootstrap_state(CHAT_B) is None
        assert (
            store.get_ai_analysis(CHAT_B, 1, "fake", "fake-model", version) is None
        )
    finally:
        store.close()


async def test_partial_history_failure_processes_only_successful_chat(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        seed(store, CHAT_B, 1)
        cfg = make_config(tmp_path)
        provider = provider_ok()
        runner = FakeRunner(
            [CHAT_A, CHAT_B],
            history_sync_enabled=True,
            history_results=[
                SimpleNamespace(chat_id=CHAT_A, ok=True),
                SimpleNamespace(chat_id=CHAT_B, ok=False),
            ],
        )

        await run_until_bootstrap_done(
            cfg, store, provider, chat_ids=(CHAT_A, CHAT_B), runner=runner
        )

        version = effective_analysis_version(provider.prompt_version, "")
        assert store.get_bootstrap_state(CHAT_A)["status"] == "done"
        assert store.get_bootstrap_state(CHAT_B) is None
        assert (
            store.get_ai_analysis(CHAT_A, 1, "fake", "fake-model", version) is not None
        )
        assert (
            store.get_ai_analysis(CHAT_B, 1, "fake", "fake-model", version) is None
        )
    finally:
        store.close()


async def test_fully_successful_history_sync_preserves_processing(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        seed(store, CHAT_B, 1)
        cfg = make_config(tmp_path)
        provider = provider_ok()
        runner = FakeRunner(
            [CHAT_A, CHAT_B],
            history_sync_enabled=True,
            history_results=[
                SimpleNamespace(chat_id=CHAT_A, ok=True),
                SimpleNamespace(chat_id=CHAT_B, ok=True),
            ],
        )

        await run_until_bootstrap_done(
            cfg, store, provider, chat_ids=(CHAT_A, CHAT_B), runner=runner
        )

        version = effective_analysis_version(provider.prompt_version, "")
        assert store.get_bootstrap_state(CHAT_A)["status"] == "done"
        assert store.get_bootstrap_state(CHAT_B)["status"] == "done"
        assert (
            store.get_ai_analysis(CHAT_A, 1, "fake", "fake-model", version) is not None
        )
        assert (
            store.get_ai_analysis(CHAT_B, 1, "fake", "fake-model", version) is not None
        )
        assert len(provider.calls) == 2
    finally:
        store.close()


async def test_restart_keeps_snapshot_and_skips_success(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in (1, 2, 3):
            seed(store, CHAT_A, message_id)
        cfg = make_config(tmp_path)

        first = provider_ok()
        await run_until_bootstrap_done(cfg, store, first)
        state = store.get_bootstrap_state(CHAT_A)

        second = provider_ok()
        await run_until_bootstrap_done(cfg, store, second)

        assert second.calls == []  # a stored success is never re-analysed
        after = store.get_bootstrap_state(CHAT_A)
        assert after["selected_count"] == state["selected_count"]
        assert after["steady_after_message_id"] == state["steady_after_message_id"]
    finally:
        store.close()


async def test_shared_budget_stops_bootstrap_and_keeps_queue(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        for message_id in (1, 2, 3):
            seed(store, CHAT_A, message_id)
        budget = ApiRequestBudget(store, limit=2, clock=lambda: 0)
        provider = BudgetProvider(budget)
        cfg = make_config(tmp_path)

        created = {}
        runner = FakeRunner([CHAT_A])
        task = asyncio.create_task(
            run_application(
                cfg,
                store=store,
                runner=runner,
                provider=provider,
                catalog=CATALOG,
                worker_factory=recording_factory(created),
            )
        )
        await wait_until(lambda: "worker" in created)
        await wait_until(lambda: provider.calls == 2)
        runner.release()
        await task

        assert budget.used() == 2
        version = effective_analysis_version(provider.prompt_version, "")
        # One message is still queued and was not marked as analysed.
        assert store.count_ai_bootstrap("budget", "budget-model", version) == 1
    finally:
        store.close()


# --- cleanup when worker.stop() fails ---------------------------------------


class RecordingStopWorker:
    """Tiny worker whose stop() performs a scripted action."""

    def __init__(self, stop_action):
        self._stop_action = stop_action
        self.stop_calls = 0
        self.started = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stop_calls += 1
        await self._stop_action()


async def test_worker_stop_runtime_error_still_shuts_down_runner(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        runner = FakeRunner([CHAT_A])
        runner.release()
        worker = RecordingStopWorker(
            lambda: _raise(RuntimeError("worker stop failed"))
        )
        cfg = make_config(tmp_path)

        await run_application(
            cfg,
            store=store,
            runner=runner,
            provider=provider_ok(),
            catalog=CATALOG,
            worker_factory=lambda *args, **kwargs: worker,
        )

        assert worker.stop_calls == 1
        assert runner.events == ["runner.start", "runner.wait", "runner.shutdown"]
    finally:
        store.close()


async def _raise(exc):
    raise exc


async def test_worker_stop_cancelled_still_shuts_down_and_propagates(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        runner = FakeRunner([CHAT_A])
        runner.release()
        worker = RecordingStopWorker(
            lambda: _raise(asyncio.CancelledError())
        )
        cfg = make_config(tmp_path)

        with pytest.raises(asyncio.CancelledError):
            await run_application(
                cfg,
                store=store,
                runner=runner,
                provider=provider_ok(),
                catalog=CATALOG,
                worker_factory=lambda *args, **kwargs: worker,
            )

        assert worker.stop_calls == 1
        assert runner.events[-1] == "runner.shutdown"
    finally:
        store.close()


async def test_cancellation_during_worker_stop_still_shuts_down(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)
        runner = FakeRunner([CHAT_A])
        runner.release()
        entered = asyncio.Event()
        never = asyncio.Event()

        async def block_forever():
            entered.set()
            await never.wait()

        worker = RecordingStopWorker(block_forever)
        cfg = make_config(tmp_path)
        task = asyncio.create_task(
            run_application(
                cfg,
                store=store,
                runner=runner,
                provider=provider_ok(),
                catalog=CATALOG,
                worker_factory=lambda *args, **kwargs: worker,
            )
        )

        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert runner.events[-1] == "runner.shutdown"
    finally:
        store.close()


async def test_worker_stop_error_does_not_hide_application_error(tmp_path):
    store = MessageStore(tmp_path / "m.db")
    try:
        seed(store, CHAT_A, 1)

        class WaitFailingRunner(FakeRunner):
            async def wait(self):
                self.events.append("runner.wait")
                raise ValueError("wait failed")

        runner = WaitFailingRunner([CHAT_A])
        worker = RecordingStopWorker(
            lambda: _raise(RuntimeError("worker stop failed"))
        )
        cfg = make_config(tmp_path)

        with pytest.raises(ValueError, match="wait failed"):
            await run_application(
                cfg,
                store=store,
                runner=runner,
                provider=provider_ok(),
                catalog=CATALOG,
                worker_factory=lambda *args, **kwargs: worker,
            )

        assert runner.events[-1] == "runner.shutdown"
    finally:
        store.close()
