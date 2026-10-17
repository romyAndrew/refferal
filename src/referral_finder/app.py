"""Application lifecycle wiring: Telegram ingestion plus the optional AI worker.

Ordering and failure isolation are the point of this module:

1. ``runner.start()`` connects to Telegram, registers the handler, catches up and
   runs the bounded history sync. If it fails, no AI worker is started.
2. Only after a successful start are the configured/resolved chats snapshotted
   and the :class:`BootstrapSteadyWorker` launched — and only when
   ``LLM_ENABLED`` is true and the provider/catalog are valid. Chats whose
   startup history sync did not complete are excluded from the snapshot so a
   watermark is never fixed over missing history.
3. ``runner.wait()`` blocks until Telegram disconnects.
4. On the way out the worker is stopped first, then ``runner.shutdown()`` drains
   the writer and closes the ``MessageStore`` last.

An AI configuration/availability problem never stops Telegram collection:
provider and catalog errors are logged with a generic message (no secrets, no
message text) and the worker is simply skipped. The worker never closes the
store itself; the lifecycle owner (``runner.shutdown``) does.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .companies import CompanyCatalogError, load_default_catalog
from .llm.base import LLMConfigError
from .llm.bootstrap import BootstrapSteadyWorker
from .llm.factory import provider_from_config

logger = logging.getLogger(__name__)


async def run_application(
    config,
    *,
    store,
    runner,
    provider=None,
    provider_factory=provider_from_config,
    catalog=None,
    catalog_loader=load_default_catalog,
    worker_factory=BootstrapSteadyWorker,
):
    """Run Telegram ingestion with the AI worker, in a safe shutdown order.

    ``store`` and ``runner`` are built (and owned) by the caller. This function
    never closes ``store`` directly: ``runner.shutdown()`` does, after the
    worker has stopped, so no background analysis can race the closed database.
    """
    worker = None
    primary_error: Optional[BaseException] = None
    cleanup_cancel: Optional[asyncio.CancelledError] = None
    try:
        await runner.start()
        if config.llm_enabled:
            try:
                worker = await _start_ai_worker(
                    config,
                    store,
                    runner,
                    provider=provider,
                    provider_factory=provider_factory,
                    catalog=catalog,
                    catalog_loader=catalog_loader,
                    worker_factory=worker_factory,
                )
            except Exception as exc:  # noqa: BLE001 - AI must never stop Telegram
                # Generic, argument-free message: no key, no text, no ids.
                logger.warning(
                    "AI worker initialization failed (%s); continuing without AI.",
                    type(exc).__name__,
                )
                worker = None
        await runner.wait()
    except BaseException as exc:
        # Remember the application error, but let the finally block run the full
        # cleanup before it propagates.
        primary_error = exc
        raise
    finally:
        # A cancellation or an error while stopping the worker must never skip
        # runner.shutdown(). The nested try/finally guarantees the runner is
        # cleaned up; a cancellation is remembered and re-raised afterwards.
        try:
            if worker is not None:
                try:
                    await worker.stop()
                except asyncio.CancelledError as exc:
                    cleanup_cancel = exc
                except Exception:
                    logger.exception("Error while stopping the AI worker.")
        finally:
            try:
                await runner.shutdown()
            except asyncio.CancelledError as exc:
                cleanup_cancel = cleanup_cancel or exc
            except Exception:
                logger.exception("Error while shutting down the Telegram runner.")
                if primary_error is None and cleanup_cancel is None:
                    raise

        if cleanup_cancel is not None:
            if primary_error is not None:
                raise cleanup_cancel from primary_error
            raise cleanup_cancel


async def _start_ai_worker(
    config,
    store,
    runner,
    *,
    provider,
    provider_factory,
    catalog,
    catalog_loader,
    worker_factory,
) -> Optional[object]:
    """Build, snapshot and start the AI worker; return ``None`` when skipped."""
    chat_ids = runner.configured_chat_ids()
    history_sync_enabled = getattr(runner, "history_sync_enabled", False)
    if callable(history_sync_enabled):
        history_sync_enabled = history_sync_enabled()
    if history_sync_enabled:
        # Only a chat with an explicit successful result may be snapshotted.
        # A missing result is not proof of success: both explicit failures and
        # chats absent from the result list stay unsnapshotted, so no watermark
        # can be fixed over an incomplete history. When history sync is
        # disabled, the previous behavior (no filtering) is preserved.
        results = getattr(runner, "history_results", ())
        confirmed_ids = {
            int(result.chat_id)
            for result in results
            if getattr(result, "ok", False)
        }
        skipped_ids = [
            chat_id for chat_id in chat_ids if int(chat_id) not in confirmed_ids
        ]
        if skipped_ids:
            logger.warning(
                "AI worker skips %d chat(s) without a confirmed history sync.",
                len(skipped_ids),
            )
            chat_ids = tuple(
                chat_id for chat_id in chat_ids if int(chat_id) in confirmed_ids
            )
    if not chat_ids:
        logger.warning("AI worker skipped: no chat with a completed history sync.")
        return None

    if provider is None:
        try:
            provider = provider_factory(config, store=store)
        except LLMConfigError:
            logger.warning(
                "AI worker skipped: LLM provider is not configured correctly."
            )
            return None

    if catalog is None:
        try:
            catalog = catalog_loader()
        except CompanyCatalogError:
            logger.warning(
                "AI worker skipped: the company catalog could not be loaded."
            )
            return None

    worker = worker_factory(
        store,
        provider,
        chat_ids,
        enabled=True,
        criteria=config.llm_criteria,
        bootstrap_limit=config.llm_bootstrap_limit,
        poll_interval=config.llm_poll_interval,
        batch_size=config.llm_batch_size,
        max_attempts=config.llm_max_attempts,
        catalog=catalog,
    )
    await worker.start()
    return worker
