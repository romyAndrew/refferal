"""One-message live smoke test for the DeepSeek API.

Deliberately separate from the pipeline and the collector: it always analyses a
single built-in synthetic message and never touches Telegram, SQLite, or user
data. A real HTTP call happens **only** when ``--live`` is passed *and*
``LLM_ENABLED=true`` *and* ``DEEPSEEK_API_KEY`` is present in the environment.

Safety properties:

* The API key is read only from ``DEEPSEEK_API_KEY`` and registered with the log
  secret filter; it is never printed or placed in exceptions.
* ``--live`` requires ``--db`` pointing at the application's existing database,
  so the one-off request reserves the same persistent daily budget as the app.
  A missing ``--db`` or a missing database file refuses the request and never
  creates a fresh budget store.
* The message still flows through ``scrub`` before the provider sees it, and no
  Telegram identifiers/links/metadata are attached.
* The summary shows the model, criteria version and the *planned* request count
  (``requests_planned=1``); ``requests=1`` is only printed once the provider call
  has actually completed (success or handled error). The result prints only the
  validated category, relevance, status and usage. The full request and raw
  provider response are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..config import DEFAULT_LLM_MAX_API_REQUESTS_PER_DAY
from ..logging_setup import register_secret
from ..prefilter import prefilter
from ..scrubbing import scrub
from ..storage import MessageStore
from .base import LLMConfigError, LLMError, LLMRequest
from .budget import ApiRequestBudget
from .deepseek import DEFAULT_BASE_URL, DeepSeekProvider, HttpTransport
from .factory import DEEPSEEK_API_KEY_ENV
from .versioning import effective_analysis_version

SMOKE_MESSAGE = (
    "Ищу подработку курьером, подскажите, есть ли реферальная программа?"
)
DEFAULT_MODEL = "deepseek-flash"
_TRUE_VALUES = {"1", "true", "yes", "on"}

# Small bounds are enough for one classification.
_MAX_INPUT_CHARS = 1000
_MAX_OUTPUT_TOKENS = 200


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m referral_finder.llm.smoke",
        description=(
            "Send one synthetic message to DeepSeek (requires --live, "
            "LLM_ENABLED=true and DEEPSEEK_API_KEY)."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="actually perform the single HTTP request (otherwise nothing is sent)",
    )
    parser.add_argument("--criteria", default="", help="search criteria text")
    parser.add_argument(
        "--model", default=None, help="override the model id (default deepseek-flash)"
    )
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "path to the application's existing SQLite database; required for "
            "--live. Its persistent daily budget is shared with the app. The "
            "file must already exist (it is never created here)."
        ),
    )
    return parser


def _env_enabled(env: Mapping[str, str]) -> bool:
    return (env.get("LLM_ENABLED") or "").strip().lower() in _TRUE_VALUES


def _limit_from_env(env: Mapping[str, str]) -> int:
    raw = env.get("LLM_MAX_API_REQUESTS_PER_DAY")
    if raw is None or not str(raw).strip():
        return DEFAULT_LLM_MAX_API_REQUESTS_PER_DAY
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise LLMConfigError("LLM_MAX_API_REQUESTS_PER_DAY must be an integer") from exc
    if value < 1:
        raise LLMConfigError("LLM_MAX_API_REQUESTS_PER_DAY must be at least 1")
    return value


def _build_live_provider(env: Mapping[str, str], model: str, transport, store):
    api_key = env[DEEPSEEK_API_KEY_ENV]
    register_secret(api_key)
    base_url = env.get("LLM_BASE_URL") or DEFAULT_BASE_URL
    budget = ApiRequestBudget(store, _limit_from_env(env))
    return DeepSeekProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        transport=transport,
        max_input_chars=_MAX_INPUT_CHARS,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        budget=budget,
    )


def _analyze(provider, args) -> int:
    version = effective_analysis_version(provider.prompt_version, args.criteria)
    print(
        f"live=1 model={provider.model} criteria_version={version} "
        "requests_planned=1"
    )

    gate = prefilter(SMOKE_MESSAGE)
    if not gate.matched:
        print("status=prefilter_skipped requests=0")
        return 1
    cleaned = scrub(SMOKE_MESSAGE)
    request = LLMRequest(text=cleaned.text, criteria=args.criteria)

    try:
        result = asyncio.run(provider.analyze(request))
    except LLMError as exc:
        # Only the error type, never the key or the request/response body. The
        # call was attempted, so it is counted, but the error status makes it
        # clear that no successful response was obtained.
        print(f"status=error error={type(exc).__name__} requests=1")
        return 1
    except Exception as exc:  # noqa: BLE001 - report type only, no details
        print(f"status=error error={type(exc).__name__} requests=1")
        return 1

    print(
        f"status=ok category={result.category} relevant={result.relevant} requests=1"
    )
    usage = getattr(provider, "last_usage", None)
    if usage is not None:
        print("usage=" + json.dumps(usage, ensure_ascii=False))
    return 0


def run(
    argv: Optional[Sequence[str]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    transport: Optional[HttpTransport] = None,
    provider=None,
) -> int:
    args = _build_parser().parse_args(argv)
    env = os.environ if env is None else env
    model = args.model or env.get("LLM_MODEL") or DEFAULT_MODEL

    if not args.live:
        print(f"live=0 model={model} requests=0 note=pass --live to actually call")
        return 0

    if provider is not None:
        # Injected provider (offline tests): no runtime guard needed.
        return _analyze(provider, args)

    if not _env_enabled(env):
        print("error=LLM_ENABLED is not true", file=sys.stderr)
        return 2
    if not env.get(DEEPSEEK_API_KEY_ENV):
        print(f"error={DEEPSEEK_API_KEY_ENV} is not set", file=sys.stderr)
        return 2
    if not args.db:
        print("error=--db is required for --live (daily budget)", file=sys.stderr)
        return 2

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        # Never silently create a fresh budget DB at a wrong path.
        print("error=budget database not found; no request sent", file=sys.stderr)
        return 2

    store = MessageStore(db_path)
    try:
        try:
            provider = _build_live_provider(env, model, transport, store)
        except LLMConfigError as exc:
            print(f"error={exc}", file=sys.stderr)
            return 2
        return _analyze(provider, args)
    finally:
        store.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
