"""One-message live smoke test for the DeepSeek API.

Deliberately separate from the pipeline and the collector: it always analyses a
single built-in synthetic message and never touches Telegram, SQLite, or user
data. A real HTTP call happens **only** when ``--live`` is passed *and*
``LLM_ENABLED=true`` *and* ``DEEPSEEK_API_KEY`` is present in the environment.

Safety properties:

* The API key is read only from ``DEEPSEEK_API_KEY`` and registered with the log
  secret filter; it is never printed or placed in exceptions.
* The message still flows through ``scrub`` before the provider sees it, and no
  Telegram identifiers/links/metadata are attached.
* The summary shows the model, criteria version and request count (1); the result
  prints only the validated category, relevance, status and usage. The full
  request and raw provider response are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Mapping, Optional, Sequence

from ..logging_setup import register_secret
from ..prefilter import prefilter
from ..scrubbing import scrub
from .base import LLMError, LLMRequest
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
    return parser


def _env_enabled(env: Mapping[str, str]) -> bool:
    return (env.get("LLM_ENABLED") or "").strip().lower() in _TRUE_VALUES


def _build_live_provider(env: Mapping[str, str], model: str, transport):
    api_key = env[DEEPSEEK_API_KEY_ENV]
    register_secret(api_key)
    base_url = env.get("LLM_BASE_URL") or DEFAULT_BASE_URL
    return DeepSeekProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        transport=transport,
        max_input_chars=_MAX_INPUT_CHARS,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
    )


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

    if provider is None:
        if not _env_enabled(env):
            print("error=LLM_ENABLED is not true", file=sys.stderr)
            return 2
        if not env.get(DEEPSEEK_API_KEY_ENV):
            print(f"error={DEEPSEEK_API_KEY_ENV} is not set", file=sys.stderr)
            return 2
        provider = _build_live_provider(env, model, transport)

    version = effective_analysis_version(provider.prompt_version, args.criteria)
    print(f"live=1 model={provider.model} criteria_version={version} requests=1")

    gate = prefilter(SMOKE_MESSAGE)
    if not gate.matched:
        print("status=prefilter_skipped")
        return 1
    cleaned = scrub(SMOKE_MESSAGE)
    request = LLMRequest(text=cleaned.text, criteria=args.criteria)

    try:
        result = asyncio.run(provider.analyze(request))
    except LLMError as exc:
        # Only the error type, never the key or the request/response body.
        print(f"status=error error={type(exc).__name__}")
        return 1
    except Exception as exc:  # noqa: BLE001 - report type only, no details
        print(f"status=error error={type(exc).__name__}")
        return 1

    print(
        f"status=ok category={result.category} relevant={result.relevant}"
    )
    usage = getattr(provider, "last_usage", None)
    if usage is not None:
        print("usage=" + json.dumps(usage, ensure_ascii=False))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
