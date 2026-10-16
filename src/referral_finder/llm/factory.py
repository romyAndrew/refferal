"""Provider factory.

Creates an offline :class:`~referral_finder.llm.fake.FakeProvider` unless AI is
explicitly enabled, and only then a real :class:`DeepSeekProvider`. The API key
is read **only** from the ``DEEPSEEK_API_KEY`` environment variable (never from
``.env`` loaded values, never printed) and is registered with the log secret
filter before the provider can make any call.

This module never performs a network request itself.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

from ..config import Config
from ..logging_setup import register_secret
from .base import LLMConfigError, LLMProvider
from .budget import ApiRequestBudget
from .deepseek import DeepSeekProvider, HttpTransport
from .fake import FakeProvider

# Environment variable that holds the real DeepSeek key.
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"


def provider_from_config(
    config: Config,
    *,
    store=None,
    transport: Optional[HttpTransport] = None,
    environ: Optional[Mapping[str, str]] = None,
    clock=None,
) -> LLMProvider:
    """Build a provider from ``config`` (offline FakeProvider by default).

    A real DeepSeek provider always gets a persistent ``ApiRequestBudget`` backed
    by ``store``; a live provider without a store/budget is refused so no runtime
    path can bypass the daily limit.
    """
    env = os.environ if environ is None else environ

    if not config.llm_enabled:
        return FakeProvider()

    provider_name = (config.llm_provider or "").strip().lower()
    if provider_name in ("", "fake", "none"):
        return FakeProvider()
    if provider_name != "deepseek":
        raise LLMConfigError(f"unsupported LLM provider: {config.llm_provider!r}")

    api_key = env.get(DEEPSEEK_API_KEY_ENV)
    if not api_key:
        raise LLMConfigError(f"{DEEPSEEK_API_KEY_ENV} is not set in the environment")

    if store is None:
        raise LLMConfigError(
            "a MessageStore is required to build the DeepSeek provider "
            "(daily request budget)"
        )
    budget = ApiRequestBudget(
        store, config.llm_max_api_requests_per_day, clock=clock
    )

    # Mask the key before the provider can possibly be used.
    register_secret(api_key)

    return DeepSeekProvider(
        api_key=api_key,
        model=config.llm_model,
        base_url=config.llm_base_url,
        transport=transport,
        timeout=config.llm_timeout,
        max_retries=config.llm_max_retries,
        max_input_chars=config.llm_max_input_chars,
        max_output_tokens=config.llm_max_output_tokens,
        budget=budget,
    )
