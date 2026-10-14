"""Command-line entry point.

Loads configuration and, when it is valid, runs the Telethon client until it
disconnects. No Telegram/network access happens at import time; a connection is
only opened when this module is executed.

Note (stage 2): actually running this opens a real Telegram connection and may
prompt for the login code on first use. That live run is performed separately
after explicit approval.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional, Sequence

from .config import Config, ConfigError, load_config
from .logging_setup import setup_logging
from .telegram_client import TelegramRunner


async def _run(config: Config) -> None:
    runner = TelegramRunner(config)
    try:
        await runner.run_forever()
    finally:
        await runner.shutdown()


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        print(
            "Create a .env file (copy .env.example) or export the required "
            "environment variables before running.",
            file=sys.stderr,
        )
        return 2

    setup_logging(
        config.log_level,
        secrets=(config.api_hash, str(config.session_path)),
    )
    logger = logging.getLogger("referral_finder")
    logger.info("%s", config.safe_repr())

    if not config.chats:
        logger.error(
            "No chats configured in %s. Add one numeric chat ID per line.",
            config.chats_file,
        )
        return 2

    try:
        asyncio.run(_run(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user; shutting down.")
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        logger.exception("Fatal error while running the Telegram client.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
