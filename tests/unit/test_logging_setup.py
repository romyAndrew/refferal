"""Tests for logging setup and secret masking.

These tests do not touch the network or real credentials.
"""

import contextlib
import io
import logging

from referral_finder.logging_setup import SecretFilter, register_secret, setup_logging


def _make_record(msg, args=()):
    return logging.LogRecord(
        name="unit",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_secret_filter_masks_message():
    log_filter = SecretFilter(["S3CR3T"])
    record = _make_record("token is %s", ("S3CR3T",))
    assert log_filter.filter(record) is True
    assert "S3CR3T" not in record.getMessage()
    assert "***" in record.getMessage()


def test_secret_filter_without_secrets_is_noop():
    log_filter = SecretFilter([])
    record = _make_record("plain message")
    assert log_filter.filter(record) is True
    assert record.getMessage() == "plain message"


def test_setup_logging_masks_secret_on_stderr():
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(buffer):
            setup_logging(level="INFO", secrets=["S3CR3T"])
            logging.getLogger("unit.test").info("token is S3CR3T here")
    finally:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)

    output = buffer.getvalue()
    assert "S3CR3T" not in output
    assert "***" in output


def test_register_secret_after_setup_masks_future_records():
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(buffer):
            setup_logging(level="INFO", secrets=[])
            register_secret("AFTERKEY")
            logging.getLogger("unit.after").info("value AFTERKEY here")
    finally:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)

    output = buffer.getvalue()
    assert "AFTERKEY" not in output
    assert "***" in output


def test_register_secret_before_setup_is_merged():
    buffer = io.StringIO()
    register_secret("BEFOREKEY")
    try:
        with contextlib.redirect_stderr(buffer):
            setup_logging(level="INFO", secrets=[])
            logging.getLogger("unit.before").info("value BEFOREKEY here")
    finally:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)

    output = buffer.getvalue()
    assert "BEFOREKEY" not in output
    assert "***" in output
