"""Tests for best-effort text scrubbing (no network, no DB)."""

from referral_finder.scrubbing import SCRUBBING_VERSION, ScrubResult, scrub


# --- contract / boundary ---------------------------------------------------


def test_version_is_explicit():
    assert isinstance(SCRUBBING_VERSION, str) and SCRUBBING_VERSION


def test_empty_and_none_yield_empty_without_rules():
    assert scrub("") == ScrubResult("", ())
    assert scrub(None) == ScrubResult("", ())
    assert scrub("   ").rule_ids == ()


def test_no_match_keeps_text_and_has_no_rules():
    result = scrub("Ищу подработку курьером, оплата ежедневно")
    assert result.text == "Ищу подработку курьером, оплата ежедневно"
    assert result.rule_ids == ()


def test_result_never_contains_original_fragments():
    result = scrub("пишите на ivan@example.com")
    assert "ivan@example.com" not in result.text
    assert "ivan" not in result.text
    assert result.rule_ids == ("scrub.email",)


# --- individual rules ------------------------------------------------------


def test_email_is_replaced():
    result = scrub("контакт: test.user+tag@sub.example.ru")
    assert result.text == "контакт: [email]"
    assert result.rule_ids == ("scrub.email",)


def test_phone_numbers_are_replaced():
    for text in (
        "+7 999 123-45-67",
        "8 (999) 123-45-67",
        "+79991234567",
        "тел. +1 (202) 555-0142 звоните",
    ):
        result = scrub(text)
        assert "[phone]" in result.text, text
        assert "scrub.phone" in result.rule_ids, text


def test_short_numbers_are_preserved():
    for text in ("оплата 500 руб", "в 12:30", "код 12345"):
        result = scrub(text)
        assert "[phone]" not in result.text, text
        assert result.rule_ids == (), text


def test_telegram_links_are_replaced_before_generic_urls():
    for text in (
        "https://t.me/someuser",
        "t.me/someuser/42",
        "https://t.me/c/123456789/55",
        "tg://resolve?domain=someuser",
    ):
        result = scrub(text)
        assert "scrub.telegram_link" in result.rule_ids, text
        assert "someuser" not in result.text, text


def test_generic_urls_are_replaced():
    result = scrub("смотри https://example.com/page?x=1 и www.site.org")
    assert "scrub.url" in result.rule_ids
    assert "example.com" not in result.text
    assert "site.org" not in result.text


def test_username_is_replaced():
    result = scrub("напиши мне @telegram_user, пожалуйста")
    assert "[username]" in result.text
    assert "scrub.username" in result.rule_ids
    assert "telegram_user" not in result.text


def test_email_is_not_misread_as_username():
    result = scrub("ivan.petrov@example.com")
    # The whole email is removed first; no stray @username is left behind.
    assert "scrub.username" not in result.rule_ids
    assert "@" not in result.text


def test_technical_ids_are_replaced():
    for text in ("-1001234567890", "chat_id=123456789", "user_id: 987654321"):
        result = scrub(text)
        assert "scrub.tech_id" in result.rule_ids, text
        assert "123456789" not in result.text, text


# --- mixed text ------------------------------------------------------------


def test_mixed_text_reports_all_rule_types():
    text = (
        "Привет, пиши @helper или на mail@test.ru, тел +7 999 111-22-33, "
        "ссылка https://t.me/helper/10, id чата -1001234567890"
    )
    result = scrub(text)
    assert set(result.rule_ids) >= {
        "scrub.username",
        "scrub.email",
        "scrub.phone",
        "scrub.telegram_link",
        "scrub.tech_id",
    }
    for fragment in ("helper", "mail@test.ru", "111-22-33", "t.me", "-1001234567890"):
        assert fragment not in result.text, fragment
    # Everyday words survive.
    assert "Привет" in result.text
