"""Tests for the independent keyword prefilter (no network, no DB)."""

from referral_finder.prefilter import (
    PREFILTER_VERSION,
    PrefilterResult,
    matches,
    prefilter,
)


# --- contract / boundary cases -------------------------------------------


def test_version_is_explicit():
    assert isinstance(PREFILTER_VERSION, str) and PREFILTER_VERSION


def test_empty_text_is_not_matched():
    result = prefilter("")
    assert result == PrefilterResult(False, ())
    assert result.matched is False
    assert result.rule_ids == ()
    assert result.version == PREFILTER_VERSION


def test_none_and_whitespace_are_not_matched():
    assert prefilter(None).matched is False
    assert prefilter("   \t\n ").matched is False


def test_punctuation_only_is_not_matched():
    assert prefilter("!!! ??? ... — ()").matched is False


def test_not_matched_result_has_no_rule_ids():
    result = prefilter("Сегодня отличная погода, идём гулять")
    assert result.matched is False
    assert result.rule_ids == ()


def test_matches_wrapper_mirrors_prefilter():
    assert matches("дайте реферальную ссылку") is True
    assert matches("просто болтаем о погоде") is False


def test_deterministic():
    text = "Ищу подработку курьером, дайте рефку"
    assert prefilter(text) == prefilter(text)


# --- positive matches -----------------------------------------------------


def test_referral_russian_matches():
    result = prefilter("Кто может скинуть реферальную ссылку?")
    assert result.matched is True
    assert "prefilter.referral" in result.rule_ids


def test_referral_transliteration_matches():
    result = prefilter("gde vzyat refku, podskazhite")
    assert result.matched is True
    assert "prefilter.referral" in result.rule_ids


def test_courier_word_form_matches():
    result = prefilter("Ищу подработку курьером")
    assert result.matched is True
    assert "prefilter.courier" in result.rule_ids


def test_delivery_matches():
    result = prefilter("Есть вакансии в доставке рядом с домом")
    assert result.matched is True
    assert "prefilter.delivery" in result.rule_ids


def test_job_kind_matches():
    assert "prefilter.job_kind" in prefilter("нужен пеший курьер").rule_ids
    assert "prefilter.job_kind" in prefilter("катаюсь велокурьером").rule_ids


def test_generic_job_phrase_matches():
    assert "prefilter.generic_job" in prefilter("Ищу работу, подскажите").rule_ids
    assert "prefilter.generic_job" in prefilter("Предлагаю работу с оплатой").rule_ids
    assert "prefilter.generic_job" in prefilter("Ищем подработку на вечер").rule_ids
    assert "prefilter.generic_job" in prefilter("Открыта вакансия").rule_ids


def test_generic_job_english_phrase_matches():
    assert "prefilter.generic_job" in prefilter("looking for a job in the city").rule_ids
    assert "prefilter.generic_job" in prefilter("hiring couriers, weekly pay").rule_ids


def test_case_and_punctuation_are_irrelevant():
    result = prefilter("КУРЬЕР!!! СРОЧНО???")
    assert result.matched is True
    assert "prefilter.courier" in result.rule_ids


def test_mixed_script_referral_matches():
    # "рeфералку" mixes Cyrillic with a Latin 'e'; normalization folds it.
    result = prefilter("рeфералку дадите?")
    assert result.matched is True
    assert "prefilter.referral" in result.rule_ids


# --- negative / precision guardrails --------------------------------------


def test_unrelated_topics_are_not_matched():
    for text in (
        "Кто смотрел вчерашний матч?",
        "Завтра едем на дачу, кто с нами?",
        "Продаю детскую коляску, недорого",
    ):
        assert prefilter(text).matched is False, text


def test_bare_work_words_are_not_matched():
    # Generic words alone must not open the gate; only precise phrases do.
    for text in (
        "Работа над проектом идёт полным ходом",
        "Это хорошая работа, мне нравится",
        "Nice work!",
        "That is a good job",
        "work in progress",
    ):
        assert prefilter(text).matched is False, text


def test_referral_negation_still_opens_the_gate():
    # The prefilter is recall-oriented: it does not resolve intent, so a
    # referral word is still routed onward (the AI stage decides later).
    assert prefilter("Рефералка не нужна, спасибо").matched is True
