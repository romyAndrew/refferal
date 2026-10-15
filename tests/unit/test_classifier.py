"""Tests for the rule-based classifier (no network, no DB)."""

import json
from pathlib import Path

import pytest

from referral_finder.classifier import (
    CATEGORIES,
    CLASSIFIER_VERSION,
    IRRELEVANT,
    JOB_INTEREST,
    RECRUITMENT_OFFER,
    REFERRAL_OFFER,
    REFERRAL_REQUEST,
    UNCERTAIN,
    Classification,
    classify,
    normalize,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "classifier_synthetic.json"


def _synthetic():
    with DATA.open(encoding="utf-8") as fh:
        return json.load(fh)


# --- normalization -------------------------------------------------------


def test_normalize_lowercases_folds_yo_and_whitespace():
    assert normalize("  ЁЖИК   Привет  ") == "ежик привет"


def test_normalize_folds_stray_latin_in_mixed_token():
    # "рeфералка" mixes Cyrillic and Latin; the Latin 'e' folds to Cyrillic.
    assert normalize("рeфералка") == "рефералка"


# --- results -------------------------------------------------------------


def test_empty_text_is_uncertain():
    result = classify("")
    assert result.category == UNCERTAIN
    assert result.score == 0.0
    assert result.matched_rules == ()


def test_result_has_version_and_explanatory_rules():
    result = classify("Дайте реферальную ссылку")
    assert isinstance(result, Classification)
    assert result.category == REFERRAL_REQUEST
    assert result.classifier_version == CLASSIFIER_VERSION
    assert "req.referral.give" in result.matched_rules
    assert 0.0 < result.score <= 1.0  # heuristic, bounded


def test_request_and_offer_are_distinguished():
    assert classify("Дайте реферальную ссылку").category == REFERRAL_REQUEST
    assert classify("Вот моя реферальная ссылка").category == REFERRAL_OFFER


def test_referral_offer_and_recruitment_are_distinguished():
    assert classify("Держи рефку").category == REFERRAL_OFFER
    assert classify("Набираю курьеров").category == RECRUITMENT_OFFER


def test_bare_referral_mention_is_uncertain():
    assert classify("реферальная ссылка").category == UNCERTAIN


def test_negation_is_irrelevant():
    assert classify("Рефералка не нужна, спасибо").category == IRRELEVANT


def test_past_experience_is_irrelevant():
    assert classify("Вчера оформился по реферальной ссылке, всё супер").category == IRRELEVANT


def test_clearly_unrelated_is_irrelevant():
    assert classify("Сегодня отличная погода").category == IRRELEVANT


def test_conflicting_strong_intents_are_uncertain():
    result = classify("Ищу работу курьером, дайте реферальную ссылку")
    assert result.category == UNCERTAIN


def test_deterministic():
    text = "Набираю курьеров, оплата ежедневно"
    assert classify(text) == classify(text)


def test_only_known_categories():
    for item in _synthetic():
        result = classify(item["text"])
        assert result.category in CATEGORIES


# --- synthetic functional/regression set ---------------------------------
# NOTE: synthetic examples only exercise the rules. They are NOT evidence of
# real-world quality; that requires a separate hand-labeled sample.


@pytest.mark.parametrize("item", _synthetic(), ids=lambda item: item["group"])
def test_synthetic_expectations(item):
    result = classify(item["text"])
    assert result.category == item["category"], (
        f"{item['text']!r} -> {result.category} "
        f"(rules={result.matched_rules}) expected {item['category']}"
    )
