"""Tests for effective analysis versioning (prompt version + criteria hash)."""

from referral_finder.llm.versioning import (
    CRITERIA_HASH_LENGTH,
    criteria_hash,
    effective_analysis_version,
    normalize_criteria,
)

PROMPT = "prompt-2026.10"


def test_normalize_criteria_trims_and_collapses_whitespace():
    assert normalize_criteria("  курьер   доставка ") == "курьер доставка"
    assert normalize_criteria(None) == ""


def test_criteria_hash_is_stable_and_length_bounded():
    assert criteria_hash("a") == criteria_hash("a")
    assert len(criteria_hash("anything")) == CRITERIA_HASH_LENGTH


def test_whitespace_variants_share_the_same_hash():
    assert criteria_hash("курьер доставка") == criteria_hash("  курьер   доставка \n")


def test_same_criteria_same_version():
    assert effective_analysis_version(PROMPT, "курьер") == effective_analysis_version(
        PROMPT, "курьер"
    )


def test_changed_criteria_produce_a_distinct_version():
    a = effective_analysis_version(PROMPT, "курьер")
    b = effective_analysis_version(PROMPT, "пеший курьер")
    assert a != b


def test_empty_criteria_is_deterministic_and_keeps_prompt_prefix():
    version = effective_analysis_version(PROMPT, "")
    assert version == effective_analysis_version(PROMPT, "   ")
    assert version.startswith(PROMPT + "+c")


def test_version_does_not_embed_raw_criteria_text():
    version = effective_analysis_version(PROMPT, "секретное слово курьер")
    assert "секретное" not in version
    assert "курьер" not in version
