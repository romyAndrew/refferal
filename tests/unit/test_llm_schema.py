"""Tests for the structured AI-analysis contract (no network)."""

import pytest

from referral_finder.llm import (
    STATUS_ERROR,
    STATUS_OK,
    AnalysisResult,
    ValidationError,
    ai_analysis_row,
    expected_relevant,
    validate_analysis,
)


def test_valid_relevant_category():
    result = validate_analysis(
        {
            "category": "job_interest",
            "relevant": True,
            "confidence": 0.8,
            "explanation": "ищет работу курьером",
        }
    )
    assert result == AnalysisResult("job_interest", True, 0.8, "ищет работу курьером")


def test_valid_irrelevant_category():
    result = validate_analysis(
        {"category": "irrelevant", "relevant": False, "confidence": 0, "explanation": ""}
    )
    assert result.category == "irrelevant"
    assert result.relevant is False


def test_uncertain_abstains_and_is_not_negative():
    result = validate_analysis(
        {"category": "uncertain", "relevant": None, "confidence": 0.2, "explanation": "?"}
    )
    assert result.category == "uncertain"
    assert result.relevant is None
    # A missing "relevant" is also treated as abstain for uncertain.
    assert validate_analysis(
        {"category": "uncertain", "confidence": 0.1, "explanation": ""}
    ).relevant is None


def test_confidence_is_normalised_to_float():
    result = validate_analysis(
        {"category": "job_interest", "relevant": True, "confidence": 1, "explanation": ""}
    )
    assert result.confidence == 1.0


def test_expected_relevant_mapping():
    assert expected_relevant("referral_request") is True
    assert expected_relevant("recruitment_offer") is True
    assert expected_relevant("irrelevant") is False
    assert expected_relevant("uncertain") is None


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        {"category": "unknown", "relevant": False, "confidence": 0.5, "explanation": ""},
        {"category": "job_interest", "relevant": True, "confidence": True, "explanation": ""},
        {"category": "job_interest", "relevant": True, "confidence": 1.5, "explanation": ""},
        {"category": "job_interest", "relevant": True, "confidence": -0.1, "explanation": ""},
        {"category": "job_interest", "relevant": True, "confidence": 0.5, "explanation": 3},
        {"category": "job_interest", "relevant": None, "confidence": 0.5, "explanation": ""},
        {"category": "job_interest", "relevant": False, "confidence": 0.5, "explanation": ""},
        {"category": "irrelevant", "relevant": "no", "confidence": 0.5, "explanation": ""},
        {"category": "uncertain", "relevant": True, "confidence": 0.5, "explanation": ""},
    ],
)
def test_invalid_payloads_raise(payload):
    with pytest.raises(ValidationError):
        validate_analysis(payload)


def test_ai_analysis_row_ok():
    result = AnalysisResult("job_interest", True, 0.5, "reason")
    row = ai_analysis_row(
        1,
        2,
        result,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version="p1",
        prompt_hash="h1",
        analyzed_at_utc=100,
    )
    assert row == (1, 2, "deepseek", "deepseek-flash", "p1", "h1", STATUS_OK, 1,
                   "job_interest", 0.5, "reason", None, 100)


def test_ai_analysis_row_uncertain_abstains():
    result = AnalysisResult("uncertain", None, 0.1, "?")
    row = ai_analysis_row(
        1, 2, result, provider="p", model="m", prompt_version="v",
        prompt_hash="h", analyzed_at_utc=5,
    )
    assert row[7] is None  # relevant stays NULL, not 0


def test_ai_analysis_row_error_without_result():
    row = ai_analysis_row(
        1, 2, None, provider="p", model="m", prompt_version="v",
        prompt_hash="h", analyzed_at_utc=7, status=STATUS_ERROR, error="timeout",
    )
    assert row[6] == STATUS_ERROR
    assert row[8] is None and row[9] is None and row[10] is None
    assert row[11] == "timeout"
