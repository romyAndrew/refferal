"""Tests for the company reference catalog (offline, no network)."""

from pathlib import Path

import pytest

from referral_finder.companies import (
    CompanyCatalogError,
    CompaniesCatalog,
    load_companies,
    load_default_catalog,
    parse_companies,
)
from referral_finder.prefilter import (
    PREFILTER_VERSION,
    effective_prefilter_version,
    prefilter,
)

REAL_CATALOG = Path(__file__).resolve().parents[2] / "config" / "companies.yaml"

REQUIRED_IDS = {
    "yandex_eda",
    "yandex_lavka",
    "yandex_delivery",
    "vkusvill",
    "kuper",
    "samokat",
    "x5",
    "pyaterochka",
    "perekrestok",
    "chizhik",
    "ozon",
    "ozon_fresh",
    "wildberries",
    "magnit",
    "lenta",
    "metro",
    "auchan",
    "dixy",
    "globus",
    "okey",
    "azbuka_vkusa",
    "cdek",
    "dostavista",
    "peshkariki",
}


def _catalog(text: str) -> CompaniesCatalog:
    return parse_companies(text)


def test_real_catalog_loads_and_covers_all_groups():
    catalog = load_companies(REAL_CATALOG)
    ids = {company.id for company in catalog.companies}
    assert REQUIRED_IDS <= ids


def test_alternative_spellings_share_the_canonical_company():
    catalog = load_companies(REAL_CATALOG)
    # Kuper and its historical name SberMarket are the same company.
    assert catalog.canonicalize("Есть знакомые в СберМаркете?") == ("kuper",)
    assert catalog.canonicalize("кто работает в Купере?") == ("kuper",)
    # Spelling/yo variants.
    assert catalog.canonicalize("ВкусВилл") == ("vkusvill",)
    assert catalog.canonicalize("Вкусвилл") == ("vkusvill",)
    assert catalog.canonicalize("Пятерочка") == ("pyaterochka",)
    assert catalog.canonicalize("Пятёрочка") == ("pyaterochka",)


def test_most_specific_alias_wins():
    catalog = load_companies(REAL_CATALOG)
    # "Ozon Fresh" must not also be reported as plain "Ozon".
    assert catalog.canonicalize("Ozon Fresh") == ("ozon_fresh",)
    assert catalog.canonicalize("Озон Фреш") == ("ozon_fresh",)


def test_unknown_company_still_matches_universal_rule():
    catalog = load_companies(REAL_CATALOG)
    result = prefilter("Кто работает в Ромашке?", catalog)
    assert result.matched is True
    assert "prefilter.company_person" in result.rule_ids


def test_catalog_assisted_detection_covers_extra_phrasings():
    catalog = load_companies(REAL_CATALOG)
    # Not covered by the universal structural rule, but a person cue next to a
    # known alias is enough.
    result = prefilter("Кто-нибудь из СберМаркета здесь?", catalog)
    assert result.matched is True
    assert "prefilter.company_alias_person" in result.rule_ids


def test_bare_brand_mention_is_not_person_search():
    catalog = load_companies(REAL_CATALOG)
    person_ids = {"prefilter.company_person", "prefilter.company_alias_person"}
    for text in (
        "Яндекс доставил заказ",
        "Какой телефон службы поддержки Яндекса?",
        "Купил продукты в Самокате",
    ):
        result = prefilter(text, catalog)
        # A bare brand mention must never make it a person/employee search.
        assert person_ids.isdisjoint(result.rule_ids), text


def test_changing_catalog_changes_effective_version():
    catalog_a = _catalog(
        "companies:\n  - id: foo\n    canonical: Foo\n    aliases:\n      - Foo\n"
    )
    catalog_b = _catalog(
        "companies:\n  - id: foo\n    canonical: Foo\n    aliases:\n"
        "      - Foo\n      - Foo Bar\n"
    )
    assert effective_prefilter_version(None) == PREFILTER_VERSION
    assert effective_prefilter_version(catalog_a) != effective_prefilter_version(catalog_b)
    assert effective_prefilter_version(catalog_a).startswith(PREFILTER_VERSION)


def test_content_hash_is_stable_under_reordering():
    one = _catalog(
        "companies:\n  - id: foo\n    canonical: Foo\n    aliases:\n"
        "      - Foo\n      - Bar\n"
    )
    two = _catalog(
        "companies:\n  - id: foo\n    canonical: Foo\n    aliases:\n"
        "      - Bar\n      - Foo\n"
    )
    assert one.content_hash == two.content_hash


def test_missing_catalog_is_a_hard_error(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(CompanyCatalogError):
        load_companies(missing)
    # The default path must not silently fall back to an empty catalog.
    with pytest.raises(CompanyCatalogError):
        load_default_catalog(missing)
    with pytest.raises(CompanyCatalogError):
        load_default_catalog(environ={"TG_COMPANIES_FILE": str(missing)})


def test_default_catalog_is_cwd_independent(monkeypatch, tmp_path):
    # Running from an unrelated working directory must still find the catalog.
    monkeypatch.chdir(tmp_path)
    catalog = load_default_catalog()
    ids = {company.id for company in catalog.companies}
    assert REQUIRED_IDS <= ids


def test_explicit_env_path_overrides_default(tmp_path):
    catalog = load_default_catalog(environ={"TG_COMPANIES_FILE": str(REAL_CATALOG)})
    assert {c.id for c in catalog.companies} >= REQUIRED_IDS


def test_invalid_catalog_file_raises_with_line_number(tmp_path):
    bad = tmp_path / "companies.yaml"
    bad.write_text(
        "companies:\n  - id: foo\n    canonical: Foo\n    aliases:\n"
        "      - Foo\n    unexpected: x\n",
        encoding="utf-8",
    )
    with pytest.raises(CompanyCatalogError) as excinfo:
        load_companies(bad)
    assert "line" in str(excinfo.value)


def test_parser_rejects_malformed_and_ambiguous_content():
    with pytest.raises(CompanyCatalogError):
        _catalog("companies:\n  - id: foo\n    canonical: Foo\n    aliases:\n")
    with pytest.raises(CompanyCatalogError):
        _catalog("companies:\n  - id: foo\n    canonical: Foo\n    aliases:\n      - Foo\n"
                 "  - id: bar\n    canonical: Bar\n    aliases:\n      - Foo\n")
    with pytest.raises(CompanyCatalogError):
        _catalog("companies:\n  - id: foo\n    canonical: Foo\n    aliases:\n"
                 "      - Foo\n    unexpected: x\n")
    with pytest.raises(CompanyCatalogError):
        _catalog("# only comments\n")
