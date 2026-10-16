"""Company reference catalog.

Loads ``config/companies.yaml`` and provides alias → canonical-company lookup
and a catalog content hash used to version the prefilter.

To avoid a new runtime dependency (and any network access) the module ships a
**strict, restricted** reader for a small YAML subset instead of PyYAML. The
supported subset is intentionally tiny and fully validated; anything else raises
:class:`CompanyCatalogError` with a line number. Supported syntax:

* full-line comments (``# ...``) and blank lines;
* ``version: N`` (optional) and ``companies:`` at the top level;
* ``  - id: <id>`` starting a company (2-space indent);
* ``    canonical: <name>`` (4-space indent);
* ``    aliases:`` followed by ``      - <alias>`` items (6-space indent);
* optional single/double quotes around values.

The catalog never makes a bare brand mention a person search; it only helps
detect alternative spellings once a person/employee cue is present.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from .classifier import normalize

logger = logging.getLogger(__name__)

# Environment variable for an explicit catalog path (optional).
COMPANIES_FILE_ENV = "TG_COMPANIES_FILE"
DEFAULT_COMPANIES_FILENAME = "companies.yaml"

# Older code/CLI referred to a CWD-relative default; kept as a documented name.
DEFAULT_COMPANIES_PATH = "config/companies.yaml"


class CompanyCatalogError(ValueError):
    """Raised when the companies catalog is missing, malformed or ambiguous."""


def project_root() -> Path:
    """Repository root, derived from the package location (CWD-independent)."""
    return Path(__file__).resolve().parents[2]


def resolve_companies_path(
    path=None, environ: Optional[Mapping[str, str]] = None
) -> Path:
    """Resolve the catalog path.

    Precedence: explicit ``path`` > ``TG_COMPANIES_FILE`` > the repository's
    ``config/companies.yaml``. The default does not depend on the current
    working directory.
    """
    env = os.environ if environ is None else environ
    if path is not None:
        return Path(path).expanduser()
    explicit = env.get(COMPANIES_FILE_ENV)
    if explicit and explicit.strip():
        return Path(explicit.strip()).expanduser()
    return project_root() / "config" / DEFAULT_COMPANIES_FILENAME


class Company:
    """One canonical company with its writing variants."""

    __slots__ = ("id", "canonical", "aliases")

    def __init__(self, company_id: str, canonical: str, aliases: Sequence[str]):
        self.id = company_id
        self.canonical = canonical
        self.aliases = tuple(aliases)


# Person/employee cues used together with a known company alias. Kept focused
# (no bare "есть"/"ищу") so a brand mention alone never opens the gate.
_PERSON_CUE = re.compile(
    r"(?:"
    r"\bкто\b|\bкого\b|\bкому\b|\bкем\b|"
    r"знаком\w*|сотрудник\w*|работник\w*|человек\w*|коллег\w*|"
    r"работает|работают|трудится|трудятся|"
    r"\bwho\b|\banyone\b|\banybody\b|"
    r"know(?:s)?\s+(?:anyone|someone|somebody|a\s+person)|"
    r"employee\w*|\bstaff\b|\bworks\b|\bworking\b|"
    r"sotrudnik\w*|znakom\w*|rabotaet|rabotayut"
    r")",
    re.IGNORECASE,
)

_CUE_WINDOW = 48


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value


def _compute_hash(companies: Sequence[Company]) -> str:
    """Stable hash of the normalized catalog content (order-independent)."""
    parts: List[str] = []
    for company in sorted(companies, key=lambda c: c.id):
        parts.append("id=" + company.id)
        parts.append("canonical=" + normalize(company.canonical))
        for alias in sorted({normalize(a) for a in company.aliases if a}):
            parts.append("alias=" + alias)
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


class CompaniesCatalog:
    """Resolved company catalog: lookup, person-search detection, content hash."""

    def __init__(self, companies: Iterable[Company]):
        self._companies: Tuple[Company, ...] = tuple(companies)
        self._by_id = {c.id: c for c in self._companies}
        if len(self._by_id) != len(self._companies):
            raise CompanyCatalogError("duplicate company id in companies.yaml")

        alias_map: dict = {}
        for company in self._companies:
            names = set(company.aliases) | {company.canonical}
            for alias in names:
                key = normalize(alias)
                if not key:
                    continue
                existing = alias_map.get(key)
                if existing is not None and existing != company.id:
                    raise CompanyCatalogError(
                        f"alias {alias!r} maps to both {existing!r} and {company.id!r}"
                    )
                alias_map[key] = company.id
        self._alias_map = alias_map
        self._patterns: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
            # Allow a trailing word tail so inflected forms match ("в Купере",
            # "в СберМаркете"); the cue requirement keeps this from over-firing.
            (company_id, re.compile(r"(?<!\w)" + re.escape(alias) + r"\w*"))
            for alias, company_id in alias_map.items()
        )
        self.content_hash = _compute_hash(self._companies) if self._companies else ""

    @property
    def companies(self) -> Tuple[Company, ...]:
        return self._companies

    def get(self, company_id: str) -> Optional[Company]:
        return self._by_id.get(company_id)

    def canonicalize(self, text: str) -> Tuple[str, ...]:
        """Return canonical company ids mentioned in ``text`` (most specific first)."""
        normalized = normalize(text)
        if not normalized:
            return ()
        matches: List[Tuple[int, int, str]] = []
        for company_id, pattern in self._patterns:
            for match in pattern.finditer(normalized):
                matches.append((match.start(), match.end(), company_id))
        # Prefer the longest match for overlapping aliases (e.g. "Ozon Fresh"
        # over "Ozon").
        matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        accepted: List[Tuple[int, int, str]] = []
        for start, end, company_id in matches:
            if any(not (end <= s or start >= e) for s, e, _ in accepted):
                continue
            accepted.append((start, end, company_id))
        seen: List[str] = []
        for _, _, company_id in sorted(accepted):
            if company_id not in seen:
                seen.append(company_id)
        return tuple(seen)

    def has_person_search(self, normalized_text: str) -> bool:
        """True when a known company alias co-occurs with a person/employee cue."""
        if not normalized_text:
            return False
        for _, pattern in self._patterns:
            for match in pattern.finditer(normalized_text):
                window = normalized_text[
                    max(0, match.start() - _CUE_WINDOW): match.end() + _CUE_WINDOW
                ]
                if _PERSON_CUE.search(window):
                    return True
        return False


def parse_companies(text: str) -> CompaniesCatalog:
    """Parse the restricted YAML subset. Raises :class:`CompanyCatalogError`."""
    companies: List[Company] = []
    current: Optional[dict] = None
    in_aliases = False

    def _finish() -> None:
        nonlocal current
        if current is None:
            return
        company_id = current["id"]
        canonical = current["canonical"]
        aliases = current["aliases"]
        if not company_id:
            raise CompanyCatalogError("company entry without an id")
        if not canonical:
            raise CompanyCatalogError(f"company {company_id!r} without a canonical name")
        if not aliases:
            raise CompanyCatalogError(f"company {company_id!r} without aliases")
        companies.append(Company(company_id, canonical, tuple(dict.fromkeys(aliases))))
        current = None

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if re.match(r"^version:\s*\S+\s*$", line) or line.strip() == "companies:":
            continue
        match = re.match(r"^  - id:\s*(.+?)\s*$", line)
        if match:
            _finish()
            current = {
                "id": _scalar(match.group(1)),
                "canonical": None,
                "aliases": [],
            }
            in_aliases = False
            continue
        if current is None:
            raise CompanyCatalogError(f"companies.yaml line {lineno}: unexpected content")
        match = re.match(r"^    canonical:\s*(.+?)\s*$", line)
        if match:
            current["canonical"] = _scalar(match.group(1))
            in_aliases = False
            continue
        if re.match(r"^    aliases:\s*$", line):
            in_aliases = True
            continue
        match = re.match(r"^      - (.+?)\s*$", line)
        if match and in_aliases:
            current["aliases"].append(_scalar(match.group(1)))
            continue
        raise CompanyCatalogError(f"companies.yaml line {lineno}: unsupported syntax")

    _finish()
    if not companies:
        raise CompanyCatalogError("companies.yaml contains no companies")
    return CompaniesCatalog(companies)


def load_companies(path) -> CompaniesCatalog:
    """Load and validate the catalog from an explicit ``path`` (must exist)."""
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise CompanyCatalogError(f"companies catalog not found: {catalog_path}")
    return parse_companies(catalog_path.read_text(encoding="utf-8"))


def load_default_catalog(
    path=None, environ: Optional[Mapping[str, str]] = None
) -> CompaniesCatalog:
    """Load the catalog from the resolved path.

    A missing catalog is a hard error, never a silent fallback: silently using
    an empty catalog would change the effective prefilter version (and reuse
    stale decisions) and drop a whole detection rule without any warning.
    """
    catalog_path = resolve_companies_path(path, environ)
    if not catalog_path.exists():
        raise CompanyCatalogError(
            "companies catalog not found: "
            f"{catalog_path} (set {COMPANIES_FILE_ENV} to point at it)"
        )
    return load_companies(catalog_path)
