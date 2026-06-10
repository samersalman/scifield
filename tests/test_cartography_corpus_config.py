"""Unit tests for the V2-S08 corpus-config single source of truth.

No corpus I/O beyond reading the committed ``conf/corpus/{v1,v2}.yaml`` rosters. These
tests pin the two deliberate-duplication seams flagged in the Batch-2 spec review so a
silent drift fails loudly:

* ``flow.CANONICAL_JOURNAL_SLUGS`` (the frozen v1 ten, kept as an independent literal
  for the freeze guarantee) must stay set-equal to ``get_canonical_journal_slugs("v1")``.
* ``get_generalist_slugs()`` (a literal, since the tier is not in the YAML schema) must
  remain a subset of the v2 roster.
"""

from __future__ import annotations

import pytest

from scifield.cartography import corpus_config
from scifield.cartography.flow import CANONICAL_JOURNAL_SLUGS

_EXPECTED_GENERALISTS = frozenset(
    {
        "n_engl_j_med",
        "lancet",
        "jama",
        "bmj",
        "ann_intern_med",
        "nature",
        "science",
        "proc_natl_acad_sci_usa",
    }
)


def test_v1_roster_set_equals_flow_literal() -> None:
    """Drift guard: the config-derived v1 roster matches the frozen flow literal."""
    assert set(corpus_config.get_canonical_journal_slugs("v1")) == set(CANONICAL_JOURNAL_SLUGS)


def test_v1_roster_is_the_ten_unique() -> None:
    slugs = corpus_config.get_canonical_journal_slugs("v1")
    assert len(slugs) == 10
    assert len(set(slugs)) == 10


def test_v2_roster_is_seventy_eight_unique() -> None:
    slugs = corpus_config.get_canonical_journal_slugs("v2")
    assert len(slugs) == 78
    assert len(set(slugs)) == 78


def test_v2_keeps_the_v1_ten() -> None:
    """Tier-0 = the v1 ten, kept verbatim — so v2 must be a superset of v1."""
    v1 = set(corpus_config.get_canonical_journal_slugs("v1"))
    v2 = set(corpus_config.get_canonical_journal_slugs("v2"))
    assert v1 <= v2


def test_generalists_exact_eight() -> None:
    assert corpus_config.get_generalist_slugs() == _EXPECTED_GENERALISTS


def test_generalists_subset_of_v2_roster() -> None:
    """Drift guard: every generalist slug really exists in the v2 corpus config."""
    v2 = set(corpus_config.get_canonical_journal_slugs("v2"))
    assert corpus_config.get_generalist_slugs() <= v2


def test_all_known_equals_v2_roster() -> None:
    """v2 ⊇ v1, so the union across known versions is exactly the v2 roster."""
    v2 = set(corpus_config.get_canonical_journal_slugs("v2"))
    assert set(corpus_config.all_known_slugs()) == v2


def test_resolve_data_version_default_is_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(corpus_config.ENV_VAR, raising=False)
    assert corpus_config.resolve_data_version() == "v1"


def test_resolve_data_version_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(corpus_config.ENV_VAR, "v2")
    assert corpus_config.resolve_data_version() == "v2"


def test_resolve_data_version_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(corpus_config.ENV_VAR, "v2")
    assert corpus_config.resolve_data_version("v1") == "v1"


def test_unknown_version_raises() -> None:
    with pytest.raises(FileNotFoundError):
        corpus_config.get_canonical_journal_slugs("nonexistent_version_xyz")
