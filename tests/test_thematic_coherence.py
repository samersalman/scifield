"""Tests for :mod:`scifield.thematic.coherence`.

Tiny synthetic corpus — two topics with disjoint vocabulary — to confirm
the gensim wrapper returns finite floats for both NPMI and C_v and
tolerates degenerate inputs (empty word lists) without crashing.
"""

from __future__ import annotations

import math

import pytest

from scifield.thematic.coherence import compute_coherence, tokenise_for_coherence

pytest.importorskip("gensim")


def _toy_corpus() -> list[str]:
    """Two clearly-separated topics, repeated enough for gensim to estimate co-occurrence."""
    return [
        "cat dog pet animal",
        "dog cat pet kitten",
        "puppy dog pet animal",
        "cat kitten animal pet",
        "dog pet animal cat",
        "math physics science research",
        "physics math equation theory",
        "science research equation physics",
        "theory math science equation",
        "research physics science math",
    ] * 3


def test_tokenise_lowercases_and_strips_stopwords() -> None:
    tokens = tokenise_for_coherence(["The cat AND dog are pets!"])
    assert "the" not in tokens[0]
    assert "and" not in tokens[0]
    assert "are" not in tokens[0]
    assert "cat" in tokens[0]
    assert "dog" in tokens[0]
    assert "pets" in tokens[0]


def test_compute_coherence_returns_finite_floats() -> None:
    texts = tokenise_for_coherence(_toy_corpus())
    topics = [
        ["cat", "dog", "pet", "animal", "kitten"],
        ["math", "physics", "science", "research", "equation"],
    ]
    out = compute_coherence(topics, texts, top_n=5, metrics=("c_npmi", "c_v"))
    assert set(out.keys()) == {"c_npmi", "c_v"}
    for metric, value in out.items():
        assert isinstance(value, float), f"{metric} not float"
        assert math.isfinite(value), f"{metric} is non-finite: {value}"


def test_compute_coherence_skips_empty_word_lists() -> None:
    texts = tokenise_for_coherence(_toy_corpus())
    topics = [
        [],
        ["cat", "dog", "pet"],
        [""],
    ]
    out = compute_coherence(topics, texts, top_n=5, metrics=("c_npmi",))
    assert "c_npmi" in out
    assert math.isfinite(out["c_npmi"])


def test_compute_coherence_all_empty_returns_nan() -> None:
    texts = tokenise_for_coherence(_toy_corpus())
    out = compute_coherence([[], [""]], texts, top_n=5, metrics=("c_npmi", "c_v"))
    assert math.isnan(out["c_npmi"])
    assert math.isnan(out["c_v"])
