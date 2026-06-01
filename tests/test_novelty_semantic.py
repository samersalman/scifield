"""Unit tests for the pure numeric core of semantic novelty (V1-S10).

These exercise the embedding-distance math on tiny hand-built arrays — no
real corpus, no FAISS, no DuckDB. The public ``compute_semantic_novelty`` is
imported to assert the symbol exists and is wired, but its full I/O path is
left to the orchestrator's integration run.
"""

from __future__ import annotations

import numpy as np
import pytest

from scifield.novelty.semantic import (  # noqa: F401  (compute_* imported for API contract)
    _l2_normalize,
    _mean_novelty_by_field,
    _min_novelty_within_field,
    compute_semantic_novelty,
)


def _unit(rows: list[list[float]]) -> np.ndarray:
    """L2-normalise a small hand-written matrix to mimic real inputs."""
    return _l2_normalize(np.asarray(rows, dtype=np.float32))


def test_mean_novelty_matches_hand_computed_centroid() -> None:
    # Single field, three years. Vectors chosen for easy dot products.
    # p0 (year 2000) = e_x, p1 (year 2001) = e_y, p2 (year 2002) = e_x.
    vectors = _unit([[1, 0], [0, 1], [1, 0]])
    topic_ids = np.array([7, 7, 7])
    years = np.array([2000, 2001, 2002])

    mean, n_prior = _mean_novelty_by_field(vectors, topic_ids, years)

    # p0: no prior -> NaN, n_prior 0.
    assert n_prior[0] == 0
    assert np.isnan(mean[0])
    # p1: prior {p0=e_x}, centroid = e_x, cos(e_y, e_x)=0 -> nov = 1.
    assert n_prior[1] == 1
    assert mean[1] == pytest.approx(1.0)
    # p2: priors {e_x, e_y}, centroid=(0.5,0.5), cos(e_x,centroid)=0.5 -> nov=0.5.
    assert n_prior[2] == 2
    assert mean[2] == pytest.approx(0.5)


def test_same_year_papers_are_not_priors() -> None:
    # Two papers in the SAME year must not see each other as priors.
    vectors = _unit([[1, 0], [0, 1]])
    topic_ids = np.array([3, 3])
    years = np.array([2010, 2010])

    mean, n_prior = _mean_novelty_by_field(vectors, topic_ids, years)

    assert n_prior[0] == 0
    assert n_prior[1] == 0
    assert np.isnan(mean[0])
    assert np.isnan(mean[1])


def test_earliest_in_field_has_zero_prior_and_null_scores() -> None:
    # Two distinct fields; the chronologically-first paper of each is earliest.
    vectors = _unit([[1, 0], [0, 1], [1, 1], [1, 0]])
    topic_ids = np.array([1, 1, 2, 2])
    years = np.array([1999, 2001, 2000, 2002])

    mean, n_prior = _mean_novelty_by_field(vectors, topic_ids, years)
    mn = _min_novelty_within_field(vectors, topic_ids, years)

    # Earliest of field 1 (row 0) and field 2 (row 2): no prior.
    for earliest in (0, 2):
        assert n_prior[earliest] == 0
        assert np.isnan(mean[earliest])
        assert np.isnan(mn[earliest])
    # Later papers have a prior.
    assert n_prior[1] == 1
    assert n_prior[3] == 1
    assert not np.isnan(mn[1])
    assert not np.isnan(mn[3])


def test_min_novelty_equals_one_minus_nearest_prior_cosine() -> None:
    # Field with a focal paper closer to one prior than another.
    # p0=e_x (2000), p1=e_y (2000), p2 (2001) sits at 45deg between them.
    vectors = _unit([[1, 0], [0, 1], [1, 1]])
    topic_ids = np.array([5, 5, 5])
    years = np.array([2000, 2000, 2001])

    mn = _min_novelty_within_field(vectors, topic_ids, years)

    # p2 cosine to each prior is cos(45deg) = 1/sqrt(2); nearest -> same.
    expected = 1.0 - (1.0 / np.sqrt(2.0))
    assert mn[2] == pytest.approx(expected, abs=1e-6)
    # The two year-2000 papers have no prior.
    assert np.isnan(mn[0])
    assert np.isnan(mn[1])


def test_paper_identical_to_prior_has_min_novelty_near_zero() -> None:
    # p1 is a byte-identical re-statement of prior p0 -> distance ~ 0.
    vectors = _unit([[0.3, 0.7, 0.1], [0.3, 0.7, 0.1]])
    topic_ids = np.array([9, 9])
    years = np.array([2005, 2006])

    mn = _min_novelty_within_field(vectors, topic_ids, years)

    assert mn[1] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(mn[0])


def test_min_novelty_respects_field_partition() -> None:
    # Identical vectors but in DIFFERENT fields must not count as priors.
    vectors = _unit([[1, 0], [1, 0]])
    topic_ids = np.array([1, 2])
    years = np.array([2000, 2001])

    mn = _min_novelty_within_field(vectors, topic_ids, years)

    assert np.isnan(mn[0])
    assert np.isnan(mn[1])  # row 1's only earlier paper is in another field


def test_min_novelty_fields_filter_skips_excluded_fields() -> None:
    # When `fields` is restricted, rows outside it keep NaN even if they have
    # a within-field prior. Used to exclude noise (handled by FAISS).
    vectors = _unit([[1, 0], [1, 0], [0, 1], [0, 1]])
    topic_ids = np.array([1, 1, -1, -1])
    years = np.array([2000, 2001, 2000, 2001])

    mn = _min_novelty_within_field(vectors, topic_ids, years, fields=np.array([1]))

    assert mn[1] == pytest.approx(0.0, abs=1e-6)  # field 1 processed
    assert np.isnan(mn[3])  # field -1 excluded despite having a prior


def test_l2_normalize_unit_rows_and_zero_safe() -> None:
    out = _l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    # Zero row stays zero (no NaN).
    assert np.all(out[1] == 0.0)
