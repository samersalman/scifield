"""Unit tests for the V2 v1↔v2 topic crosswalk.

No corpus I/O: every fixture is hand-built numpy/dict data whose centroids, member
overlaps, and expected mappings are known by construction. The module is pure, so the
builder script (``V2/scripts/build_crosswalk.py``) is not exercised here.

The load-bearing test is :func:`test_split_is_captured_in_top_k`: a v1 topic is
constructed to *split* into two v2 topics (one v1 cluster occupies the same region as
two v2 sub-clusters), and we assert the crosswalk surfaces **both** v2 topics — the
split-detection guarantee this module exists to provide, the case a clean-permutation
mapping would silently lose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scifield.cartography.crosswalk import (
    CROSSWALK_COLUMNS,
    NOISE_LABEL,
    build_crosswalk,
    centroid_cosine_matrix,
    pmid_jaccard,
    topic_centroids,
)

_SEED = 20260609


def _unit(*components: float) -> np.ndarray:
    """A float64 vector (not necessarily unit — centroid fns normalize themselves)."""
    return np.asarray(components, dtype=np.float64)


# --------------------------------------------------------------------------- #
# topic_centroids                                                             #
# --------------------------------------------------------------------------- #
def test_centroids_are_the_per_topic_mean_and_noise_is_excluded() -> None:
    """Centroid == mean of a topic's rows; the noise label (-1) is dropped."""
    embeddings = np.array(
        [
            [2.0, 0.0],  # topic 0
            [4.0, 0.0],  # topic 0  -> mean (3, 0) -> normalized (1, 0)
            [0.0, 5.0],  # topic 1  -> normalized (0, 1)
            [9.0, 9.0],  # noise -1 -> must NOT appear
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, NOISE_LABEL])

    cents = topic_centroids(embeddings, labels, normalize=True)
    assert set(cents) == {0, 1}  # noise excluded
    np.testing.assert_allclose(cents[0], [1.0, 0.0])
    np.testing.assert_allclose(cents[1], [0.0, 1.0])

    # Without normalization we get the raw mean.
    raw = topic_centroids(embeddings, labels, normalize=False)
    np.testing.assert_allclose(raw[0], [3.0, 0.0])


def test_centroids_validate_shapes() -> None:
    """Non-2-D embeddings or a label/row mismatch raise ValueError."""
    with pytest.raises(ValueError):
        topic_centroids(np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError):
        topic_centroids(np.zeros((3, 2)), np.zeros(2))


# --------------------------------------------------------------------------- #
# centroid_cosine_matrix                                                      #
# --------------------------------------------------------------------------- #
def test_cosine_matrix_orientation_and_values() -> None:
    """rows=v1, cols=v2; identical centroids -> 1.0, orthogonal -> 0.0."""
    v1 = {0: _unit(1.0, 0.0), 2: _unit(0.0, 1.0)}  # ids deliberately not contiguous
    v2 = {5: _unit(1.0, 0.0), 7: _unit(0.0, 1.0)}

    matrix, v1_ids, v2_ids = centroid_cosine_matrix(v1, v2)
    assert v1_ids == [0, 2]  # ascending
    assert v2_ids == [5, 7]
    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix[0, 0], 1.0)  # v1 0 vs v2 5 (both x-axis)
    np.testing.assert_allclose(matrix[0, 1], 0.0)  # v1 0 vs v2 7 (orthogonal)
    np.testing.assert_allclose(matrix[1, 1], 1.0)  # v1 2 vs v2 7 (both y-axis)


def test_cosine_matrix_empty_side_is_well_formed() -> None:
    """An empty centroid map yields a correctly-shaped empty matrix, no crash."""
    matrix, v1_ids, v2_ids = centroid_cosine_matrix({}, {5: _unit(1.0, 0.0)})
    assert matrix.shape == (0, 1)
    assert v1_ids == []
    assert v2_ids == [5]


# --------------------------------------------------------------------------- #
# pmid_jaccard                                                                #
# --------------------------------------------------------------------------- #
def test_jaccard_math_on_known_sets() -> None:
    """Jaccard = |∩| / |∪|; only overlapping pairs are returned."""
    v1 = {0: {1, 2, 3, 4}, 1: {90, 91}}
    v2 = {5: {3, 4, 5, 6}, 6: {100}}  # v1[0]∩v2[5]={3,4}; v1[1] & v2[6] share nothing

    jac = pmid_jaccard(v1, v2)
    # |{3,4}| / |{1,2,3,4,5,6}| = 2/6
    assert jac[(0, 5)] == pytest.approx(2 / 6)
    # No overlapping pair other than (0, 5).
    assert set(jac) == {(0, 5)}


def test_jaccard_identical_sets_is_one() -> None:
    """Identical member sets give Jaccard 1.0."""
    jac = pmid_jaccard({0: {1, 2, 3}}, {5: {1, 2, 3}})
    assert jac[(0, 5)] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# build_crosswalk — clean map + the SPLIT guarantee                          #
# --------------------------------------------------------------------------- #
def _split_and_clean_fixture() -> tuple[dict, dict, dict, dict]:
    """Construct a clean 1→1 map (v1 0 → v2 5) and a 1→2 split (v1 1 → v2 {2,3}).

    Geometry (2-D unit circle):
      * v1 topic 0 sits at angle 0  -> v2 topic 5 sits at angle 0  (clean match).
      * v1 topic 1 sits between two v2 sub-clusters: v2 topic 2 at +small angle and
        v2 topic 3 at -small angle, both very close to v1 1 (the split).
    Membership mirrors the geometry: v1 0 and v2 5 share most PMIDs; v1 1's papers are
    partitioned across v2 2 and v2 3.
    """
    a = np.deg2rad(8.0)  # small angle so both split halves keep high cosine to v1 1
    v1_centroids = {
        0: _unit(1.0, 0.0),
        1: _unit(np.cos(np.deg2rad(90.0)), np.sin(np.deg2rad(90.0))),  # straight up
    }
    v2_centroids = {
        5: _unit(1.0, 0.0),  # == v1 0
        2: _unit(np.cos(np.deg2rad(90.0) + a), np.sin(np.deg2rad(90.0) + a)),
        3: _unit(np.cos(np.deg2rad(90.0) - a), np.sin(np.deg2rad(90.0) - a)),
    }
    v1_members = {
        0: {1, 2, 3, 4},
        1: {10, 11, 12, 13},  # split below
    }
    v2_members = {
        5: {1, 2, 3, 4},  # clean: same as v1 0
        2: {10, 11},  # half of v1 1
        3: {12, 13},  # other half of v1 1
    }
    return v1_centroids, v2_centroids, v1_members, v2_members


def test_clean_one_to_one_maps_v1_0_to_v2_5() -> None:
    """v1 topic 0 maps to v2 topic 5 at rank 1 with cosine≈1 and Jaccard==1."""
    cw = build_crosswalk(*_split_and_clean_fixture(), top_k=3, min_cosine=0.3)

    top0 = cw[(cw["v1_topic"] == 0) & (cw["rank"] == 1)].iloc[0]
    assert int(top0["v2_topic"]) == 5
    assert top0["cosine"] == pytest.approx(1.0, abs=1e-9)
    assert top0["jaccard"] == pytest.approx(1.0)
    assert int(top0["n_overlap"]) == 4
    assert int(top0["n_v1"]) == 4 and int(top0["n_v2"]) == 4


def test_split_is_captured_in_top_k() -> None:
    """v1 topic 1 SPLITS: the crosswalk returns BOTH v2 topics {2, 3}.

    This is the guarantee a clean-permutation mapping would lose. Both halves clear the
    cosine threshold (small angular offset) and each shares exactly half of v1 1's
    papers, so both appear as candidates for v1 1.
    """
    cw = build_crosswalk(*_split_and_clean_fixture(), top_k=3, min_cosine=0.3)

    v1_1 = cw[cw["v1_topic"] == 1]
    assert set(v1_1["v2_topic"]) == {2, 3}  # the split is captured, both present
    # Each split half overlaps exactly 2 of v1 1's 4 papers -> Jaccard 2/4.
    for _, row in v1_1.iterrows():
        assert int(row["n_overlap"]) == 2
        assert row["jaccard"] == pytest.approx(2 / 4)
    # v1 0 did NOT bleed into the split (clean topic stays 1→1 at rank 1).
    assert int(cw[(cw["v1_topic"] == 0) & (cw["rank"] == 1)]["v2_topic"].iloc[0]) == 5


def test_merge_is_visible_as_one_v2_under_many_v1() -> None:
    """Two v1 topics pointing at the same v2 topic shows as a merge in the long table."""
    # Both v1 topics sit at the same point as the single v2 topic 9.
    v1_c = {0: _unit(1.0, 0.0), 1: _unit(1.0, 0.0)}
    v2_c = {9: _unit(1.0, 0.0)}
    v1_m = {0: {1, 2}, 1: {3, 4}}
    v2_m = {9: {1, 2, 3, 4}}

    cw = build_crosswalk(v1_c, v2_c, v1_m, v2_m, top_k=3, min_cosine=0.3)
    merge = cw.groupby("v2_topic").size()
    assert int(merge.loc[9]) == 2  # v2 topic 9 is the home of two v1 topics


# --------------------------------------------------------------------------- #
# determinism, schema, thresholds                                            #
# --------------------------------------------------------------------------- #
def test_output_schema_and_sorted_determinism() -> None:
    """Columns/dtypes are fixed; output is sorted by (v1_topic, rank) and stable."""
    fixture = _split_and_clean_fixture()
    cw1 = build_crosswalk(*fixture, top_k=3, min_cosine=0.3)
    cw2 = build_crosswalk(*fixture, top_k=3, min_cosine=0.3)

    assert list(cw1.columns) == CROSSWALK_COLUMNS
    assert cw1["v1_topic"].dtype == np.dtype("int64")
    assert cw1["cosine"].dtype == np.dtype("float64")
    # Sorted by (v1_topic, rank).
    sorted_view = cw1.sort_values(["v1_topic", "rank"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(cw1, sorted_view)
    # Determinism: identical inputs -> identical frame.
    pd.testing.assert_frame_equal(cw1, cw2)


def test_top_k_caps_candidates_per_v1_topic() -> None:
    """top_k limits the number of v2 candidates emitted for a v1 topic."""
    # One v1 topic near three equally-plausible v2 topics; top_k=2 keeps only two.
    v1_c = {0: _unit(1.0, 0.0)}
    v2_c = {
        1: _unit(np.cos(np.deg2rad(2)), np.sin(np.deg2rad(2))),
        2: _unit(np.cos(np.deg2rad(4)), np.sin(np.deg2rad(4))),
        3: _unit(np.cos(np.deg2rad(6)), np.sin(np.deg2rad(6))),
    }
    v1_m = {0: {1, 2, 3}}
    v2_m = {1: {1}, 2: {2}, 3: {3}}

    cw = build_crosswalk(v1_c, v2_c, v1_m, v2_m, top_k=2, min_cosine=0.3)
    assert len(cw) == 2
    assert list(cw["rank"]) == [1, 2]
    # Ranked by cosine desc: the nearest (angle 2) is rank 1.
    assert int(cw.iloc[0]["v2_topic"]) == 1


def test_min_cosine_threshold_filters_distant_topics() -> None:
    """A v2 topic below min_cosine and with no shared PMIDs is dropped."""
    v1_c = {0: _unit(1.0, 0.0)}
    v2_c = {5: _unit(1.0, 0.0), 6: _unit(0.0, 1.0)}  # 6 is orthogonal (cosine 0)
    v1_m = {0: {1, 2}}
    v2_m = {5: {1, 2}, 6: {99}}  # 6 shares nothing -> jaccard 0

    cw = build_crosswalk(v1_c, v2_c, v1_m, v2_m, top_k=3, min_cosine=0.3, min_jaccard=0.0)
    assert set(cw["v2_topic"]) == {5}  # topic 6 filtered (cosine 0 < 0.3, jaccard 0)


def test_min_jaccard_can_rescue_an_off_centroid_overlap() -> None:
    """A pair below min_cosine is KEPT if it clears min_jaccard (either-threshold rule)."""
    # v2 topic 6 is orthogonal (cosine 0 < min_cosine) but shares ALL of v1 0's papers.
    v1_c = {0: _unit(1.0, 0.0)}
    v2_c = {6: _unit(0.0, 1.0)}
    v1_m = {0: {1, 2, 3}}
    v2_m = {6: {1, 2, 3}}  # jaccard 1.0

    cw = build_crosswalk(v1_c, v2_c, v1_m, v2_m, top_k=3, min_cosine=0.3, min_jaccard=0.5)
    assert set(cw["v2_topic"]) == {6}  # rescued by Jaccard despite zero cosine
    assert cw.iloc[0]["jaccard"] == pytest.approx(1.0)


def test_empty_inputs_yield_well_formed_empty_frame() -> None:
    """Empty centroid/member maps return the empty frame with the canonical schema."""
    cw = build_crosswalk({}, {}, {}, {})
    assert cw.empty
    assert list(cw.columns) == CROSSWALK_COLUMNS
    assert cw["v1_topic"].dtype == np.dtype("int64")
    assert cw["cosine"].dtype == np.dtype("float64")
