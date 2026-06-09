"""Unit tests for the V2 cartography rigor primitives (permutation + jackknife).

No I/O, no global RNG: every stochastic call takes an explicit seed, so the
assertions are deterministic. The structured-vs-noise contrast for
:func:`permutation_test` is the load-bearing check — a clearly non-null statistic
must land in the tail (small p, large |z|); a noise statistic must sit in the bulk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scifield.cartography.nulls import (
    attribution_flip_rate,
    jackknife_drop_one,
    permutation_test,
    permute_labels,
    rank_stability,
)

_SEED = 20260609


# --------------------------------------------------------------------------- #
# permute_labels
# --------------------------------------------------------------------------- #
def test_permute_labels_global_preserves_multiset_and_other_columns() -> None:
    """Global shuffle keeps the label multiset and every other column unchanged."""
    df = pd.DataFrame({"x": range(8), "label": list("AABBCCDD"), "keep": list("zyxwvuts")})
    out = permute_labels(df, label_col="label", seed=_SEED)
    # Same labels, same counts.
    assert sorted(out["label"]) == sorted(df["label"])
    # Other columns untouched, index preserved, input not mutated.
    pd.testing.assert_series_equal(out["x"], df["x"])
    pd.testing.assert_series_equal(out["keep"], df["keep"])
    assert list(df["label"]) == list("AABBCCDD")  # original intact


def test_permute_labels_is_deterministic_for_a_seed() -> None:
    """Same seed -> identical replicate; different seed -> (very likely) different."""
    df = pd.DataFrame({"label": list("ABCDEFGH"), "g": [0] * 8})
    a = permute_labels(df, label_col="label", seed=_SEED)
    b = permute_labels(df, label_col="label", seed=_SEED)
    c = permute_labels(df, label_col="label", seed=_SEED + 1)
    assert list(a["label"]) == list(b["label"])
    assert list(a["label"]) != list(c["label"])


def test_permute_labels_within_group_preserves_per_group_composition() -> None:
    """Within-group shuffle keeps each group's label multiset intact."""
    df = pd.DataFrame(
        {
            "topic": [0, 0, 0, 1, 1, 1],
            "journal": ["a", "b", "c", "a", "b", "c"],
        }
    )
    out = permute_labels(df, label_col="journal", group_col="topic", seed=_SEED)
    for t in (0, 1):
        before = sorted(df.loc[df["topic"] == t, "journal"])
        after = sorted(out.loc[out["topic"] == t, "journal"])
        assert before == after


# --------------------------------------------------------------------------- #
# permutation_test
# --------------------------------------------------------------------------- #
def test_permutation_test_structured_beats_null() -> None:
    """An observation far in the right tail gets a small p and large positive z."""
    rng = np.random.default_rng(_SEED)
    null = rng.normal(0, 1, size=5000)
    res = permutation_test(6.0, null, alternative="greater")
    assert res["p_value"] < 0.01
    assert res["z"] > 4
    assert res["n_perm"] == 5000


def test_permutation_test_noise_does_not_beat_null() -> None:
    """An observation drawn from the null sits in the bulk (large two-sided p)."""
    rng = np.random.default_rng(_SEED)
    null = rng.normal(0, 1, size=5000)
    res = permutation_test(0.05, null, alternative="two-sided")
    assert res["p_value"] > 0.2
    assert abs(res["z"]) < 1.0


def test_permutation_test_add_one_estimator_never_zero() -> None:
    """Even a maximal observation yields p = 1/(1+n_perm), never exactly 0."""
    null = np.zeros(99)
    res = permutation_test(100.0, null, alternative="greater")
    assert res["p_value"] == 1 / (1 + 99)
    assert res["p_value"] > 0.0


def test_permutation_test_drops_nan_replicates() -> None:
    """NaN null replicates are excluded from n_perm and the tally."""
    null = np.array([0.0, 1.0, np.nan, 2.0, np.nan])
    res = permutation_test(0.5, null, alternative="greater")
    assert res["n_perm"] == 3


def test_permutation_test_rejects_bad_alternative() -> None:
    """An unknown alternative raises ValueError."""
    try:
        permutation_test(1.0, np.array([0.0, 1.0]), alternative="sideways")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------- #
# jackknife_drop_one
# --------------------------------------------------------------------------- #
def _scores_excluding(held_out: object) -> pd.DataFrame:
    """compute_fn fixture: rank the other entities, dropping held_out."""
    base = {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.5}
    kept = {k: v for k, v in base.items() if k != held_out}
    return pd.DataFrame({"entity": list(kept), "score": list(kept.values())})


def test_jackknife_drop_one_produces_one_run_per_entity() -> None:
    """N unique entities -> N runs, each tagged with the dropped entity."""
    entities = ["a", "b", "c", "d"]
    out = jackknife_drop_one(entities, _scores_excluding)
    assert out["held_out"].nunique() == 4
    assert set(out["held_out"]) == set(entities)
    # Each run excludes its held-out entity from the result body.
    for ent in entities:
        run = out[out["held_out"] == ent]
        assert ent not in set(run["entity"])
    # Each run has the other 3 entities.
    assert (out.groupby("held_out").size() == 3).all()


def test_jackknife_drop_one_dedups_entities() -> None:
    """Duplicate entities are collapsed so each is dropped exactly once."""
    out = jackknife_drop_one(["a", "a", "b"], _scores_excluding)
    assert out["held_out"].nunique() == 2


def test_jackknife_drop_one_empty_entities() -> None:
    """No entities -> an empty frame carrying just the held_out column."""
    out = jackknife_drop_one([], _scores_excluding)
    assert out.empty
    assert list(out.columns) == ["held_out"]


# --------------------------------------------------------------------------- #
# attribution_flip_rate
# --------------------------------------------------------------------------- #
def test_attribution_flip_rate_bounds_and_stability() -> None:
    """Flip rate is in [0, 1]; a perfectly stable attribution flips 0%."""
    stable = pd.DataFrame(
        {
            "held_out": ["a", "b", "c"],
            "topic": [0, 0, 0],
            "origin": ["x", "x", "x"],  # same origin in every run
        }
    )
    rate = attribution_flip_rate(stable, key_col="topic", value_col="origin")
    assert rate == 0.0
    assert 0.0 <= rate <= 1.0


def test_attribution_flip_rate_detects_a_flip() -> None:
    """A key whose attribution changes across runs counts toward the flip rate."""
    df = pd.DataFrame(
        {
            "held_out": ["a", "b", "a", "b"],
            "topic": [0, 0, 1, 1],
            "origin": ["x", "x", "y", "z"],  # topic 0 stable, topic 1 flips
        }
    )
    rate = attribution_flip_rate(df, key_col="topic", value_col="origin")
    assert rate == 0.5  # 1 of 2 keys flips


def test_attribution_flip_rate_nan_is_a_distinct_value() -> None:
    """A key that gains/loses an attribution (NaN vs value) counts as a flip."""
    df = pd.DataFrame(
        {
            "held_out": ["a", "b"],
            "topic": [0, 0],
            "origin": ["x", np.nan],
        }
    )
    assert attribution_flip_rate(df, key_col="topic", value_col="origin") == 1.0


def test_attribution_flip_rate_empty_is_zero() -> None:
    """An empty frame is vacuously stable (flip rate 0)."""
    empty = pd.DataFrame({"topic": [], "origin": []})
    assert attribution_flip_rate(empty, key_col="topic", value_col="origin") == 0.0


# --------------------------------------------------------------------------- #
# rank_stability
# --------------------------------------------------------------------------- #
def test_rank_stability_identical_orderings_is_one() -> None:
    """If every run ranks shared entities the same way, stability == 1.0."""
    df = pd.DataFrame(
        {
            "held_out": ["x", "x", "x", "y", "y", "y"],
            "entity": ["a", "b", "c", "a", "b", "c"],
            "score": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
        }
    )
    assert rank_stability(df, key_col="entity", score_col="score") == 1.0


def test_rank_stability_reversed_orderings_is_negative() -> None:
    """Opposite orderings across runs give a negative mean correlation."""
    df = pd.DataFrame(
        {
            "held_out": ["x", "x", "x", "y", "y", "y"],
            "entity": ["a", "b", "c", "a", "b", "c"],
            "score": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],  # reversed in run y
        }
    )
    assert rank_stability(df, key_col="entity", score_col="score") < 0


def test_rank_stability_single_run_is_one() -> None:
    """A single run is trivially self-consistent (stability 1.0)."""
    df = pd.DataFrame({"held_out": ["x", "x"], "entity": ["a", "b"], "score": [1.0, 2.0]})
    assert rank_stability(df, key_col="entity", score_col="score") == 1.0
