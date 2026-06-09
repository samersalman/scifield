"""Unit tests for the V1-S15 F2-enrichment archetype-by-impact statistic.

No network / no data files: every fixture is a small deterministic synthetic
pandas DataFrame built with a fixed-seed ``np.random.default_rng`` (or
hand-computable constants). The pure :func:`kruskal_archetype_impact` is the only
thing exercised; the I/O notebook (``notebooks/12``) is deliberately NOT run.

Three contract fixtures are covered:

* (a) four clearly-separated archetype impact distributions -> small p AND a
  positive epsilon-squared, with the expected median ordering;
* (b) four identical-distribution groups -> non-significant p and ~0 effect size;
* (c) a truth-test of the ``epsilon_squared = (H - k + 1) / (n - k)`` formula on a
  hand-computable ``H``/``k``/``n`` (and a fully hand-worked 3-group case where H
  is computed independently from the rank sums).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scifield.findings.impact import kruskal_archetype_impact

# The four canonical dual-novelty archetype labels (semantic first, structural
# second), ordered low-impact -> high-impact as established by Gate G3.
_DN = "disruptive-novel"
_NC = "novel-consolidating"
_CD = "conventional-disruptive"
_INC = "incremental"
_G3_IMPACT_ORDER = [_DN, _NC, _CD, _INC]


def _frame(label_to_values: dict[str, np.ndarray]) -> pd.DataFrame:
    """Stack ``{archetype_label: impact_values}`` into a tidy two-column frame."""
    labels: list[str] = []
    impacts: list[float] = []
    for label, values in label_to_values.items():
        labels.extend([label] * len(values))
        impacts.extend(values.tolist())
    return pd.DataFrame({"arch_mean_cd5": labels, "cited_by_pctile_within_year": impacts})


# --------------------------------------------------------------------------- #
# (a) clearly-separated groups -> significant, positive effect, expected order
# --------------------------------------------------------------------------- #


def test_separated_groups_significant_with_expected_median_order() -> None:
    """Four well-separated impact distributions -> tiny p, positive epsilon^2.

    Group locations follow the G3 inversion (disruptive-novel lowest ->
    incremental highest); the test must recover both the significance and that
    median ordering.
    """
    rng = np.random.default_rng(0)
    n = 400
    # Means 0.30 < 0.45 < 0.55 < 0.70 with tight spread -> clean separation.
    data = {
        _DN: rng.normal(0.30, 0.05, n),
        _NC: rng.normal(0.45, 0.05, n),
        _CD: rng.normal(0.55, 0.05, n),
        _INC: rng.normal(0.70, 0.05, n),
    }
    df = _frame(data)

    res = kruskal_archetype_impact(
        df, archetype_col="arch_mean_cd5", impact_col="cited_by_pctile_within_year"
    )

    assert res["k"] == 4
    assert res["n"] == 4 * n
    # Strong separation -> overwhelmingly significant and a clearly positive effect.
    assert res["p_value"] < 1e-50
    assert res["H"] > 0.0
    assert res["epsilon_squared"] > 0.5

    # Median ordering reconfirms the G3 inversion: dn < nc < cd < inc.
    medians = [res["per_group"][g]["median"] for g in _G3_IMPACT_ORDER]
    assert medians == sorted(medians)
    assert res["per_group"][_DN]["median"] < res["per_group"][_INC]["median"]

    # epsilon_squared is exactly (H - k + 1) / (n - k) from the returned fields.
    expected_eps = (res["H"] - res["k"] + 1) / (res["n"] - res["k"])
    assert res["epsilon_squared"] == pytest.approx(expected_eps)
    # Per-group n's are reported and sum to the total.
    assert sum(res["per_group"][g]["n"] for g in _G3_IMPACT_ORDER) == res["n"]


# --------------------------------------------------------------------------- #
# (b) identical distributions -> non-significant, ~0 effect size
# --------------------------------------------------------------------------- #


def test_identical_groups_non_significant_and_near_zero_effect() -> None:
    """Four draws from ONE distribution -> p well above 0.05 and epsilon^2 ~ 0.

    Under the null, the expected value of H is ``k - 1`` (= 3 here), so the
    expected epsilon^2 is ~0; a single fixed seed lands comfortably non-significant.
    """
    rng = np.random.default_rng(0)
    n = 500
    data = {g: rng.normal(0.5, 0.10, n) for g in _G3_IMPACT_ORDER}
    df = _frame(data)

    res = kruskal_archetype_impact(
        df, archetype_col="arch_mean_cd5", impact_col="cited_by_pctile_within_year"
    )

    assert res["k"] == 4
    assert res["n"] == 4 * n
    assert res["p_value"] > 0.05  # no real group effect
    # Rank-based effect size hugs zero (|eps^2| small; can be slightly negative).
    assert abs(res["epsilon_squared"]) < 0.01
    # Formula consistency holds in the null regime too.
    expected_eps = (res["H"] - res["k"] + 1) / (res["n"] - res["k"])
    assert res["epsilon_squared"] == pytest.approx(expected_eps)


# --------------------------------------------------------------------------- #
# (b') nulls are dropped, not imputed; the universe shrinks accordingly
# --------------------------------------------------------------------------- #


def test_null_label_and_null_impact_rows_are_dropped() -> None:
    """Rows with a null archetype OR a null impact are excluded from n and groups."""
    df = pd.DataFrame(
        {
            "arch_mean_cd5": [_DN, _DN, _INC, _INC, None, _CD, _NC],
            "cited_by_pctile_within_year": [0.1, 0.2, 0.8, 0.9, 0.5, np.nan, 0.4],
        }
    )
    res = kruskal_archetype_impact(
        df, archetype_col="arch_mean_cd5", impact_col="cited_by_pctile_within_year"
    )

    # None-label row and the NaN-impact (conventional-disruptive) row both drop:
    # surviving groups are dn(2), inc(2), nc(1) -> k=3, n=5.
    assert res["k"] == 3
    assert res["n"] == 5
    assert set(res["per_group"]) == {_DN, _INC, _NC}
    assert res["per_group"][_DN]["n"] == 2
    assert res["per_group"][_INC]["n"] == 2
    assert res["per_group"][_NC]["n"] == 1
    assert res["per_group"][_DN]["median"] == pytest.approx(0.15)


def test_fewer_than_two_groups_raises() -> None:
    """A single surviving group cannot be Kruskal-tested -> ValueError."""
    df = pd.DataFrame(
        {
            "arch_mean_cd5": [_DN, _DN, None],
            "cited_by_pctile_within_year": [0.1, 0.2, 0.9],
        }
    )
    with pytest.raises(ValueError, match="2 archetype groups"):
        kruskal_archetype_impact(
            df, archetype_col="arch_mean_cd5", impact_col="cited_by_pctile_within_year"
        )


# --------------------------------------------------------------------------- #
# (c) epsilon_squared formula truth-test on a hand-computable H/k/n
# --------------------------------------------------------------------------- #


def test_epsilon_squared_formula_matches_hand_computation() -> None:
    """Hand-worked 3-group Kruskal case: independent H -> epsilon^2 by formula.

    Groups (no ties): A=[1,2,3], B=[4,5,6], C=[7,8,9] over N=9 ranked values.
    Ranks equal the values (already sorted, distinct), so rank sums are
    R_A=6, R_B=15, R_C=24. The Kruskal statistic with no ties is

        H = 12 / (N (N+1)) * sum(R_i^2 / n_i) - 3 (N+1)
          = 12 / (9*10) * (6^2/3 + 15^2/3 + 24^2/3) - 3*10
          = (12/90) * (12 + 75 + 192) - 30
          = (2/15) * 279 - 30 = 37.2 - 30 = 7.2

    With k=3, N=9: epsilon^2 = (H - k + 1)/(N - k) = (7.2 - 2)/6 = 5.2/6.
    """
    df = pd.DataFrame(
        {
            "arch_mean_cd5": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "cited_by_pctile_within_year": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    res = kruskal_archetype_impact(
        df, archetype_col="arch_mean_cd5", impact_col="cited_by_pctile_within_year"
    )

    assert res["k"] == 3
    assert res["n"] == 9
    # Independently hand-computed H for this tie-free design.
    assert res["H"] == pytest.approx(7.2)
    # The formula, evaluated by hand, equals 5.2 / 6.
    assert res["epsilon_squared"] == pytest.approx(5.2 / 6.0)
    # And it equals (H - k + 1)/(n - k) computed from the returned fields.
    assert res["epsilon_squared"] == pytest.approx(
        (res["H"] - res["k"] + 1) / (res["n"] - res["k"])
    )


def test_epsilon_squared_zero_crossing_when_H_below_k_minus_one() -> None:
    """Boundary: a near-null interleaving gives small H < k-1 -> epsilon^2 <= 0.

    Three tie-free groups whose values are fully interleaved across the joint rank
    order produce a tiny H (well below the null expectation ``k - 1 = 2``). The
    rank-based epsilon^2 then sits at or below zero, demonstrating the formula's
    zero-crossing on REAL (non-degenerate) ranks -- the all-tie corner is avoided
    on purpose because scipy returns NaN there (0/0 tie correction).
    """
    # Interleaved so each group spans the full range -> rank sums nearly equal.
    df = pd.DataFrame(
        {
            "arch_mean_cd5": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "cited_by_pctile_within_year": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    res = kruskal_archetype_impact(
        df, archetype_col="arch_mean_cd5", impact_col="cited_by_pctile_within_year"
    )
    assert res["k"] == 3
    assert res["n"] == 9
    # H stays below the null expectation k-1 -> non-positive epsilon^2.
    assert res["H"] < res["k"] - 1
    assert res["epsilon_squared"] <= 0.0
    # Formula stays exact on these real ranks.
    expected_eps = (res["H"] - res["k"] + 1) / (res["n"] - res["k"])
    assert res["epsilon_squared"] == pytest.approx(expected_eps)
