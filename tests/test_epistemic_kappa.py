"""Synthetic-only tests for V1-S07 inter-rater agreement helpers.

No real labels are loaded here — per the V1-S07 plan, ``kappa.py`` may
only run against synthetic data in this session. Every fixture below is
hand-built in-process.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from scifield.epistemic.kappa import (
    cohens_kappa,
    krippendorffs_alpha,
    per_field_summary,
)


def test_cohens_kappa_perfect_agreement() -> None:
    assert cohens_kappa([1, 1, 0, 0, 1], [1, 1, 0, 0, 1]) == 1.0


def test_cohens_kappa_perfect_disagreement_binary() -> None:
    # Binary inversion -> kappa == -1.0
    assert cohens_kappa([1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]) == -1.0


def test_cohens_kappa_matches_sklearn_on_hand_built_confusion() -> None:
    # 10-pair example with confusion matrix [[4, 1], [1, 4]]:
    #   4x (0,0), 1x (0,1), 1x (1,0), 4x (1,1).
    a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    b = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    expected = cohen_kappa_score(a, b)
    got = cohens_kappa(a, b)
    assert math.isclose(got, expected, abs_tol=1e-6)
    # Sanity: this confusion matrix yields kappa = 0.6 by hand.
    assert math.isclose(got, 0.6, abs_tol=1e-6)


def test_cohens_kappa_drops_none_pairs() -> None:
    # Pair-wise: (1,1), (None,1) -> drop, (1,1), (1,1) -> 3 usable pairs,
    # all perfect agreement.
    got = cohens_kappa([1, None, 1, 1], [1, 1, 1, 1])
    # After drop: a=[1,1,1], b=[1,1,1] -> single unique label -> nan
    assert math.isnan(got)


def test_cohens_kappa_drops_none_pairs_with_variation() -> None:
    # After dropping None pairs there should still be 3 comparable pairs
    # with mixed labels.
    a = [1, None, 0, 1, None]
    b = [1, 1, 0, 0, 0]
    # Drop indices 1 and 4 -> a=[1,0,1], b=[1,0,0]
    expected = cohen_kappa_score([1, 0, 1], [1, 0, 0])
    got = cohens_kappa(a, b)
    assert math.isclose(got, expected, abs_tol=1e-6)


def test_cohens_kappa_constant_field_returns_nan() -> None:
    # All-True constant: no variation -> nan, not a crash.
    assert math.isnan(cohens_kappa([True, True, True, True], [True, True, True, True]))


def test_cohens_kappa_fewer_than_two_pairs_returns_nan() -> None:
    assert math.isnan(cohens_kappa([1], [1]))
    assert math.isnan(cohens_kappa([], []))


def test_cohens_kappa_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([1, 0], [1, 0, 1])


def test_krippendorffs_alpha_perfect_agreement() -> None:
    assert krippendorffs_alpha([1, 0, 1, 0, 1], [1, 0, 1, 0, 1]) == 1.0


def test_krippendorffs_alpha_perfect_agreement_strings() -> None:
    # String labels exercise the integer-encoding path.
    a = ["RCT", "cohort", "RCT", "review", "cohort"]
    b = ["RCT", "cohort", "RCT", "review", "cohort"]
    assert krippendorffs_alpha(a, b) == 1.0


def test_krippendorffs_alpha_drops_none_pairs() -> None:
    # After drop: 3 usable pairs all matching -> single class -> nan.
    got = krippendorffs_alpha([1, None, 1, 1], [1, 1, 1, 1])
    assert math.isnan(got)


def test_krippendorffs_alpha_constant_field_returns_nan() -> None:
    assert math.isnan(krippendorffs_alpha([True, True, True], [True, True, True]))


def test_krippendorffs_alpha_fewer_than_two_pairs_returns_nan() -> None:
    assert math.isnan(krippendorffs_alpha([1], [1]))


def test_per_field_summary_columns_and_row_count() -> None:
    pairs: dict[str, tuple[Sequence[Any], Sequence[Any]]] = {
        "study_design": (["RCT", "cohort"], ["RCT", "cohort"]),
        "has_control": ([True, False], [True, False]),
        "sample_size": ([100, 200], [100, 200]),
    }
    df = per_field_summary(pairs)
    assert list(df.columns) == ["field", "n", "n_agree", "kappa", "alpha"]
    assert len(df) == len(pairs)
    assert list(df["field"]) == ["study_design", "has_control", "sample_size"]


def test_per_field_summary_values_match_helpers_two_field_ten_pairs() -> None:
    # Field 1: study_design (string enum), 10 pairs, mostly agree, one swap.
    sd_a = ["RCT", "RCT", "cohort", "cohort", "review", "review", "RCT", "cohort", "RCT", "RCT"]
    sd_b = ["RCT", "RCT", "cohort", "cohort", "review", "review", "RCT", "cohort", "RCT", "cohort"]

    # Field 2: has_control (bool), 10 pairs with one None on rater A and
    # one disagreement on the surviving pairs.
    hc_a = [True, True, False, False, True, None, True, False, True, False]
    hc_b = [True, True, False, False, True, True, True, False, False, False]

    pairs: dict[str, tuple[Sequence[Any], Sequence[Any]]] = {
        "study_design": (sd_a, sd_b),
        "has_control": (hc_a, hc_b),
    }
    df = per_field_summary(pairs)

    # study_design row
    row_sd = df.loc[df["field"] == "study_design"].iloc[0]
    assert row_sd["n"] == 10
    assert row_sd["n_agree"] == 9
    assert math.isclose(row_sd["kappa"], cohens_kappa(sd_a, sd_b), abs_tol=1e-12)
    assert math.isclose(row_sd["alpha"], krippendorffs_alpha(sd_a, sd_b), abs_tol=1e-12)

    # has_control row (after dropping the None pair, 9 usable pairs, 8 agree)
    row_hc = df.loc[df["field"] == "has_control"].iloc[0]
    assert row_hc["n"] == 9
    assert row_hc["n_agree"] == 8
    assert math.isclose(row_hc["kappa"], cohens_kappa(hc_a, hc_b), abs_tol=1e-12)
    assert math.isclose(row_hc["alpha"], krippendorffs_alpha(hc_a, hc_b), abs_tol=1e-12)


def test_per_field_summary_handles_constant_field_without_raising() -> None:
    pairs: dict[str, tuple[Sequence[Any], Sequence[Any]]] = {
        "all_true": ([True, True, True], [True, True, True]),
        "varies": ([1, 0, 1], [1, 0, 1]),
    }
    df = per_field_summary(pairs)
    constant_row = df.loc[df["field"] == "all_true"].iloc[0]
    assert constant_row["n"] == 3
    assert constant_row["n_agree"] == 3
    assert math.isnan(constant_row["kappa"])
    assert math.isnan(constant_row["alpha"])


def test_per_field_summary_returns_pandas_dataframe() -> None:
    pairs: dict[str, tuple[list, list]] = {"f": ([1, 0], [1, 0])}
    df = per_field_summary(pairs)
    assert isinstance(df, pd.DataFrame)
    # Smoke: alpha/kappa dtype is numeric (could be float64 / NaN-friendly).
    assert np.issubdtype(df["kappa"].dtype, np.floating)
    assert np.issubdtype(df["alpha"].dtype, np.floating)
