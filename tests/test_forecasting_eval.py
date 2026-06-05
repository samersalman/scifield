"""Unit tests for the V1-S14 sealed-test-set evaluation + Gate G4 stats.

These exercise the pure pre-registered statistics and the test-orchestration
plumbing in :mod:`scifield.forecasting.evaluate` on tiny hand-built numpy/pandas
fixtures — no real parquet, no DuckDB, no network, no GPU, and (crucially) the
injected HGT is a tiny FAKE forecaster, never the real torch model. Mirrors the
pure-fixture style of ``tests/test_forecasting_baselines.py``.

The expected values are HAND-COMPUTED from the plan's prose (Brier =
``(score - label)**2``; equal-width reliability bins with a ``1.0`` score in the
last bin; the H1∧H2 truth table), NOT round-tripped through the implementation:

- A constant-sign Brier diff (model strictly closer to every label than baseline)
  must yield ``direction == "favors_model"`` and a finite, small p-value.
- The all-zero-diff degenerate case must yield ``nan`` stats + ``"tie"`` WITHOUT
  raising (this scipy build returns ``(0.0, 1.0)`` rather than raising on an
  all-zero input, so the module pre-checks the all-zero case explicitly).
- Mismatched ``(topic_id, origin_year)`` keys must align on the INTERSECTION only.
- ``calibration_table`` on a per-bin self-consistent fixture has
  ``observed_freq == mean_predicted`` per non-empty bin, ``sum(n) == N``, and
  exactly ``n_bins`` rows.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from scifield.forecasting.baselines.base import ForecastPrediction
from scifield.forecasting.data import MLP_FEATURES, NODE_FEATURES
from scifield.forecasting.evaluate import (
    calibration_table,
    gate_g4_verdict,
    paired_brier_wilcoxon,
    paired_score_wilcoxon,
)

# NOTE: the three public functions are named ``test_*`` per the pinned V1-S14 API,
# which collides with pytest's default test-collection (pytest would try to run the
# imported callables as test cases and fail looking for ``features``/``per_unit``
# fixtures). Alias them to non-``test_`` names HERE so pytest ignores them; the
# public symbols in ``evaluate.py`` are unchanged.
from scifield.forecasting.evaluate import test_metrics_table as build_test_metrics_table
from scifield.forecasting.evaluate import test_per_unit_scores as build_test_per_unit_scores
from scifield.forecasting.evaluate import test_sensitivity_table as build_test_sensitivity_table

# The baselines fit inside test_metrics_table / test_per_unit_scores are CPU-torch;
# pin to one intra-op thread (matches evaluate's guard) for reliability/determinism.
try:  # pragma: no cover - torch is a hard dep, but stay defensive
    import torch

    torch.set_num_threads(1)
except ImportError:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #


def _per_unit_side(
    topic_id: list[int],
    origin_year: list[int],
    emergence_score: list[float],
    emergent: list[int],
    *,
    emergent_additive: list[int] | None = None,
    emergent_count_surge: list[int] | None = None,
    share_forecast: list[float] | None = None,
    forward_share: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Build one model's per-unit staged-array dict (test_per_unit_scores shape)."""
    n = len(topic_id)
    return {
        "topic_id": np.asarray(topic_id, dtype=np.int64),
        "origin_year": np.asarray(origin_year, dtype=np.int64),
        "emergence_score": np.asarray(emergence_score, dtype=float),
        "emergent": np.asarray(emergent),
        "emergent_additive": np.asarray(
            emergent if emergent_additive is None else emergent_additive
        ),
        "emergent_count_surge": np.asarray(
            emergent if emergent_count_surge is None else emergent_count_surge
        ),
        "share_forecast": np.asarray(
            [0.0] * n if share_forecast is None else share_forecast, dtype=float
        ),
        "forward_share": np.asarray(
            [0.0] * n if forward_share is None else forward_share, dtype=float
        ),
    }


class _FakeForecaster:
    """A minimal injected 'hgt': emergence_score = sigmoid of a chosen feature col.

    Mirrors the real HGT's INJECTION contract (``name == "hgt"`` and a
    ``predict(features) -> ForecastPrediction`` of the right length) without any
    torch. ``fit`` deliberately raises so a test fails loudly if the orchestration
    ever retrains the injected model.
    """

    name = "hgt"

    def __init__(self, score_col: str = "share_growth_3yr", share_col: str = "share_3yr_mean"):
        self.score_col = score_col
        self.share_col = share_col

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> _FakeForecaster:
        raise AssertionError("injected hgt must never be fit by the orchestration")

    def predict(self, features: pd.DataFrame) -> ForecastPrediction:
        raw = features[self.score_col].to_numpy(dtype=float)
        score = 1.0 / (1.0 + np.exp(-raw))  # squash to [0, 1] like the real model
        share = np.clip(features[self.share_col].to_numpy(dtype=float), 0.0, None)
        return ForecastPrediction(emergence_score=score, share_forecast=share)


# --------------------------------------------------------------------------- #
# paired_brier_wilcoxon
# --------------------------------------------------------------------------- #


def test_paired_brier_known_sign_favors_model() -> None:
    # Model is strictly closer to every label than the baseline:
    #   label=1 rows: model 0.9 (brier .01) vs baseline 0.4 (brier .36) -> diff -.35
    #   label=0 rows: model 0.1 (brier .01) vs baseline 0.6 (brier .36) -> diff -.35
    # => every diff < 0 => median < 0 => "favors_model"; 5 same-sign pairs => p=0.0625.
    keys_t = [1, 2, 3, 4, 5]
    keys_y = [2018] * 5
    emergent = [1, 0, 1, 0, 1]
    model = _per_unit_side(keys_t, keys_y, [0.9, 0.1, 0.9, 0.1, 0.9], emergent)
    base = _per_unit_side(keys_t, keys_y, [0.4, 0.6, 0.4, 0.6, 0.4], emergent)
    per_unit = {"hgt": model, "no_graph": base}

    res = paired_brier_wilcoxon(per_unit)
    assert res["n_pairs"] == 5
    assert res["direction"] == "favors_model"
    assert res["median_diff"] == pytest.approx(-0.35)
    assert res["pvalue"] == pytest.approx(0.0625)  # smallest 2-sided p for n=5
    assert isinstance(res["statistic"], float) and not math.isnan(res["statistic"])


def test_paired_brier_favors_baseline_sign() -> None:
    # Flip the roles: now the baseline is closer, so every diff > 0 -> "favors_baseline".
    keys_t = [1, 2, 3, 4, 5]
    keys_y = [2018] * 5
    emergent = [1, 0, 1, 0, 1]
    model = _per_unit_side(keys_t, keys_y, [0.4, 0.6, 0.4, 0.6, 0.4], emergent)
    base = _per_unit_side(keys_t, keys_y, [0.9, 0.1, 0.9, 0.1, 0.9], emergent)
    res = paired_brier_wilcoxon({"hgt": model, "no_graph": base})
    assert res["direction"] == "favors_baseline"
    assert res["median_diff"] == pytest.approx(0.35)


def test_paired_brier_clearly_separated_small_pvalue() -> None:
    # 10 same-sign (negative) diffs -> two-sided p = 2 * 2^-10 ~ 0.00195 (< 0.05).
    n = 10
    keys_t = list(range(n))
    keys_y = [2018] * n
    emergent = [1, 0] * (n // 2)
    model = _per_unit_side(keys_t, keys_y, [0.95 if e else 0.05 for e in emergent], emergent)
    base = _per_unit_side(keys_t, keys_y, [0.45 if e else 0.55 for e in emergent], emergent)
    res = paired_brier_wilcoxon({"hgt": model, "no_graph": base})
    assert res["n_pairs"] == n
    assert res["direction"] == "favors_model"
    assert res["pvalue"] < 0.05


def test_paired_brier_all_zero_diff_is_tie_nan() -> None:
    # Identical scores on both sides -> diff all zero -> degenerate -> nan + tie,
    # and NO exception (this scipy build returns (0.0, 1.0) rather than raising).
    keys_t = [1, 2, 3, 4]
    keys_y = [2018] * 4
    emergent = [1, 0, 1, 0]
    same = [0.7, 0.2, 0.8, 0.3]
    model = _per_unit_side(keys_t, keys_y, same, emergent)
    base = _per_unit_side(keys_t, keys_y, same, emergent)
    res = paired_brier_wilcoxon({"hgt": model, "no_graph": base})
    assert res["n_pairs"] == 4  # counted before zero-dropping
    assert math.isnan(res["statistic"])
    assert math.isnan(res["pvalue"])
    assert res["median_diff"] == 0.0
    assert res["direction"] == "tie"


def test_paired_brier_aligns_on_intersection_only() -> None:
    # model has keys {(1,18),(2,18),(3,18),(4,18)}; baseline {(2,18),(3,18),(4,18),(9,18)}.
    # Intersection = {(2,18),(3,18),(4,18)} -> n_pairs == 3. The non-shared rows
    # ((1,18) only in model, (9,18) only in baseline) are dropped.
    model = _per_unit_side([1, 2, 3, 4], [18, 18, 18, 18], [0.9, 0.9, 0.9, 0.9], [1, 1, 0, 0])
    base = _per_unit_side([2, 3, 4, 9], [18, 18, 18, 18], [0.4, 0.4, 0.6, 0.6], [1, 0, 0, 1])
    res = paired_brier_wilcoxon({"hgt": model, "no_graph": base})
    assert res["n_pairs"] == 3


def test_paired_brier_alignment_is_by_key_not_position() -> None:
    # Same 3 keys but the baseline rows are SHUFFLED relative to the model. A
    # positional impl would mis-pair; key alignment must give the correct diff.
    # keys/labels: (1,18)->1, (2,18)->0, (3,18)->1.
    model = _per_unit_side([1, 2, 3], [18, 18, 18], [0.9, 0.1, 0.9], [1, 0, 1])
    # baseline rows reordered: (3,18) then (1,18) then (2,18); scores 0.4/0.4/0.6.
    base = _per_unit_side([3, 1, 2], [18, 18, 18], [0.4, 0.4, 0.6], [1, 1, 0])
    res = paired_brier_wilcoxon({"hgt": model, "no_graph": base})
    # Per key: brier_model .01 each; brier_base .36 each => diff -.35 each.
    assert res["n_pairs"] == 3
    assert res["median_diff"] == pytest.approx(-0.35)
    assert res["direction"] == "favors_model"


# --------------------------------------------------------------------------- #
# paired_score_wilcoxon
# --------------------------------------------------------------------------- #


def test_paired_score_model_higher_direction() -> None:
    # Raw-score diff: model uniformly +0.3 above baseline -> median > 0 -> "model_higher".
    keys_t = [1, 2, 3, 4, 5]
    keys_y = [2019] * 5
    emergent = [1, 0, 1, 0, 1]
    model = _per_unit_side(keys_t, keys_y, [0.6, 0.5, 0.7, 0.4, 0.8], emergent)
    base = _per_unit_side(keys_t, keys_y, [0.3, 0.2, 0.4, 0.1, 0.5], emergent)
    res = paired_score_wilcoxon({"hgt": model, "no_graph": base})
    assert res["n_pairs"] == 5
    assert res["direction"] == "model_higher"
    assert res["median_diff"] == pytest.approx(0.3)
    assert not math.isnan(res["pvalue"])


def test_paired_score_baseline_higher_direction() -> None:
    # Model uniformly below baseline -> median < 0 -> "baseline_higher".
    keys_t = [1, 2, 3, 4, 5]
    keys_y = [2019] * 5
    emergent = [1, 0, 1, 0, 1]
    model = _per_unit_side(keys_t, keys_y, [0.3, 0.2, 0.4, 0.1, 0.5], emergent)
    base = _per_unit_side(keys_t, keys_y, [0.6, 0.5, 0.7, 0.4, 0.8], emergent)
    res = paired_score_wilcoxon({"hgt": model, "no_graph": base})
    assert res["direction"] == "baseline_higher"
    assert res["median_diff"] == pytest.approx(-0.3)


def test_paired_score_all_zero_is_tie_nan() -> None:
    keys_t = [1, 2, 3]
    keys_y = [2019] * 3
    same = [0.5, 0.5, 0.5]
    model = _per_unit_side(keys_t, keys_y, same, [1, 0, 1])
    base = _per_unit_side(keys_t, keys_y, same, [1, 0, 1])
    res = paired_score_wilcoxon({"hgt": model, "no_graph": base})
    assert res["direction"] == "tie"
    assert math.isnan(res["pvalue"])
    assert res["median_diff"] == 0.0


def test_paired_score_empty_intersection_is_tie() -> None:
    # No shared keys -> n_pairs == 0 -> degenerate tie (nan), no raise.
    model = _per_unit_side([1, 2], [2018, 2018], [0.9, 0.1], [1, 0])
    base = _per_unit_side([3, 4], [2018, 2018], [0.4, 0.6], [1, 0])
    res = paired_score_wilcoxon({"hgt": model, "no_graph": base})
    assert res["n_pairs"] == 0
    assert res["direction"] == "tie"
    assert math.isnan(res["pvalue"])


# --------------------------------------------------------------------------- #
# calibration_table
# --------------------------------------------------------------------------- #

_CALIBRATION_COLUMNS = [
    "bin_index",
    "bin_lo",
    "bin_hi",
    "n",
    "mean_predicted",
    "observed_freq",
]


def test_calibration_perfect_per_bin() -> None:
    # Two bins (n_bins=2): edges [0, 0.5, 1].
    # Bin 0 ([0,0.5)): 10 scores all 0.2, exactly 2 labels==1 -> obs 0.2 == pred 0.2.
    # Bin 1 ([0.5,1]): 10 scores all 0.8, exactly 8 labels==1 -> obs 0.8 == pred 0.8.
    scores = np.array([0.2] * 10 + [0.8] * 10)
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    table = calibration_table(scores, labels, n_bins=2)

    assert list(table.columns) == _CALIBRATION_COLUMNS
    assert len(table) == 2
    assert int(table["n"].sum()) == len(scores)
    # Bin edges.
    assert table["bin_lo"].tolist() == pytest.approx([0.0, 0.5])
    assert table["bin_hi"].tolist() == pytest.approx([0.5, 1.0])
    # Per non-empty bin: observed_freq == mean_predicted (perfect calibration).
    for _, row in table.iterrows():
        assert row["n"] == 10
        assert row["observed_freq"] == pytest.approx(row["mean_predicted"])
    assert table.loc[0, "mean_predicted"] == pytest.approx(0.2)
    assert table.loc[1, "mean_predicted"] == pytest.approx(0.8)
    assert table.loc[0, "observed_freq"] == pytest.approx(0.2)
    assert table.loc[1, "observed_freq"] == pytest.approx(0.8)


def test_calibration_score_one_lands_in_last_bin() -> None:
    # A score of exactly 1.0 must be assigned to the LAST bin (inclusive right edge).
    scores = np.array([0.0, 1.0])
    labels = np.array([0, 1])
    table = calibration_table(scores, labels, n_bins=4)
    assert len(table) == 4
    assert table.loc[0, "n"] == 1  # 0.0 -> first bin
    assert table.loc[3, "n"] == 1  # 1.0 -> last bin
    assert table.loc[1, "n"] == 0
    assert table.loc[2, "n"] == 0
    # Empty bins -> nan means.
    assert math.isnan(table.loc[1, "mean_predicted"])
    assert math.isnan(table.loc[2, "observed_freq"])


def test_calibration_empty_input_all_empty_rows() -> None:
    table = calibration_table(np.array([]), np.array([]), n_bins=5)
    assert len(table) == 5
    assert (table["n"] == 0).all()
    assert table["mean_predicted"].isna().all()
    assert table["observed_freq"].isna().all()


def test_calibration_clips_out_of_range_scores() -> None:
    # Scores outside [0,1] are clipped for binning robustness: -0.2 -> bin 0, 1.5 -> last.
    scores = np.array([-0.2, 1.5])
    labels = np.array([0, 1])
    table = calibration_table(scores, labels, n_bins=4)
    assert table.loc[0, "n"] == 1
    assert table.loc[3, "n"] == 1
    # Clipped value used for the mean (-0.2 -> 0.0, 1.5 -> 1.0).
    assert table.loc[0, "mean_predicted"] == pytest.approx(0.0)
    assert table.loc[3, "mean_predicted"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# gate_g4_verdict — the H1 ∧ H2 truth table
# --------------------------------------------------------------------------- #


def _metrics(auc_hgt: float, auc_comparator: float) -> pd.DataFrame:
    # Minimal test_metrics_table-shaped frame: just the model + emergence_auc cols,
    # with a no_graph row and an hgt row (other baselines omitted intentionally).
    return pd.DataFrame(
        {
            "model": ["no_graph", "hgt"],
            "emergence_auc": [auc_comparator, auc_hgt],
        }
    )


def test_gate_g4_all_pass() -> None:
    # delta = (0.77 - 0.71)*100 = 6pp > 5; p=0.01 < 0.05 -> H1, H2, overall all PASS.
    res = gate_g4_verdict(_metrics(0.77, 0.71), {"pvalue": 0.01, "direction": "favors_model"})
    assert res["delta_pp"] == pytest.approx(6.0)
    assert res["h1_pass"] is True
    assert res["h2_pass"] is True
    assert res["overall_pass"] is True
    assert res["h2_direction"] == "favors_model"
    assert res["mechanical_recommendation"] == "PASS (F3 success)"


def test_gate_g4_h1_fails_margin() -> None:
    # delta = 3pp (not > 5) -> H1 fails even though p is significant -> NULL.
    res = gate_g4_verdict(_metrics(0.74, 0.71), {"pvalue": 0.01, "direction": "favors_model"})
    assert res["delta_pp"] == pytest.approx(3.0)
    assert res["h1_pass"] is False
    assert res["h2_pass"] is True
    assert res["overall_pass"] is False
    assert res["mechanical_recommendation"] == "NULL FINDING (F3 reported as null)"


def test_gate_g4_h2_fails_significance() -> None:
    # delta = 6pp (H1 ok) but p=0.20 (not < 0.05) -> H2 fails -> NULL.
    res = gate_g4_verdict(_metrics(0.77, 0.71), {"pvalue": 0.20, "direction": "favors_model"})
    assert res["h1_pass"] is True
    assert res["h2_pass"] is False
    assert res["overall_pass"] is False
    assert res["mechanical_recommendation"] == "NULL FINDING (F3 reported as null)"


def test_gate_g4_nan_auc_fails_h1_without_raising() -> None:
    # A single-class test slice -> nan AUC -> H1 fails (delta is nan); must not raise.
    res = gate_g4_verdict(
        _metrics(float("nan"), 0.71), {"pvalue": 0.01, "direction": "favors_model"}
    )
    assert res["h1_pass"] is False
    assert res["overall_pass"] is False
    assert res["mechanical_recommendation"] == "NULL FINDING (F3 reported as null)"


def test_gate_g4_nan_pvalue_fails_h2() -> None:
    # Degenerate Wilcoxon (tie) -> nan p -> H2 fails even with a big AUC margin.
    res = gate_g4_verdict(_metrics(0.85, 0.71), {"pvalue": float("nan"), "direction": "tie"})
    assert res["h1_pass"] is True
    assert res["h2_pass"] is False
    assert res["overall_pass"] is False
    assert res["h2_direction"] == "tie"


def test_gate_g4_exact_margin_boundary_is_strict() -> None:
    # delta exactly == margin is NOT > margin -> H1 fails (strict inequality).
    # 0.6 - 0.1 == 0.5 exactly in float64, so delta_pp == 50.0 exactly; use a 50pp
    # margin so we hit the boundary cleanly without float fuzz.
    res = gate_g4_verdict(
        _metrics(0.6, 0.1), {"pvalue": 0.01, "direction": "favors_model"}, margin_pp=50.0
    )
    assert res["delta_pp"] == pytest.approx(50.0)
    assert res["h1_pass"] is False


def test_gate_g4_custom_comparator() -> None:
    # comparator=mlp resolves the right row; full return-key set is present.
    metrics = pd.DataFrame({"model": ["mlp", "hgt"], "emergence_auc": [0.60, 0.70]})
    res = gate_g4_verdict(metrics, {"pvalue": 0.01, "direction": "favors_model"}, comparator="mlp")
    assert res["comparator"] == "mlp"
    assert res["auc_comparator"] == pytest.approx(0.60)
    assert res["delta_pp"] == pytest.approx(10.0)
    assert set(res.keys()) == {
        "auc_hgt",
        "auc_comparator",
        "comparator",
        "delta_pp",
        "margin_pp",
        "h1_pass",
        "pvalue",
        "alpha",
        "h2_pass",
        "h2_direction",
        "overall_pass",
        "mechanical_recommendation",
    }


# --------------------------------------------------------------------------- #
# test_sensitivity_table
# --------------------------------------------------------------------------- #

_SENSITIVITY_COLUMNS = [
    "variant",
    "label_column",
    "emergence_auc",
    "n_test",
    "n_test_pos",
    "delta_vs_primary_pp",
]


def test_sensitivity_three_rows_and_primary_delta_zero() -> None:
    # Scores perfectly rank the PRIMARY label -> primary AUC 1.0.
    # additive label flips one pair so AUC drops (delta < 0); count_surge keeps
    # perfect ranking under a different positive set -> AUC 1.0 (delta 0).
    scores = [0.1, 0.2, 0.8, 0.9]
    emergent = [0, 0, 1, 1]  # primary: perfectly separated by score -> AUC 1.0
    additive = [1, 0, 0, 1]  # one low-score positive (0.1) -> imperfect -> AUC < 1.0
    count_surge = [0, 0, 1, 1]  # same as primary -> AUC 1.0
    per_unit = {
        "hgt": _per_unit_side(
            [1, 2, 3, 4],
            [2018, 2018, 2018, 2018],
            scores,
            emergent,
            emergent_additive=additive,
            emergent_count_surge=count_surge,
        )
    }
    table = build_test_sensitivity_table(per_unit)

    assert list(table.columns) == _SENSITIVITY_COLUMNS
    assert list(table["variant"]) == ["primary", "additive_jump", "count_surge"]
    assert list(table["label_column"]) == [
        "emergent",
        "emergent_additive",
        "emergent_count_surge",
    ]
    assert (table["n_test"] == 4).all()

    primary = table.set_index("variant").loc["primary"]
    additive_row = table.set_index("variant").loc["additive_jump"]
    surge_row = table.set_index("variant").loc["count_surge"]

    assert primary["emergence_auc"] == pytest.approx(1.0)
    assert primary["delta_vs_primary_pp"] == 0.0  # primary delta is exactly 0
    assert primary["n_test_pos"] == 2

    # Additive AUC strictly below primary -> negative delta.
    assert additive_row["emergence_auc"] < 1.0
    assert additive_row["delta_vs_primary_pp"] < 0.0
    assert additive_row["delta_vs_primary_pp"] == pytest.approx(
        (additive_row["emergence_auc"] - 1.0) * 100
    )
    # count_surge ranks perfectly too -> AUC 1.0, delta 0.
    assert surge_row["emergence_auc"] == pytest.approx(1.0)
    assert surge_row["delta_vs_primary_pp"] == pytest.approx(0.0)


def test_sensitivity_single_class_variant_is_nan_delta() -> None:
    # A variant label that is all-zero -> AUC nan -> delta nan-safe (not a crash).
    scores = [0.1, 0.4, 0.9]
    per_unit = {
        "hgt": _per_unit_side(
            [1, 2, 3],
            [2018, 2018, 2018],
            scores,
            [0, 1, 1],  # primary: both classes -> finite AUC
            emergent_count_surge=[0, 0, 0],  # single-class -> nan AUC
        )
    }
    table = build_test_sensitivity_table(per_unit).set_index("variant")
    assert not math.isnan(table.loc["primary", "emergence_auc"])
    assert math.isnan(table.loc["count_surge", "emergence_auc"])
    assert math.isnan(table.loc["count_surge", "delta_vs_primary_pp"])
    assert table.loc["count_surge", "n_test_pos"] == 0


# --------------------------------------------------------------------------- #
# test_metrics_table / test_per_unit_scores — orchestration plumbing
# --------------------------------------------------------------------------- #

_MODEL_ORDER = ["naive", "arima", "mlp", "no_graph", "hgt"]
_TEST_METRIC_COLUMNS = [
    "model",
    "emergence_auc",
    "share_mape",
    "n_train",
    "n_test",
    "n_test_pos",
    "fallback_frac",
]
_PER_UNIT_KEYS = {
    "topic_id",
    "origin_year",
    "emergence_score",
    "emergent",
    "emergent_additive",
    "emergent_count_surge",
    "share_forecast",
    "forward_share",
}


def _materialized_frame() -> pd.DataFrame:
    """Synthetic materialized frame: TRAIN + TEST rows, both classes in TEST.

    Carries ``topic_id, origin_year, split`` + all NODE_FEATURES + the label cols
    (the PRIMARY ``emergent`` plus the two sensitivity labels ``emergent_additive``
    / ``emergent_count_surge``) + ``forward_share, volume_ok, label_complete``. All
    rows are ``volume_ok=True & label_complete=True`` (the labeled slice). A ``val``
    and an ``excluded`` row are added to confirm the TEST mask excludes them.
    """
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []

    def add(split: str, topic: int, year: int, emergent: int) -> None:
        row: dict[str, object] = {
            "topic_id": topic,
            "origin_year": year,
            "split": split,
            "emergent": emergent,
            "emergent_additive": int(rng.random() > 0.5),
            "emergent_count_surge": int(rng.random() > 0.5),
            "forward_share": abs(float(rng.normal())) + 0.05,  # > 0 so MAPE defined
            "volume_ok": True,
            "label_complete": True,
        }
        for col in NODE_FEATURES:
            row[col] = float(rng.normal())
        # Keep the rank-/series-bearing columns sane and positive.
        row["share_3yr_mean"] = abs(float(rng.normal())) + 0.01
        row["share_growth_3yr"] = abs(float(rng.normal())) + 0.5
        row["share_last"] = abs(float(rng.normal())) + 0.01
        rows.append(row)

    # Train: 4 topics across 2010..2016 (enough per-topic history for ARIMA).
    for tid in range(4):
        for y in range(2010, 2017):
            add("train", tid, y, int(rng.random() > 0.5))
    # Test: 4 topics across 2020..2021 with BOTH classes present.
    classes = [1, 0, 1, 0, 1, 0, 1, 0]
    i = 0
    for tid in range(4):
        for y in (2020, 2021):
            add("test", tid, y, classes[i])
            i += 1
    # A val row and an excluded row that the TEST mask must ignore.
    add("val", 0, 2018, 1)
    add("excluded", 1, 2019, 0)
    return pd.DataFrame(rows)


def _eval_cfg() -> OmegaConf:
    learned = {"hidden": 4, "epochs": 2, "lr": 0.01, "batch_size": 256, "seed": 1729}
    return OmegaConf.create(
        {
            "label": {"gamma": 1.5},
            "features": {
                "mlp_columns": list(MLP_FEATURES),
                "node_columns": list(NODE_FEATURES),
            },
            "baselines": {
                "naive": {},
                "arima": {"order": [1, 1, 0]},
                "mlp": dict(learned),
                "no_graph": dict(learned),
            },
        }
    )


def test_metrics_table_five_rows_and_contract() -> None:
    frame = _materialized_frame()
    metrics = build_test_metrics_table(frame, _eval_cfg(), _FakeForecaster())

    # EXACT columns + 5 rows in the locked order (hgt last).
    assert list(metrics.columns) == _TEST_METRIC_COLUMNS
    assert list(metrics["model"]) == _MODEL_ORDER
    assert len(metrics) == 5

    n_train_expected = int((frame["split"] == "train").sum())
    n_test_expected = int((frame["split"] == "test").sum())
    n_test_pos_expected = int(((frame["split"] == "test") & (frame["emergent"] == 1)).sum())
    assert (metrics["n_train"] == n_train_expected).all()
    assert (metrics["n_test"] == n_test_expected).all()
    assert (metrics["n_test_pos"] == n_test_pos_expected).all()
    # The val/excluded rows are NOT in the test slice (8 test rows, not 10).
    assert n_test_expected == 8
    assert n_test_pos_expected == 4


def test_metrics_table_fallback_frac_nan_except_arima() -> None:
    metrics = build_test_metrics_table(
        _materialized_frame(), _eval_cfg(), _FakeForecaster()
    ).set_index("model")
    for name in ("naive", "mlp", "no_graph", "hgt"):
        assert math.isnan(metrics.loc[name, "fallback_frac"])
    arima_ff = metrics.loc["arima", "fallback_frac"]
    assert not math.isnan(arima_ff)
    assert 0.0 <= arima_ff <= 1.0


def test_metrics_table_hgt_auc_is_finite_on_both_class_test() -> None:
    # Test slice has both classes -> the injected hgt's AUC is a real number, and
    # its emergence scores are in [0,1] (the FakeForecaster squashes via sigmoid).
    metrics = build_test_metrics_table(
        _materialized_frame(), _eval_cfg(), _FakeForecaster()
    ).set_index("model")
    assert not math.isnan(metrics.loc["hgt", "emergence_auc"])
    assert not math.isnan(metrics.loc["hgt", "share_mape"])


def test_per_unit_scores_keys_lengths_and_order() -> None:
    frame = _materialized_frame()
    per_unit = build_test_per_unit_scores(frame, _eval_cfg(), _FakeForecaster())

    assert set(per_unit.keys()) == set(_MODEL_ORDER)
    n_test = int((frame["split"] == "test").sum())
    for name in _MODEL_ORDER:
        arrays = per_unit[name]
        assert set(arrays.keys()) == _PER_UNIT_KEYS, f"{name} key set"
        lengths = {len(v) for v in arrays.values()}
        assert lengths == {n_test}, f"{name} ragged: {lengths}"
        assert np.all(np.isfinite(arrays["emergence_score"]))
        assert np.all(np.isfinite(arrays["share_forecast"]))
    # The shared label arrays are identical across models (they come off the slice).
    for name in _MODEL_ORDER:
        assert np.array_equal(per_unit[name]["emergent"], per_unit["naive"]["emergent"])
        assert np.array_equal(per_unit[name]["topic_id"], per_unit["naive"]["topic_id"])


def test_per_unit_scores_feeds_paired_brier_end_to_end() -> None:
    # The staged dict must drop straight into paired_brier_wilcoxon (hgt vs no_graph)
    # with the right pair count and a real (or degenerate) result, no exception.
    frame = _materialized_frame()
    per_unit = build_test_per_unit_scores(frame, _eval_cfg(), _FakeForecaster())
    res = paired_brier_wilcoxon(per_unit)
    n_test = int((frame["split"] == "test").sum())
    assert res["n_pairs"] == n_test
    assert res["direction"] in {"favors_model", "favors_baseline", "tie"}
