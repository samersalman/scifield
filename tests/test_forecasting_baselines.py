"""Unit tests for the V1-S12 forecasting baselines (pure fixtures, no real I/O).

These exercise the baseline scoring metrics, the four baseline predictors, and the
``evaluate`` plumbing on tiny hand-built numpy/pandas frames — no real parquet, no
DuckDB, no network. The learned baselines run at ``epochs=1..2`` so the whole module
stays fast. Mirrors the pure-fixture style of ``tests/test_novelty_semantic.py`` and
the config/fixture conventions of ``tests/test_forecasting_data.py``.

Encoded module assumptions (see HANDOFF NOTES at the bottom of this file):
- ``share_mape``'s default ``eps=1e-8`` floors the denominator, so the prose formula
  ``mean(|f-t|/t)`` is only exact when ``eps=0.0`` is passed (asserted both ways).
- ``ArimaPerTopic`` builds each row's series from the years ``<= t`` it has observed;
  a row whose series is shorter than ``min_series_len`` (=4) forces the naive fallback,
  and those fallback rows are byte-identical to ``NaiveMovingAverage(gamma).predict()``.
- A *constant long* series does NOT fall back via exception (it converges with se≈0 and
  uses the hard indicator), so we only assert it produces finite output without raising.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from scipy.special import expit

from scifield.forecasting.baselines.arima import ArimaPerTopic
from scifield.forecasting.baselines.base import (
    ForecastPrediction,
    Predictor,
    emergence_auc,
    share_mape,
)
from scifield.forecasting.baselines.evaluate import per_topic_scores, run_baselines
from scifield.forecasting.baselines.mlp import MlpForecaster, _TorchForecaster
from scifield.forecasting.baselines.naive import NaiveMovingAverage
from scifield.forecasting.baselines.no_graph import NoGraphForecaster
from scifield.forecasting.data import MLP_FEATURES, NODE_FEATURES

# Baselines are CPU-torch; pin to one intra-op thread (matches evaluate's guard) so
# the learned baselines are reliable + deterministic when run outside run_baselines.
try:  # pragma: no cover - torch is a hard dep, but stay defensive
    import torch

    torch.set_num_threads(1)
except ImportError:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
# base.py — emergence_auc
# --------------------------------------------------------------------------- #


def test_emergence_auc_perfectly_separable_is_one() -> None:
    # Positives strictly out-rank negatives -> AUC 1.0.
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert emergence_auc(scores, labels) == pytest.approx(1.0)


def test_emergence_auc_partially_separable_known_value() -> None:
    # One mis-ranked pair: positives {0.2, 0.9}, negatives {0.1, 0.8}.
    # Concordant pairs: 0.2>0.1, 0.9>0.1, 0.9>0.8 (3); discordant: 0.2<0.8 (1).
    # AUC = 3/4.
    scores = np.array([0.1, 0.8, 0.2, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert emergence_auc(scores, labels) == pytest.approx(0.75)


def test_emergence_auc_single_class_is_nan() -> None:
    # ROC-AUC undefined with one class present -> nan guard (no raise).
    assert math.isnan(emergence_auc(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1])))
    assert math.isnan(emergence_auc(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0])))


# --------------------------------------------------------------------------- #
# base.py — share_mape
# --------------------------------------------------------------------------- #


def test_share_mape_matches_prose_formula_with_eps_zero() -> None:
    # All targets positive; eps=0 gives the exact prose mean(|f-t|/t).
    forecast = np.array([1.0, 2.0, 0.0])
    target = np.array([2.0, 4.0, 5.0])
    # |1-2|/2 + |2-4|/4 + |0-5|/5 = 0.5 + 0.5 + 1.0 = 2.0 ; mean = 2/3.
    assert share_mape(forecast, target, eps=0.0) == pytest.approx(2.0 / 3.0)


def test_share_mape_default_eps_floors_denominator() -> None:
    # The default eps=1e-8 only perturbs the denominator slightly; assert it is
    # close to the exact value but acknowledges the floor.
    forecast = np.array([1.0, 2.0, 0.0])
    target = np.array([2.0, 4.0, 5.0])
    assert share_mape(forecast, target) == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_share_mape_only_counts_positive_targets() -> None:
    # The middle row has target 0 and is dropped entirely (not inflated).
    forecast = np.array([1.0, 2.0, 0.0])
    target = np.array([2.0, 0.0, 5.0])
    # Kept rows: |1-2|/2 = 0.5, |0-5|/5 = 1.0 -> mean 0.75.
    assert share_mape(forecast, target, eps=0.0) == pytest.approx(0.75)


def test_share_mape_no_positive_target_is_nan() -> None:
    assert math.isnan(share_mape(np.array([1.0, 2.0]), np.array([0.0, 0.0])))
    # Negative targets are also excluded -> still nan.
    assert math.isnan(share_mape(np.array([1.0, 2.0]), np.array([-1.0, 0.0])))


# --------------------------------------------------------------------------- #
# naive.py — NaiveMovingAverage
# --------------------------------------------------------------------------- #


def _naive_features() -> pd.DataFrame:
    # share_3yr_mean carried forward as the share_forecast; share_growth_3yr drives
    # the logistic emergence score. Last row injects NaN in BOTH consumed columns.
    return pd.DataFrame(
        {
            "share_3yr_mean": [0.20, 0.05, np.nan],
            "share_growth_3yr": [0.5, 2.0, np.nan],
        }
    )


def test_naive_is_a_predictor() -> None:
    assert isinstance(NaiveMovingAverage(), Predictor)
    assert NaiveMovingAverage().name == "naive"


def test_naive_predict_matches_hand_computed_values() -> None:
    gamma = 1.5
    model = NaiveMovingAverage(gamma=gamma)
    # fit is a no-op and returns self (chainable).
    assert model.fit(_naive_features(), pd.DataFrame()) is model
    pred = model.predict(_naive_features())
    assert isinstance(pred, ForecastPrediction)

    # share_forecast == share_3yr_mean (NaN -> 0.0).
    assert pred.share_forecast == pytest.approx([0.20, 0.05, 0.0])

    # emergence_score == expit(share_growth_3yr - gamma) (NaN growth -> expit(-gamma)).
    expected = expit(np.array([0.5, 2.0, 0.0]) - gamma)
    assert pred.emergence_score == pytest.approx(expected)
    # Spot-check the closed forms: expit(-1.0) and expit(0.5).
    assert pred.emergence_score[0] == pytest.approx(expit(-1.0))
    assert pred.emergence_score[1] == pytest.approx(expit(0.5))


def test_naive_outputs_are_nan_safe_and_aligned() -> None:
    pred = NaiveMovingAverage().predict(_naive_features())
    assert len(pred.emergence_score) == 3
    assert len(pred.share_forecast) == 3
    assert np.all(np.isfinite(pred.emergence_score))
    assert np.all(np.isfinite(pred.share_forecast))


# --------------------------------------------------------------------------- #
# arima.py — ArimaPerTopic (fallback nuance)
# --------------------------------------------------------------------------- #


def _arima_labels(n: int) -> pd.DataFrame:
    return pd.DataFrame({"emergent": [0] * n, "forward_share": [0.1] * n})


def _arima_mixed_frame() -> pd.DataFrame:
    """Topic 0 SHORT (2 origin years) + topic 1 LONG (8 origin years).

    For each ``(topic, origin_year=t)`` the series is the observed ``share_last``
    over years ``<= t``. Topic 0 never reaches ``min_series_len=4`` observed years,
    and topic 1's earliest origins also have short series, so several rows fall back.
    """
    rows: list[dict[str, float]] = []

    def add(topic: int, t: int, share_last: float, s3: float, growth: float) -> None:
        rows.append(
            {
                "topic_id": topic,
                "origin_year": t,
                "share_last": share_last,
                "share_3yr_mean": s3,
                "share_growth_3yr": growth,
            }
        )

    # Topic 0: only two origin years -> series length < 4 always -> fallback.
    add(0, 2010, 0.10, 0.10, 1.0)
    add(0, 2011, 0.12, 0.11, 1.1)
    # Topic 1: eight varied origin years.
    vals = [0.05, 0.06, 0.08, 0.07, 0.09, 0.11, 0.10, 0.13]
    for i, y in enumerate(range(2008, 2016)):
        s3 = float(np.mean(vals[max(0, i - 2) : i + 1]))
        add(1, y, vals[i], s3, 1.2)
    return pd.DataFrame(rows)


def test_arima_is_a_predictor() -> None:
    assert isinstance(ArimaPerTopic(), Predictor)
    assert ArimaPerTopic().name == "arima"
    assert ArimaPerTopic().order == (1, 1, 0)


def test_arima_predicts_finite_aligned_output() -> None:
    feats = _arima_mixed_frame()
    model = ArimaPerTopic(order=(1, 1, 0), gamma=1.5, min_series_len=4)
    assert model.fit(feats, _arima_labels(len(feats))) is model
    pred = model.predict(feats)

    assert isinstance(pred, ForecastPrediction)
    # 1:1 alignment with input rows.
    assert len(pred.emergence_score) == len(feats)
    assert len(pred.share_forecast) == len(feats)
    # Guaranteed finite (no NaN/inf).
    assert np.all(np.isfinite(pred.emergence_score))
    assert np.all(np.isfinite(pred.share_forecast))


def test_arima_short_series_forces_naive_fallback() -> None:
    feats = _arima_mixed_frame()
    gamma = 1.5
    model = ArimaPerTopic(order=(1, 1, 0), gamma=gamma, min_series_len=4)
    model.fit(feats, _arima_labels(len(feats)))
    pred = model.predict(feats)

    # Short / early-origin rows force the fallback; diagnostics expose it.
    assert model.n_total_ == len(feats)
    assert model.n_fallback_ > 0

    # The fallback rows are byte-identical to NaiveMovingAverage(gamma) on the SAME
    # rows. Identify them as the rows where ARIMA output equals the naive output.
    naive = NaiveMovingAverage(gamma=gamma).predict(feats)
    fell_back = np.isclose(pred.emergence_score, naive.emergence_score) & np.isclose(
        pred.share_forecast, naive.share_forecast
    )
    # At least the two topic-0 rows (always short) must have fallen back.
    assert fell_back[:2].all()
    # Number of equal-to-naive rows is at least the reported fallback count.
    assert int(fell_back.sum()) >= model.n_fallback_


def test_arima_constant_long_series_is_finite_without_raising() -> None:
    # A degenerate/constant LONG series does NOT raise; per the impl author it
    # converges (se≈0) and uses the hard indicator rather than throwing. We assert
    # only that it produces finite output and never crashes.
    rows = [
        {
            "topic_id": 2,
            "origin_year": y,
            "share_last": 0.20,
            "share_3yr_mean": 0.20,
            "share_growth_3yr": 1.0,
        }
        for y in range(2000, 2012)  # 12 constant origin years
    ]
    feats = pd.DataFrame(rows)
    model = ArimaPerTopic(order=(1, 1, 0), gamma=1.5, min_series_len=4)
    model.fit(feats, _arima_labels(len(feats)))
    pred = model.predict(feats)  # must not raise

    assert model.n_total_ == len(feats)
    assert np.all(np.isfinite(pred.emergence_score))
    assert np.all(np.isfinite(pred.share_forecast))


# --------------------------------------------------------------------------- #
# mlp.py / no_graph.py — torch-CPU learned baselines
# --------------------------------------------------------------------------- #


def _learned_frame(n: int = 22, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tiny frame with all NODE_FEATURES, NaN in block-B cols, mixed labels."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "topic_id": np.arange(n) % 5,
        "origin_year": 2000 + (np.arange(n) // 5),
    }
    for col in NODE_FEATURES:
        data[col] = rng.normal(size=n)
    # Inject NaN into some block-B columns to exercise impute-on-train.
    for col in ("sem_nov_mean_3yr", "cd5_3yr", "cited_by_pctile_3yr"):
        data[col] = data[col].copy()
        data[col][:3] = np.nan
    # share_3yr_mean must be present (and some forward_share > 0 below).
    data["share_3yr_mean"] = np.abs(rng.normal(size=n)) + 0.01
    features = pd.DataFrame(data)
    labels = pd.DataFrame(
        {
            "emergent": (rng.random(n) > 0.5).astype(int),  # mixed classes
            "forward_share": np.abs(rng.normal(size=n)),  # some > 0
        }
    )
    return features, labels


@pytest.mark.parametrize(
    ("cls", "feature_columns", "expected_n_features"),
    [
        (MlpForecaster, tuple(MLP_FEATURES), 9),
        (NoGraphForecaster, tuple(NODE_FEATURES), 16),
    ],
)
def test_learned_baseline_fit_predict_contract(
    cls: type, feature_columns: tuple[str, ...], expected_n_features: int
) -> None:
    features, labels = _learned_frame()
    model = cls(
        feature_columns=feature_columns,
        hidden=4,
        epochs=2,
        lr=0.01,
        batch_size=256,
        seed=1729,
    )
    assert isinstance(model, Predictor)
    assert isinstance(model, _TorchForecaster)
    assert len(model.feature_columns) == expected_n_features

    out = model.fit(features, labels).predict(features)
    assert isinstance(out, ForecastPrediction)

    # Length == n rows.
    assert len(out.emergence_score) == len(features)
    assert len(out.share_forecast) == len(features)
    # All finite despite injected NaN (impute on TRAIN works).
    assert np.all(np.isfinite(out.emergence_score))
    assert np.all(np.isfinite(out.share_forecast))
    # emergence_score is a sigmoid output in [0, 1]; share_forecast >= 0.
    assert out.emergence_score.min() >= 0.0
    assert out.emergence_score.max() <= 1.0
    assert out.share_forecast.min() >= 0.0


@pytest.mark.parametrize(
    ("cls", "feature_columns"),
    [
        (MlpForecaster, tuple(MLP_FEATURES)),
        (NoGraphForecaster, tuple(NODE_FEATURES)),
    ],
)
def test_learned_baseline_is_deterministic(cls: type, feature_columns: tuple[str, ...]) -> None:
    features, labels = _learned_frame()

    def run() -> ForecastPrediction:
        model = cls(feature_columns=feature_columns, hidden=4, epochs=2, seed=1729)
        assert isinstance(model, _TorchForecaster)
        return model.fit(features, labels).predict(features)

    a = run()
    b = run()
    # Same seed + same inputs => bit-identical arrays on CPU.
    assert np.array_equal(a.emergence_score, b.emergence_score)
    assert np.array_equal(a.share_forecast, b.share_forecast)


def test_mlp_and_no_graph_differ_in_feature_columns() -> None:
    # MlpForecaster uses 9 (block A); NoGraphForecaster uses all 16 node features.
    mlp = MlpForecaster()
    no_graph = NoGraphForecaster()
    assert len(mlp.feature_columns) == 9
    assert len(no_graph.feature_columns) == 16
    assert tuple(mlp.feature_columns) == tuple(MLP_FEATURES)
    assert tuple(no_graph.feature_columns) == tuple(NODE_FEATURES)
    assert mlp.feature_columns != no_graph.feature_columns


# --------------------------------------------------------------------------- #
# evaluate.py — run_baselines / per_topic_scores plumbing
# --------------------------------------------------------------------------- #

_BASELINE_NAMES = ("naive", "arima", "mlp", "no_graph")
_METRIC_COLUMNS = (
    "baseline",
    "emergence_auc",
    "share_mape",
    "n_train",
    "n_val",
    "n_val_pos",
    "fallback_frac",
)


def _materialized_frame() -> pd.DataFrame:
    """Synthetic materialized frame: train + val rows, both emergent classes in val.

    Carries ``topic_id, origin_year, split`` + all NODE_FEATURES + the label cols
    (``emergent, forward_share, volume_ok, label_complete``). ``share_3yr_mean`` and
    ``share_growth_3yr`` are part of NODE_FEATURES already; ``share_last`` is too.
    All rows here are ``volume_ok=True & label_complete=True`` (the labeled slice).
    """
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []

    def add(split: str, topic: int, year: int, emergent: int) -> None:
        row: dict[str, object] = {
            "topic_id": topic,
            "origin_year": year,
            "split": split,
            "emergent": emergent,
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

    # Train: 4 topics across 2010..2016 (gives ARIMA enough per-topic history).
    for tid in range(4):
        for y in range(2010, 2017):
            add("train", tid, y, int(rng.random() > 0.5))
    # Val: 4 topics across 2018..2019 with BOTH classes present.
    classes = [1, 0, 1, 0, 1, 0, 1, 0]
    i = 0
    for tid in range(4):
        for y in (2018, 2019):
            add("val", tid, y, classes[i])
            i += 1
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


def test_run_baselines_metrics_table_contract() -> None:
    frame = _materialized_frame()
    metrics = run_baselines(frame, _eval_cfg())

    # EXACT columns, in order.
    assert list(metrics.columns) == list(_METRIC_COLUMNS)
    # Exactly the 4 baselines, in the locked order.
    assert list(metrics["baseline"]) == list(_BASELINE_NAMES)
    assert len(metrics) == 4

    n_train_expected = int((frame["split"] == "train").sum())
    n_val_expected = int((frame["split"] == "val").sum())
    n_val_pos_expected = int(((frame["split"] == "val") & (frame["emergent"] == 1)).sum())
    # n_train / n_val match the labeled slice sizes (all rows are labeled here).
    assert (metrics["n_train"] == n_train_expected).all()
    assert (metrics["n_val"] == n_val_expected).all()
    assert (metrics["n_val_pos"] == n_val_pos_expected).all()


def test_run_baselines_fallback_frac_nan_except_arima() -> None:
    metrics = run_baselines(_materialized_frame(), _eval_cfg()).set_index("baseline")

    # fallback_frac is NaN for the three non-arima baselines.
    for name in ("naive", "mlp", "no_graph"):
        assert math.isnan(metrics.loc[name, "fallback_frac"])
    # ...and a (finite) float for arima.
    arima_ff = metrics.loc["arima", "fallback_frac"]
    assert isinstance(float(arima_ff), float)
    assert not math.isnan(arima_ff)
    assert 0.0 <= arima_ff <= 1.0


def test_run_baselines_auc_defined_on_both_class_val() -> None:
    # Val carries both emergent classes, so AUC is a real number (not nan) for the
    # deterministic baselines; mape is finite (all forward_share > 0).
    metrics = run_baselines(_materialized_frame(), _eval_cfg()).set_index("baseline")
    for name in _BASELINE_NAMES:
        assert not math.isnan(metrics.loc[name, "emergence_auc"])
        assert not math.isnan(metrics.loc[name, "share_mape"])


def test_per_topic_scores_staging_contract() -> None:
    frame = _materialized_frame()
    scores = per_topic_scores(frame, _eval_cfg())

    # Dict keyed by exactly the 4 baseline names.
    assert set(scores.keys()) == set(_BASELINE_NAMES)

    n_val = int((frame["split"] == "val").sum())
    required = (
        "topic_id",
        "origin_year",
        "emergence_score",
        "emergent",
        "share_forecast",
        "forward_share",
    )
    for name in _BASELINE_NAMES:
        arrays = scores[name]
        for key in required:
            assert key in arrays, f"{name} missing {key}"
            assert len(arrays[key]) == n_val, f"{name}.{key} length"
        # Staged arrays must be finite where they are model outputs.
        assert np.all(np.isfinite(arrays["emergence_score"]))
        assert np.all(np.isfinite(arrays["share_forecast"]))
