"""V1-S12 baseline evaluation harness — train, score, and stage diagnostics.

This module is the thin orchestration layer over the four baselines
(:class:`~scifield.forecasting.baselines.naive.NaiveMovingAverage`,
:class:`~scifield.forecasting.baselines.arima.ArimaPerTopic`,
:class:`~scifield.forecasting.baselines.mlp.MlpForecaster`,
:class:`~scifield.forecasting.baselines.no_graph.NoGraphForecaster`). It consumes
the materialized ``(topic_id, origin_year, NODE_FEATURES..., label cols, split)``
frame produced by :func:`scifield.forecasting.data.materialize`, fits every
baseline on the TRAIN labeled slice, scores it on the VAL labeled slice, and
emits two things:

* :func:`run_baselines` — a tidy metrics table (one row per baseline) with the
  validation emergence-AUC + share-MAPE and run diagnostics; and
* :func:`per_topic_scores` — the per-validation-row score arrays staged for the
  V1-S14 Wilcoxon signed-rank comparison (the test is **not** run here).

Slice definitions (LOCKED, contract §7)
---------------------------------------
* TRAIN slice = ``split == "train" & volume_ok & label_complete``.
* VAL slice   = ``split == "val"   & volume_ok & label_complete``.

Both metrics flow through the pre-registered guards in
:mod:`scifield.forecasting.baselines.base`: :func:`emergence_auc` returns ``nan``
on a single-class validation slice and :func:`share_mape` is computed only over
rows with a strictly positive ``forward_share`` target.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from scifield.forecasting.baselines.base import emergence_auc, share_mape

__all__ = ["per_topic_scores", "run_baselines"]

#: LOCKED — the metrics-table columns, one row per baseline, in this order.
_METRIC_COLUMNS = (
    "baseline",
    "emergence_auc",
    "share_mape",
    "n_train",
    "n_val",
    "n_val_pos",
    "fallback_frac",
)


def _pin_torch_single_threaded() -> None:
    """Force torch onto a single intra-op thread before any MLP fit.

    The torch-CPU baselines (``mlp`` / ``no_graph``) segfault during training on
    this macOS box with the default multi-threaded intra-op pool (a known
    libomp/torch threading race on Darwin). Pinning torch to one thread makes the
    learned baselines run reliably and keeps them deterministic; it also belongs
    here rather than in the (locked) baseline modules so every entry point
    (:func:`run_baselines`, :func:`per_topic_scores`, the CLI, the notebook, the
    tests) gets the same guard. Idempotent; a no-op if torch is unavailable.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep, but stay safe
        return
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)


def _labeled_slice(features: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return the labeled rows for one split: ``split & volume_ok & label_complete``.

    Parameters
    ----------
    features:
        The materialized frame (carries ``split``, ``volume_ok``,
        ``label_complete``).
    split_name:
        ``"train"`` or ``"val"``.

    Returns
    -------
    pandas.DataFrame
        The masked rows with a fresh ``RangeIndex`` (so the predictor outputs,
        which align 1:1 with the input row ORDER, line up with positional
        indexing).
    """
    mask = (
        (features["split"] == split_name)
        & features["volume_ok"].astype(bool)
        & features["label_complete"].astype(bool)
    )
    return features.loc[mask].reset_index(drop=True)


def _build_predictors(cfg: Any) -> list[Any]:
    """Construct the four baselines from ``cfg`` (lazy-imported per the contract).

    Hyperparameters come straight from ``cfg.baselines.*`` and the feature
    columns from ``cfg.features.{mlp_columns,node_columns}``. The predictor
    classes are imported inside the function body so importing this module never
    drags in torch / statsmodels.

    Parameters
    ----------
    cfg:
        The forecasting ``DictConfig`` (``conf/forecasting/v1.yaml``).

    Returns
    -------
    list
        ``[NaiveMovingAverage, ArimaPerTopic, MlpForecaster, NoGraphForecaster]``
        in that order (the row order of the metrics table).
    """
    from scifield.forecasting.baselines.arima import ArimaPerTopic
    from scifield.forecasting.baselines.mlp import MlpForecaster
    from scifield.forecasting.baselines.naive import NaiveMovingAverage
    from scifield.forecasting.baselines.no_graph import NoGraphForecaster

    _pin_torch_single_threaded()

    naive = NaiveMovingAverage(gamma=float(cfg.label.gamma))
    arima = ArimaPerTopic(
        order=cast("tuple[int, int, int]", tuple(int(x) for x in cfg.baselines.arima.order)),
        gamma=float(cfg.label.gamma),
    )
    mlp = MlpForecaster(
        feature_columns=tuple(str(c) for c in cfg.features.mlp_columns),
        hidden=int(cfg.baselines.mlp.hidden),
        epochs=int(cfg.baselines.mlp.epochs),
        lr=float(cfg.baselines.mlp.lr),
        batch_size=int(cfg.baselines.mlp.batch_size),
        seed=int(cfg.baselines.mlp.seed),
    )
    no_graph = NoGraphForecaster(
        feature_columns=tuple(str(c) for c in cfg.features.node_columns),
        hidden=int(cfg.baselines.no_graph.hidden),
        epochs=int(cfg.baselines.no_graph.epochs),
        lr=float(cfg.baselines.no_graph.lr),
        batch_size=int(cfg.baselines.no_graph.batch_size),
        seed=int(cfg.baselines.no_graph.seed),
    )
    return [naive, arima, mlp, no_graph]


def run_baselines(features: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Fit + score the four baselines on the temporal split; return the metrics.

    Each baseline is fit on the TRAIN labeled slice
    (``split=="train" & volume_ok & label_complete``) and evaluated on the VAL
    labeled slice (``split=="val" & ...``). The FULL validation feature frame —
    including ``share_3yr_mean`` and ``topic_id`` / ``origin_year`` / ``share_last``
    — is passed to ``predict`` so :class:`ArimaPerTopic` can reassemble its
    per-topic series. ``fallback_frac`` is read off the fitted
    :class:`ArimaPerTopic` AFTER ``predict`` (``n_fallback_ / n_total_``) and is
    ``NaN`` for the other three baselines.

    Parameters
    ----------
    features:
        The materialized ``(topic_id, origin_year, NODE_FEATURES..., label cols,
        split)`` frame from :func:`scifield.forecasting.data.materialize`.
    cfg:
        The forecasting ``DictConfig``.

    Returns
    -------
    pandas.DataFrame
        One row per baseline with the LOCKED columns ``baseline,
        emergence_auc, share_mape, n_train, n_val, n_val_pos, fallback_frac``
        (in that order).
    """
    train = _labeled_slice(features, "train")
    val = _labeled_slice(features, "val")

    n_train = int(len(train))
    n_val = int(len(val))

    val_emergent = val["emergent"].to_numpy() if n_val else np.empty(0)
    val_forward = val["forward_share"].to_numpy() if n_val else np.empty(0)
    n_val_pos = int((val_emergent == 1).sum()) if n_val else 0

    train_labels = train[["emergent", "forward_share"]]

    rows: list[dict[str, Any]] = []
    for predictor in _build_predictors(cfg):
        predictor.fit(train, train_labels)
        pred = predictor.predict(val)

        auc = emergence_auc(pred.emergence_score, val_emergent)
        mape = share_mape(pred.share_forecast, val_forward)

        # fallback_frac is only meaningful for arima (read AFTER predict()).
        fallback_frac = float("nan")
        n_total = getattr(predictor, "n_total_", 0)
        if predictor.name == "arima" and n_total:
            fallback_frac = float(predictor.n_fallback_) / float(n_total)

        rows.append(
            {
                "baseline": str(predictor.name),
                "emergence_auc": float(auc),
                "share_mape": float(mape),
                "n_train": n_train,
                "n_val": n_val,
                "n_val_pos": n_val_pos,
                "fallback_frac": fallback_frac,
            }
        )

    return pd.DataFrame(rows, columns=list(_METRIC_COLUMNS))


def per_topic_scores(features: pd.DataFrame, cfg: Any) -> dict:
    """Stage per-validation-row score arrays for the V1-S14 Wilcoxon test.

    Fits each baseline on the TRAIN labeled slice and scores the VAL labeled
    slice, returning the raw arrays (no Wilcoxon — that is V1-S14). The keys
    ``topic_id`` / ``origin_year`` carry the row identity so a later session can
    pair baselines on the same ``(topic, origin_year)`` samples.

    Parameters
    ----------
    features:
        The materialized frame (see :func:`run_baselines`).
    cfg:
        The forecasting ``DictConfig``.

    Returns
    -------
    dict
        Maps each baseline ``name`` to a dict of ``np.ndarray`` over the VAL
        labeled slice: at minimum ``topic_id, origin_year, emergence_score,
        emergent, share_forecast, forward_share`` (all the same length and
        order).
    """
    val = _labeled_slice(features, "val")
    train = _labeled_slice(features, "train")
    train_labels = train[["emergent", "forward_share"]]

    topic_id = val["topic_id"].to_numpy() if len(val) else np.empty(0, dtype=np.int64)
    origin_year = val["origin_year"].to_numpy() if len(val) else np.empty(0, dtype=np.int64)
    emergent = val["emergent"].to_numpy() if len(val) else np.empty(0)
    forward_share = val["forward_share"].to_numpy() if len(val) else np.empty(0)

    out: dict[str, dict[str, np.ndarray]] = {}
    for predictor in _build_predictors(cfg):
        predictor.fit(train, train_labels)
        pred = predictor.predict(val)
        out[str(predictor.name)] = {
            "topic_id": np.asarray(topic_id),
            "origin_year": np.asarray(origin_year),
            "emergence_score": np.asarray(pred.emergence_score, dtype=float),
            "emergent": np.asarray(emergent),
            "share_forecast": np.asarray(pred.share_forecast, dtype=float),
            "forward_share": np.asarray(forward_share, dtype=float),
        }
    return out
