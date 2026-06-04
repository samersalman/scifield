"""V1-S12 forecasting baseline interface + shared scoring metrics.

This module pins the common contract every V1-S12 baseline (and the future
V1-S13 GNN) implements, so :mod:`scifield.forecasting.baselines.evaluate` can
treat them interchangeably. A baseline is a :class:`Predictor`: it learns from
a training slice and, for any feature frame, emits BOTH

``emergence_score``
    A higher-is-more-emergent score used only for ROC-AUC. The scale is
    arbitrary because AUC is rank-based; any monotonic transform is equivalent.

``share_forecast``
    A point prediction of ``forward_3yr_mean_share`` in the original share
    space, scored by :func:`share_mape`.

Consumption convention (uniform across all baselines)
-----------------------------------------------------
Baselines read their supervision from two LOCKED label columns produced by
:func:`scifield.forecasting.data.build_labels`:

- the binary emergence label is ``labels["emergent"]`` (int 0/1);
- the share regression target is ``labels["forward_share"]`` (== the
  ``forward_3yr_mean_share`` mean-share-over-``[t+1..t+3]`` target).

:meth:`Predictor.predict` returns a :class:`ForecastPrediction` whose two
arrays are aligned **1:1 with the input feature rows** — same length, same
order — so callers can concatenate them with the input frame's keys without
re-joining.

Both metrics carry pre-registered guards so a degenerate validation slice
returns ``nan`` rather than raising: :func:`emergence_auc` is ``nan`` when the
labels are single-class, and :func:`share_mape` is computed only over rows
where the target is strictly positive (``nan`` if there are none).
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

__all__ = ["ForecastPrediction", "Predictor", "emergence_auc", "share_mape"]


class ForecastPrediction(NamedTuple):
    """A baseline's twin outputs for one feature frame.

    Attributes
    ----------
    emergence_score:
        Float array, one entry per input row; higher means more likely
        emergent. Any monotonic scale is fine — :func:`emergence_auc` is
        rank-based.
    share_forecast:
        Float array, one entry per input row; the predicted
        ``forward_3yr_mean_share`` in the original share space (the MAPE
        target space, scored by :func:`share_mape`).
    """

    emergence_score: np.ndarray
    share_forecast: np.ndarray


@runtime_checkable
class Predictor(Protocol):
    """Common interface for every V1-S12 baseline and the V1-S13 GNN.

    Implementations expose a stable ``name`` (used as the row key in the
    metrics table), learn from a training slice via :meth:`fit`, and produce
    aligned emergence + share predictions via :meth:`predict`.

    The supervision columns are fixed by convention: :meth:`fit` reads the
    binary emergence label from ``labels["emergent"]`` and the share target
    from ``labels["forward_share"]``. :meth:`predict` returns a
    :class:`ForecastPrediction` whose arrays are aligned 1:1 with the input
    ``features`` rows (same length and order).
    """

    name: str

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> Predictor:
        """Learn from a training slice and return ``self`` for chaining.

        Parameters
        ----------
        features:
            Training feature rows (one row per ``(topic_id, origin_year)``).
        labels:
            Aligned label frame; the emergence label is read from
            ``labels["emergent"]`` and the share target from
            ``labels["forward_share"]``.

        Returns
        -------
        Predictor
            ``self``.
        """
        ...

    def predict(self, features: pd.DataFrame) -> ForecastPrediction:
        """Score ``features``, returning arrays aligned 1:1 with its rows.

        Parameters
        ----------
        features:
            Feature rows to score.

        Returns
        -------
        ForecastPrediction
            ``emergence_score`` and ``share_forecast`` arrays, each the same
            length and order as ``features``.
        """
        ...


def emergence_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC of emergence ``scores`` against binary ``labels``.

    Scored with :func:`sklearn.metrics.roc_auc_score`, which is rank-based, so
    the absolute scale of ``scores`` does not matter. When ``labels`` contain
    only a single class (all 0 or all 1), ROC-AUC is undefined; per the
    pre-registered guard this returns ``float("nan")`` instead of raising.

    Parameters
    ----------
    scores:
        Predicted emergence scores; higher means more likely emergent.
    labels:
        Binary ground-truth labels (the ``emergent`` column), aligned to
        ``scores``.

    Returns
    -------
    float
        The ROC-AUC, or ``float("nan")`` if ``labels`` are single-class.
    """
    labels = np.asarray(labels)
    if np.unique(labels).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores)))


def share_mape(forecast: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Mean absolute percentage error over rows with a positive target.

    Per the pre-registered guard, MAPE is computed ONLY where ``target > 0``::

        mean(|forecast - target| / target)   over rows with target > 0

    Rows with a zero (or non-positive) target are excluded entirely rather than
    inflating the error through a near-zero denominator. If no row has a
    positive target, the metric is undefined and ``float("nan")`` is returned.

    Parameters
    ----------
    forecast:
        Predicted ``forward_3yr_mean_share`` values.
    target:
        Ground-truth ``forward_share`` values, aligned to ``forecast``.
    eps:
        Small floor added to the denominator for numerical safety on the kept
        (strictly positive) rows. Default ``1e-8``.

    Returns
    -------
    float
        The mean absolute percentage error over positive-target rows, or
        ``float("nan")`` if there are no such rows.
    """
    forecast = np.asarray(forecast, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = target > 0
    if not np.any(mask):
        return float("nan")
    f = forecast[mask]
    t = target[mask]
    return float(np.mean(np.abs(f - t) / (t + eps)))
