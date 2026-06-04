"""V1-S12 naive moving-average baseline — the trivial forecasting floor.

:class:`NaiveMovingAverage` is the cheapest predictor that still emits the two
outputs the :class:`~scifield.forecasting.baselines.base.Predictor` contract
demands, with NO learned parameters:

``share_forecast``
    The topic's trailing 3-year mean share (``share_3yr_mean``) carried forward
    unchanged — a persistence forecast of ``forward_3yr_mean_share``.

``emergence_score``
    A logistic squash of the trailing share growth ratio, centred on the
    emergence threshold: ``expit(share_growth_3yr - gamma)``. Because
    :func:`~scifield.forecasting.baselines.base.emergence_auc` is rank-based,
    ANY monotonic function of ``share_growth_3yr`` yields the same AUC; the
    ``- gamma`` shift and the logistic squash only make the raw score a
    loosely-calibrated "probability-like" number, they do not change ranking.

This baseline learns nothing — :meth:`fit` is a no-op — so it is also the
fallback every :class:`~scifield.forecasting.baselines.arima.ArimaPerTopic` row
drops to when its series is too short or its fit fails.

NaN handling
------------
The two consumed feature columns can be NaN (e.g. a topic-year with no prior
window). Both are coerced through :func:`numpy.nan_to_num` before use, so the
returned arrays are guaranteed finite: a NaN ``share_3yr_mean`` forecasts ``0``
share, and a NaN ``share_growth_3yr`` maps to ``expit(0 - gamma)`` rather than
NaN. Scores are never NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from scifield.forecasting.baselines.base import ForecastPrediction

__all__ = ["NaiveMovingAverage"]


class NaiveMovingAverage:
    """Persistence forecast + logistic-growth emergence score (no parameters).

    Conforms to :class:`~scifield.forecasting.baselines.base.Predictor`.

    Parameters
    ----------
    gamma:
        Emergence threshold the growth ratio is centred on in the logistic
        score (``expit(share_growth_3yr - gamma)``). Default ``1.5`` (matches
        ``cfg.label.gamma``). Rank-irrelevant for AUC; affects only the score's
        calibration offset.
    """

    name: str = "naive"

    def __init__(self, gamma: float = 1.5) -> None:
        self.gamma = float(gamma)

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> NaiveMovingAverage:
        """No-op: the naive baseline learns nothing. Returns ``self``.

        Parameters
        ----------
        features, labels:
            Accepted for interface conformance and ignored.

        Returns
        -------
        NaiveMovingAverage
            ``self``.
        """
        return self

    def predict(self, features: pd.DataFrame) -> ForecastPrediction:
        """Forecast trailing-mean share + logistic-growth emergence score.

        Parameters
        ----------
        features:
            Feature rows; must carry ``share_3yr_mean`` and ``share_growth_3yr``
            (both NaN-tolerated).

        Returns
        -------
        ForecastPrediction
            ``emergence_score`` and ``share_forecast`` arrays, each aligned 1:1
            with ``features`` rows and guaranteed finite (no NaN).
        """
        share_forecast = np.nan_to_num(features["share_3yr_mean"].to_numpy(dtype=float), nan=0.0)
        growth = np.nan_to_num(features["share_growth_3yr"].to_numpy(dtype=float), nan=0.0)
        emergence_score = expit(growth - self.gamma)
        return ForecastPrediction(
            emergence_score=np.asarray(emergence_score, dtype=float),
            share_forecast=np.asarray(share_forecast, dtype=float),
        )
