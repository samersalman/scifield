"""V1-S12 per-topic ARIMA baseline — a classical time-series floor.

:class:`ArimaPerTopic` fits an independent ARIMA model to each topic's yearly
leaf-share series and forecasts the next ``horizon`` (3) years, with a NAIVE
fallback wherever ARIMA is inapplicable or fails.

The series
----------
For a row ``(topic g, origin_year t)`` the share series is the LOCKED
``share_last`` value (= share at year ``t``) for every year ``<= t`` the topic
is observed. It is assembled from two leakage-safe sources and deduped by year:

* the training rows seen in :meth:`fit` (stored as ``{origin_year: share_last}``
  per topic), and
* the rows of the *current* ``features`` frame for the same topic with
  ``origin_year <= t``.

Only years ``<= t`` ever enter the series, so a forecast made at origin ``t``
never sees a future share — the same no-leakage discipline
:mod:`scifield.forecasting.data` enforces at the feature level.

Forecast → outputs
------------------
ARIMA's 3-step forecast gives a predictive mean per step and a predictive
standard error per step. We collapse them to:

``share_forecast``
    The mean of the 3 forecast means — a point prediction of
    ``forward_3yr_mean_share``.

``emergence_score``
    ``P(mean forecast >= gamma * trailing_mean)`` under a normal predictive
    distribution: ``norm.cdf((mean_fc - gamma * trailing_mean) / se)`` with
    ``se`` the mean forecast standard error. When ``se <= 0`` (degenerate
    predictive variance) the probability collapses to the hard indicator
    ``float(mean_fc >= gamma * trailing_mean)``. ``trailing_mean`` is the row's
    ``share_3yr_mean`` when present, else the mean of the last 3 series points.

Naive fallback (never crash the run)
------------------------------------
A row falls back to :class:`~scifield.forecasting.baselines.naive.NaiveMovingAverage`
(applied to that single row) when its series is shorter than ``min_series_len``,
OR when fitting/forecasting raises ANY exception, OR when the result is
non-finite. Every fallback increments ``self.n_fallback_``; ``self.n_total_`` is
the total number of scored rows, so ``evaluate.py`` can report
``fallback_frac = n_fallback_ / n_total_``. A pathological topic degrades to the
naive floor for its rows; it never raises.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA

from scifield.forecasting.baselines.base import ForecastPrediction
from scifield.forecasting.baselines.naive import NaiveMovingAverage

__all__ = ["ArimaPerTopic"]


class ArimaPerTopic:
    """Per-topic ARIMA share forecaster with a naive fallback.

    Conforms to :class:`~scifield.forecasting.baselines.base.Predictor`.

    Parameters
    ----------
    order:
        ARIMA ``(p, d, q)`` order applied to every topic. Default ``(1, 1, 0)``
        (matches ``cfg.baselines.arima.order``).
    gamma:
        Emergence threshold used in the predictive-CDF emergence score and in
        the naive fallback. Default ``1.5`` (matches ``cfg.label.gamma``).
    min_series_len:
        Minimum number of observed years required to attempt an ARIMA fit.
        Series shorter than this fall back to naive without trying ARIMA.
        Default ``4``.

    Attributes
    ----------
    n_fallback_:
        Number of rows in the most recent :meth:`predict` call that used the
        naive fallback (set after :meth:`predict`).
    n_total_:
        Total number of rows scored in the most recent :meth:`predict` call.
        ``fallback_frac = n_fallback_ / n_total_``.
    """

    name: str = "arima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 0),
        gamma: float = 1.5,
        min_series_len: int = 4,
    ) -> None:
        self.order = tuple(int(x) for x in order)
        self.gamma = float(gamma)
        self.min_series_len = int(min_series_len)
        # Per-topic training share series as {origin_year: share_last}.
        self._train_series: dict[int, dict[int, float]] = {}
        # Naive fallback (shares gamma) for short / failed / non-finite rows.
        self._naive = NaiveMovingAverage(gamma=self.gamma)
        self.n_fallback_: int = 0
        self.n_total_: int = 0

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> ArimaPerTopic:
        """Store each topic's training yearly share series; prep the fallback.

        The yearly share series is ``share_last`` (share at year ``t``) keyed by
        ``origin_year``, since ``share_last`` equals the share at the row's
        origin year. No ARIMA is fit here — fitting happens lazily per row in
        :meth:`predict`, once the full ``<= t`` series is assembled.

        Parameters
        ----------
        features:
            Training feature rows; must carry ``topic_id``, ``origin_year`` and
            ``share_last``.
        labels:
            Accepted for interface conformance (the naive fallback ignores it).

        Returns
        -------
        ArimaPerTopic
            ``self``.
        """
        self._train_series = {}
        if {"topic_id", "origin_year", "share_last"}.issubset(features.columns):
            sub = features[["topic_id", "origin_year", "share_last"]]
            for topic_id, grp in sub.groupby("topic_id", sort=False):
                series = {
                    int(y): float(s)
                    for y, s in zip(
                        grp["origin_year"].to_numpy(),
                        grp["share_last"].to_numpy(),
                        strict=True,
                    )
                    if np.isfinite(s)
                }
                self._train_series[int(topic_id)] = series
        # Fit the (stateless) naive fallback for interface symmetry.
        self._naive.fit(features, labels)
        self.n_fallback_ = 0
        self.n_total_ = 0
        return self

    def predict(self, features: pd.DataFrame) -> ForecastPrediction:
        """Per-row 3-step ARIMA forecast with naive fallback; set diagnostics.

        Parameters
        ----------
        features:
            Feature rows to score; must carry ``topic_id``, ``origin_year`` and
            ``share_last`` (and ``share_3yr_mean`` / ``share_growth_3yr`` for the
            trailing mean and naive fallback).

        Returns
        -------
        ForecastPrediction
            ``emergence_score`` and ``share_forecast`` arrays aligned 1:1 with
            ``features`` rows, guaranteed finite. Sets ``self.n_fallback_`` and
            ``self.n_total_``.
        """
        n = len(features)
        emergence = np.empty(n, dtype=float)
        share_fc = np.empty(n, dtype=float)

        # Pre-index the CURRENT frame's share_last by topic for the union series.
        cur_by_topic: dict[int, dict[int, float]] = {}
        if {"topic_id", "origin_year", "share_last"}.issubset(features.columns):
            for topic_id, grp in features.groupby("topic_id", sort=False):
                cur_by_topic[int(topic_id)] = {
                    int(y): float(s)
                    for y, s in zip(
                        grp["origin_year"].to_numpy(),
                        grp["share_last"].to_numpy(),
                        strict=True,
                    )
                    if np.isfinite(s)
                }

        n_fallback = 0
        # Precompute the naive fallback over the whole frame (cheap, vectorized);
        # individual fallback rows copy from it so the math is byte-identical.
        naive_pred = self._naive.predict(features)

        topic_arr = (
            features["topic_id"].to_numpy() if "topic_id" in features.columns else np.full(n, -1)
        )
        year_arr = (
            features["origin_year"].to_numpy() if "origin_year" in features.columns else np.zeros(n)
        )
        trailing_arr = (
            features["share_3yr_mean"].to_numpy(dtype=float)
            if "share_3yr_mean" in features.columns
            else np.full(n, np.nan)
        )

        for i in range(n):
            g = int(topic_arr[i])
            t = int(year_arr[i])

            series = self._build_series(g, t, cur_by_topic)

            use_fallback = len(series) < self.min_series_len
            if not use_fallback:
                trailing_mean = trailing_arr[i]
                if not np.isfinite(trailing_mean):
                    trailing_mean = float(np.mean(series[-3:]))
                fc = self._forecast_row(series, trailing_mean)
                if fc is None:
                    use_fallback = True
                else:
                    e_score, s_fore = fc
                    emergence[i] = e_score
                    share_fc[i] = s_fore

            if use_fallback:
                emergence[i] = float(naive_pred.emergence_score[i])
                share_fc[i] = float(naive_pred.share_forecast[i])
                n_fallback += 1

        self.n_total_ = int(n)
        self.n_fallback_ = int(n_fallback)
        return ForecastPrediction(
            emergence_score=np.asarray(emergence, dtype=float),
            share_forecast=np.asarray(share_fc, dtype=float),
        )

    def _build_series(
        self, topic_id: int, t: int, cur_by_topic: dict[int, dict[int, float]]
    ) -> np.ndarray:
        """Assemble the leakage-safe ``share_last`` series for years ``<= t``.

        Unions the stored training series for ``topic_id`` with the current
        frame's rows for that topic, keeps only years ``<= t``, dedups by year
        (current frame wins on collisions), and sorts ascending.
        """
        merged: dict[int, float] = {}
        for year, share in self._train_series.get(topic_id, {}).items():
            if year <= t:
                merged[year] = share
        for year, share in cur_by_topic.get(topic_id, {}).items():
            if year <= t:
                merged[year] = share
        if not merged:
            return np.empty(0, dtype=float)
        years_sorted = sorted(merged)
        return np.array([merged[y] for y in years_sorted], dtype=float)

    def _forecast_row(self, series: np.ndarray, trailing_mean: float) -> tuple[float, float] | None:
        """Fit ARIMA on ``series`` and return ``(emergence_score, share_forecast)``.

        Returns ``None`` (signalling a naive fallback) on ANY exception or any
        non-finite intermediate / result.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(series, order=self.order).fit()
                fc = res.get_forecast(3)
                mean_fc = float(np.asarray(fc.predicted_mean).mean())
                se = float(np.mean(np.asarray(fc.se_mean)))
        except Exception:
            return None

        if not np.isfinite(mean_fc) or not np.isfinite(se):
            return None

        threshold = self.gamma * float(trailing_mean)
        if se > 0:
            emergence_score = float(norm.cdf((mean_fc - threshold) / se))
        else:
            emergence_score = float(mean_fc >= threshold)

        if not np.isfinite(emergence_score) or not np.isfinite(mean_fc):
            return None
        return emergence_score, mean_fc
