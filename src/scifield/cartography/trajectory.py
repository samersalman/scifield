"""V2 cartography — the per-topic **trajectory / forecast** layer.

This is the V2-S10 "future" layer on top of the V2 leaf-topic cartography. Where
:mod:`scifield.cartography.cascade` looks *backward* (where a topic originated, how it
diffused across the journal panel), this module looks *forward*: for each leaf topic it
fits the recent annual time series and projects a short ``horizon`` (default +5 years,
2026→2030) with uncertainty bands.

Two quantities, both projected
------------------------------
* **share** (headline) — the topic's fraction of the year's research output
  (``n_papers / total_papers_year``). Projecting the *share* removes the corpus-growth
  confound (every topic looks like it is "rising" if you only watch raw counts while the
  whole corpus grows), so share is the quantity the map foregrounds.
* **volume** (secondary context) — the raw ``n_papers`` per year. Carried alongside so a
  reader can see absolute scale, not just relative momentum.

Transforms (keep bands in valid range)
---------------------------------------
The series are fit in a transformed space so the back-transformed bands cannot leave the
quantity's natural range:

* **share** is fit as **logit** ``log(p / (1 - p))`` after epsilon-clipping into
  ``(EPS, 1 - EPS)`` (``EPS = 1e-6``); the forecast and its band are back-transformed
  with :func:`scipy.special.expit`, so the bands stay inside ``[0, 1]`` and a share can
  legitimately be ``0`` in early years.
* **volume** is fit as **log1p** ``log(1 + v)`` (handles ``v == 0`` cleanly without a
  separate epsilon); the forecast/band are back-transformed with ``expm1`` and clipped at
  ``0``, so the bands stay ``>= 0``.

Models (state-space first, robust fallback)
-------------------------------------------
The primary model is a local-linear-trend **structural time-series / state-space** model
(:class:`statsmodels.tsa.statespace.structural.UnobservedComponents`,
``level="local linear trend"``), fit by maximum likelihood (default L-BFGS, no RNG) and
forecast with its analytic confidence band. It is **deterministic**: refitting the same
series gives byte-identical output. When a topic has fewer than ``min_years`` observed
points, *or* the state-space fit fails to converge / returns a non-finite forecast, the
topic falls back to :func:`_fit_loglinear` — an ordinary-least-squares fit on the
transformed series with an analytic Student-``t`` **prediction interval** that widens with
the forecast horizon (the standard ``1 + 1/n + (x - x̄)² / Sxx`` term). The chosen path is
recorded in the ``model`` column (``"state_space_llt"`` vs ``"loglinear_fallback"``).

Regular annual grid
-------------------
Each topic's observed series is reindexed onto a **gap-free annual grid** over
``[first_observed_year, last_obs_year]`` (missing years filled with ``n_papers = 0``,
``share = 0``) before fitting, so the state-space model and the fixed ``+horizon`` are
well-defined even when a topic skips a year.

Purity
------
Every function takes in-memory frames and returns frames; no file/network/GPU. The build
driver (``V2/scripts/build_trajectory.py``) reads ``archetypes.parquet``, calls these two
functions, joins real ``label`` / ``size`` from the topic hierarchy into ``summary``, and
writes the parquet. This layer is **leaf-only**: the ``grain`` column is carried as the
constant ``"leaf"`` to match the roles/cascade table convention.

Conventions: ``from __future__ import annotations``; numpy-style docstrings; numpy +
pandas + scipy + statsmodels imported lazily inside functions; ruff/black line-length 100.
Real counts are COMPUTED, never hardcoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "EPS",
    "FLAT_SLOPE_THRESHOLD",
    "build_topic_year_series",
    "fit_trajectories",
]

# Epsilon used to clip a share into ``(EPS, 1 - EPS)`` before the logit transform, so a
# 0 or 1 share is representable in log-odds space without ``±inf``.
EPS: float = 1e-6

# |slope of projected share per year| below which a topic's ``direction`` is "flat".
# Chosen at 1e-4 share/yr: at corpus scale (~28k papers/yr) that is ~3 papers/yr of
# share-equivalent drift — below the noise floor of a single leaf topic's year-to-year
# wobble, so only a sustained, visible trend reads as rising/falling.
FLAT_SLOPE_THRESHOLD: float = 1e-4

# Discrete trend grain carried on every row to match the roles/cascade table convention
# (this forecast layer is leaf-only).
_GRAIN: str = "leaf"

# The two model-path labels recorded in ``summary.model``.
_MODEL_STATE_SPACE: str = "state_space_llt"
_MODEL_FALLBACK: str = "loglinear_fallback"


def build_topic_year_series(
    papers: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
    year_col: str = "year",
    max_complete_year: int = 2025,
) -> pd.DataFrame:
    """Long per-(topic, year) count + corpus share series from a per-paper frame.

    Counts papers per ``(topic, year)``, computes the year's total over the **real**
    topics, and forms each topic's ``share = n_papers / total_papers_year``. BERTopic
    noise (``topic_key == -1``) is removed from **both** the per-topic numerator and the
    per-year share denominator, and any ``year > max_complete_year`` (the partial trailing
    year — 2026 in the V2 corpus) is dropped entirely so an incomplete year never enters
    the fit.

    Parameters
    ----------
    papers :
        Per-paper frame carrying at least ``[topic_key, year_col]`` (e.g.
        ``archetypes.parquet``: ``pmid, year, topic_id, ...``). One row per paper; extra
        columns are ignored.
    topic_key :
        Topic-grain column. Default ``"topic_id"`` (leaf).
    year_col :
        Publication-year column. Default ``"year"``.
    max_complete_year :
        Last fully-observed year; rows with ``year > max_complete_year`` are dropped.
        Default ``2025`` (the V2 corpus's 2026 is a ~3.9k-paper partial year).

    Returns
    -------
    pandas.DataFrame
        One row per ``(topic_id, year)`` for every real topic/complete year, sorted by
        ``topic_id`` then ``year``, with columns:

        ``topic_id`` : int64 — the topic id (noise ``-1`` removed).
        ``year`` : int64 — the publication year (``<= max_complete_year``).
        ``n_papers`` : int64 — papers in this topic this year.
        ``total_papers_year`` : int64 — papers across **all real topics** this year
            (the share denominator; excludes ``-1``).
        ``share`` : float64 — ``n_papers / total_papers_year`` in ``[0, 1]``.

    Notes
    -----
    Only ``(topic, year)`` pairs that actually occur appear here (no gap-filling yet);
    :func:`fit_trajectories` reindexes each topic onto a gap-free annual grid before
    fitting. Empty input yields an empty, well-formed frame.
    """
    import pandas as pd

    out_cols = ["topic_id", "year", "n_papers", "total_papers_year", "share"]
    missing = [c for c in (topic_key, year_col) if c not in papers.columns]
    if missing:
        raise ValueError(f"papers missing required column(s): {missing}")

    work = papers.loc[:, [topic_key, year_col]].copy()
    work[topic_key] = pd.to_numeric(work[topic_key], errors="coerce")
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work = work.dropna(subset=[topic_key, year_col])
    if work.empty:
        return _empty_series_frame()

    work[topic_key] = work[topic_key].astype("int64")
    work[year_col] = work[year_col].astype("int64")

    # Drop BERTopic noise from the numerator AND the share denominator, and drop the
    # partial trailing year, BEFORE any counting.
    work = work.loc[work[topic_key] != -1]
    work = work.loc[work[year_col] <= int(max_complete_year)]
    if work.empty:
        return _empty_series_frame()

    counts = (
        work.groupby([topic_key, year_col], dropna=False)
        .size()
        .reset_index(name="n_papers")
        .rename(columns={topic_key: "topic_id", year_col: "year"})
    )
    year_totals = counts.groupby("year", dropna=False)["n_papers"].sum()
    counts["total_papers_year"] = counts["year"].map(year_totals).astype("int64")
    counts["share"] = (counts["n_papers"] / counts["total_papers_year"]).astype("float64")

    counts["topic_id"] = counts["topic_id"].astype("int64")
    counts["year"] = counts["year"].astype("int64")
    counts["n_papers"] = counts["n_papers"].astype("int64")
    counts = counts.sort_values(["topic_id", "year"]).reset_index(drop=True)
    return counts[out_cols]


def _empty_series_frame() -> pd.DataFrame:
    """Well-formed empty :func:`build_topic_year_series` output (typed columns)."""
    import pandas as pd

    return pd.DataFrame(
        {
            "topic_id": pd.Series([], dtype="int64"),
            "year": pd.Series([], dtype="int64"),
            "n_papers": pd.Series([], dtype="int64"),
            "total_papers_year": pd.Series([], dtype="int64"),
            "share": pd.Series([], dtype="float64"),
        }
    )


def _empty_series_out_frame() -> pd.DataFrame:
    """Well-formed empty ``series_out`` frame (observed+projected long rows)."""
    import pandas as pd

    return pd.DataFrame(
        {
            "grain": pd.Series([], dtype="object"),
            "topic_id": pd.Series([], dtype="int64"),
            "year": pd.Series([], dtype="int64"),
            "kind": pd.Series([], dtype="object"),
            "share": pd.Series([], dtype="float64"),
            "share_lo": pd.Series([], dtype="float64"),
            "share_hi": pd.Series([], dtype="float64"),
            "volume": pd.Series([], dtype="float64"),
            "volume_lo": pd.Series([], dtype="float64"),
            "volume_hi": pd.Series([], dtype="float64"),
        }
    )


def _empty_summary_frame() -> pd.DataFrame:
    """Well-formed empty ``summary`` frame (one row per topic; here zero rows)."""
    import pandas as pd

    return pd.DataFrame(
        {
            "grain": pd.Series([], dtype="object"),
            "topic_id": pd.Series([], dtype="int64"),
            "label": pd.Series([], dtype="object"),
            "size": pd.Series([], dtype="Int64"),
            "n_years_observed": pd.Series([], dtype="int64"),
            "last_obs_year": pd.Series([], dtype="int64"),
            "last_obs_share": pd.Series([], dtype="float64"),
            "model": pd.Series([], dtype="object"),
            "slope_share_per_yr": pd.Series([], dtype="float64"),
            "direction": pd.Series([], dtype="object"),
            "horizon_year": pd.Series([], dtype="int64"),
            "proj_share": pd.Series([], dtype="float64"),
            "proj_share_lo": pd.Series([], dtype="float64"),
            "proj_share_hi": pd.Series([], dtype="float64"),
            "proj_volume": pd.Series([], dtype="float64"),
            "proj_volume_lo": pd.Series([], dtype="float64"),
            "proj_volume_hi": pd.Series([], dtype="float64"),
            "fit_ok": pd.Series([], dtype="bool"),
        }
    )


def _logit_clip(p: Any) -> Any:
    """Epsilon-clipped logit of a share array (``log(p / (1 - p))`` on ``(EPS, 1-EPS)``)."""
    import numpy as np

    arr = np.clip(np.asarray(p, dtype="float64"), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def _fit_state_space(
    y: Any,
    *,
    horizon: int,
    ci_level: float,
) -> tuple[Any, Any, Any, bool]:
    """Fit a local-linear-trend state-space model; forecast ``horizon`` with a band.

    Fits :class:`statsmodels.tsa.statespace.structural.UnobservedComponents` with
    ``level="local linear trend"`` by maximum likelihood (default L-BFGS, **no RNG** —
    deterministic), then ``get_forecast(horizon)`` for the predicted mean and
    ``conf_int(alpha=1 - ci_level)`` for the analytic band.

    Parameters
    ----------
    y :
        1-D transformed observed series (logit-share or log1p-volume) on a regular annual
        grid, length ``>= 2``.
    horizon :
        Number of future steps to forecast.
    ci_level :
        Central probability of the band (e.g. ``0.80`` → 10th/90th percentiles).

    Returns
    -------
    mean : numpy.ndarray
        Length-``horizon`` forecast mean in transformed space.
    lo : numpy.ndarray
        Lower band in transformed space.
    hi : numpy.ndarray
        Upper band in transformed space.
    converged : bool
        ``False`` if MLE did not converge or any of ``mean``/``lo``/``hi`` is non-finite
        (the caller then falls back to :func:`_fit_loglinear`).

    Notes
    -----
    A perfectly linear / constant transformed series often makes the likelihood ridge
    flat and MLE reports non-convergence; that is exactly the signal to fall back. The
    fit explores extreme variances harmlessly under ``np.errstate`` + suppressed
    statsmodels warnings.
    """
    import warnings

    import numpy as np

    arr = np.asarray(y, dtype="float64")
    if arr.size < 2 or not np.isfinite(arr).all():
        return arr[:0], arr[:0], arr[:0], False

    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        from statsmodels.tsa.statespace.structural import UnobservedComponents
    except ImportError:  # pragma: no cover - statsmodels is a declared dependency
        return arr[:0], arr[:0], arr[:0], False

    try:
        with np.errstate(over="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            model = UnobservedComponents(arr, level="local linear trend")
            res = model.fit(disp=False)
            forecast = res.get_forecast(int(horizon))
            mean = np.asarray(forecast.predicted_mean, dtype="float64")
            band = np.asarray(forecast.conf_int(alpha=1.0 - float(ci_level)), dtype="float64")
    except (ValueError, np.linalg.LinAlgError):
        return arr[:0], arr[:0], arr[:0], False

    lo = band[:, 0]
    hi = band[:, 1]
    converged = bool(res.mle_retvals.get("converged", False))
    if not (np.isfinite(mean).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
        return arr[:0], arr[:0], arr[:0], False
    return mean, lo, hi, converged


def _fit_loglinear(
    y: Any,
    *,
    horizon: int,
    ci_level: float,
) -> tuple[Any, Any, Any, bool]:
    """Robust OLS fit on the transformed series with a horizon-widening ``t`` band.

    Ordinary least-squares line ``ŷ(x) = a + b·x`` (via :func:`scipy.stats.linregress`,
    **no RNG**) on the transformed observed series, with a classical Student-``t``
    **prediction interval** at each future ``x``::

        half(x) = t_crit · s · sqrt(1 + 1/n + (x - x̄)² / Sxx)

    where ``s`` is the residual standard deviation, ``Sxx = Σ(x - x̄)²``, ``t_crit`` the
    two-sided ``ci_level`` quantile on ``n - 2`` dof. The ``(x - x̄)²`` term makes the
    band **widen with the forecast horizon** (in transformed space) — the fallback's whole
    point.

    Parameters
    ----------
    y :
        1-D transformed observed series, length ``>= 2``.
    horizon :
        Number of future steps to forecast (future ``x = n, n+1, ..., n+horizon-1``).
    ci_level :
        Central probability of the prediction interval.

    Returns
    -------
    mean : numpy.ndarray
        Length-``horizon`` forecast mean in transformed space.
    lo, hi : numpy.ndarray
        Prediction-interval bounds in transformed space.
    fit_ok : bool
        ``True`` if the linear fit produced finite output; ``False`` (degenerate input)
        signals the caller to emit a flat NaN-band fallback.

    Notes
    -----
    With ``n == 2`` (zero residual dof) ``s`` and ``t_crit`` are taken as ``0`` so the
    band degenerates to the point line rather than dividing by zero — still finite and
    ordered. Fully deterministic.
    """
    import numpy as np
    from scipy import stats

    arr = np.asarray(y, dtype="float64")
    n = int(arr.size)
    if n < 2 or not np.isfinite(arr).all():
        return arr[:0], arr[:0], arr[:0], False

    x = np.arange(n, dtype="float64")
    xbar = float(x.mean())
    sxx = float(((x - xbar) ** 2).sum())
    fit = stats.linregress(x, arr)
    slope = float(fit.slope)
    intercept = float(fit.intercept)

    resid = arr - (intercept + slope * x)
    dof = n - 2
    if dof > 0:
        s = float(np.sqrt(float((resid**2).sum()) / dof))
        t_crit = float(stats.t.ppf(1.0 - (1.0 - float(ci_level)) / 2.0, dof))
    else:
        s = 0.0
        t_crit = 0.0

    xf = np.arange(n, n + int(horizon), dtype="float64")
    mean = intercept + slope * xf
    if sxx > 0.0:
        se = s * np.sqrt(1.0 + 1.0 / n + (xf - xbar) ** 2 / sxx)
    else:  # pragma: no cover - x is a contiguous range so Sxx > 0 for n >= 2
        se = np.full(int(horizon), s, dtype="float64")
    half = t_crit * se
    lo = mean - half
    hi = mean + half
    if not (np.isfinite(mean).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
        return arr[:0], arr[:0], arr[:0], False
    return mean, lo, hi, True


def _fit_one_quantity(
    y_obs: Any,
    *,
    horizon: int,
    ci_level: float,
    min_years: int,
    transform: str,
) -> tuple[Any, Any, Any, str, bool]:
    """Fit + forecast one quantity (share or volume), back-transformed to natural range.

    Applies the quantity's transform, picks the state-space model when there are
    ``>= min_years`` points and it converges, else the log-linear fallback, then
    back-transforms the mean/band and clips to the quantity's valid range.

    Parameters
    ----------
    y_obs :
        Observed natural-scale series on the regular grid (share in ``[0, 1]`` or raw
        volume ``>= 0``).
    horizon, ci_level, min_years :
        As in :func:`fit_trajectories`.
    transform :
        ``"logit"`` (share) or ``"log1p"`` (volume) — selects the (back-)transform and
        the natural-range clip.

    Returns
    -------
    mean, lo, hi : numpy.ndarray
        Length-``horizon`` back-transformed forecast and band, range-clipped.
    model : str
        ``"state_space_llt"`` or ``"loglinear_fallback"``.
    fit_ok : bool
        Whether a usable forecast was produced (always finite when ``True``).
    """
    import numpy as np

    natural = np.asarray(y_obs, dtype="float64")
    # logit for share (bands -> [0, 1]); log1p for volume (bands -> >= 0, 0-safe).
    y = _logit_clip(natural) if transform == "logit" else np.log1p(np.clip(natural, 0.0, None))

    model = _MODEL_FALLBACK
    mean: Any = None
    lo: Any = None
    hi: Any = None
    used_state_space = False
    if natural.size >= int(min_years):
        s_mean, s_lo, s_hi, converged = _fit_state_space(y, horizon=horizon, ci_level=ci_level)
        if converged and s_mean.size == int(horizon):
            mean, lo, hi, model = s_mean, s_lo, s_hi, _MODEL_STATE_SPACE
            used_state_space = True

    if not used_state_space:
        mean, lo, hi, ok = _fit_loglinear(y, horizon=horizon, ci_level=ci_level)
        model = _MODEL_FALLBACK
        if not ok or mean.size != int(horizon):
            # Degenerate series: flat-line the last observed transformed value, NaN band.
            last = float(y[-1]) if y.size else 0.0
            mean = np.full(int(horizon), last, dtype="float64")
            lo = np.full(int(horizon), np.nan, dtype="float64")
            hi = np.full(int(horizon), np.nan, dtype="float64")

    if transform == "logit":
        from scipy.special import expit

        bt_mean = expit(mean)
        bt_lo = expit(lo)
        bt_hi = expit(hi)
        # expit already lands in [0, 1]; clip defensively against fp drift.
        bt_mean = np.clip(bt_mean, 0.0, 1.0)
        bt_lo = np.clip(bt_lo, 0.0, 1.0)
        bt_hi = np.clip(bt_hi, 0.0, 1.0)
    else:
        bt_mean = np.clip(np.expm1(mean), 0.0, None)
        bt_lo = np.clip(np.expm1(lo), 0.0, None)
        bt_hi = np.clip(np.expm1(hi), 0.0, None)

    fit_ok = bool(np.isfinite(bt_mean).all())
    return bt_mean, bt_lo, bt_hi, model, fit_ok


def fit_trajectories(
    series: pd.DataFrame,
    *,
    horizon: int = 5,
    ci_level: float = 0.80,
    min_years: int = 8,
    last_obs_year: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit + project each topic's share and volume trajectory ``horizon`` years ahead.

    For every topic in ``series`` (the output of :func:`build_topic_year_series`):

    1. reindex its observed years onto a **gap-free annual grid** ending at
       ``last_obs_year`` (missing years filled with ``n_papers = 0``, ``share = 0``);
    2. fit and forecast **share** (logit space) and **volume** (log1p space) with
       :func:`_fit_state_space` (``>= min_years`` points and convergence) or the
       :func:`_fit_loglinear` fallback, back-transforming so share-bands stay in
       ``[0, 1]`` and volume-bands stay ``>= 0``;
    3. emit the long ``series_out`` (observed + projected rows) and a per-topic
       ``summary``.

    The fit is **deterministic**: running twice on the same ``series`` yields
    byte-identical frames.

    Parameters
    ----------
    series :
        Long ``[topic_id, year, n_papers, total_papers_year, share]`` from
        :func:`build_topic_year_series`.
    horizon :
        Number of future years to project (default ``5`` → 2026‥2030 with
        ``last_obs_year=2025``).
    ci_level :
        Central probability of the projection band (default ``0.80``).
    min_years :
        Minimum observed grid years for the state-space model; below this a topic uses
        the log-linear fallback (default ``8``).
    last_obs_year :
        The last observed year; the grid ends here and the horizon runs
        ``last_obs_year + 1 … last_obs_year + horizon``. Default ``2025``.

    Returns
    -------
    series_out : pandas.DataFrame
        One row per ``(grain, topic_id, year, kind)``, ``kind ∈ {"observed",
        "projected"}``, sorted by ``topic_id`` then ``year``, with columns
        ``[grain, topic_id, year, kind, share, share_lo, share_hi, volume, volume_lo,
        volume_hi]``. Bands (``*_lo`` / ``*_hi``) are **null on observed rows**, populated
        on projected rows. ``grain`` is the constant ``"leaf"``.
    summary : pandas.DataFrame
        One row per ``(grain, topic_id)``, sorted by ``topic_id``, with columns
        ``[grain, topic_id, label, size, n_years_observed, last_obs_year, last_obs_share,
        model, slope_share_per_yr, direction, horizon_year, proj_share, proj_share_lo,
        proj_share_hi, proj_volume, proj_volume_lo, proj_volume_hi, fit_ok]``.
        ``direction ∈ {"rising", "flat", "falling"}`` from the sign of
        ``slope_share_per_yr`` against :data:`FLAT_SLOPE_THRESHOLD`; ``horizon_year =
        last_obs_year + horizon``; ``proj_*`` are the final-horizon-year projection +
        band. ``label`` (``""``) and ``size`` (``<NA>``) are placeholders — the build
        driver joins the real values from the topic hierarchy afterward.

    Notes
    -----
    ``slope_share_per_yr`` is the OLS slope of the **observed share** over its grid years
    (natural scale, not the transformed-space coefficient), so it reads in
    share-per-year units. Empty input yields two well-formed empty frames.
    """
    import numpy as np
    import pandas as pd

    required = ["topic_id", "year", "n_papers", "share"]
    missing = [c for c in required if c not in series.columns]
    if missing:
        raise ValueError(f"series missing required column(s): {missing}")
    if series.empty:
        return _empty_series_out_frame(), _empty_summary_frame()

    work = series.loc[:, required].copy()
    work["topic_id"] = work["topic_id"].astype("int64")
    work["year"] = work["year"].astype("int64")
    work["n_papers"] = pd.to_numeric(work["n_papers"], errors="coerce").fillna(0.0)
    work["share"] = pd.to_numeric(work["share"], errors="coerce").fillna(0.0).astype("float64")
    work = work.loc[work["year"] <= int(last_obs_year)]
    if work.empty:
        return _empty_series_out_frame(), _empty_summary_frame()

    horizon = int(horizon)
    horizon_year = int(last_obs_year) + horizon
    future_years = list(range(int(last_obs_year) + 1, horizon_year + 1))

    series_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for topic_id, grp in work.groupby("topic_id", dropna=False):
        topic_id = int(topic_id)
        g = grp.sort_values("year")
        first_year = int(g["year"].min())
        # Regular annual grid [first_year .. last_obs_year], gaps filled with zeros.
        grid_years = np.arange(first_year, int(last_obs_year) + 1, dtype="int64")
        grid = g.set_index("year").reindex(grid_years).rename_axis("year").reset_index()
        grid["n_papers"] = grid["n_papers"].fillna(0.0).astype("float64")
        grid["share"] = grid["share"].fillna(0.0).astype("float64")

        share_obs = grid["share"].to_numpy(dtype="float64")
        vol_obs = grid["n_papers"].to_numpy(dtype="float64")
        n_years = int(grid.shape[0])
        last_obs_share = float(share_obs[-1])

        # --- forecast both quantities ---
        s_mean, s_lo, s_hi, s_model, s_ok = _fit_one_quantity(
            share_obs,
            horizon=horizon,
            ci_level=ci_level,
            min_years=min_years,
            transform="logit",
        )
        v_mean, v_lo, v_hi, _v_model, v_ok = _fit_one_quantity(
            vol_obs,
            horizon=horizon,
            ci_level=ci_level,
            min_years=min_years,
            transform="log1p",
        )
        model = s_model  # the share path is the headline; record its model.
        fit_ok = bool(s_ok and v_ok)

        # --- observed share slope (natural scale, share/yr) for direction ---
        gx = (grid_years - grid_years[0]).astype("float64")
        if gx.size >= 2 and float(((gx - gx.mean()) ** 2).sum()) > 0.0:
            from scipy import stats

            slope = float(stats.linregress(gx, share_obs).slope)
        else:
            slope = 0.0
        if slope > FLAT_SLOPE_THRESHOLD:
            direction = "rising"
        elif slope < -FLAT_SLOPE_THRESHOLD:
            direction = "falling"
        else:
            direction = "flat"

        # --- observed rows (bands null) ---
        for yr, sh, vol in zip(grid_years.tolist(), share_obs, vol_obs, strict=True):
            series_rows.append(
                {
                    "grain": _GRAIN,
                    "topic_id": topic_id,
                    "year": int(yr),
                    "kind": "observed",
                    "share": float(sh),
                    "share_lo": np.nan,
                    "share_hi": np.nan,
                    "volume": float(vol),
                    "volume_lo": np.nan,
                    "volume_hi": np.nan,
                }
            )
        # --- projected rows (bands populated) ---
        for k, yr in enumerate(future_years):
            series_rows.append(
                {
                    "grain": _GRAIN,
                    "topic_id": topic_id,
                    "year": int(yr),
                    "kind": "projected",
                    "share": float(s_mean[k]),
                    "share_lo": float(s_lo[k]),
                    "share_hi": float(s_hi[k]),
                    "volume": float(v_mean[k]),
                    "volume_lo": float(v_lo[k]),
                    "volume_hi": float(v_hi[k]),
                }
            )

        summary_rows.append(
            {
                "grain": _GRAIN,
                "topic_id": topic_id,
                "label": "",
                "size": pd.NA,
                "n_years_observed": n_years,
                "last_obs_year": int(last_obs_year),
                "last_obs_share": last_obs_share,
                "model": model,
                "slope_share_per_yr": slope,
                "direction": direction,
                "horizon_year": horizon_year,
                "proj_share": float(s_mean[-1]),
                "proj_share_lo": float(s_lo[-1]),
                "proj_share_hi": float(s_hi[-1]),
                "proj_volume": float(v_mean[-1]),
                "proj_volume_lo": float(v_lo[-1]),
                "proj_volume_hi": float(v_hi[-1]),
                "fit_ok": fit_ok,
            }
        )

    series_out = _finalize_series_out(series_rows)
    summary = _finalize_summary(summary_rows)
    return series_out, summary


def _finalize_series_out(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build the typed, sorted ``series_out`` frame from accumulated row dicts."""
    import pandas as pd

    if not rows:
        return _empty_series_out_frame()
    out = pd.DataFrame(rows)
    out["grain"] = out["grain"].astype("object")
    out["topic_id"] = out["topic_id"].astype("int64")
    out["year"] = out["year"].astype("int64")
    out["kind"] = out["kind"].astype("object")
    for c in ("share", "share_lo", "share_hi", "volume", "volume_lo", "volume_hi"):
        out[c] = out[c].astype("float64")
    # observed before projected within a topic (kind sorts "observed" < "projected"),
    # then by year — but sort explicitly on (topic_id, year) with observed first.
    out["_krank"] = (out["kind"] == "projected").astype("int64")
    out = out.sort_values(["topic_id", "_krank", "year"]).drop(columns="_krank")
    out = out.reset_index(drop=True)
    return out[
        [
            "grain",
            "topic_id",
            "year",
            "kind",
            "share",
            "share_lo",
            "share_hi",
            "volume",
            "volume_lo",
            "volume_hi",
        ]
    ]


def _finalize_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build the typed, sorted ``summary`` frame from accumulated row dicts."""
    import pandas as pd

    if not rows:
        return _empty_summary_frame()
    out = pd.DataFrame(rows)
    out["grain"] = out["grain"].astype("object")
    out["topic_id"] = out["topic_id"].astype("int64")
    out["label"] = out["label"].astype("object")
    out["size"] = out["size"].astype("Int64")
    out["n_years_observed"] = out["n_years_observed"].astype("int64")
    out["last_obs_year"] = out["last_obs_year"].astype("int64")
    out["horizon_year"] = out["horizon_year"].astype("int64")
    out["model"] = out["model"].astype("object")
    out["direction"] = out["direction"].astype("object")
    out["fit_ok"] = out["fit_ok"].astype("bool")
    for c in (
        "last_obs_share",
        "slope_share_per_yr",
        "proj_share",
        "proj_share_lo",
        "proj_share_hi",
        "proj_volume",
        "proj_volume_lo",
        "proj_volume_hi",
    ):
        out[c] = out[c].astype("float64")
    out = out.sort_values("topic_id").reset_index(drop=True)
    return out[
        [
            "grain",
            "topic_id",
            "label",
            "size",
            "n_years_observed",
            "last_obs_year",
            "last_obs_share",
            "model",
            "slope_share_per_yr",
            "direction",
            "horizon_year",
            "proj_share",
            "proj_share_lo",
            "proj_share_hi",
            "proj_volume",
            "proj_volume_lo",
            "proj_volume_hi",
            "fit_ok",
        ]
    ]
