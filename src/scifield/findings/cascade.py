"""V1-S15 F1 — pre-registered epistemic-cascade lead/lag analysis (PURE logic).

This module implements the F1 contract
(``docs/preregistrations/PR3_epistemic_cascade.md``,
``/tmp/v1s15_f1_contract.md``): does within-topic evidence *quality/maturity*
LEAD research-volume growth (rigor precedes the surge) or LAG it (attention
precedes rigor)? The whole module is **pure** — it receives an already-joined,
already-leaf-filtered in-memory tidy frame and never touches disk, network, GPU,
or any external service. Notebook 11 does the I/O and calls these helpers.

Universe (assembled by the notebook, not here)
----------------------------------------------
Papers in ``epistemic_extracted.parquet`` (``pmid, study_design``) INNER-joined
to ``novelty_semantic.parquet`` (``pmid, topic_id, year``) on ``pmid``, LEAF
topics only (``topic_id != -1``). The pure entry point
:func:`build_topic_year_series` expects exactly the tidy columns
``pmid, topic_id, year, study_design`` (extra columns are ignored).

Two per-topic ANNUAL series
---------------------------
* **volume(topic, y)** — paper count for that topic in year ``y`` over ALL
  designs.
* **quality(topic, y)** PRIMARY — **mean evidence-tier** over the papers that
  year whose design is graded, NaN-skipping. The ordinal :data:`TIER_MAP`
  (higher = stronger) grades ``RCT=4, cohort=3, case_control=2,
  case_series=1``; ``review`` and ``other`` carry **no tier** (NaN) and are
  EXCLUDED from the mean. A year with zero graded papers → ``mean_tier`` is NaN.
* **quality ROBUSTNESS series** — **RCT_share(topic, y)** = ``#RCT / #all
  papers that topic-year`` (denominator = ALL designs, incl. review + other).

Qualifying topic
----------------
``≥ v_min`` papers TOTAL **and** ``≥ min_years`` distinct years each with
``≥ min_count`` papers (TOTAL count). See :func:`qualifying_topics`. With the
pinned defaults (30 / 8 / 5) this is the 138-topic denominator for the
holds-fraction.

Series construction for the lead/lag tests (per qualifying topic)
-----------------------------------------------------------------
1. Restrict to the contiguous span ``[min year with ≥ min_count papers, max
   year with ≥ min_count papers]`` (trim thin leading/trailing years).
2. The PRIMARY quality series may have internal NaN years (no graded papers):
   **linear-interpolate internal gaps**. If ``> max_nan_frac`` of in-span years
   are NaN-tier, the topic is recorded **quality-non-evaluable** (kept in the
   results table with an explicit reason — NOT silently dropped — and it stays
   in the qualifying denominator, counting as non-significant; conservative).
   See :func:`prepare_series`.
3. **Difference each series once (Δ)** for stationarity before Granger; the
   differencing happens *inside* :func:`granger_pair` / :func:`panel_granger`.

Tests (per evaluable qualifying topic)
--------------------------------------
* **CCF** — cross-correlation of (quality, volume) over integer lags
  ``∈ [-max_lag, +max_lag]``; report the **peak-|corr| lag** and its sign. A
  NEGATIVE peak lag = quality LEADS volume (rigor precedes the surge); POSITIVE
  = quality LAGS. See :func:`cross_correlation`.
* **Granger causality**, BOTH directions, on the Δ-differenced series, at fixed
  ``lag=3`` (statsmodels ``grangercausalitytests``, statistic ``ssr_ftest``,
  the single p-value at lag 3). A degenerate/non-fittable topic → ``NaN`` in
  both directions. ``quality → volume`` p = "quality leads";
  ``volume → quality`` p = "volume leads / quality lags". See
  :func:`granger_pair`.

Multiple comparisons & decision rule
------------------------------------
Pool ALL per-topic per-direction p-values and apply Benjamini-Hochberg FDR at
``q < 0.05`` (:func:`bh_fdr`). A topic shows a **directional** quality↔volume
relationship iff **exactly one** direction is FDR-significant; both = "coupled,
no clear direction" (not directional); neither = none. :func:`decide_f1`
returns the verdict: F1 HOLDS iff (a) ``≥ holds_frac`` of qualifying topics are
FDR-significant directional **and** (b) the pooled :func:`panel_granger` test
agrees on the dominant direction.

Operationalization choices the contract left open (documented identically in
PR3)
----------------------------------------------------------------------------
* **CCF estimator** — Pearson correlation between the two series with one
  standardized at zero-mean/unit-std on the *overlapping* window for each
  integer lag; ties in ``|corr|`` are broken toward the lag of **smallest
  absolute value**, and among equal absolute lags toward the **negative**
  (quality-leads) sign, so a perfectly symmetric/degenerate CCF reports lag 0.
* **CCF input** — computed on the Δ-differenced series (same stationarised
  series fed to Granger), so the lead/lag estimators are consistent.
* **Panel "dominant direction"** — taken from the per-topic *directional*
  majority (lead vs lag among topics with exactly one significant direction).
  An exact per-topic lead/lag tie (``n_lead == n_lag``, zero OR positive) ⇒ no
  dominant direction ⇒ F1 does NOT hold; there is NO panel tie-break. "Panel
  agrees" requires the panel block-F to be significant (``p < panel_alpha``) in
  that same direction.
* **Granger / panel min length** — a topic series shorter than the minimum
  fittable length for ``lag`` (``2*lag + 2`` post-differencing observations)
  yields ``NaN`` p-values rather than raising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "TIER_MAP",
    "bh_fdr",
    "build_topic_year_series",
    "classify_direction",
    "cross_correlation",
    "decide_f1",
    "evidence_tier",
    "granger_pair",
    "panel_granger",
    "prepare_series",
    "qualifying_topics",
]

#: Ordinal evidence-tier map (higher = stronger study design). ``review`` and
#: ``other`` are deliberately ABSENT — they carry no tier and grade to NaN.
TIER_MAP: dict[str, int] = {
    "RCT": 4,
    "cohort": 3,
    "case_control": 2,
    "case_series": 1,
}


# --------------------------------------------------------------------------- #
# Pure logic core (no I/O -- unit-testable on in-memory data)
# --------------------------------------------------------------------------- #


def evidence_tier(design: object, *, tier_map: dict[str, int] = TIER_MAP) -> float:
    """Map one study-design label to its ordinal evidence tier.

    Parameters
    ----------
    design :
        A study-design label, e.g. ``"RCT"`` / ``"cohort"`` / ``"review"``. Any
        value not present in ``tier_map`` (including ``"review"``, ``"other"``,
        ``None``, or NaN) grades to ``NaN`` — it carries no tier and is excluded
        from the mean-tier series.
    tier_map :
        The ordinal grade dict; defaults to the pinned :data:`TIER_MAP`.

    Returns
    -------
    float
        ``float(tier_map[design])`` when graded, else ``float('nan')``.

    Notes
    -----
    Returns a float (never an int) so the result composes cleanly with
    NaN-skipping means.
    """
    try:
        return float(tier_map[design])  # type: ignore[index]
    except (KeyError, TypeError):
        return float("nan")


def build_topic_year_series(
    df: pd.DataFrame,
    *,
    tier_map: dict[str, int] = TIER_MAP,
    topic_col: str = "topic_id",
    year_col: str = "year",
    design_col: str = "study_design",
) -> pd.DataFrame:
    """Collapse a tidy paper frame to one row per ``(topic, year)``.

    Pure and I/O-free. The denominators are the FULL topic-year paper count
    (all designs); the mean-tier numerator NaN-skips ungraded designs (``review``
    / ``other`` and anything outside ``tier_map``).

    Parameters
    ----------
    df :
        Tidy paper frame with at least ``[topic_col, year_col, design_col]`` (one
        row per paper). Extra columns are ignored. ``year`` may be any orderable
        integer-like dtype.
    tier_map :
        Ordinal evidence-tier dict; defaults to :data:`TIER_MAP`.
    topic_col, year_col, design_col :
        Column names; defaulted to ``topic_id`` / ``year`` / ``study_design``.

    Returns
    -------
    pandas.DataFrame
        One row per ``(topic_id, year)`` with columns:

        * ``volume`` — paper count that topic-year over ALL designs.
        * ``mean_tier`` — NaN-skipping mean evidence tier; ``NaN`` when the
          topic-year has zero graded papers.
        * ``rct_share`` — ``#RCT / volume`` (denominator = all designs).
        * ``n_graded`` — number of graded (non-NaN-tier) papers that topic-year.

        Sorted by ``(topic_id, year)``; a plain ``RangeIndex``.

    Notes
    -----
    Years with zero papers for a topic are simply absent (no zero-filled rows);
    span-trimming and gap-interpolation happen later in :func:`prepare_series`.
    """
    work = df[[topic_col, year_col, design_col]].copy()
    work["tier"] = work[design_col].map(lambda d: evidence_tier(d, tier_map=tier_map))
    work["is_rct"] = (work[design_col] == "RCT").astype("float64")
    work["is_graded"] = work["tier"].notna().astype("float64")

    grouped = work.groupby([topic_col, year_col], sort=True)
    out = grouped.agg(
        volume=(design_col, "size"),
        mean_tier=("tier", "mean"),
        n_rct=("is_rct", "sum"),
        n_graded=("is_graded", "sum"),
    ).reset_index()

    out["rct_share"] = out["n_rct"].to_numpy(dtype="float64") / out["volume"].to_numpy(
        dtype="float64"
    )
    out["n_graded"] = out["n_graded"].astype("int64")
    out["volume"] = out["volume"].astype("int64")
    # Pandas ``mean`` of an all-NaN group already yields NaN; make it explicit.
    out["mean_tier"] = out["mean_tier"].astype("float64")
    out = out.drop(columns=["n_rct"])

    cols = [topic_col, year_col, "volume", "mean_tier", "rct_share", "n_graded"]
    return out[cols].sort_values([topic_col, year_col]).reset_index(drop=True)


def qualifying_topics(
    series: pd.DataFrame,
    *,
    v_min: int = 30,
    min_years: int = 8,
    min_count: int = 5,
    topic_col: str = "topic_id",
) -> list:
    """Return the sorted list of topics meeting the qualifying-volume rule.

    A topic qualifies iff it has ``≥ v_min`` papers TOTAL **and** ``≥ min_years``
    distinct years each with ``≥ min_count`` papers (TOTAL count). Pure.

    Parameters
    ----------
    series :
        A ``(topic, year)`` frame as built by :func:`build_topic_year_series`
        (must carry ``topic_col`` and a ``volume`` column).
    v_min :
        Minimum total paper count per topic. Pre-registered default ``30``.
    min_years :
        Minimum number of distinct years meeting the per-year threshold.
        Pre-registered default ``8``.
    min_count :
        Per-year paper-count threshold. Pre-registered default ``5``.
    topic_col :
        Topic id column name; defaults to ``topic_id``.

    Returns
    -------
    list
        Sorted list of qualifying topic ids (the holds-fraction denominator).
    """
    qualifying: list = []
    for topic, grp in series.groupby(topic_col, sort=True):
        total = int(grp["volume"].sum())
        n_dense_years = int((grp["volume"] >= min_count).sum())
        if total >= v_min and n_dense_years >= min_years:
            qualifying.append(topic)
    return qualifying


def prepare_series(
    topic_series: pd.DataFrame,
    *,
    quality_col: str = "mean_tier",
    max_nan_frac: float = 0.25,
    min_count: int = 5,
    year_col: str = "year",
):
    """Trim, gap-interpolate, and evaluability-flag one topic's two series.

    Pure. The span is the contiguous ``[min, max]`` of years whose ``volume`` is
    ``≥ min_count`` (thin leading/trailing years trimmed); the volume series is
    reindexed to the full integer span (missing years → ``volume`` 0). Internal
    NaN quality years are **linearly interpolated**; if ``> max_nan_frac`` of the
    in-span years are NaN before interpolation, the topic is flagged
    ``"non_evaluable"`` (still returned, not dropped). Differencing is deferred
    to the test functions.

    Parameters
    ----------
    topic_series :
        The ``(year, volume, quality_col, ...)`` rows for ONE topic (as sliced
        from :func:`build_topic_year_series`). Need not be contiguous in year.
    quality_col :
        Which quality series to prepare — ``"mean_tier"`` (PRIMARY) or
        ``"rct_share"`` (robustness). Defaults to ``"mean_tier"``.
    max_nan_frac :
        Fraction of in-span NaN-quality years above which the topic is
        non-evaluable. Pre-registered default ``0.25``.
    min_count :
        Per-year threshold defining the dense span. Default ``5``.
    year_col :
        Year column name; defaults to ``year``.

    Returns
    -------
    tuple[pandas.Series, pandas.Series, str]
        ``(quality, volume, status)`` aligned on a contiguous integer-year
        index. ``status`` is one of:

        * ``"ok"`` — evaluable; ``quality`` is gap-interpolated and finite
          across the (interior) span.
        * ``"non_evaluable"`` — ``> max_nan_frac`` of in-span years had no graded
          papers; series still returned for transparency but the caller treats
          the topic as non-significant.
        * ``"too_short"`` — fewer than 2 dense years, so no span exists.

    Notes
    -----
    ``rct_share`` never has structural NaNs (its denominator is the full
    topic-year count), so for the robustness series ``status`` is effectively
    always ``"ok"`` (or ``"too_short"``); the NaN-fraction branch is a no-op
    there. Edge NaNs that remain after interpolation (when the very first/last
    in-span year is NaN) are filled by nearest-value extrapolation so the
    returned ``quality`` has no NaNs for an ``"ok"`` topic.
    """
    import numpy as np
    import pandas as pd

    s = topic_series.sort_values(year_col)
    years = s[year_col].to_numpy()
    volume_raw = s["volume"].to_numpy(dtype="float64")
    quality_raw = s[quality_col].to_numpy(dtype="float64")

    dense = np.where(volume_raw >= min_count)[0]
    if dense.size < 2:
        empty = pd.Series(dtype="float64")
        return empty, empty, "too_short"

    y_lo = int(years[dense[0]])
    y_hi = int(years[dense[-1]])
    full_years = np.arange(y_lo, y_hi + 1)

    by_year_vol = dict(zip((int(y) for y in years), volume_raw, strict=True))
    by_year_qual = dict(zip((int(y) for y in years), quality_raw, strict=True))

    volume = pd.Series(
        [float(by_year_vol.get(int(y), 0.0)) for y in full_years],
        index=full_years,
        dtype="float64",
    )
    quality = pd.Series(
        [by_year_qual.get(int(y), float("nan")) for y in full_years],
        index=full_years,
        dtype="float64",
    )

    n_span = len(full_years)
    n_nan = int(quality.isna().sum())
    status = "ok"
    if n_span > 0 and (n_nan / n_span) > max_nan_frac:
        status = "non_evaluable"

    # Linear-interpolate internal gaps; nearest-fill any remaining edge NaNs so
    # an evaluable series is fully finite for the lead/lag estimators.
    quality = quality.interpolate(method="linear", limit_direction="both")

    return quality, volume, status


def cross_correlation(
    quality,
    volume,
    *,
    max_lag: int = 5,
    difference: bool = True,
):
    """Cross-correlation of ``(quality, volume)`` over integer lags.

    Pure. For each integer lag ``L ∈ [-max_lag, +max_lag]`` the Pearson
    correlation is computed on the overlapping window of ``quality[t]`` against
    ``volume[t - L]``. By the contract's sign convention a **NEGATIVE** peak lag
    means quality LEADS volume (rigor precedes the surge — volume echoes earlier
    quality); **POSITIVE** means quality LAGS.

    Parameters
    ----------
    quality, volume :
        Equal-length aligned series (``pandas.Series`` or array-like). Typically
        the :func:`prepare_series` outputs.
    max_lag :
        Largest absolute lag considered. Pre-registered default ``5``.
    difference :
        If ``True`` (default) both series are first-differenced (``Δ``) before
        correlating, matching the stationarised input fed to Granger.

    Returns
    -------
    tuple[int, float, dict[int, float]]
        ``(peak_lag, peak_corr, ccf)`` where ``ccf`` maps every lag to its
        correlation (``NaN`` where a window is too short or degenerate).
        ``peak_lag`` is the ``argmax`` of ``|corr|``; ties break toward the lag
        of smallest ``|lag|`` and then toward the negative (quality-leads) sign,
        so a flat/degenerate CCF reports ``(0, nan-or-0, ...)``.

    Notes
    -----
    A lag whose overlap has fewer than 2 finite paired points, or zero variance
    in either leg, yields ``NaN`` for that lag and is never selected as the peak
    unless every lag is ``NaN`` (then ``peak_lag = 0``, ``peak_corr = nan``).
    """
    import numpy as np

    q = np.asarray(quality, dtype="float64")
    v = np.asarray(volume, dtype="float64")
    if difference:
        q = np.diff(q)
        v = np.diff(v)

    n = min(q.size, v.size)
    q = q[:n]
    v = v[:n]

    ccf: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        # ``ccf[lag] = corr(quality[t], volume[t - lag])``. With this convention
        # a NEGATIVE lag = quality LEADS volume: if volume echoes earlier quality
        # (``volume[t] == quality[t - k]``, k > 0) the peak lands at ``lag = -k``.
        shift = -lag
        if shift < 0:
            a = q[-shift:]
            b = v[: n + shift]
        elif shift > 0:
            a = q[: n - shift]
            b = v[shift:]
        else:
            a = q
            b = v
        ccf[lag] = _pearson(a, b)

    finite_lags = [(lag, c) for lag, c in ccf.items() if np.isfinite(c)]
    if not finite_lags:
        return 0, float("nan"), ccf

    # Peak = max |corr|; tie-break by smallest |lag|, then negative sign first.
    peak_lag, peak_corr = max(
        finite_lags,
        key=lambda kv: (abs(kv[1]), -abs(kv[0]), -np.sign(kv[0])),
    )
    return int(peak_lag), float(peak_corr), ccf


def _pearson(a, b) -> float:
    """Pearson correlation of two equal-length arrays; ``NaN`` if degenerate."""
    import numpy as np

    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def granger_pair(
    quality,
    volume,
    *,
    lag: int = 3,
):
    """Bidirectional fixed-lag Granger causality on the Δ-differenced series.

    Pure. Both series are first-differenced internally, then
    ``statsmodels.tsa.stattools.grangercausalitytests`` is run at fixed
    ``maxlag=[lag]`` taking the single ``ssr_ftest`` p-value. Wrapped in
    try/except: any degenerate / non-fittable topic returns ``NaN`` in both
    directions (recorded, treated as non-significant). All statsmodels output
    printing and deprecation warnings are suppressed.

    Parameters
    ----------
    quality, volume :
        Equal-length aligned series (typically :func:`prepare_series` outputs).
    lag :
        Fixed Granger lag order. Pre-registered default ``3``.

    Returns
    -------
    tuple[float, float]
        ``(p_q_to_v, p_v_to_q)``:

        * ``p_q_to_v`` — p-value for ``quality → volume`` ("quality leads").
        * ``p_v_to_q`` — p-value for ``volume → quality`` ("volume leads /
          quality lags").

        Either is ``NaN`` when its regression cannot be fit (too few
        post-differencing observations — needs ``≥ 2*lag + 2`` — or a singular
        design).

    Notes
    -----
    ``grangercausalitytests(data[:, [target, cause]], ...)`` tests whether the
    SECOND column Granger-causes the FIRST, so the ``quality → volume`` test
    passes ``column_stack([Δvolume, Δquality])``.
    """
    import numpy as np

    q = np.diff(np.asarray(quality, dtype="float64"))
    v = np.diff(np.asarray(volume, dtype="float64"))
    p_q_to_v = _granger_one(target=v, cause=q, lag=lag)
    p_v_to_q = _granger_one(target=q, cause=v, lag=lag)
    return p_q_to_v, p_v_to_q


def _granger_one(*, target, cause, lag: int) -> float:
    """One-directional ``ssr_ftest`` p-value (cause → target); ``NaN`` on failure."""
    import warnings

    import numpy as np

    target = np.asarray(target, dtype="float64")
    cause = np.asarray(cause, dtype="float64")
    n = min(target.size, cause.size)
    target = target[:n]
    cause = cause[:n]

    # Need enough observations to fit ``lag`` lags with residual df left over.
    if n < 2 * lag + 2:
        return float("nan")
    if not (np.all(np.isfinite(target)) and np.all(np.isfinite(cause))):
        return float("nan")
    if np.std(target) == 0.0 or np.std(cause) == 0.0:
        return float("nan")

    from statsmodels.tsa.stattools import grangercausalitytests

    data = np.column_stack([target, cause])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data, maxlag=[lag], verbose=False)
        p = float(res[lag][0]["ssr_ftest"][1])
    except Exception:
        return float("nan")
    if not np.isfinite(p):
        return float("nan")
    return p


def bh_fdr(pvalues, *, q: float = 0.05):
    """Benjamini-Hochberg FDR reject mask at level ``q``.

    Pure and truth-testable. ``NaN`` p-values are carried through as **not
    rejected** (``False``) without participating in the BH ranking; only finite
    p-values count toward the number of hypotheses ``m``.

    Parameters
    ----------
    pvalues :
        Array-like of p-values (may contain ``NaN``).
    q :
        Target false-discovery rate. Pre-registered default ``0.05``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, same length/order as ``pvalues``; ``True`` where the
        hypothesis is rejected under BH at ``q``. All-``NaN`` (or empty) input
        yields an all-``False`` mask.

    Notes
    -----
    Standard step-up BH: sort the ``m`` finite p-values ascending, find the
    largest rank ``k`` with ``p_(k) <= (k/m) * q``, and reject every hypothesis
    with ``p <= p_(k)``.
    """
    import numpy as np

    p = np.asarray(pvalues, dtype="float64")
    reject = np.zeros(p.shape, dtype=bool)
    finite_idx = np.where(np.isfinite(p))[0]
    m = finite_idx.size
    if m == 0:
        return reject

    pv = p[finite_idx]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    ranks = np.arange(1, m + 1)
    thresh = ranks / m * q
    passing = np.where(ranked <= thresh)[0]
    if passing.size == 0:
        return reject

    k = int(passing.max())  # 0-based index of the largest passing rank
    p_cut = ranked[k]
    reject[finite_idx] = pv <= p_cut
    return reject


def classify_direction(*, leads_sig: bool, lags_sig: bool) -> str:
    """Map a topic's two FDR-significance flags to a direction category.

    Parameters
    ----------
    leads_sig :
        Whether ``quality → volume`` ("quality leads") is FDR-significant.
    lags_sig :
        Whether ``volume → quality`` ("quality lags") is FDR-significant.

    Returns
    -------
    str
        ``"lead"`` (exactly quality→volume), ``"lag"`` (exactly volume→quality),
        ``"coupled"`` (both — no clear direction), or ``"none"`` (neither). Only
        ``"lead"``/``"lag"`` count as a directional relationship per the
        decision rule.
    """
    if leads_sig and lags_sig:
        return "coupled"
    if leads_sig:
        return "lead"
    if lags_sig:
        return "lag"
    return "none"


def panel_granger(
    differenced_by_topic,
    *,
    lag: int = 3,
):
    """Pooled fixed-effects panel-Granger on the within-topic Δ-series.

    Pure. Pools every topic's *already-differenced* ``(Δquality, Δvolume)``
    series into one stacked panel with TOPIC fixed effects (topic dummies) and
    runs two regressions:

    * regress ``Δvolume_t`` on lags ``1..lag`` of ``Δvolume`` and lags
      ``1..lag`` of ``Δquality`` → block-F on the ``Δquality`` lags =
      ``p_{quality→volume}``.
    * the symmetric regression → ``p_{volume→quality}``.

    Parameters
    ----------
    differenced_by_topic :
        Iterable of ``(dquality, dvolume)`` array pairs, ONE per evaluable
        qualifying topic, each already first-differenced (i.e. pass
        ``np.diff`` of the :func:`prepare_series` outputs). Pairs too short to
        contribute a single lagged row are skipped.
    lag :
        Number of own/cross lags. Pre-registered default ``3``.

    Returns
    -------
    tuple[float, float]
        ``(p_q_to_v, p_v_to_q)`` block-F p-values; ``NaN`` if the pooled design
        cannot be fit (no usable rows / singular).

    Notes
    -----
    Within-topic lagging never crosses a topic boundary (each topic's lagged
    design matrix is built independently, then stacked), so no spurious
    cross-topic autoregression is introduced. A constant plus ``T-1`` topic
    dummies absorb the topic means (the within transformation).
    """
    blocks_qv = _build_panel(differenced_by_topic, target="volume", lag=lag)
    blocks_vq = _build_panel(differenced_by_topic, target="quality", lag=lag)
    p_q_to_v = _panel_block_ftest(blocks_qv, cross_prefix="dq", lag=lag)
    p_v_to_q = _panel_block_ftest(blocks_vq, cross_prefix="dv", lag=lag)
    return p_q_to_v, p_v_to_q


def _build_panel(differenced_by_topic, *, target: str, lag: int):
    """Stack per-topic lagged design rows with topic ids into one panel dict."""
    import numpy as np

    y_rows: list[float] = []
    own_rows: list[list[float]] = []  # lags of the target series itself
    cross_rows: list[list[float]] = []  # lags of the other series
    topic_ids: list[int] = []

    for tid, (dq, dv) in enumerate(differenced_by_topic):
        dq = np.asarray(dq, dtype="float64")
        dv = np.asarray(dv, dtype="float64")
        n = min(dq.size, dv.size)
        dq = dq[:n]
        dv = dv[:n]
        if n < lag + 1:
            continue
        if target == "volume":
            y_series, own_series, cross_series = dv, dv, dq
        else:
            y_series, own_series, cross_series = dq, dq, dv
        if not (np.all(np.isfinite(dq)) and np.all(np.isfinite(dv))):
            continue
        for t in range(lag, n):
            y_rows.append(float(y_series[t]))
            own_rows.append([float(own_series[t - k]) for k in range(1, lag + 1)])
            cross_rows.append([float(cross_series[t - k]) for k in range(1, lag + 1)])
            topic_ids.append(tid)

    return {
        "y": np.asarray(y_rows, dtype="float64"),
        "own": np.asarray(own_rows, dtype="float64").reshape(len(own_rows), lag),
        "cross": np.asarray(cross_rows, dtype="float64").reshape(len(cross_rows), lag),
        "topic": np.asarray(topic_ids, dtype="int64"),
    }


def _panel_block_ftest(panel, *, cross_prefix: str, lag: int) -> float:
    """OLS with topic FE; block-F p-value on the cross-series lag columns."""
    import numpy as np

    y = panel["y"]
    own = panel["own"]
    cross = panel["cross"]
    topic = panel["topic"]
    if y.size == 0:
        return float("nan")

    # Topic dummies, dropping the first level so the constant is identified.
    uniq = np.unique(topic)
    if uniq.size >= 2:
        dummy_levels = uniq[1:]
        dummies = np.column_stack([(topic == lev).astype("float64") for lev in dummy_levels])
    else:
        dummies = np.empty((y.size, 0), dtype="float64")

    const = np.ones((y.size, 1), dtype="float64")
    # Column order: const | own lags | cross lags | topic dummies.
    x = np.column_stack([const, own, cross, dummies])
    n_const = 1
    cross_start = n_const + own.shape[1]
    cross_idx = list(range(cross_start, cross_start + cross.shape[1]))

    import statsmodels.api as sm

    try:
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            model = sm.OLS(y, x).fit()
            restriction = np.zeros((len(cross_idx), x.shape[1]), dtype="float64")
            for i, j in enumerate(cross_idx):
                restriction[i, j] = 1.0
            ftest = model.f_test(restriction)
            p = float(ftest.pvalue)
    except Exception:
        return float("nan")
    if not np.isfinite(p):
        return float("nan")
    return p


def decide_f1(
    per_topic_results,
    panel_result,
    *,
    n_qualifying: int,
    holds_frac: float = 0.20,
    panel_alpha: float = 0.05,
) -> dict:
    """Apply the pre-registered F1 decision rule to the assembled results.

    Pure. F1 HOLDS iff BOTH (a) at least ``holds_frac`` of the ``n_qualifying``
    topics show an FDR-significant DIRECTIONAL (``"lead"`` or ``"lag"``)
    quality↔volume Granger relationship, and (b) the pooled panel-Granger test
    agrees on the dominant direction (panel block-F significant at
    ``panel_alpha`` in the same direction as the per-topic directional majority).

    Parameters
    ----------
    per_topic_results :
        Iterable of per-topic dicts, each with a ``"direction"`` key whose value
        is one of ``"lead"`` / ``"lag"`` / ``"coupled"`` / ``"none"`` (e.g. from
        :func:`classify_direction`). Topics flagged non-evaluable / too-short
        must appear here with ``"none"`` so they sit in the denominator as
        non-significant. Extra keys are ignored.
    panel_result :
        ``(p_q_to_v, p_v_to_q)`` from :func:`panel_granger` (``NaN`` allowed).
    n_qualifying :
        Size of the qualifying denominator (138 on the real corpus).
    holds_frac :
        Minimum directional fraction. Pre-registered default ``0.20`` (``≥ 28``
        of ``138``).
    panel_alpha :
        Panel significance threshold. Pre-registered default ``0.05``.

    Returns
    -------
    dict
        Verdict with keys:

        * ``"holds"`` (bool) — F1 holds.
        * ``"n_directional"`` (int) — count of ``"lead"`` + ``"lag"`` topics.
        * ``"frac_directional"`` (float) — ``n_directional / n_qualifying``.
        * ``"dominant_direction"`` (str) — ``"quality_leads"`` /
          ``"quality_lags"`` (the per-topic directional majority) or ``"none"``.
          An exact per-topic lead/lag tie (``n_lead == n_lag``, zero OR
          positive) ⇒ no dominant direction ⇒ F1 does NOT hold; there is NO
          panel tie-break. ``"none"`` is also returned when no directional
          topics exist.
        * ``"panel_agrees"`` (bool) — panel significant at ``panel_alpha`` in the
          dominant direction.
        * ``"n_lead"`` / ``"n_lag"`` / ``"n_coupled"`` / ``"n_none"`` (int) — the
          full per-topic direction split.
        * ``"panel_p_quality_leads"`` / ``"panel_p_quality_lags"`` (float) — the
          two panel p-values, echoed for the report.
        * ``"frac_threshold"`` (float) — ``holds_frac``, echoed.

    Notes
    -----
    The ``frac_directional`` gate uses the literal ``frac_directional >=
    holds_frac`` comparison on the exact ratio; at the pinned boundary
    ``28/138 ≈ 0.2029 ≥ 0.20`` holds while ``27/138 ≈ 0.1957 < 0.20`` does not.
    """
    import numpy as np

    directions = [r.get("direction", "none") for r in per_topic_results]
    n_lead = sum(1 for d in directions if d == "lead")
    n_lag = sum(1 for d in directions if d == "lag")
    n_coupled = sum(1 for d in directions if d == "coupled")
    n_none = sum(1 for d in directions if d == "none")

    n_directional = n_lead + n_lag
    frac_directional = (n_directional / n_qualifying) if n_qualifying > 0 else 0.0

    p_q_to_v, p_v_to_q = panel_result
    p_q_to_v = float(p_q_to_v)
    p_v_to_q = float(p_v_to_q)

    # Dominant per-topic direction. Any exact lead/lag tie (zero OR positive)
    # yields NO dominant direction (there is no panel tie-break): the
    # ``dominant != "none"`` guard below then forces ``holds=False`` — the
    # conservative reading.
    if n_lead > n_lag:
        dominant = "quality_leads"
    elif n_lag > n_lead:
        dominant = "quality_lags"
    else:
        # n_lead == n_lag (including > 0): no dominant direction.
        dominant = "none"

    if dominant == "quality_leads":
        panel_p = p_q_to_v
    elif dominant == "quality_lags":
        panel_p = p_v_to_q
    else:
        panel_p = float("nan")
    panel_agrees = bool(np.isfinite(panel_p) and panel_p < panel_alpha)

    holds = bool(frac_directional >= holds_frac and panel_agrees and dominant != "none")

    return {
        "holds": holds,
        "n_directional": int(n_directional),
        "frac_directional": float(frac_directional),
        "dominant_direction": dominant,
        "panel_agrees": panel_agrees,
        "n_lead": int(n_lead),
        "n_lag": int(n_lag),
        "n_coupled": int(n_coupled),
        "n_none": int(n_none),
        "panel_p_quality_leads": p_q_to_v,
        "panel_p_quality_lags": p_v_to_q,
        "frac_threshold": float(holds_frac),
    }
