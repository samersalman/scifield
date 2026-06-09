"""Unit tests for the PR3D F1 *levels* diagnostic additions (pure logic).

These cover the three additive functions in :mod:`scifield.findings.cascade`
locked by ``docs/preregistrations/PR3D_levels_diagnostic.md``:

* :func:`toda_yamamoto_pair` — the augmented-VAR (``p = k + d_max = 4``,
  ``trend="c"``) by-hand Wald χ²(``df = k = 3``) Granger test on **LEVEL** series,
  restricting ONLY the first ``k=3`` cross-lags (the ``d_max=1`` augmenting lag is
  estimated but excluded).
* :func:`engle_granger_ecm_pair` — Engle–Granger cointegration + the Δvolume
  error-correction (speed-of-adjustment) loading sign/p (robustness/informational).
* :func:`panel_toda_yamamoto` — the level-panel mirror of ``panel_granger``
  (first-``k`` cross-lag block Wald on levels, with topic fixed effects).

No network / no data files: every fixture is a small hand-built / fixed-seed
``np.random.default_rng`` synthetic series with explicit assertions. The decisive
contrast (the differenced test being BLIND to a pure level cascade that TY
detects) is exercised on a hand-built cointegrated "stock-of-quality" level
cascade. The size / power behaviour at scale lives in
``tests/test_findings_cascade_powersim.py``; this file pins the per-function
contract.
"""

from __future__ import annotations

import numpy as np

from scifield.findings.cascade import (
    engle_granger_ecm_pair,
    granger_pair,
    panel_toda_yamamoto,
    toda_yamamoto_pair,
)

_SEED = 20260609


# --------------------------------------------------------------------------- #
# Synthetic DGP helpers (fixed-seed, hand-documented)
# --------------------------------------------------------------------------- #


def _ar1(n: int, rng, *, phi: float = 0.8, scale: float = 1.0) -> np.ndarray:
    """A persistent stationary AR(1) level series (the quality-level process)."""
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + scale * rng.standard_normal()
    return x


def _direct_level_cascade(n: int, rng, *, beta: float = 2.0, lag: int = 2):
    """quality = AR(1); volume_t = beta*quality_{t-lag} + noise (cascade in LEVELS).

    Both the differenced and the levels test have power here; used to confirm TY
    recovers the injected LEAD with a small p_q_to_v and a large p_v_to_q.
    """
    q = _ar1(n, rng)
    v = np.zeros(n)
    for t in range(n):
        v[t] = (beta * q[t - lag] if t - lag >= 0 else 0.0) + 0.5 * rng.standard_normal()
    return q, v


def _stock_level_cascade(n: int, rng, *, beta: float = 2.0, lag: int = 2):
    """A cointegration-flavoured level cascade the DIFFERENCED test is blind to.

    quality = AR(1); ``stock`` is a slow integrator of quality's LEVEL
    (``stock_t = 0.9*stock_{t-1} + 0.1*quality_t``); volume tracks the lagged
    stock: ``volume_t = beta*stock_{t-lag} + small noise``. Because volume tracks
    the *level* (not the lagged *change*) of quality, the once-differenced Granger
    test has little power while the Toda–Yamamoto levels test detects the lead.
    """
    q = _ar1(n, rng)
    stock = np.zeros(n)
    for t in range(1, n):
        stock[t] = 0.9 * stock[t - 1] + 0.1 * q[t]
    v = np.zeros(n)
    for t in range(n):
        v[t] = (beta * stock[t - lag] if t - lag >= 0 else 0.0) + 0.3 * rng.standard_normal()
    return q, v


def _cointegrated_ecm(n: int, rng, *, beta: float = 2.0, alpha: float = 0.7):
    """Strong error-correction pair: volume reverts to beta*quality each period.

    quality is a slow random walk; ``volume_t = volume_{t-1} +
    alpha*(beta*quality_t - volume_{t-1}) + noise`` so volume actively adjusts
    toward the long-run equilibrium ``beta*quality`` — cointegrated, with a
    NEGATIVE error-correction loading in the Δvolume equation.
    """
    q = np.cumsum(0.5 * rng.standard_normal(n))
    v = np.zeros(n)
    v[0] = beta * q[0]
    for t in range(1, n):
        target = beta * q[t]
        v[t] = v[t - 1] + alpha * (target - v[t - 1]) + 0.3 * rng.standard_normal()
    return q, v


# --------------------------------------------------------------------------- #
# 1. TY recovers an injected level LEAD
# --------------------------------------------------------------------------- #


def test_toda_yamamoto_recovers_injected_level_lead() -> None:
    """A strong quality->volume LEVEL cascade -> small p_q_to_v, large p_v_to_q."""
    rng = np.random.default_rng(_SEED)
    q, v = _direct_level_cascade(60, rng, beta=2.0, lag=2)
    p_q_to_v, p_v_to_q = toda_yamamoto_pair(q, v)
    assert np.isfinite(p_q_to_v) and np.isfinite(p_v_to_q)
    # quality LEADS volume in the levels -> that direction is highly significant.
    assert p_q_to_v < 0.01
    # ... and clearly smaller than the reverse (volume does not lead quality).
    assert p_q_to_v < p_v_to_q
    assert p_v_to_q > 0.05


def test_toda_yamamoto_default_order_and_keywords() -> None:
    """The locked signature defaults (k=3, d_max=1) are honoured and keyword-only."""
    rng = np.random.default_rng(_SEED + 1)
    q, v = _direct_level_cascade(50, rng, beta=2.0, lag=2)
    # Explicit defaults must match the implicit ones byte-for-byte.
    a = toda_yamamoto_pair(q, v)
    b = toda_yamamoto_pair(q, v, k=3, d_max=1)
    assert a[0] == b[0] and a[1] == b[1]


# --------------------------------------------------------------------------- #
# 2. The DIFFERENCED test is BLIND to a pure level-cascade that TY detects
# --------------------------------------------------------------------------- #


def test_differenced_blind_but_ty_detects_level_cascade() -> None:
    """DECISIVE CONTRAST: on a cointegrated stock-of-quality level cascade the
    once-differenced Granger test is NOT significant while the TY levels test IS.

    This is the whole rationale for PR3D: a relationship that lives in the levels
    (volume tracks the accumulated *level* of quality, not its lagged *changes*)
    is invisible to the differenced test PR3 ran but recoverable by Toda–Yamamoto.
    Seed 11 / T=40 is chosen so the contrast is unambiguous on this single
    fixture: the differenced p_q_to_v is ~0.68 (nowhere near significant) while
    the TY levels p_q_to_v is ~0.001 in the correct quality->volume direction.
    """
    rng = np.random.default_rng(11)
    q, v = _stock_level_cascade(40, rng, beta=2.0, lag=2)

    pg_q_to_v, _pg_v_to_q = granger_pair(q, v, lag=3)
    pt_q_to_v, pt_v_to_q = toda_yamamoto_pair(q, v)

    assert np.isfinite(pt_q_to_v)
    # The differenced test fails to flag the (real) quality->volume relationship ...
    assert np.isfinite(pg_q_to_v) and pg_q_to_v > 0.05
    # ... while the TY levels test detects it strongly in the correct direction.
    assert pt_q_to_v < 0.05
    assert pt_q_to_v < pt_v_to_q


# --------------------------------------------------------------------------- #
# 2b. The by-hand Wald R-matrix reproduces statsmodels test_causality
# --------------------------------------------------------------------------- #


def test_byhand_wald_matches_statsmodels_test_causality() -> None:
    """GUARD the R-matrix arithmetic (the highest-risk piece of the batch).

    Fits ``VAR(y).fit(4, trend="c")`` on a fixed-seed 2-col LEVEL series and
    builds the by-hand Wald restricting ALL 4 of the cause's lags in the effect
    equation (df = 4) using the SAME C-order flatten (``row*neqs + eq``) and the
    SAME restriction index (``row = 1 + (lag-1)*neqs + cause_col``) that
    :func:`scifield.findings.cascade._ty_one` uses. With all ``p`` lags
    restricted (not just the first ``k``), this must reproduce statsmodels'
    ``res.test_causality(..., kind="wald")`` on BOTH ``.test_statistic`` and
    ``.pvalue`` to rtol≈1e-6 — the arithmetic statsmodels uses for the full-lag
    restriction. (The ~4 index lines are replicated inline here; ``_ty_one`` is
    NOT modified or imported for its internals.)
    """
    from scipy.stats import chi2
    from statsmodels.tsa.api import VAR

    rng = np.random.default_rng(_SEED)
    n = 60
    q = np.zeros(n)
    v = np.zeros(n)
    for t in range(1, n):
        q[t] = 0.8 * q[t - 1] + rng.standard_normal()
        # genuine cross-dependence so the Wald is well away from zero
        v[t] = 0.5 * q[t - 1] + 0.3 * v[t - 1] + rng.standard_normal()
    y = np.column_stack([q, v])  # col 0 = cause (q), col 1 = effect (v)

    p = 4  # restrict ALL p lags here (df = p = 4), not just the first k
    res = VAR(y).fit(p, trend="c")
    neqs = res.neqs
    params = np.asarray(res.params, dtype="float64")
    beta = params.flatten(order="C")  # cov index = row*neqs + eq (same as _ty_one)
    cov = np.asarray(res.cov_params(), dtype="float64")

    cause_col, effect_col = 0, 1
    # SAME restriction index + C-order flatten _ty_one uses, over all p lags:
    rows = [1 + (lag - 1) * neqs + cause_col for lag in range(1, p + 1)]
    flat_idx = [r * neqs + effect_col for r in rows]
    restriction = np.zeros((len(flat_idx), beta.size), dtype="float64")
    for i, j in enumerate(flat_idx):
        restriction[i, j] = 1.0

    rb = restriction @ beta
    mid = restriction @ cov @ restriction.T
    wald = float(rb.T @ np.linalg.solve(mid, rb))
    pval = float(chi2.sf(wald, df=p))

    # statsmodels restricts ALL lags of the cause in the effect equation.
    sm = res.test_causality(caused=effect_col, causing=cause_col, kind="wald")
    assert np.isclose(
        wald, sm.test_statistic, rtol=1e-6
    ), f"by-hand Wald {wald} != statsmodels {sm.test_statistic}"
    assert np.isclose(pval, sm.pvalue, rtol=1e-6), f"by-hand p {pval} != statsmodels {sm.pvalue}"


# --------------------------------------------------------------------------- #
# 3. Degenerate guards (too-short / singular) -> (nan, nan), never raises
# --------------------------------------------------------------------------- #


def test_toda_yamamoto_too_short_returns_nan_both_directions() -> None:
    """obs <= 2p+2 (=10 at p=4) -> (nan, nan) without raising."""
    # 10 observations: exactly the guard boundary (10 <= 10 -> NaN).
    q = np.arange(10, dtype="float64")
    v = np.arange(10, dtype="float64") * 0.5
    p_q_to_v, p_v_to_q = toda_yamamoto_pair(q, v)
    assert np.isnan(p_q_to_v) and np.isnan(p_v_to_q)


def test_toda_yamamoto_eleven_obs_is_fittable() -> None:
    """11 observations (> 2p+2) is the first fittable length -> finite or NaN, no raise."""
    rng = np.random.default_rng(_SEED + 2)
    q, v = _direct_level_cascade(11, rng, beta=2.0, lag=2)
    p_q_to_v, p_v_to_q = toda_yamamoto_pair(q, v)
    # Must not raise; with this many params it may be finite or NaN (singular), but
    # both must be valid floats in [0, 1] or NaN.
    for p in (p_q_to_v, p_v_to_q):
        assert np.isnan(p) or (0.0 <= p <= 1.0)


def test_toda_yamamoto_constant_series_returns_nan() -> None:
    """A zero-variance (constant) series -> singular VAR -> (nan, nan), no raise."""
    q = np.ones(40)
    v = np.ones(40)
    p_q_to_v, p_v_to_q = toda_yamamoto_pair(q, v)
    assert np.isnan(p_q_to_v) and np.isnan(p_v_to_q)


# --------------------------------------------------------------------------- #
# 5. Engle–Granger ECM: cointegrated pair vs independent pair
# --------------------------------------------------------------------------- #


def test_engle_granger_ecm_detects_cointegration_and_negative_loading() -> None:
    """A strongly error-correcting pair -> cointegrated=True, negative EC coef, small p."""
    rng = np.random.default_rng(1)
    q, v = _cointegrated_ecm(50, rng, beta=2.0, alpha=0.7)
    out = engle_granger_ecm_pair(q, v)
    assert set(out.keys()) == {"cointegrated", "ec_coef", "ec_pvalue", "coint_pvalue"}
    assert out["cointegrated"] is True
    assert out["coint_pvalue"] < 0.05
    # Speed-of-adjustment loading in the Δvolume equation is negative (reverts).
    assert np.isfinite(out["ec_coef"]) and out["ec_coef"] < 0.0
    assert np.isfinite(out["ec_pvalue"]) and out["ec_pvalue"] < 0.05


def test_engle_granger_ecm_independent_pair_not_cointegrated() -> None:
    """Two independent random walks -> not cointegrated; EC fields NaN."""
    rng = np.random.default_rng(123)
    q = np.cumsum(rng.standard_normal(50))
    v = np.cumsum(rng.standard_normal(50))
    out = engle_granger_ecm_pair(q, v)
    assert out["cointegrated"] is False
    assert out["coint_pvalue"] >= 0.05
    # Not cointegrated -> EC coefficient / p are NaN (no ECM fitted).
    assert np.isnan(out["ec_coef"]) and np.isnan(out["ec_pvalue"])


def test_engle_granger_ecm_degenerate_returns_nan_false() -> None:
    """A too-short / constant pair -> not cointegrated, NaN EC fields, no raise."""
    out = engle_granger_ecm_pair(np.ones(6), np.arange(6, dtype="float64"))
    assert out["cointegrated"] is False
    assert np.isnan(out["ec_coef"]) and np.isnan(out["ec_pvalue"])


# --------------------------------------------------------------------------- #
# 6. panel_toda_yamamoto recovers an injected level lead across topics
# --------------------------------------------------------------------------- #


def test_panel_toda_yamamoto_recovers_dominant_level_direction() -> None:
    """Pooled topics with a quality->volume LEVEL lead -> panel p_q_to_v << p_v_to_q."""
    rng = np.random.default_rng(_SEED)
    level_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(8):
        q, v = _direct_level_cascade(40, rng, beta=2.0, lag=2)
        level_pairs.append((q, v))
    p_q_to_v, p_v_to_q = panel_toda_yamamoto(level_pairs)
    assert np.isfinite(p_q_to_v) and np.isfinite(p_v_to_q)
    assert p_q_to_v < 0.05
    assert p_q_to_v < p_v_to_q


def test_panel_toda_yamamoto_default_keywords() -> None:
    """Explicit (k=3, d_max=1) matches the defaulted call."""
    rng = np.random.default_rng(_SEED + 3)
    pairs = [_direct_level_cascade(40, rng, beta=2.0, lag=2) for _ in range(5)]
    a = panel_toda_yamamoto(pairs)
    b = panel_toda_yamamoto(pairs, k=3, d_max=1)
    assert a[0] == b[0] and a[1] == b[1]


def test_panel_toda_yamamoto_empty_input_returns_nan() -> None:
    """No usable topic series -> (nan, nan), no raise (mirrors panel_granger)."""
    p_q_to_v, p_v_to_q = panel_toda_yamamoto([])
    assert np.isnan(p_q_to_v) and np.isnan(p_v_to_q)


def test_panel_toda_yamamoto_skips_too_short_topics() -> None:
    """Topics too short to contribute a lagged row are skipped, not raised on."""
    rng = np.random.default_rng(_SEED + 4)
    pairs = [_direct_level_cascade(40, rng, beta=2.0, lag=2) for _ in range(6)]
    # Add two degenerate too-short pairs that must be silently skipped.
    pairs.append((np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0])))
    pairs.append((np.array([0.0, 0.0]), np.array([0.0, 0.0])))
    p_q_to_v, p_v_to_q = panel_toda_yamamoto(pairs)
    assert np.isfinite(p_q_to_v) and np.isfinite(p_v_to_q)
    assert p_q_to_v < 0.05
