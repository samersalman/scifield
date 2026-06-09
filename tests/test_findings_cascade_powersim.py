"""Unit tests for the PR3D power-simulation module (pure, fixed-seed).

Covers ``scifield.findings.cascade_powersim`` locked by
``docs/preregistrations/PR3D_levels_diagnostic.md`` §7:

* :func:`simulate_series` — the four DGPs (``null`` / ``level_cascade`` /
  ``diff_cascade`` / ``level_cascade_coint``) returning ``(quality_levels,
  volume_levels)``. ``level_cascade_coint`` is the cointegrated integrator DGP the
  §8 verdict reads off — differencing-blind, TY-recoverable.
* :func:`run_powersim` — runs BOTH ``granger_pair`` (PR3 differenced) AND
  ``toda_yamamoto_pair`` (TY levels) on the SAME simulated series for every
  ``(dgp, T, beta)`` cell and returns a tidy detection-rate frame.

No network / no data files; modest ``n_rep`` for speed (the full ``n_rep=500``
run happens later in notebook 15). The size (β=0) test uses ``n_rep>=400``.

Size behaviour observed empirically (with the AR(1) φ=0.8 base, chi²(df=k) Wald
mandated by the spec — NOT the size-corrected F form):

* differenced test: ≈0.05 at T ∈ {22, 30, 40} (clean).
* TY levels test: ≈0.05 at the HEADLINE T=30 / T=40, but the χ² Wald
  **over-rejects at short T** (T=15, and to a lesser extent T=22). This is the
  documented small-sample inflation the plan anticipates; it is RECORDED here,
  and only the headline T ∈ {30, 40} are hard-asserted for TY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scifield.findings.cascade_powersim import run_powersim, simulate_series

_SEED = 20260609

# Loose size tolerance: chi²(df=k) Wald is asymptotic; mild over-rejection at
# realistic finite T is expected and reportable. Headline TY cells (T>=30) and
# all differenced cells must sit within this band.
_SIZE_TOL = 0.10


# --------------------------------------------------------------------------- #
# simulate_series — shapes, dtypes, determinism, DGP semantics
# --------------------------------------------------------------------------- #


def test_simulate_series_shapes_and_dtypes() -> None:
    """Each DGP returns two equal-length float64 LEVEL arrays of length T."""
    for dgp in ("null", "level_cascade", "diff_cascade"):
        rng = np.random.default_rng(_SEED)
        q, v = simulate_series(dgp, T=30, beta=1.0, lag=2, rng=rng)
        assert isinstance(q, np.ndarray) and isinstance(v, np.ndarray)
        assert q.shape == (30,) and v.shape == (30,)
        assert q.dtype == np.float64 and v.dtype == np.float64
        assert np.all(np.isfinite(q)) and np.all(np.isfinite(v))


def test_simulate_series_is_deterministic_given_rng() -> None:
    """Same seed -> byte-identical series (the simulation is reproducible)."""
    q1, v1 = simulate_series("level_cascade", T=30, beta=1.0, rng=np.random.default_rng(_SEED))
    q2, v2 = simulate_series("level_cascade", T=30, beta=1.0, rng=np.random.default_rng(_SEED))
    assert np.array_equal(q1, q2) and np.array_equal(v1, v2)


def test_simulate_series_unknown_dgp_raises() -> None:
    """An unknown DGP name is a programming error -> ValueError."""
    try:
        simulate_series("not_a_dgp", T=30, beta=1.0, rng=np.random.default_rng(_SEED))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown dgp")


def test_simulate_series_null_quality_does_not_drive_volume() -> None:
    """Under 'null', volume carries no quality signal -> TY direction not significant on average."""
    from scifield.findings.cascade import toda_yamamoto_pair

    rng = np.random.default_rng(_SEED)
    sig = 0
    n = 200
    for _ in range(n):
        q, v = simulate_series("null", T=40, beta=0.0, rng=rng)
        p_q_to_v, _ = toda_yamamoto_pair(q, v)
        sig += int(np.isfinite(p_q_to_v) and p_q_to_v < 0.05)
    # Size at T=40 for TY is mild; well under half the replications.
    assert sig / n < _SIZE_TOL


def test_simulate_series_beta_zero_is_null_like() -> None:
    """beta=0 in level_cascade / diff_cascade injects no cascade (the size cells)."""
    from scifield.findings.cascade import granger_pair

    rng = np.random.default_rng(_SEED)
    sig = 0
    n = 300
    for _ in range(n):
        q, v = simulate_series("level_cascade", T=40, beta=0.0, rng=rng)
        p_q_to_v, _ = granger_pair(q, v, lag=3)
        sig += int(np.isfinite(p_q_to_v) and p_q_to_v < 0.05)
    assert sig / n < _SIZE_TOL


# --------------------------------------------------------------------------- #
# run_powersim — frame schema
# --------------------------------------------------------------------------- #


def test_run_powersim_frame_schema_and_cells() -> None:
    """The returned frame has the locked columns and one row per (dgp,T,beta,test)."""
    df = run_powersim(
        T_grid=(22, 30),
        beta_grid=(0.0, 1.0),
        dgps=("null", "level_cascade"),
        n_rep=40,
        lag=2,
        alpha=0.05,
        seed=_SEED,
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["dgp", "T", "beta", "test", "detect_rate", "n_rep"]
    # One row per (dgp, T, beta, test) with test in {differenced, levels}.
    assert set(df["test"].unique()) == {"differenced", "levels"}
    # 2 dgps * 2 T * 2 beta * 2 tests = 16 rows.
    assert len(df) == 16
    assert (df["n_rep"] == 40).all()
    assert ((df["detect_rate"] >= 0.0) & (df["detect_rate"] <= 1.0)).all()
    assert set(df["dgp"].unique()) == {"null", "level_cascade"}


def test_run_powersim_is_reproducible() -> None:
    """Same seed -> identical detection-rate frame (run-once reproducibility)."""
    a = run_powersim(
        T_grid=(30,),
        beta_grid=(0.0, 1.0),
        dgps=("level_cascade",),
        n_rep=50,
        lag=2,
        alpha=0.05,
        seed=_SEED,
    )
    b = run_powersim(
        T_grid=(30,),
        beta_grid=(0.0, 1.0),
        dgps=("level_cascade",),
        n_rep=50,
        lag=2,
        alpha=0.05,
        seed=_SEED,
    )
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# 3. Size control on the null DGP (the decisive size requirement)
# --------------------------------------------------------------------------- #


def test_size_control_on_null_dgp_headline_lengths() -> None:
    """β=0 false-positive rate ≈ 0.05 at the HEADLINE T ∈ {30, 40} for BOTH tests.

    The differenced test is additionally clean at T=22; the TY χ² Wald is only
    asserted at the headline T>=30 (its short-T over-rejection is recorded by
    ``test_short_T_size_is_recorded`` below, not hard-asserted, per the plan).
    """
    df = run_powersim(
        T_grid=(22, 30, 40),
        beta_grid=(0.0,),
        dgps=("null",),
        n_rep=500,
        lag=2,
        alpha=0.05,
        seed=_SEED,
    )
    size = df.set_index(["test", "T"])["detect_rate"]

    # Differenced test: clean at all three realistic lengths.
    for t in (22, 30, 40):
        assert (
            size[("differenced", t)] <= _SIZE_TOL
        ), f"differenced size at T={t} = {size[('differenced', t)]:.3f} > {_SIZE_TOL}"

    # TY levels test: headline lengths only (T >= 30).
    for t in (30, 40):
        assert size[("levels", t)] <= _SIZE_TOL, (
            f"TY levels size at T={t} = {size[('levels', t)]:.3f} > {_SIZE_TOL} "
            "(headline lengths must control size)"
        )


def test_short_T_size_is_recorded_not_asserted() -> None:
    """RECORD (do not hard-assert ≈0.05) the TY short-T size at T ∈ {8, 15}.

    The plan anticipates TY χ² Wald over-rejection at very short T; this test
    simply confirms those size cells are present and finite so notebook 15 can
    report them openly. It is a documentation guard, not a pass/fail on the rate.
    """
    df = run_powersim(
        T_grid=(8, 15),
        beta_grid=(0.0,),
        dgps=("null",),
        n_rep=400,
        lag=2,
        alpha=0.05,
        seed=_SEED,
    )
    size = df.set_index(["test", "T"])["detect_rate"]
    for t in (8, 15):
        for test in ("differenced", "levels"):
            assert np.isfinite(size[(test, t)])
            assert 0.0 <= size[(test, t)] <= 1.0
    # T=8 <= 2p+2: TY cannot fit a VAR(4) -> NaN guard -> 0 detections recorded.
    assert size[("levels", 8)] == 0.0
    # At T=15 the TY χ² Wald is expected to over-reject (documented finding):
    # we record it is materially above nominal, not that it equals 0.05.
    assert size[("levels", 15)] > size[("differenced", 15)]


# --------------------------------------------------------------------------- #
# 4. The differenced test DETECTS a diff_cascade (fairness sanity)
# --------------------------------------------------------------------------- #


def test_differenced_test_detects_diff_cascade() -> None:
    """FAIRNESS: on diff_cascade the differenced test has good power at T=30.

    Confirms the differenced test is the right tool for a difference-cascade and
    is not simply always blind — guarding against a simulation rigged to favour TY.
    """
    df = run_powersim(
        T_grid=(30,),
        beta_grid=(2.0,),
        dgps=("diff_cascade",),
        n_rep=200,
        lag=2,
        alpha=0.05,
        seed=_SEED,
    )
    diff_power = df[(df["test"] == "differenced") & (df["T"] == 30)]["detect_rate"].iloc[0]
    assert diff_power > 0.5


def test_ty_powered_for_level_cascade_at_headline_T() -> None:
    """SANITY for the verdict table: TY is well-powered for level_cascade at T=30,
    and at least as powered as the differenced test there (the level-cascade case)."""
    df = run_powersim(
        T_grid=(30,),
        beta_grid=(1.0,),
        dgps=("level_cascade",),
        n_rep=200,
        lag=2,
        alpha=0.05,
        seed=_SEED,
    )
    sub = df.set_index("test")["detect_rate"]
    assert sub["levels"] > 0.5
    # On a level cascade, TY should detect at least as often as the differenced test.
    assert sub["levels"] >= sub["differenced"] - 0.05


# --------------------------------------------------------------------------- #
# 5. The DISCRIMINATING DGP: level_cascade_coint is differencing-blind
# --------------------------------------------------------------------------- #


def test_level_cascade_coint_is_differencing_blind_at_headline_T() -> None:
    """DECISIVE CONTRAST the §8 verdict reads off: on the cointegrated integrator
    DGP (``level_cascade_coint``) at the headline T=30, the LEVELS (TY) detection
    rate clearly exceeds the DIFFERENCED detection rate.

    This guards the grid's only differencing-blind DGP — volume tracks a slow
    integrator of the quality LEVEL, so the differenced test is under-powered while
    the Toda–Yamamoto levels test recovers the lead. The verdict table (PR3D §8.1)
    routes its "differenced under-powered / levels well-powered" leg through THIS
    DGP at T=30, so the contrast must stay test-guarded. (Modest n_rep for speed;
    the full n_rep=500 run lives in notebook 15.)
    """
    df = run_powersim(
        T_grid=(30,),
        beta_grid=(2.0,),
        dgps=("level_cascade_coint",),
        n_rep=120,
        lag=2,
        alpha=0.05,
        seed=_SEED,
    )
    sub = df.set_index("test")["detect_rate"]
    # Levels test is well-powered here ...
    assert sub["levels"] > 0.6
    # ... while the differenced test is materially under-powered (the blindness),
    # so levels clearly exceeds differenced — the differencing-blind contrast.
    assert sub["levels"] > sub["differenced"] + 0.2
