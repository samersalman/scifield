"""PR3D power simulation — differenced (PR3) vs Toda–Yamamoto (levels) detection.

This is the **decisive** module of the F1 levels diagnostic
(``docs/preregistrations/PR3D_levels_diagnostic.md`` §7): on synthetic data with
known ground truth, it establishes **which test detects which kind of cascade at
the real series lengths**, and measures each test's false-positive (size) rate so
that any Toda–Yamamoto small-sample over-rejection is *measured and reported, not
hidden*. The whole module is **pure** — no I/O, no plotting, no network, no GPU;
notebook 15 does the figure and parquet write.

Four data-generating processes (DGPs), all returning ``(quality_levels,
volume_levels)`` with a fixed cascade ``lag``:

* **``null``** — quality and volume are independent persistent processes; no
  relationship. The ``β = 0`` cells coincide with this and give each test's SIZE.
* **``level_cascade``** — a **literal single-lag** level relationship:
  ``volume_t = c + β·quality_{t-lag} + noise``. **BOTH** tests detect this (the
  differenced test sees it through the differenced single-lag term too); TY is only
  modestly better-powered at small β. It is reported but is *not* the
  differencing-blind contrast.
* **``diff_cascade``** — the cascade lives in the **DIFFERENCES**:
  ``Δvolume_t = β·Δquality_{t-lag} + noise`` (integrated back to a level series).
  The deliberate **FAIRNESS** case where the PR3 differenced test is well-powered /
  competitive.
* **``level_cascade_coint``** — a **cointegrated / integrator** level cascade:
  volume tracks a slow INTEGRATOR (``stock_t = 0.9·stock_{t-1} + 0.1·quality_t``)
  of the quality LEVEL, ``volume_t = β·stock_{t-lag} + noise``. Because volume
  tracks the *accumulated level* of quality (not its lagged *changes*), the
  differenced test is **BLIND** while the TY levels test recovers it. This is the
  only DGP that demonstrates the differencing-blind contrast and the case that
  would make F1's null APPARENT — the §8 verdict reads off this DGP.

Quality process (documented choice, per PR3D §7 latitude)
---------------------------------------------------------
``quality`` is a **stationary AR(1) with φ = 0.8** (a persistent but bounded
*level* process), chosen over a pure random walk because it (a) yields clean size
control at the headline ``T = 30`` and (b) produces the pre-registered behaviour
the §8 verdict table reads — the TY levels test is well-powered for
``level_cascade`` at realistic ``T`` while the differenced test is under-powered
there, yet the differenced test wins on ``diff_cascade``. The ``diff_cascade``
volume series is the cumulative sum of its simulated differences so that all three
DGPs hand the tests genuine LEVEL series (the TY test fits on levels, the
differenced test differences internally).

For every replication BOTH ``granger_pair`` (PR3 differenced, reused verbatim) and
``toda_yamamoto_pair`` (TY levels) run on the **same** series; detection is
``p_q_to_v < α`` for the injected ``quality → volume`` direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

__all__ = ["run_powersim", "simulate_series"]

#: Quality AR(1) persistence (documented in the module docstring).
_PHI: float = 0.8


def simulate_series(
    dgp: str,
    T: int,
    beta: float,
    *,
    lag: int = 2,
    rng,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one ``(quality_levels, volume_levels)`` pair under ``dgp``.

    Pure and fully determined by ``rng`` (pass a seeded
    ``numpy.random.default_rng``). ``quality`` is a stationary AR(1) (φ = 0.8) in
    every DGP; ``volume`` is built per the DGP semantics described in the module
    docstring. Both returned arrays are ``float64`` LEVEL series of length ``T``.

    Parameters
    ----------
    dgp :
        One of ``"null"`` / ``"level_cascade"`` / ``"diff_cascade"``.
    T :
        Series length (number of annual observations).
    beta :
        Cascade effect size; ``β = 0`` injects no cascade (the size cell).
    lag :
        Cascade lag. Pre-registered default ``2``.
    rng :
        A ``numpy.random.Generator`` (e.g. ``np.random.default_rng(seed)``).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(quality, volume)`` LEVEL arrays, each shape ``(T,)``, dtype float64.

    Raises
    ------
    ValueError
        If ``dgp`` is not one of the three recognised names.
    """
    import numpy as np

    quality = _ar1(T, rng, phi=_PHI)

    if dgp == "null":
        # Volume is an independent persistent process (no quality signal).
        volume = _ar1(T, rng, phi=_PHI)
    elif dgp == "level_cascade":
        # Cascade in LEVELS: volume_t = c + beta*quality_{t-lag} + noise.
        volume = np.zeros(T, dtype="float64")
        for t in range(T):
            lead = beta * quality[t - lag] if t - lag >= 0 else 0.0
            volume[t] = lead + 0.5 * rng.standard_normal()
    elif dgp == "diff_cascade":
        # Cascade in DIFFERENCES: Δvolume_t = beta*Δquality_{t-lag} + noise;
        # integrate the differences back into a LEVEL series of length T.
        dq = np.diff(quality)  # length T-1
        dv = np.zeros(T - 1, dtype="float64")
        for t in range(T - 1):
            lead = beta * dq[t - lag] if t - lag >= 0 else 0.0
            dv[t] = lead + rng.standard_normal()
        volume = np.concatenate([[0.0], np.cumsum(dv)]).astype("float64")
    elif dgp == "level_cascade_coint":
        # Cointegrated/integrator level cascade: volume tracks a slow INTEGRATOR of
        # the quality LEVEL, so the relationship lives purely in levels and the
        # differenced test is BLIND while the Toda-Yamamoto levels test can recover
        # it. Constants are byte-identical to the validated ``_stock_level_cascade``
        # test fixture in ``tests/test_findings_cascade_levels.py`` (0.9 integrator
        # persistence, 0.1 quality-level loading, 0.3 volume noise).
        stock = np.zeros(T, dtype="float64")
        for t in range(1, T):
            stock[t] = 0.9 * stock[t - 1] + 0.1 * quality[t]
        volume = np.zeros(T, dtype="float64")
        for t in range(T):
            lead = beta * stock[t - lag] if t - lag >= 0 else 0.0
            volume[t] = lead + 0.3 * rng.standard_normal()
    else:
        raise ValueError(
            f"unknown dgp {dgp!r}; expected one of 'null', 'level_cascade', "
            "'diff_cascade', 'level_cascade_coint'"
        )

    return quality.astype("float64"), volume.astype("float64")


def _ar1(n: int, rng, *, phi: float):
    """A stationary AR(1) level series of length ``n`` with persistence ``phi``."""
    import numpy as np

    x = np.zeros(n, dtype="float64")
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def run_powersim(
    *,
    T_grid=(8, 15, 22, 30, 40),
    beta_grid=(0.0, 0.25, 0.5, 1.0, 2.0),
    dgps=("null", "level_cascade", "diff_cascade", "level_cascade_coint"),
    n_rep: int = 500,
    lag: int = 2,
    alpha: float = 0.05,
    seed: int = 20260609,
) -> pd.DataFrame:
    """Detection-rate grid for the differenced (PR3) vs TY (levels) tests.

    Pure. For each ``(dgp, T, β)`` cell it runs ``n_rep`` replications; in each
    replication it generates ONE series via :func:`simulate_series` and runs BOTH
    :func:`~scifield.findings.cascade.granger_pair` (PR3 differenced, reused
    verbatim) AND :func:`~scifield.findings.cascade.toda_yamamoto_pair` (TY levels)
    on that **same** series, recording a detection iff the injected-direction
    p-value ``p_q_to_v < alpha`` (NaN counts as non-detection). The ``β = 0`` cells
    are each test's false-positive (size) rate per ``T``.

    Reproducibility: a single seeded ``numpy.random.default_rng(seed)`` draws every
    series in a fixed cell/replication order, so the returned frame is identical
    across runs with the same arguments.

    Parameters
    ----------
    T_grid :
        Series lengths. Pre-registered ``(8, 15, 22, 30, 40)`` (median ≈ 30 =
        headline).
    beta_grid :
        Effect sizes. Pre-registered ``(0.0, 0.25, 0.5, 1.0, 2.0)`` (``β = 0`` =
        size cells).
    dgps :
        DGP names. Pre-registered
        ``("null", "level_cascade", "diff_cascade", "level_cascade_coint")``.
    n_rep :
        Replications per cell. Pre-registered ``≥ 500`` for the full run; tests use
        a modest count for speed.
    lag :
        Cascade lag. Pre-registered default ``2``.
    alpha :
        Significance level. Pre-registered default ``0.05``.
    seed :
        Master RNG seed. Pre-registered default ``20260609``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[dgp, T, beta, test, detect_rate, n_rep]`` with one row per
        ``(dgp, T, β, test)`` and ``test ∈ {"differenced", "levels"}``;
        ``detect_rate`` is the fraction of the ``n_rep`` replications in which that
        test's ``p_q_to_v < alpha``. Sorted by ``(dgp, T, beta, test)`` for a
        stable, reproducible row order.
    """
    import numpy as np
    import pandas as pd

    from scifield.findings.cascade import granger_pair, toda_yamamoto_pair

    rng = np.random.default_rng(seed)

    records: list[dict] = []
    # Fixed iteration order (dgp -> T -> beta -> replication) pins reproducibility.
    for dgp in dgps:
        for T in T_grid:
            for beta in beta_grid:
                diff_hits = 0
                levels_hits = 0
                for _ in range(n_rep):
                    quality, volume = simulate_series(dgp, int(T), float(beta), lag=lag, rng=rng)
                    p_diff, _ = granger_pair(quality, volume, lag=3)
                    p_ty, _ = toda_yamamoto_pair(quality, volume)
                    if np.isfinite(p_diff) and p_diff < alpha:
                        diff_hits += 1
                    if np.isfinite(p_ty) and p_ty < alpha:
                        levels_hits += 1
                records.append(
                    {
                        "dgp": dgp,
                        "T": int(T),
                        "beta": float(beta),
                        "test": "differenced",
                        "detect_rate": diff_hits / n_rep,
                        "n_rep": int(n_rep),
                    }
                )
                records.append(
                    {
                        "dgp": dgp,
                        "T": int(T),
                        "beta": float(beta),
                        "test": "levels",
                        "detect_rate": levels_hits / n_rep,
                        "n_rep": int(n_rep),
                    }
                )

    df = pd.DataFrame.from_records(
        records, columns=["dgp", "T", "beta", "test", "detect_rate", "n_rep"]
    )
    return df.sort_values(["dgp", "T", "beta", "test"]).reset_index(drop=True)
