"""V1-S15 F2-enrichment: dual-novelty archetype-by-impact statistic.

Finding F2 (the dual-novelty 2x2) already **holds** -- Gate G3 (PROCEED, signed
2026-05-31) established the two novelty axes are independent and flagged the
*surprising* impact-by-archetype inversion: the most novel-and-disruptive papers
sit LOWER on age-fair within-year citation impact than incremental ones. This
module does NOT re-gate F2; it quantifies that already-established inversion with
a single omnibus test so the F2 narrative can cite a statistic and a rank-based
effect size.

:func:`kruskal_archetype_impact` runs a Kruskal-Wallis H-test of an impact metric
across the four :data:`~scifield.novelty.archetypes.QUADRANT_LABELS` archetype
groups (a nonparametric ANOVA on ranks -- appropriate because the impact metric
is a bounded within-year percentile, not a Gaussian), and reports the rank-based
effect size epsilon-squared. It is pure and I/O-free: the caller (notebook 12)
does the parquet read and passes the in-memory frame.

Effect size
-----------
``epsilon_squared = (H - k + 1) / (n - k)`` where ``H`` is the Kruskal-Wallis
statistic, ``k`` the number of groups, and ``n`` the total sample size across
groups. This is the rank-based epsilon-squared (Tomczak & Tomczak 2014): it is
~0 under the null (groups drawn from one distribution) and approaches 1 as group
separation grows, on roughly the same 0-1 scale as eta-squared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["kruskal_archetype_impact"]


def kruskal_archetype_impact(
    df: pd.DataFrame,
    *,
    archetype_col: str,
    impact_col: str,
) -> dict:
    """Kruskal-Wallis test of an impact metric across the archetype groups.

    Drops rows whose archetype label *or* impact value is null, groups the
    surviving rows by archetype, and runs a Kruskal-Wallis H-test across the
    groups (one omnibus test, no pairwise correction). Returns the statistic, the
    p-value, the rank-based effect size epsilon-squared, and the per-group median
    and count -- enough to read off (and reconfirm) the impact-by-archetype
    ordering.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame carrying at least ``archetype_col`` and ``impact_col``. Other
        columns are ignored. Not mutated.
    archetype_col : str
        Name of the categorical archetype-label column (e.g. ``"arch_mean_cd5"``).
        Rows with a null label are excluded (they are not a 2x2 group).
    impact_col : str
        Name of the numeric impact column (e.g.
        ``"cited_by_pctile_within_year"``). Rows with a null value are excluded.

    Returns
    -------
    dict
        A dict with:

        ``"H"`` : float
            The Kruskal-Wallis H statistic.
        ``"p_value"`` : float
            The omnibus p-value.
        ``"k"`` : int
            The number of distinct archetype groups actually present after the
            null-drop.
        ``"n"`` : int
            The total number of rows across all groups (the non-null universe).
        ``"epsilon_squared"`` : float
            The rank-based effect size ``(H - k + 1) / (n - k)``.
        ``"per_group"`` : dict[str, dict]
            Per-archetype ``{"median": float, "n": int}``, keyed by label.

    Raises
    ------
    ValueError
        If fewer than two groups remain after dropping nulls (Kruskal-Wallis
        needs at least two groups), or if any surviving group is empty.

    Notes
    -----
    Group order is the column's category order as returned by ``groupby`` (sorted
    labels); the test statistic is order-invariant, so this only affects the
    iteration order of ``per_group``. ``epsilon_squared`` is computed from the
    returned ``H``/``k``/``n`` exactly, so a caller can re-derive it by hand.
    """
    import numpy as np
    from scipy.stats import kruskal

    # Drop rows null on EITHER the archetype label or the impact value.
    work = df[[archetype_col, impact_col]].copy()
    work = work[work[archetype_col].notna() & work[impact_col].notna()]

    # One numeric array per archetype group, in sorted-label order.
    groups: list[np.ndarray] = []
    per_group: dict[str, dict] = {}
    for label, sub in work.groupby(archetype_col, sort=True):
        values = sub[impact_col].to_numpy(dtype="float64")
        if values.size == 0:  # pragma: no cover - notna() filter precludes this
            raise ValueError(f"archetype group {label!r} is empty after the null-drop")
        groups.append(values)
        per_group[str(label)] = {"median": float(np.median(values)), "n": int(values.size)}

    k = len(groups)
    if k < 2:
        raise ValueError(
            f"Kruskal-Wallis needs >= 2 archetype groups; got {k} after dropping nulls."
        )

    n = int(sum(g.size for g in groups))
    h_stat, p_value = kruskal(*groups)

    # Rank-based effect size (Tomczak & Tomczak 2014): ~0 under the null, -> 1 as
    # groups separate. Derived verbatim from H, k, n so it is hand-checkable.
    epsilon_squared = (float(h_stat) - k + 1) / (n - k)

    return {
        "H": float(h_stat),
        "p_value": float(p_value),
        "k": int(k),
        "n": n,
        "epsilon_squared": float(epsilon_squared),
        "per_group": per_group,
    }
