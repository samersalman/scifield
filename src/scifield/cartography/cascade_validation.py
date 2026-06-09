"""V2 cartography — cascade **validation** (the make-or-break rigor layer).

This module implements the four rigor methods of the locked
``V2/docs/cartography/validation_protocol.md`` against the V2-S03 cascade outputs,
to decide whether the inter-journal cascade structure is **real** or a
**panel / temporal artifact** (the primary input to Gate G6). It is pure and
I/O-free: every function takes in-memory frames and returns a frame or a verdict
dict; reading parquet, writing tables, and drawing live in
``V2/scripts/run_cascade_validation.py`` and the notebook.

The four methods (validation_protocol §1–4)
-------------------------------------------
* **Method B — null / permutation** (:func:`permutation_verdict`). The
  gate-critical check. Build the permutation null with
  :func:`scifield.cartography.nulls.permute_labels` at the locked
  ``n_perm=1000`` / ``seed=20260609``; the primary null is **Null-1** (journal
  label shuffle *within topic*), the secondary is **Null-2** (first-appearance
  year shuffle within topic). The primary test statistic is **S1** (the spread —
  population std — across the ten per-journal seeding scores); the secondary is
  **S2** (net directed lead-lag asymmetry summed over journal pairs). The add-one,
  upper-tail p-value plus a z-score and percentile come from
  :func:`scifield.cartography.nulls.permutation_test`.
* **Method C — drop-one-journal jackknife** (:func:`jackknife_origins`). Re-run
  :func:`scifield.cartography.cascade.origin_attribution` ten times, each omitting
  one journal, and measure ``attribution_flip_rate`` — but a topic whose origin
  journal *is* the dropped journal is **forced** to reattribute, so those cells are
  excluded from the flip-rate numerator and counted separately as
  ``forced_reattribution_count`` (validation_protocol §3). This module owns the
  jackknife of **origin attributions**; V2-S05 owns the jackknife of **role
  scores** (``role_rank_correlation``).
* **Method D — granularity sensitivity** (:func:`granularity_consistency`). The
  per-journal seeding-score ranking at the leaf grain (149 ``topic_id``) vs the mid
  grain (96 ``mid_level_id``); the comparison statistic is their Spearman ρ against
  the 0.5 robustness bar.
* **Method A — held-out years** (:func:`heldout_consistency`). *Supporting only.*
  Per-journal seeding-score ranking learned on 1995–2018 vs recomputed on
  2019–2025; Spearman ρ. Topics first appearing after 2018 are unscorable in the
  train window and excluded (count reported).

The 1995 left-censoring threat (carry into every verdict)
---------------------------------------------------------
The corpus starts in 1995, so ~80% of topics are "all-tied at 1995" — present in
several journals at panel start with no resolvable ordering (S03 caveat #1). A 1995
origin means **"present at panel start"**, never "born in 1995". The within-topic
label shuffle (Null-1) preserves each topic's year footprint and so *partly*
controls for this, but every method here also reports a **resolvable-subset**
sensitivity: the same verdict recomputed on the topics with ≥ 2 distinct
first-appearance years (:func:`resolvable_topics`). If the structure survives only
on the all-tied topics, it is an artifact and must be reported as such.

Conventions: ``from __future__ import annotations``; numpy-style docstrings; numpy +
pandas + scipy only; ruff/black line-length 100. Real counts are COMPUTED, never
hardcoded. Locked constants live in :data:`N_PERM`, :data:`SEED`,
:data:`PANEL_JOURNALS`, :data:`HELDOUT_TRAIN_MAX_YEAR`, :data:`HELDOUT_MIN_YEAR`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "GRANULARITY_RHO_BAR",
    "HELDOUT_MIN_YEAR",
    "HELDOUT_TRAIN_MAX_YEAR",
    "N_PERM",
    "PANEL_JOURNALS",
    "SEED",
    "granularity_consistency",
    "heldout_consistency",
    "jackknife_origins",
    "permutation_verdict",
    "resolvable_topics",
    "run_all_validations",
    "seeding_spread_s1",
    "leadlag_asymmetry_s2",
]

# --------------------------------------------------------------------------- #
# Locked constants (validation_protocol §0). Do NOT silently change — if a value
# looks wrong, implement it as locked and flag the concern in the doc + log.
# --------------------------------------------------------------------------- #
N_PERM: int = 1000
SEED: int = 20260609
GRANULARITY_RHO_BAR: float = 0.5
HELDOUT_TRAIN_MAX_YEAR: int = 2018
HELDOUT_MIN_YEAR: int = 2019

# The canonical ten journal slugs (V2/CONTEXT.md §3). S1 is the spread across these
# ten per-journal seeding scores; reindexing to this fixed panel keeps the spread
# comparable across permutations (a permutation that empties a journal's score is
# simply absent from the spread, never imputed).
PANEL_JOURNALS: tuple[str, ...] = (
    "ann_surg",
    "arthroscopy",
    "br_j_surg",
    "clin_orthop_relat_res",
    "j_am_coll_surg",
    "j_arthroplasty",
    "j_bone_joint_surg_am",
    "jama_surg",
    "spine",
    "surgery",
)


# --------------------------------------------------------------------------- #
# Test statistics (S1 primary, S2 secondary). Pure, computed from a flow slice.
# --------------------------------------------------------------------------- #
def seeding_spread_s1(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
    panel: tuple[str, ...] = PANEL_JOURNALS,
) -> float:
    """S1 — spread (population std) across the per-journal seeding scores.

    The primary cascade statistic (validation_protocol §2). Each journal's
    seeding score is its mean normalized-lead across topics
    (:func:`scifield.findings.seeding.seeding_score`); a *real* cascade makes some
    journals consistent leaders and others consistent followers, so the ten scores
    spread out (high std). A no-ordering null collapses them toward a common mean
    (low std).

    Parameters
    ----------
    flow_or_first :
        A flow-table slice carrying at least ``[topic_key, "journal_slug",
        "year"]`` (or a frame with ``"first_year"``, which is renamed to ``"year"``
        for the seeding primitive — the per-topic ordering is identical either
        way).
    topic_key :
        Topic-grain column (``"topic_id"`` leaf / ``"mid_level_id"`` mid). Default
        ``"topic_id"``.
    panel :
        The journal slugs the spread is taken over. Default :data:`PANEL_JOURNALS`
        (the ten). Journals absent from the slice are dropped from the spread (never
        imputed), so a permutation that happens to leave a journal scoreless simply
        narrows the panel for that replicate.

    Returns
    -------
    float
        The population standard deviation (``ddof=0``) of the per-journal seeding
        scores over the journals present in the slice. ``0.0`` if fewer than two
        journals are scored.

    Notes
    -----
    Pure. Uses ``ddof=0`` so the statistic and its permutation replicates are on the
    same footing (the protocol's "spread", not an inferential sample std).
    """
    import numpy as np

    from scifield.findings.seeding import seeding_score

    work = flow_or_first
    if "year" not in work.columns and "first_year" in work.columns:
        work = work.rename(columns={"first_year": "year"})
    if topic_key != "topic_id":
        work = work.rename(columns={topic_key: "topic_id"})

    scores = seeding_score(work, entity_col="journal_slug")
    if scores.empty:
        return 0.0
    s = scores.set_index("journal_slug")["seeding_score"].reindex(list(panel))
    vals = s.to_numpy(dtype="float64")
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return 0.0
    return float(np.std(vals, ddof=0))


def leadlag_asymmetry_s2(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
) -> float:
    """S2 — net directed lead-lag asymmetry summed over journal pairs.

    The secondary cascade statistic (validation_protocol §2). From the directed
    seeding network (:func:`scifield.cartography.cascade.cascade_edges`, edge
    ``weight = n_precedes / n_shared``), sum over **unordered** journal pairs of
    ``|weight(A->B) - weight(B->A)|``. A real cascade has directional edges (A leads
    B more than B leads A) → high asymmetry; a symmetric, no-ordering structure →
    low asymmetry.

    Parameters
    ----------
    flow_or_first :
        A flow-table slice or first-appearance frame (``"first_year"`` is renamed to
        ``"year"`` for the edge builder).
    topic_key :
        Topic-grain column. Default ``"topic_id"``.

    Returns
    -------
    float
        ``sum_{A<B} |weight(A->B) - weight(B->A)|`` over journal pairs that share at
        least one topic. ``0.0`` when no two journals co-occur.

    Notes
    -----
    Each unordered pair is counted once. Pairs where one direction is absent treat
    the missing weight as ``0.0``. Pure.
    """
    from scifield.cartography.cascade import cascade_edges

    work = flow_or_first
    if "year" not in work.columns and "first_year" in work.columns:
        work = work.rename(columns={"first_year": "year"})

    edges = cascade_edges(work, topic_key=topic_key)
    if edges.empty:
        return 0.0

    weights = {(row.src, row.dst): float(row.weight) for row in edges.itertuples()}
    total = 0.0
    seen: set[frozenset[str]] = set()
    for (src, dst), w_fwd in weights.items():
        pair = frozenset((src, dst))
        if pair in seen:
            continue
        seen.add(pair)
        w_rev = weights.get((dst, src), 0.0)
        total += abs(w_fwd - w_rev)
    return float(total)


_STATS: dict[str, Callable[..., float]] = {
    "S1": seeding_spread_s1,
    "S2": leadlag_asymmetry_s2,
}


# --------------------------------------------------------------------------- #
# 1995 left-censoring: resolvable-subset filter.
# --------------------------------------------------------------------------- #
def resolvable_topics(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
) -> list:
    """Topic ids with a *resolvable* ordering (≥ 2 distinct first-appearance years).

    The 1995 left-censoring guard (S03 caveat #1): a topic that first appears in
    every one of its journals in the same year (typically 1995, the panel start) has
    **no resolvable lead-lag** — it is "all-tied at panel start", not a genuine
    cascade. This returns the topics with at least two distinct per-journal
    first-appearance years, so a caller can recompute any verdict on the subset that
    actually carries ordering information.

    Parameters
    ----------
    flow_or_first :
        A flow-table slice or first-appearance frame.
    topic_key :
        Topic-grain column. Default ``"topic_id"``.

    Returns
    -------
    list
        The topic ids (values of ``topic_key``) with ≥ 2 distinct first-appearance
        years, sorted.

    Notes
    -----
    Pure. A 1995 origin is read as "present at panel start", never "born in 1995";
    excluding all-tied topics is the sensitivity that proves the structure does not
    live only in the year-one ties.
    """
    from scifield.cartography.cascade import _first_appearance_frame

    first = _first_appearance_frame(flow_or_first, topic_key=topic_key)
    if first.empty:
        return []
    distinct_years = first.groupby(topic_key)["first_year"].nunique()
    keep = distinct_years[distinct_years >= 2].index
    return sorted(keep.tolist())


def _restrict_to_resolvable(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str,
) -> pd.DataFrame:
    """Filter a flow slice to its resolvable topics (helper for the verdicts)."""
    keep = set(resolvable_topics(flow_or_first, topic_key=topic_key))
    return flow_or_first[flow_or_first[topic_key].isin(keep)].copy()


# --------------------------------------------------------------------------- #
# Method B — null / permutation verdict.
# --------------------------------------------------------------------------- #
def permutation_verdict(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str,
    stat: str = "S1",
    null: str = "within_topic",
    n_perm: int = N_PERM,
    seed: int = SEED,
    resolvable_only: bool = False,
) -> dict:
    """Permutation-null verdict for one (statistic, null) pair (Method B).

    Computes the observed statistic, builds ``n_perm`` permutation replicates with
    :func:`scifield.cartography.nulls.permute_labels`, and returns the upper-tail
    add-one p-value, z-score, and percentile from
    :func:`scifield.cartography.nulls.permutation_test`, enriched with the observed
    statistic, the number of topics used, and the resolvable-subset flag.

    Parameters
    ----------
    flow_or_first :
        A flow-table slice (``[topic_key, "journal_slug", "year"]``).
    topic_key :
        Topic-grain column (``"topic_id"`` leaf / ``"mid_level_id"`` mid).
    stat :
        ``"S1"`` (seeding-score spread, primary) or ``"S2"`` (lead-lag asymmetry,
        secondary). Default ``"S1"``.
    null :
        ``"within_topic"`` → **Null-1**: shuffle ``"journal_slug"`` within each topic
        (the primary null; destroys who-leads-whom while preserving each topic's year
        footprint and per-journal volume). ``"year_shuffle"`` → **Null-2**: shuffle
        the per-journal first-appearance ``"first_year"`` within each topic (the
        journal set fixed, the ordering destroyed). Default ``"within_topic"``.
    n_perm :
        Number of permutation replicates. Default :data:`N_PERM` (1000).
    seed :
        Base RNG seed; replicate ``i`` uses ``seed + i`` (a reproducible stream).
        Default :data:`SEED`.
    resolvable_only :
        If ``True``, restrict to topics with ≥ 2 distinct first-appearance years
        (:func:`resolvable_topics`) before computing — the 1995-censoring
        sensitivity. Default ``False`` (all topics).

    Returns
    -------
    dict
        ``{"stat", "null", "observed", "p_value", "z", "percentile", "n_perm",
        "n_topics", "resolvable_only", "null_mean", "null_std", "verdict"}``.

        * ``observed`` — the statistic on the real data.
        * ``p_value`` — add-one upper-tail (``"greater"``) p (never exactly 0).
        * ``z`` — ``(observed - mean(null)) / std(null)``.
        * ``percentile`` — fraction of null replicates strictly below ``observed``
          (in ``[0, 1]``).
        * ``n_topics`` — topics contributing to the observed statistic.
        * ``verdict`` — ``"PASS"`` (``p < 0.01`` and ``z >= 2``), ``"PARTIAL"``
          (``p < 0.05`` but ``z < 2``), else ``"FAIL"`` — the per-statistic call
          (validation_protocol §2). The *overall* Method-B rubric (S1/Null-1 is
          gate-critical; S2 / Null-2 corroborate) is assembled in
          :func:`run_all_validations`.

    Raises
    ------
    ValueError
        If ``stat`` or ``null`` is not recognised.

    Notes
    -----
    For **Null-2** the statistic is computed on the *first-appearance frame* (one
    row per (topic, journal)) so shuffling ``first_year`` within a topic is a clean
    permutation of the per-journal years; for **Null-1** the journal label is
    shuffled on the flow slice itself, then first-appearance is recomputed downstream
    by the statistic (year-subset-safe by construction). Pure: no global RNG.
    """
    import numpy as np

    from scifield.cartography import nulls
    from scifield.cartography.cascade import _first_appearance_frame

    if stat not in _STATS:
        raise ValueError(f"stat {stat!r} must be one of {sorted(_STATS)}")
    if null not in {"within_topic", "year_shuffle"}:
        raise ValueError(f"null {null!r} must be 'within_topic' or 'year_shuffle'")

    work = flow_or_first
    if resolvable_only:
        work = _restrict_to_resolvable(work, topic_key=topic_key)

    stat_fn = _STATS[stat]
    observed = stat_fn(work, topic_key=topic_key)
    n_topics = int(work[topic_key].nunique()) if not work.empty else 0

    if null == "within_topic":
        # Null-1: shuffle journal labels within topic on the flow slice.
        base = work
        label_col, group_col = "journal_slug", topic_key
    else:
        # Null-2: shuffle per-journal first-appearance years within topic.
        base = _first_appearance_frame(work, topic_key=topic_key)
        label_col, group_col = "first_year", topic_key

    null_stats = np.empty(n_perm, dtype="float64")
    for i in range(n_perm):
        replicate = nulls.permute_labels(
            base, label_col=label_col, group_col=group_col, seed=seed + i
        )
        null_stats[i] = stat_fn(replicate, topic_key=topic_key)

    test = nulls.permutation_test(observed, null_stats, alternative="greater")
    valid = null_stats[~np.isnan(null_stats)]
    percentile = float(np.mean(valid < observed)) if valid.size else float("nan")
    null_mean = float(np.mean(valid)) if valid.size else float("nan")
    null_std = float(np.std(valid, ddof=0)) if valid.size else float("nan")

    p_value = float(test["p_value"])
    z = float(test["z"])
    if p_value < 0.01 and (z >= 2.0):
        verdict = "PASS"
    elif p_value < 0.05:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "stat": stat,
        "null": null,
        "observed": float(observed),
        "p_value": p_value,
        "z": z,
        "percentile": percentile,
        "n_perm": int(test["n_perm"]),
        "n_topics": n_topics,
        "resolvable_only": bool(resolvable_only),
        "null_mean": null_mean,
        "null_std": null_std,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Method C — drop-one-journal jackknife of ORIGIN attributions.
# --------------------------------------------------------------------------- #
def jackknife_origins(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str,
    journals: tuple[str, ...] = PANEL_JOURNALS,
) -> pd.DataFrame:
    """Drop-one-journal origin attributions across the ten leave-one-out runs.

    Re-runs :func:`scifield.cartography.cascade.origin_attribution` once per journal
    in ``journals``, each time excluding that journal, and stacks the per-(topic,
    held_out) origin together with the **full-run** origin baseline. This is the raw
    material :func:`summarize_jackknife_origins` reduces to
    ``attribution_flip_rate`` + ``forced_reattribution_count`` (Method C,
    validation_protocol §3).

    Parameters
    ----------
    flow_or_first :
        A flow-table slice or first-appearance frame for one grain.
    topic_key :
        Topic-grain column (``"topic_id"`` leaf / ``"mid_level_id"`` mid).
    journals :
        The journal slugs to drop one at a time. Default :data:`PANEL_JOURNALS`
        (the ten). Only journals actually present are dropped.

    Returns
    -------
    pandas.DataFrame
        One row per (held_out journal, topic), columns ``["held_out", topic_key,
        "origin_journal_slug", "full_origin", "forced"]``: the leave-one-out origin,
        the full-run origin, and ``forced`` (``True`` iff the dropped journal *was*
        the topic's full-run origin → it must reattribute). Topics absent from a run
        (e.g. the topic only ever touched the dropped journal) carry ``NaN`` origin.

    Notes
    -----
    Pure orchestration via :func:`scifield.cartography.nulls.jackknife_drop_one`. The
    deterministic alphabetical tie-break (cascade engine) makes a non-forced
    leave-one-out origin invariant unless the drop changes which journal is earliest
    — so flips are genuine, not tie-noise.
    """
    import pandas as pd

    from scifield.cartography import nulls
    from scifield.cartography.cascade import origin_attribution

    full = origin_attribution(flow_or_first, topic_key=topic_key)
    full_origin = full.set_index(topic_key)["origin_journal_slug"]
    present = [j for j in journals if j in set(flow_or_first["journal_slug"].unique())]

    def _run(held_out: object) -> pd.DataFrame:
        sub = flow_or_first[flow_or_first["journal_slug"] != held_out]
        o = origin_attribution(sub, topic_key=topic_key)
        return o[[topic_key, "origin_journal_slug"]]

    stacked = nulls.jackknife_drop_one(present, _run)
    if stacked.empty:
        return pd.DataFrame(
            columns=["held_out", topic_key, "origin_journal_slug", "full_origin", "forced"]
        )

    stacked = stacked.merge(
        full_origin.rename("full_origin"),
        left_on=topic_key,
        right_index=True,
        how="left",
    )
    stacked["forced"] = stacked["held_out"] == stacked["full_origin"]
    cols = ["held_out", topic_key, "origin_journal_slug", "full_origin", "forced"]
    return stacked[cols].reset_index(drop=True)


def summarize_jackknife_origins(
    per_run: pd.DataFrame,
    *,
    topic_key: str,
) -> dict:
    """Reduce the jackknife-origins frame to flip rate + forced count (Method C).

    Computes the **residual** ``attribution_flip_rate`` (origin changes in
    *non-forced* leave-one-out runs, relative to the full-run origin) and the
    ``forced_reattribution_count`` (cells where the dropped journal *was* the
    topic's origin, excluded from the flip numerator), per validation_protocol §3.

    Parameters
    ----------
    per_run :
        The output of :func:`jackknife_origins` — columns ``["held_out",
        topic_key, "origin_journal_slug", "full_origin", "forced"]``.
    topic_key :
        Topic-grain column.

    Returns
    -------
    dict
        ``{"attribution_flip_rate", "forced_reattribution_count",
        "n_eligible_cells", "n_genuine_flips", "n_topics_flipped", "n_topics",
        "verdict"}``.

        * ``attribution_flip_rate`` — ``n_genuine_flips / n_eligible_cells`` over the
          non-forced (held_out, topic) cells; ``0.0`` when every non-forced origin is
          stable.
        * ``forced_reattribution_count`` — non-flip cells where the topic's origin
          journal was the one dropped (reported separately, never counted as a flip).
        * ``verdict`` — ``"PASS"`` (``flip_rate <= 0.20``), ``"PARTIAL"``
          (``0.20 < flip_rate <= 0.40``), else ``"FAIL"`` — the origin half of the
          Method-C rubric (the role half, ``role_rank_correlation``, is owned by
          V2-S05).

    Notes
    -----
    Pure. A cell is "genuine flip" iff it is **not** forced and its leave-one-out
    origin differs from the full-run origin (``NaN`` differs from any journal). The
    flip-rate denominator is the non-forced cell count, matching the protocol's
    instruction to exclude forced reattributions.
    """
    if per_run.empty:
        return {
            "attribution_flip_rate": 0.0,
            "forced_reattribution_count": 0,
            "n_eligible_cells": 0,
            "n_genuine_flips": 0,
            "n_topics_flipped": 0,
            "n_topics": 0,
            "verdict": "PASS",
        }

    forced = per_run["forced"].fillna(False).to_numpy(dtype=bool)
    forced_count = int(forced.sum())
    eligible = per_run.loc[~forced].copy()

    same = (
        eligible["origin_journal_slug"].astype("object").to_numpy()
        == eligible["full_origin"].astype("object").to_numpy()
    )
    flips_mask = ~same
    n_eligible = int(len(eligible))
    n_flips = int(flips_mask.sum())
    flip_rate = (n_flips / n_eligible) if n_eligible else 0.0
    n_topics_flipped = int(eligible.loc[flips_mask, topic_key].nunique())
    n_topics = int(per_run[topic_key].nunique())

    if flip_rate <= 0.20:
        verdict = "PASS"
    elif flip_rate <= 0.40:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "attribution_flip_rate": float(flip_rate),
        "forced_reattribution_count": forced_count,
        "n_eligible_cells": n_eligible,
        "n_genuine_flips": n_flips,
        "n_topics_flipped": n_topics_flipped,
        "n_topics": n_topics,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Method D — topic-granularity sensitivity.
# --------------------------------------------------------------------------- #
def _panel_seeding_scores(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str,
    panel: tuple[str, ...],
) -> pd.Series:
    """Per-journal seeding score reindexed to the fixed panel (helper)."""
    import pandas as pd

    from scifield.findings.seeding import seeding_score

    work = flow_or_first
    if "year" not in work.columns and "first_year" in work.columns:
        work = work.rename(columns={"first_year": "year"})
    if topic_key != "topic_id":
        work = work.rename(columns={topic_key: "topic_id"})
    scores = seeding_score(work, entity_col="journal_slug")
    if scores.empty:
        return pd.Series(dtype="float64", index=list(panel))
    return scores.set_index("journal_slug")["seeding_score"].reindex(list(panel))


def granularity_consistency(
    flow_leaf: pd.DataFrame,
    flow_mid: pd.DataFrame,
    *,
    leaf_key: str = "topic_id",
    mid_key: str = "mid_level_id",
    panel: tuple[str, ...] = PANEL_JOURNALS,
    rho_bar: float = GRANULARITY_RHO_BAR,
) -> dict:
    """Leaf-vs-mid per-journal seeding-score Spearman ρ (Method D).

    A conclusion is robust only if it survives the topic resolution. This compares
    the per-journal seeding-score ranking at the **leaf** grain (149 ``topic_id``)
    against the **mid** grain (96 ``mid_level_id``) and returns their Spearman ρ
    against the ``rho_bar`` (validation_protocol §4).

    Parameters
    ----------
    flow_leaf :
        Leaf-grain flow slice (topic key ``leaf_key``).
    flow_mid :
        Mid-grain flow slice (topic key ``mid_key``).
    leaf_key, mid_key :
        Topic-grain columns. Defaults ``"topic_id"`` / ``"mid_level_id"``.
    panel :
        Journal slugs to rank. Default :data:`PANEL_JOURNALS`.
    rho_bar :
        PASS threshold. Default :data:`GRANULARITY_RHO_BAR` (0.5).

    Returns
    -------
    dict
        ``{"spearman_rho", "n_journals", "rho_bar", "verdict", "leaf_scores",
        "mid_scores"}``. ``verdict`` — ``"PASS"`` (``ρ >= rho_bar``), ``"PARTIAL"``
        (``0.2 <= ρ < rho_bar``), else ``"FAIL"``. ``leaf_scores`` / ``mid_scores``
        are journal→score dicts (for the notebook scatter).

    Notes
    -----
    Pure. Journals scored in only one grain are dropped from the correlation pair
    (Spearman is computed on the shared, non-NaN journals).
    """
    import numpy as np
    from scipy.stats import spearmanr

    leaf_scores = _panel_seeding_scores(flow_leaf, topic_key=leaf_key, panel=panel)
    mid_scores = _panel_seeding_scores(flow_mid, topic_key=mid_key, panel=panel)

    paired = leaf_scores.rename("leaf").to_frame().join(mid_scores.rename("mid")).dropna()
    n_journals = int(len(paired))
    if n_journals < 2:
        rho = float("nan")
    else:
        rho, _ = spearmanr(paired["leaf"].to_numpy(), paired["mid"].to_numpy())
        rho = float(rho)

    if not np.isnan(rho) and rho >= rho_bar:
        verdict = "PASS"
    elif not np.isnan(rho) and rho >= 0.2:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "spearman_rho": rho,
        "n_journals": n_journals,
        "rho_bar": float(rho_bar),
        "verdict": verdict,
        "leaf_scores": {k: (None if np.isnan(v) else float(v)) for k, v in leaf_scores.items()},
        "mid_scores": {k: (None if np.isnan(v) else float(v)) for k, v in mid_scores.items()},
    }


# --------------------------------------------------------------------------- #
# Method A — held-out years (supporting only).
# --------------------------------------------------------------------------- #
def heldout_consistency(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str,
    train_max_year: int = HELDOUT_TRAIN_MAX_YEAR,
    holdout_min_year: int = HELDOUT_MIN_YEAR,
    panel: tuple[str, ...] = PANEL_JOURNALS,
) -> dict:
    """Train-vs-holdout per-journal seeding-score Spearman ρ (Method A, supporting).

    Method A (validation_protocol §1) checks that the journal lead/lag ranking holds
    *out of sample in time*: learn per-journal seeding scores on ``year <=
    train_max_year`` (1995–2018), recompute independently on ``year >=
    holdout_min_year`` (2019–2025), and correlate the two rankings (Spearman ρ).
    Topics first appearing **after** ``train_max_year`` are unscorable in the train
    window and excluded; their count is reported. Supporting evidence only — a
    PARTIAL/FAIL here does not by itself sink the gate (the window is 7 years).

    Parameters
    ----------
    flow_or_first :
        A flow-table slice carrying ``[topic_key, "journal_slug", "year"]`` (a
        ``"first_year"`` frame is unusable here — Method A needs the raw years to
        split).
    topic_key :
        Topic-grain column.
    train_max_year :
        Last year of the train window. Default :data:`HELDOUT_TRAIN_MAX_YEAR`
        (2018).
    holdout_min_year :
        First year of the holdout window. Default :data:`HELDOUT_MIN_YEAR` (2019).
    panel :
        Journal slugs to rank. Default :data:`PANEL_JOURNALS`.

    Returns
    -------
    dict
        ``{"spearman_rho", "n_journals_paired", "n_topics_excluded", "verdict",
        "train_scores", "holdout_scores", "supporting_only"}``. ``verdict`` —
        ``"PASS"`` (``ρ >= 0.5``), ``"PARTIAL"`` (``0.2 <= ρ < 0.5``), else
        ``"FAIL"``; ``supporting_only`` is ``True`` (Method A never sinks the gate
        alone).

    Raises
    ------
    ValueError
        If the input lacks a ``"year"`` column.

    Notes
    -----
    Pure. The split is temporal, never random (respecting the arrow of time). Each
    window's seeding score is recomputed independently on its own slice.
    """
    import numpy as np
    from scipy.stats import spearmanr

    from scifield.cartography.cascade import _first_appearance_frame

    if "year" not in flow_or_first.columns:
        raise ValueError("heldout_consistency needs a 'year' column to split train/holdout")

    first = _first_appearance_frame(flow_or_first, topic_key=topic_key)
    origin_year = first.groupby(topic_key)["first_year"].min()
    post_train = set(origin_year[origin_year > train_max_year].index)
    n_excluded = len(post_train)

    scorable = flow_or_first[~flow_or_first[topic_key].isin(post_train)].copy()
    train = scorable[scorable["year"] <= train_max_year]
    holdout = scorable[scorable["year"] >= holdout_min_year]

    train_scores = _panel_seeding_scores(train, topic_key=topic_key, panel=panel)
    holdout_scores = _panel_seeding_scores(holdout, topic_key=topic_key, panel=panel)

    paired = train_scores.rename("train").to_frame().join(holdout_scores.rename("holdout")).dropna()
    n_paired = int(len(paired))
    if n_paired < 2:
        rho = float("nan")
    else:
        rho, _ = spearmanr(paired["train"].to_numpy(), paired["holdout"].to_numpy())
        rho = float(rho)

    if not np.isnan(rho) and rho >= 0.5:
        verdict = "PASS"
    elif not np.isnan(rho) and rho >= 0.2:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    def _clean(series: pd.Series) -> dict:
        return {k: (None if np.isnan(v) else float(v)) for k, v in series.items()}

    return {
        "spearman_rho": rho,
        "n_journals_paired": n_paired,
        "n_topics_excluded": n_excluded,
        "verdict": verdict,
        "train_scores": _clean(train_scores),
        "holdout_scores": _clean(holdout_scores),
        "supporting_only": True,
    }


# --------------------------------------------------------------------------- #
# Orchestrator — run all four methods and assemble the structured verdict.
# --------------------------------------------------------------------------- #
def _method_b_verdict(cells: dict) -> str:
    """Combine the four S×Null cells into the Method-B rubric verdict.

    Gate-critical primary = S1 under Null-1. PASS requires that to PASS *and* be
    corroborated (PASS or PARTIAL) by S2 and by Null-2 in the same direction; a
    PARTIAL on a corroborating cell, or a non-PASS primary that is still
    significant, downgrades to PARTIAL; a FAIL on the primary is FAIL.
    """
    primary = cells[("S1", "within_topic")]["verdict"]
    s1_null2 = cells.get(("S1", "year_shuffle"), {}).get("verdict", "FAIL")
    s2_null1 = cells.get(("S2", "within_topic"), {}).get("verdict", "FAIL")
    s2_null2 = cells.get(("S2", "year_shuffle"), {}).get("verdict", "FAIL")

    if primary == "FAIL":
        return "FAIL"
    corroborators = [s1_null2, s2_null1, s2_null2]
    if primary == "PASS" and all(c == "PASS" for c in corroborators):
        return "PASS"
    # Primary present (PASS/PARTIAL) but corroboration incomplete → PARTIAL.
    return "PARTIAL"


def run_all_validations(
    flow_leaf: pd.DataFrame,
    flow_mid: pd.DataFrame,
    *,
    leaf_key: str = "topic_id",
    mid_key: str = "mid_level_id",
    n_perm: int = N_PERM,
    seed: int = SEED,
    panel: tuple[str, ...] = PANEL_JOURNALS,
    secondary_n_perm: int | None = None,
) -> dict:
    """Run Methods A–D at both grains and assemble the structured verdict (Method B/C/D + A).

    Orchestrates the full cascade-validation suite the locked protocol prescribes:
    Method B (S1/S2 × Null-1/Null-2) at both grains and on both the all-topics and
    resolvable subsets; Method C (origin jackknife) at both grains and subsets;
    Method D (granularity ρ); Method A (held-out years, supporting). Returns one
    nested dict a script can write to parquet and a notebook can plot.

    Parameters
    ----------
    flow_leaf, flow_mid :
        Leaf- and mid-grain flow slices.
    leaf_key, mid_key :
        Topic-grain columns. Defaults ``"topic_id"`` / ``"mid_level_id"``.
    n_perm :
        Permutations for the **primary** cell (S1/Null-1/leaf/all). Default
        :data:`N_PERM` (1000) — never capped.
    seed :
        Base RNG seed. Default :data:`SEED`.
    panel :
        Journal slugs. Default :data:`PANEL_JOURNALS`.
    secondary_n_perm :
        Optional smaller permutation count for the *secondary* Method-B cells (S2,
        Null-2, and the mid/resolvable variants) to keep runtime reasonable. The
        **primary** cell always uses ``n_perm``. Default ``None`` (every cell uses
        ``n_perm``).

    Returns
    -------
    dict
        ``{"method_b", "method_c", "method_d", "method_a", "overall"}`` where:

        * ``method_b`` — ``{grain: {subset: {"cells": {(stat, null): verdict_dict},
          "verdict": str}}}`` for ``grain in {"leaf", "mid"}`` and
          ``subset in {"all", "resolvable"}``.
        * ``method_c`` — ``{grain: {subset: summary_dict}}``.
        * ``method_d`` — the granularity dict.
        * ``method_a`` — ``{grain: heldout_dict}``.
        * ``overall`` — ``{"cascade_verdict": "REAL"|"PARTIAL"|"ARTIFACT",
          "rationale": str}`` synthesising the gate-critical checks.

    Notes
    -----
    Pure (no I/O). The gate-critical, make-or-break cell is S1/Null-1/leaf at both
    the all-topics and resolvable subsets; the overall ``cascade_verdict`` reads
    Method B (gate-critical) and Method C together and is **REAL** only if both PASS
    on the primary at both grains and survive the resolvable subset.
    """
    grains = {"leaf": (flow_leaf, leaf_key), "mid": (flow_mid, mid_key)}
    sec_perm = n_perm if secondary_n_perm is None else int(secondary_n_perm)

    method_b: dict = {}
    method_c: dict = {}
    for grain, (flow, key) in grains.items():
        method_b[grain] = {}
        method_c[grain] = {}
        for subset in ("all", "resolvable"):
            resolvable_only = subset == "resolvable"
            cells: dict = {}
            for stat in ("S1", "S2"):
                for null in ("within_topic", "year_shuffle"):
                    is_primary = (
                        grain == "leaf"
                        and subset == "all"
                        and stat == "S1"
                        and null == "within_topic"
                    )
                    cell_perm = n_perm if is_primary else sec_perm
                    cells[(stat, null)] = permutation_verdict(
                        flow,
                        topic_key=key,
                        stat=stat,
                        null=null,
                        n_perm=cell_perm,
                        seed=seed,
                        resolvable_only=resolvable_only,
                    )
            method_b[grain][subset] = {
                "cells": cells,
                "verdict": _method_b_verdict(cells),
            }

            sub_flow = _restrict_to_resolvable(flow, topic_key=key) if resolvable_only else flow
            per_run = jackknife_origins(sub_flow, topic_key=key, journals=panel)
            method_c[grain][subset] = summarize_jackknife_origins(per_run, topic_key=key)

    method_d = granularity_consistency(
        flow_leaf, flow_mid, leaf_key=leaf_key, mid_key=mid_key, panel=panel
    )
    method_a = {
        "leaf": heldout_consistency(flow_leaf, topic_key=leaf_key, panel=panel),
        "mid": heldout_consistency(flow_mid, topic_key=mid_key, panel=panel),
    }

    overall = _assemble_overall(method_b, method_c, method_d)
    return {
        "method_b": method_b,
        "method_c": method_c,
        "method_d": method_d,
        "method_a": method_a,
        "overall": overall,
    }


def _assemble_overall(method_b: dict, method_c: dict, method_d: dict) -> dict:
    """Synthesise REAL / PARTIAL / ARTIFACT from the gate-critical checks.

    The cascade is **REAL** only if the gate-critical primary (S1/Null-1) PASSes at
    both grains on *both* the all-topics and resolvable subsets, Method C (origin
    jackknife) PASSes, and Method D (granularity) is at least PARTIAL. If the
    primary FAILs on the all-topics set, or survives only on the all-tied topics
    (fails on the resolvable subset), it is an **ARTIFACT**. Otherwise **PARTIAL**.
    """

    def _primary(grain: str, subset: str) -> str:
        return str(method_b[grain][subset]["cells"][("S1", "within_topic")]["verdict"])

    primaries = {(g, s): _primary(g, s) for g in ("leaf", "mid") for s in ("all", "resolvable")}
    c_pass = all(
        method_c[g][s]["verdict"] == "PASS" for g in ("leaf", "mid") for s in ("all", "resolvable")
    )
    d_ok = method_d["verdict"] in {"PASS", "PARTIAL"}

    leaf_all = primaries[("leaf", "all")]
    leaf_res = primaries[("leaf", "resolvable")]

    if leaf_all == "FAIL":
        verdict = "ARTIFACT"
        why = "primary S1/Null-1 (leaf, all topics) is indistinguishable from the null"
    elif leaf_res == "FAIL":
        verdict = "ARTIFACT"
        why = (
            "primary S1/Null-1 survives on all-topics but FAILS on the resolvable "
            "subset → structure lives only in the 1995 all-tied ties"
        )
    elif all(v == "PASS" for v in primaries.values()) and c_pass and d_ok:
        verdict = "REAL"
        why = (
            "S1/Null-1 PASSes at both grains and on the resolvable subset; origin "
            "jackknife PASSes; granularity holds"
        )
    else:
        verdict = "PARTIAL"
        why = (
            "primary S1/Null-1 is significant and origins are jackknife-stable, but "
            "corroboration is incomplete (S2/Null-1 or a grain/subset is weaker)"
        )

    return {
        "cascade_verdict": verdict,
        "rationale": why,
        "primary_s1_null1_verdicts": primaries,
        "method_c_pass": bool(c_pass),
        "method_d_verdict": method_d["verdict"],
    }
