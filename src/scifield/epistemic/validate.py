"""Internal-validity & cross-model agreement lenses for Gate G2 (V1-S09).

Three *pure* functions over pandas DataFrames — no I/O, no API calls —
so the notebook (T4) can call them on the real
``data/v1/epistemic_extracted.parquet`` while the unit tests below
exercise them on synthetic in-memory frames.

The G2 redefinition (2026-05-29) drops hand-labeled κ in favour of three
label-free lenses:

* **C1 — cross-tool agreement.** Compare the DeepSeek ``study_design ==
  "RCT"`` call against the *structured* PubMed ``publication_types`` RCT
  flag (computed upstream). Reuses :func:`scifield.epistemic.kappa.cohens_kappa`.
* **C2 — model-vs-model agreement.** On the 1,981 PMIDs labeled by *both*
  DeepSeek and Claude, exact-match agreement on categorical fields and
  Spearman ρ on ``sample_size``.
* **C3 — internal-validity priors.** Six domain face-validity checks (a)–(f)
  on the deduped DeepSeek corpus, each with an explicit, commented
  plausibility band.

Conventions mirror :mod:`scifield.epistemic.kappa`: None/NaN values are
dropped pairwise before any metric, and degenerate inputs return
``float("nan")`` rather than raising.

Real-parquet column names (confirmed via ``DESCRIBE`` on
``epistemic_extracted.parquet``) and the values the functions key on:

* ``pmid`` (BIGINT)
* ``study_design`` (VARCHAR) — RCT label is ``"RCT"``
* ``sample_size`` (BIGINT, nullable)
* ``has_control`` (BOOLEAN, nullable)
* ``effect_direction`` (VARCHAR) — values ``positive``/``null``/
  ``negative``/``mixed``/``na``
* ``statistical_claim_present`` (BOOLEAN)
* ``coi_disclosed_in_abstract`` (BOOLEAN)
* ``model_id`` (VARCHAR) — full corpus is ``"deepseek-v4-flash"``;
  paired subset is ``"claude-via-claude-code"``

No scipy/statsmodels: Spearman ρ is computed via ``Series.rank()`` +
Pearson (only sklearn + krippendorff are project deps).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from scifield.epistemic.kappa import cohens_kappa

__all__ = [
    "CheckResult",
    "cross_tool_rct_agreement",
    "model_vs_model_agreement",
    "internal_validity_checks",
]


# Field-name constants — the *real* parquet column names (see module docstring).
COL_PMID = "pmid"
COL_STUDY_DESIGN = "study_design"
COL_SAMPLE_SIZE = "sample_size"
COL_HAS_CONTROL = "has_control"
COL_EFFECT_DIRECTION = "effect_direction"
COL_STAT_CLAIM = "statistical_claim_present"
COL_COI = "coi_disclosed_in_abstract"

RCT_LABEL = "RCT"


@dataclass(frozen=True)
class CheckResult:
    """One internal-validity prior outcome (frozen — safe to tabulate).

    Field order is part of the public contract (the notebook unpacks
    these into the gate table):

    1. ``key`` — short stable id, e.g. ``"a_rct_fraction"``.
    2. ``label`` — human-readable description for the gate table.
    3. ``value`` — the computed quantity being judged.
    4. ``threshold_repr`` — human-readable plausibility band string.
    5. ``passed`` — whether ``value`` falls in-band.
    """

    key: str
    label: str
    value: float
    threshold_repr: str
    passed: bool


def _spearman_rho(x: pd.Series, y: pd.Series) -> float:
    """Spearman ρ via rank + Pearson (no scipy dependency).

    Rows where *either* side is null are dropped first. Returns
    ``float("nan")`` when fewer than 2 paired observations survive or
    when either ranked vector is constant (Pearson would be undefined).
    """
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(paired) < 2:
        return float("nan")
    rx = paired["x"].rank()
    ry = paired["y"].rank()
    if rx.nunique() < 2 or ry.nunique() < 2:
        return float("nan")
    return float(rx.corr(ry))  # Pearson on ranks == Spearman ρ


def cross_tool_rct_agreement(
    deepseek_df: pd.DataFrame,
    pubtype_df: Mapping[int, bool] | pd.Series,
) -> dict[str, Any]:
    """C1: DeepSeek RCT calls vs structured PubMed RCT flag.

    Parameters
    ----------
    deepseek_df:
        DeepSeek rows carrying at least ``pmid`` and ``study_design``.
        The positive class is ``study_design == "RCT"``.
    pubtype_df:
        PMID → bool lookup (a dict or a pandas Series indexed by pmid)
        of ``is_pubmed_rct``, computed upstream from
        ``list_contains(publication_types, 'Randomized Controlled Trial')``.

    Returns
    -------
    dict
        ``{n, simple_agreement, cohens_kappa, tp, fp, fn, tn,
        sensitivity, precision}``. Only PMIDs present in *both* the
        DeepSeek frame and the PubMed lookup are scored. ``cohens_kappa``
        is ``nan`` for a degenerate (single-class) overlap.

    Notes
    -----
    Treating the PubMed structured flag as the reference, ``tp`` is a
    PubMed-RCT that DeepSeek also called RCT; ``sensitivity = tp/(tp+fn)``
    and ``precision = tp/(tp+fp)``.
    """
    lookup: dict[int, bool] = dict(pubtype_df) if not isinstance(pubtype_df, dict) else pubtype_df

    deepseek_calls: list[bool] = []
    pubmed_flags: list[bool] = []
    for _, row in deepseek_df.iterrows():
        pmid = row[COL_PMID]
        if pmid not in lookup:
            continue
        ds_val = row[COL_STUDY_DESIGN]
        if ds_val is None or (isinstance(ds_val, float) and pd.isna(ds_val)):
            continue
        deepseek_calls.append(bool(ds_val == RCT_LABEL))
        pubmed_flags.append(bool(lookup[pmid]))

    n = len(deepseek_calls)
    tp = sum(1 for d, p in zip(deepseek_calls, pubmed_flags, strict=True) if d and p)
    fp = sum(1 for d, p in zip(deepseek_calls, pubmed_flags, strict=True) if d and not p)
    fn = sum(1 for d, p in zip(deepseek_calls, pubmed_flags, strict=True) if not d and p)
    tn = sum(1 for d, p in zip(deepseek_calls, pubmed_flags, strict=True) if not d and not p)

    simple_agreement = (tp + tn) / n if n else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    # cohens_kappa reuses kappa.py — None-safe & degenerate-safe (returns nan).
    kappa = cohens_kappa(pubmed_flags, deepseek_calls)

    return {
        "n": n,
        "simple_agreement": simple_agreement,
        "cohens_kappa": kappa,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "precision": precision,
    }


def model_vs_model_agreement(paired_df: pd.DataFrame) -> dict[str, Any]:
    """C2: DeepSeek-vs-Claude agreement on the dual-labeled PMIDs.

    Parameters
    ----------
    paired_df:
        Wide frame (one row per PMID, the ~1,981 dual-labeled set) with
        ``<field>_deepseek`` / ``<field>_claude`` column pairs. Expected
        pairs:

        * ``study_design_deepseek`` / ``study_design_claude``
        * ``has_control_deepseek`` / ``has_control_claude``
        * ``sample_size_deepseek`` / ``sample_size_claude``

    Returns
    -------
    dict
        ``{"n": <rows>, "fields": {field: {...}}}``. Per categorical
        field: ``{n, n_agree, exact_match, cohens_kappa}`` over rows
        where *both* sides are non-null. For ``sample_size``:
        ``{n, spearman_rho}`` over rows where both are non-null.
        Exact-match / κ drop None pairwise (mirrors kappa.py).
    """
    n_rows = len(paired_df)
    fields: dict[str, dict[str, Any]] = {}

    for field in (COL_STUDY_DESIGN, COL_HAS_CONTROL):
        a_col, b_col = f"{field}_deepseek", f"{field}_claude"
        if a_col not in paired_df.columns or b_col not in paired_df.columns:
            continue
        sub = paired_df[[a_col, b_col]].dropna()
        n = len(sub)
        a = list(sub[a_col])
        b = list(sub[b_col])
        n_agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
        fields[field] = {
            "n": n,
            "n_agree": n_agree,
            "exact_match": (n_agree / n) if n else float("nan"),
            "cohens_kappa": cohens_kappa(a, b),
        }

    ss_a, ss_b = f"{COL_SAMPLE_SIZE}_deepseek", f"{COL_SAMPLE_SIZE}_claude"
    if ss_a in paired_df.columns and ss_b in paired_df.columns:
        sub = paired_df[[ss_a, ss_b]].dropna()
        fields[COL_SAMPLE_SIZE] = {
            "n": len(sub),
            "spearman_rho": _spearman_rho(paired_df[ss_a], paired_df[ss_b]),
        }

    return {"n": n_rows, "fields": fields}


def _dedupe_by_pmid(deepseek_df: pd.DataFrame) -> pd.DataFrame:
    """One row per PMID (there are 19 internal dup rows in the corpus).

    Keep-first after stable-sorting so the *longest* ``raw_response``
    wins per PMID: more text generally means the more complete
    extraction. PMIDs without a ``raw_response`` column fall back to
    plain keep-first on the input order.
    """
    df = deepseek_df.copy()
    if "raw_response" in df.columns:
        # Longer raw_response first, then keep-first per pmid.
        df["_resp_len"] = df["raw_response"].fillna("").astype(str).str.len()
        df = df.sort_values("_resp_len", ascending=False, kind="stable")
        df = df.drop_duplicates(subset=[COL_PMID], keep="first")
        df = df.drop(columns="_resp_len")
    else:
        df = df.drop_duplicates(subset=[COL_PMID], keep="first")
    return df


def internal_validity_checks(deepseek_df: pd.DataFrame) -> list[CheckResult]:
    """C3: six domain face-validity priors on the deduped DeepSeek corpus.

    The frame is deduped to one row per ``pmid`` first (see
    :func:`_dedupe_by_pmid`). Each prior's expected (computed) value and
    the plausibility band chosen around it:

    * **(a) RCT fraction** ``study_design == "RCT"``. Expected ≈ 6.73%.
      Band [2%, 15%]: RCTs are a small-but-real slice of the biomedical
      literature; well outside this would signal a systematic mislabel.
    * **(b) statistical-claim fraction**. Expected ≈ 66.5%. Band
      [45%, 85%]: most empirical abstracts make a statistical claim, but
      a large minority (reviews, methods, case reports) do not.
    * **(c) COI-in-abstract fraction**. Expected ≈ 0.05%. Band
      (0%, 5%]: COI text in the *abstract* is rare but must be nonzero
      (exactly-zero would imply the field never fires — a parser bug).
    * **(d) RCT ⇒ has_control**. Of RCT rows, fraction with
      ``has_control`` True. Expected ≈ 99.05%. Band [90%, 100%]: an RCT
      without a control arm is near-incoherent, so this must be very high.
    * **(e) sample_size sanity**. ``value`` is the **median** (expected
      ≈ 105, sane). ``passed`` keys on the median band [10, 1000]; the
      max (≈ 68.2M) and the list of PMIDs with ``sample_size`` > 10M are
      returned in ``threshold_repr`` for notebook adjudication (the
      notebook overrides the max-bound verdict and records the
      adjudication — see plan).
    * **(f) effect-direction 'na' fraction**. Expected ≈ 34.9% na (with
      positive ≈ 40.6%). Band [20%, 55%] on the na share: a large chunk
      of abstracts legitimately report no single effect direction.
    """
    df = _dedupe_by_pmid(deepseek_df)

    results: list[CheckResult] = []

    # (a) RCT fraction --------------------------------------------------
    sd = df[COL_STUDY_DESIGN].dropna()
    rct_frac = float((sd == RCT_LABEL).mean()) if len(sd) else float("nan")
    a_lo, a_hi = 0.02, 0.15
    results.append(
        CheckResult(
            key="a_rct_fraction",
            label="RCT fraction (study_design == 'RCT')",
            value=rct_frac,
            threshold_repr=f"in [{a_lo:.0%}, {a_hi:.0%}] (expected ≈ 6.73%)",
            passed=(not pd.isna(rct_frac)) and a_lo <= rct_frac <= a_hi,
        )
    )

    # (b) statistical-claim fraction -----------------------------------
    stat = df[COL_STAT_CLAIM].dropna()
    stat_frac = float(stat.astype(bool).mean()) if len(stat) else float("nan")
    b_lo, b_hi = 0.45, 0.85
    results.append(
        CheckResult(
            key="b_statistical_claim_fraction",
            label="Statistical-claim fraction (statistical_claim_present)",
            value=stat_frac,
            threshold_repr=f"in [{b_lo:.0%}, {b_hi:.0%}] (expected ≈ 66.5%)",
            passed=(not pd.isna(stat_frac)) and b_lo <= stat_frac <= b_hi,
        )
    )

    # (c) COI-in-abstract fraction -------------------------------------
    coi = df[COL_COI].dropna()
    coi_frac = float(coi.astype(bool).mean()) if len(coi) else float("nan")
    c_lo_excl, c_hi = 0.0, 0.05
    results.append(
        CheckResult(
            key="c_coi_fraction",
            label="COI-in-abstract fraction (coi_disclosed_in_abstract)",
            value=coi_frac,
            threshold_repr=f"in ({c_lo_excl:.0%}, {c_hi:.0%}] (expected ≈ 0.05%)",
            passed=(not pd.isna(coi_frac)) and c_lo_excl < coi_frac <= c_hi,
        )
    )

    # (d) RCT => has_control -------------------------------------------
    rct_rows = df[df[COL_STUDY_DESIGN] == RCT_LABEL]
    hc = rct_rows[COL_HAS_CONTROL].dropna()
    rct_control_frac = float(hc.astype(bool).mean()) if len(hc) else float("nan")
    d_lo, d_hi = 0.90, 1.00
    results.append(
        CheckResult(
            key="d_rct_implies_control",
            label="RCT ⇒ has_control (fraction of RCT rows with has_control True)",
            value=rct_control_frac,
            threshold_repr=f"in [{d_lo:.0%}, {d_hi:.0%}] (expected ≈ 99.05%)",
            passed=(not pd.isna(rct_control_frac)) and d_lo <= rct_control_frac <= d_hi,
        )
    )

    # (e) sample_size sanity (median-keyed; max adjudicated downstream) -
    ss = df[COL_SAMPLE_SIZE].dropna()
    ss_median = float(ss.median()) if len(ss) else float("nan")
    ss_max = float(ss.max()) if len(ss) else float("nan")
    over_10m = df.loc[df[COL_SAMPLE_SIZE] > 10_000_000, COL_PMID]
    over_10m_pmids = [int(p) for p in over_10m.tolist()]
    e_lo, e_hi = 10.0, 1000.0
    results.append(
        CheckResult(
            key="e_sample_size_median",
            label="sample_size median (sanity); max adjudicated in notebook",
            value=ss_median,
            threshold_repr=(
                f"median in [{e_lo:.0f}, {e_hi:.0f}] (expected ≈ 105); "
                f"max={ss_max:.0f} (>10M needs adjudication); "
                f">10M pmids={over_10m_pmids}"
            ),
            passed=(not pd.isna(ss_median)) and e_lo <= ss_median <= e_hi,
        )
    )

    # (f) effect-direction 'na' fraction -------------------------------
    ed = df[COL_EFFECT_DIRECTION].dropna()
    na_frac = float((ed == "na").mean()) if len(ed) else float("nan")
    f_lo, f_hi = 0.20, 0.55
    results.append(
        CheckResult(
            key="f_effect_direction_na_fraction",
            label="effect_direction 'na' fraction",
            value=na_frac,
            threshold_repr=f"in [{f_lo:.0%}, {f_hi:.0%}] (expected ≈ 34.9% na)",
            passed=(not pd.isna(na_frac)) and f_lo <= na_frac <= f_hi,
        )
    )

    return results
