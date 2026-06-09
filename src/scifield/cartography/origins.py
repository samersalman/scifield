"""V2-S06 origins of novelty — pure, I/O-free aggregation of *where novelty arises*.

This module answers "where does novelty come from?" along three axes that the
current 10-journal surgical/orthopaedic corpus supports:

1. **Sector × novelty** (:func:`novelty_by_institution_type`) — does novelty come
   from the tech/industry sector? Each paper is tagged with the *set* of author
   institution ``type`` values, and a paper is "company-affiliated" if **ANY**
   author institution has ``type == 'company'`` (the any-affiliation rule). We then
   report per-type mean/median novelty + paper counts + a dispersion (std), so the
   ``company`` row can be contrasted against ``education`` / ``healthcare`` etc.

2. **Geography × novelty** (:func:`novelty_by_country`) — same any-affiliation
   pattern keyed on institution ``country_code``. Blank country codes are **dropped,
   never imputed** (a missing country must not invent a phantom origin), and
   low-``n`` countries are flagged so a single-paper country is not over-read.

3. **Topic recombination** (:func:`recombination_events`, exploratory) — recombinant
   novelty: a paper whose *references span ≥ 2 distinct topics* is recombining
   literatures. The reference→topic resolution (``references_out.ref_openalex_id →
   archetypes.openalex_id → topic_id``) is done by the builder script; this module
   takes the already-joined ``[citing_pmid, ref_topic_id]`` frame and computes a
   per-paper recombination score (distinct-topic count + Shannon entropy over the
   referenced-topic mix) plus a per-topic recombination summary.

Aggregation grain & purity
--------------------------
Every public function is **pure and I/O-free**: it takes in-memory tidy frames and
returns a DataFrame. The :mod:`V2.scripts.build_origins` builder owns all parquet
reads/writes and the (potentially expensive) reference→topic join. Real counts are
COMPUTED here, never hardcoded.

Paper → institution-type / country tagging
------------------------------------------
A paper has *many* author institutions (``paper_institutions`` is one row per
author-position-institution). We therefore reduce to the **set** of distinct
``type`` (resp. ``country_code``) values per paper, then explode so a paper
contributes one observation to *each* sector/country it touches. Consequently the
per-type/per-country paper counts **sum to more than the paper total** (a
multi-sector paper is counted in every sector it spans) — this is the correct
"is novelty *present in* sector X" framing, documented on each function. A separate
``is_company`` boolean implements the strict any-author-company contrast.

Coverage caveats (carry into every claim)
-----------------------------------------
* **Panel-conditional.** This is the 10-journal surgical/ortho corpus, not all of
  science; "company novelty" means company-affiliated *within this corpus*.
* **Institution-type blanks.** ~35% of institutions have a blank ``type`` and ~44%
  a blank ``country_code``; blanks are excluded from the relevant axis, so the
  per-type/per-country tables are conditioned on *resolved* affiliation metadata.
* **Funding origin is DEFERRED.** OpenAlex ``grants`` are now parsed but not yet
  persisted (needs a schema change + re-harvest); citation-intent origin is
  likewise deferred (empty ``citation_intents.parquet``). See the builder + notebook.

Conventions
-----------
``from __future__ import annotations``; numpy-style docstrings; numpy + pandas only;
ruff/black line-length 100.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "COMPANY_TYPE",
    "bootstrap_delta_ci",
    "novelty_by_country",
    "novelty_by_institution_type",
    "paper_sector_tags",
    "recombination_by_topic",
    "recombination_events",
    "split_half_stability",
]

#: The ``institutions.type`` value used as the "tech/industry sector" proxy.
COMPANY_TYPE = "company"


def _blank_mask(series: pd.Series) -> pd.Series:
    """Boolean mask of *blank* entries (NaN/None or whitespace-only string).

    Parameters
    ----------
    series :
        Any pandas Series.

    Returns
    -------
    pandas.Series
        Boolean mask, ``True`` where the value is NaN/None or its stripped string
        form is empty. Used to drop blank ``type`` / ``country_code`` without
        imputation.
    """

    not_null = series.notna()
    nonempty = series.astype("string").str.strip().fillna("") != ""
    return ~(not_null & nonempty)


def _attach_paper_attribute(
    paper_inst: pd.DataFrame,
    institutions: pd.DataFrame,
    *,
    attr_col: str,
    inst_key: str = "institution_canonical_id",
) -> pd.DataFrame:
    """Map each (paper, institution) row to the institution's ``attr_col`` value.

    Joins ``paper_inst`` to ``institutions`` on ``inst_key`` and returns the tidy
    ``[pmid, attr_col]`` rows with **blank** ``attr_col`` values dropped (no
    imputation). Duplicate (pmid, value) pairs are de-duplicated so a paper with two
    company affiliations contributes the ``company`` tag exactly once.

    Parameters
    ----------
    paper_inst :
        ``paper_institutions`` frame — at least ``[pmid, inst_key]``.
    institutions :
        ``institutions`` frame — at least ``[inst_key, attr_col]``.
    attr_col :
        The institution attribute to attach (``"type"`` or ``"country_code"``).
    inst_key :
        Join key, default ``"institution_canonical_id"``.

    Returns
    -------
    pandas.DataFrame
        Distinct ``[pmid, attr_col]`` rows (one per paper × resolved attribute).
    """

    left = paper_inst.loc[:, ["pmid", inst_key]].dropna(subset=["pmid", inst_key])
    right = institutions.loc[:, [inst_key, attr_col]].drop_duplicates(subset=[inst_key])
    joined = left.merge(right, on=inst_key, how="inner")
    joined = joined.loc[~_blank_mask(joined[attr_col])]
    tagged: pd.DataFrame = joined.loc[:, ["pmid", attr_col]].drop_duplicates()
    return tagged.reset_index(drop=True)


def paper_sector_tags(
    paper_inst: pd.DataFrame,
    institutions: pd.DataFrame,
    *,
    type_col: str = "type",
    inst_key: str = "institution_canonical_id",
) -> pd.DataFrame:
    """Distinct ``[pmid, type]`` sector tags + a per-paper ``is_company`` flag.

    Implements the **any-author-affiliation rule**: a paper is tagged with every
    distinct institution ``type`` among its authors (so it appears once per sector
    it spans), and ``is_company`` is ``True`` iff *any* author institution has
    ``type == 'company'``.

    Parameters
    ----------
    paper_inst :
        ``paper_institutions`` frame with ``[pmid, inst_key]``.
    institutions :
        ``institutions`` frame with ``[inst_key, type_col]``.
    type_col :
        Institution-type column, default ``"type"``.
    inst_key :
        Join key, default ``"institution_canonical_id"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[pmid, type_col, is_company]``, distinct on ``(pmid, type_col)``.
        ``is_company`` is constant within a ``pmid`` (paper-level any-company flag),
        replicated onto each of the paper's sector rows for convenient joins. Blank
        types are excluded.
    """
    tagged = _attach_paper_attribute(paper_inst, institutions, attr_col=type_col, inst_key=inst_key)
    company_pmids = set(tagged.loc[tagged[type_col] == COMPANY_TYPE, "pmid"].unique())
    tagged = tagged.copy()
    tagged["is_company"] = tagged["pmid"].isin(company_pmids)
    return tagged.reset_index(drop=True)


def _summarise_novelty(
    df: pd.DataFrame,
    *,
    group_col: str,
    novelty_cols: list[str],
) -> pd.DataFrame:
    """Mean / median / std / n of each novelty column within ``group_col``.

    Parameters
    ----------
    df :
        Long frame carrying ``group_col`` plus the ``novelty_cols``.
    group_col :
        Grouping key (e.g. ``"type"`` or ``"country_code"``).
    novelty_cols :
        Novelty measure columns to summarise (e.g. ``["sem_nov_mean", "cd5"]``).

    Returns
    -------
    pandas.DataFrame
        One row per group value with ``n_papers`` (rows with a non-null
        ``group_col``) and, per novelty column ``c``: ``<c>_mean``, ``<c>_median``,
        ``<c>_std``, ``<c>_n`` (non-null count for *that* measure). NaNs in a
        novelty column are skipped by the mean/median/std and counted out of
        ``<c>_n``. Sorted by ``group_col``.
    """

    grouped = df.groupby(group_col, dropna=True)
    out = grouped.size().rename("n_papers").to_frame()
    for c in novelty_cols:
        agg = grouped[c].agg(["mean", "median", "std", "count"])
        out[f"{c}_mean"] = agg["mean"]
        out[f"{c}_median"] = agg["median"]
        out[f"{c}_std"] = agg["std"]
        out[f"{c}_n"] = agg["count"].astype("int64")
    out = out.reset_index().sort_values(group_col).reset_index(drop=True)
    out["n_papers"] = out["n_papers"].astype("int64")
    return out


def novelty_by_institution_type(
    paper_novelty: pd.DataFrame,
    paper_inst: pd.DataFrame,
    institutions: pd.DataFrame,
    *,
    novelty_cols: list[str],
    type_col: str = "type",
    inst_key: str = "institution_canonical_id",
) -> pd.DataFrame:
    """Novelty summarised by author-institution sector (``type``).

    Joins papers → institutions via ``paper_institutions`` using the
    **any-affiliation** rule (:func:`paper_sector_tags`): a paper contributes one
    observation to *each* distinct sector among its authors. This directly answers
    "is novelty from the tech/industry sector?" — read the ``type == 'company'`` row
    against ``education`` / ``healthcare`` etc.

    Parameters
    ----------
    paper_novelty :
        One row per paper with ``["pmid"]`` + the ``novelty_cols`` (e.g.
        ``sem_nov_mean``, ``sem_nov_min``, ``cd5``, ``cd10`` from archetypes).
        ``pmid`` is cast to string for the join.
    paper_inst :
        ``paper_institutions`` frame with ``[pmid, inst_key]`` (``pmid`` string).
    institutions :
        ``institutions`` frame with ``[inst_key, type_col]``.
    novelty_cols :
        Novelty columns to summarise.
    type_col :
        Institution-type column, default ``"type"``.
    inst_key :
        Join key, default ``"institution_canonical_id"``.

    Returns
    -------
    pandas.DataFrame
        One row per institution ``type`` with ``n_papers`` and, per novelty column,
        ``_mean`` / ``_median`` / ``_std`` / ``_n`` (see :func:`_summarise_novelty`).
        Because a multi-sector paper is counted in every sector it spans, the
        ``n_papers`` column sums to **more** than the distinct paper total — this is
        the "novelty *present in* sector X" framing. Blank types are excluded.

    Notes
    -----
    Panel-conditional and conditioned on resolved affiliation metadata (blank
    ``type`` ≈ 35% of institutions is dropped). The result is a *descriptive*
    contrast — no significance test here; the builder/notebook may add one.
    """
    import pandas as pd

    nov = paper_novelty.copy()
    nov["pmid"] = nov["pmid"].astype("string")
    pi = paper_inst.copy()
    pi["pmid"] = pi["pmid"].astype("string")

    tags = paper_sector_tags(pi, institutions, type_col=type_col, inst_key=inst_key)
    merged = tags.merge(nov, on="pmid", how="inner")
    if merged.empty:
        cols = [type_col, "n_papers"]
        for c in novelty_cols:
            cols += [f"{c}_mean", f"{c}_median", f"{c}_std", f"{c}_n"]
        return pd.DataFrame({col: [] for col in cols})
    return _summarise_novelty(merged, group_col=type_col, novelty_cols=novelty_cols)


def novelty_by_country(
    paper_novelty: pd.DataFrame,
    paper_inst: pd.DataFrame,
    institutions: pd.DataFrame,
    *,
    novelty_cols: list[str],
    country_col: str = "country_code",
    inst_key: str = "institution_canonical_id",
    low_n_threshold: int = 30,
) -> pd.DataFrame:
    """Novelty summarised by author-institution country, low-``n`` flagged.

    Mirrors :func:`novelty_by_institution_type` keyed on ``country_code`` with the
    same any-affiliation rule: a paper contributes one observation to each distinct
    country among its author institutions. **Blank country codes are dropped, never
    imputed.**

    Parameters
    ----------
    paper_novelty :
        One row per paper with ``["pmid"]`` + ``novelty_cols``.
    paper_inst :
        ``paper_institutions`` frame with ``[pmid, inst_key]``.
    institutions :
        ``institutions`` frame with ``[inst_key, country_col]``.
    novelty_cols :
        Novelty columns to summarise.
    country_col :
        ISO country-code column, default ``"country_code"``.
    inst_key :
        Join key, default ``"institution_canonical_id"``.
    low_n_threshold :
        Countries with ``n_papers < low_n_threshold`` get ``low_n == True`` so a
        thin-sample country is not over-interpreted. Default ``30``.

    Returns
    -------
    pandas.DataFrame
        One row per ``country_code`` with ``n_papers``, the per-novelty-column
        summaries, and a boolean ``low_n``. Sorted by ``country_code``. As with the
        sector table, a multi-country paper is counted in each country, so
        ``n_papers`` sums above the distinct paper total.

    Notes
    -----
    Panel-conditional; conditioned on resolved country metadata (~44% of
    institutions have a blank ``country_code``, excluded here).
    """
    import pandas as pd

    nov = paper_novelty.copy()
    nov["pmid"] = nov["pmid"].astype("string")
    pi = paper_inst.copy()
    pi["pmid"] = pi["pmid"].astype("string")

    tags = _attach_paper_attribute(pi, institutions, attr_col=country_col, inst_key=inst_key)
    merged = tags.merge(nov, on="pmid", how="inner")
    cols = [country_col, "n_papers"]
    for c in novelty_cols:
        cols += [f"{c}_mean", f"{c}_median", f"{c}_std", f"{c}_n"]
    cols += ["low_n"]
    if merged.empty:
        return pd.DataFrame({col: [] for col in cols})

    summary = _summarise_novelty(merged, group_col=country_col, novelty_cols=novelty_cols)
    summary["low_n"] = summary["n_papers"] < int(low_n_threshold)
    return summary[cols]


def split_half_stability(
    paper_novelty_with_key: pd.DataFrame,
    *,
    key_col: str,
    novelty_col: str,
    seed: int,
    n_min: int = 30,
    mode: str = "random",
    year_col: str = "year",
    year_cut: float | None = None,
) -> dict[str, object]:
    """Split-half stability of a per-key novelty ranking — robustness for a
    per-institution (country / sector) origin finding.

    Topic-granularity Method-D (leaf-vs-mid) is **N/A by construction** for a finding
    keyed on institution ``country_code`` / ``type``: those keys do not change with
    topic grain. The grain-independent analogue is *split-half stability*: split the
    papers into two halves, rank the per-``key_col`` mean novelty within each half,
    and correlate the two rankings (Spearman ρ). A finding that reproduces in two
    independent halves of the data is robust; one driven by a few papers is not.

    Two split modes:

    * ``mode="random"`` — a reproducible 50/50 random split via
      :func:`numpy.random.default_rng` seeded with ``seed``. Tests whether the
      ranking is stable to *sampling* (the general-purpose robustness check).
    * ``mode="temporal"`` — early-vs-late split at ``year_cut`` (or the median of
      ``year_col`` if ``year_cut is None``); papers in the early half go left, the
      rest right. Tests whether the ranking is stable *over time* (a stricter,
      out-of-period check). ``seed`` is unused in this mode but kept for a uniform
      signature.

    Only keys with at least ``n_min`` papers in **both** halves are ranked, so a
    thin-sample key cannot inflate or deflate ρ.

    Parameters
    ----------
    paper_novelty_with_key :
        One row per (paper, key) observation carrying ``key_col`` and ``novelty_col``
        (and ``year_col`` if ``mode="temporal"``). This is the *exploded* long frame
        (a multi-country / multi-sector paper appears once per key it touches), so
        the per-key means match the headline tables.
    key_col :
        The per-institution key to rank, e.g. ``"country_code"`` or ``"type"``.
    novelty_col :
        The novelty measure to rank on, e.g. ``"sem_nov_mean"`` or ``"cd5"``.
    seed :
        Seed for the random split (use the locked session seed for determinism).
        Ignored when ``mode="temporal"``.
    n_min :
        Minimum papers a key must have in *each* half to be ranked. Default ``30``.
    mode :
        ``"random"`` (50/50 random) or ``"temporal"`` (early-vs-late). Default
        ``"random"``.
    year_col :
        Year column used only for ``mode="temporal"``. Default ``"year"``.
    year_cut :
        Explicit early/late boundary for ``mode="temporal"`` (papers with
        ``year_col < year_cut`` are "early"). If ``None`` the median year is used.

    Returns
    -------
    dict
        ``{"spearman_rho": float, "n_keys": int, "mode": str}``. ``spearman_rho`` is
        the Spearman rank correlation between the two halves' per-key mean-novelty
        rankings over the ``n_keys`` keys shared (≥ ``n_min`` papers) by both halves.
        ``spearman_rho`` is ``nan`` when fewer than two keys qualify (ρ undefined).

    Notes
    -----
    Pure and I/O-free: the caller supplies the in-memory exploded frame. Rows with a
    null ``novelty_col`` are dropped before splitting so the per-key means use only
    measured novelty (matching ``<c>_n`` in the headline tables).
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    if mode not in {"random", "temporal"}:
        raise ValueError(f"mode must be 'random' or 'temporal', got {mode!r}")

    work = paper_novelty_with_key.dropna(subset=[key_col, novelty_col]).copy()
    work = work.loc[~_blank_mask(work[key_col])]
    if work.empty:
        return {"spearman_rho": float("nan"), "n_keys": 0, "mode": mode}

    if mode == "random":
        rng = np.random.default_rng(seed)
        assign = rng.integers(0, 2, size=len(work))
        left = work.loc[assign == 0]
        right = work.loc[assign == 1]
    else:  # temporal
        if year_col not in work.columns:
            raise ValueError(f"mode='temporal' requires '{year_col}' in the frame")
        years = pd.to_numeric(work[year_col], errors="coerce")
        work = work.loc[years.notna()].copy()
        years = years.loc[work.index]
        cut = float(years.median()) if year_cut is None else float(year_cut)
        left = work.loc[years < cut]
        right = work.loc[years >= cut]

    def _ranked_means(frame: pd.DataFrame) -> pd.Series:
        grp = frame.groupby(key_col, dropna=True)[novelty_col]
        means = grp.mean()
        counts = grp.size()
        return means.loc[counts >= int(n_min)]

    left_means = _ranked_means(left)
    right_means = _ranked_means(right)
    shared = left_means.index.intersection(right_means.index)
    n_keys = int(len(shared))
    if n_keys < 2:
        return {"spearman_rho": float("nan"), "n_keys": n_keys, "mode": mode}

    rho, _ = spearmanr(
        left_means.loc[shared].to_numpy(),
        right_means.loc[shared].to_numpy(),
    )
    return {"spearman_rho": float(rho), "n_keys": n_keys, "mode": mode}


def bootstrap_delta_ci(
    paper_novelty_with_flag: pd.DataFrame,
    *,
    flag_col: str,
    novelty_col: str,
    seed: int,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> dict[str, object]:
    """Bootstrap CI for a (flag==True − flag==False) mean-novelty delta.

    Substantiates whether a *near-null* sector finding (e.g. company minus
    non-company mean novelty ≈ 0) is a genuine null — a bootstrap CI that **spans
    zero** — versus a small-but-real effect. This is the grain-independent robustness
    number for the per-institution sector contrast, where topic-granularity Method-D
    does not apply.

    The delta is bootstrapped by resampling the two groups **independently** with
    replacement (a stratified bootstrap that preserves each group's size), recomputing
    ``mean(novelty | flag==True) − mean(novelty | flag==False)`` on each replicate.

    Parameters
    ----------
    paper_novelty_with_flag :
        One row per observation carrying a boolean ``flag_col`` (e.g. ``is_company``)
        and ``novelty_col``. Rows with a null ``novelty_col`` are dropped.
    flag_col :
        Boolean column splitting the two groups (``True`` minus ``False``).
    novelty_col :
        Novelty measure to take the group-mean delta of.
    seed :
        Seed for :func:`numpy.random.default_rng` (use the locked session seed).
    n_boot :
        Number of bootstrap replicates. Default ``1000``.
    ci :
        Central CI mass, e.g. ``0.95`` for a 95% percentile interval. Default
        ``0.95``.

    Returns
    -------
    dict
        ``{"delta": float, "ci_low": float, "ci_high": float, "spans_zero": bool,
        "n_boot": int}``. ``delta`` is the point estimate on the full data;
        ``ci_low`` / ``ci_high`` are the percentile-bootstrap bounds; ``spans_zero``
        is ``True`` iff the interval contains 0 (the near-null verdict). Any value is
        ``nan`` and ``spans_zero`` ``True`` when either group is empty (delta
        undefined).

    Notes
    -----
    Pure and I/O-free. The percentile interval uses the
    ``(1−ci)/2`` and ``1−(1−ci)/2`` quantiles of the replicate deltas.
    """
    import numpy as np

    work = paper_novelty_with_flag.dropna(subset=[novelty_col]).copy()
    flag = work[flag_col].astype("bool")
    grp_true = work.loc[flag, novelty_col].to_numpy(dtype="float64")
    grp_false = work.loc[~flag, novelty_col].to_numpy(dtype="float64")

    if grp_true.size == 0 or grp_false.size == 0:
        return {
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "spans_zero": True,
            "n_boot": int(n_boot),
        }

    delta = float(grp_true.mean() - grp_false.mean())

    rng = np.random.default_rng(seed)
    n_t = grp_true.size
    n_f = grp_false.size
    boot = np.empty(int(n_boot), dtype="float64")
    for i in range(int(n_boot)):
        bt = grp_true[rng.integers(0, n_t, size=n_t)].mean()
        bf = grp_false[rng.integers(0, n_f, size=n_f)].mean()
        boot[i] = bt - bf

    alpha = (1.0 - float(ci)) / 2.0
    ci_low = float(np.quantile(boot, alpha))
    ci_high = float(np.quantile(boot, 1.0 - alpha))
    spans_zero = bool(ci_low <= 0.0 <= ci_high)
    return {
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "spans_zero": spans_zero,
        "n_boot": int(n_boot),
    }


def recombination_events(
    references_topics: pd.DataFrame,
    *,
    citing_col: str = "citing_pmid",
    ref_topic_col: str = "ref_topic_id",
) -> pd.DataFrame:
    """Per-paper recombination score from the topics its references span.

    Recombinant novelty: a paper whose references span **≥ 2 distinct topics** is
    recombining literatures. The reference→topic frame is supplied already joined
    (``ref_openalex_id → archetypes.openalex_id → topic_id`` is done upstream by the
    builder); this function only aggregates it.

    Two scores per citing paper:

    * ``n_distinct_topics`` — count of distinct referenced topics (the simplest
      recombination breadth; ``≥ 2`` ⇒ a recombination event).
    * ``topic_entropy`` — Shannon entropy (natural log) of the *distribution* of
      references across topics, ``-Σ p log p`` where ``p`` is the share of a paper's
      resolved references falling in each topic. ``0.0`` for a single-topic paper;
      higher when references are spread evenly across many topics.

    Parameters
    ----------
    references_topics :
        Long frame, one row per (citing paper, *resolved* reference), carrying
        ``citing_col`` and ``ref_topic_col``. Rows with a missing topic (an
        unresolved reference) must be filtered upstream or are dropped here.
    citing_col :
        Citing-paper key, default ``"citing_pmid"``.
    ref_topic_col :
        Resolved referenced-topic key, default ``"ref_topic_id"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[citing_col, "n_references_resolved", "n_distinct_topics",
        "topic_entropy", "is_recombination"]``, one row per citing paper with ≥ 1
        resolved reference. ``n_references_resolved`` is the count of topic-resolved
        references; ``is_recombination`` is ``n_distinct_topics >= 2``. Sorted by
        ``n_distinct_topics`` then ``topic_entropy`` descending (most recombinant
        first). Empty input yields an empty frame with these columns.

    Notes
    -----
    Coverage: only references that resolve to a corpus topic count; references to
    out-of-corpus / topicless works are invisible, so scores are a *lower bound* on
    true recombination breadth (panel-conditional, like every origin claim).
    """
    import numpy as np
    import pandas as pd

    cols = [
        citing_col,
        "n_references_resolved",
        "n_distinct_topics",
        "topic_entropy",
        "is_recombination",
    ]
    work = references_topics.loc[:, [citing_col, ref_topic_col]].dropna(
        subset=[citing_col, ref_topic_col]
    )
    if work.empty:
        return pd.DataFrame({c: [] for c in cols})

    # Reference counts per (citing paper, referenced topic).
    per_topic = (
        work.groupby([citing_col, ref_topic_col], dropna=True).size().rename("n").reset_index()
    )

    def _entropy(counts: np.ndarray) -> float:
        total = counts.sum()
        if total <= 0:
            return 0.0
        p = counts / total
        # 0 * log 0 = 0; only positive p contribute.
        return float(-(p * np.log(p)).sum())

    rows: list[dict[str, object]] = []
    for paper, grp in per_topic.groupby(citing_col, dropna=True):
        counts = grp["n"].to_numpy(dtype="float64")
        n_distinct = int(len(counts))
        rows.append(
            {
                citing_col: paper,
                "n_references_resolved": int(counts.sum()),
                "n_distinct_topics": n_distinct,
                "topic_entropy": _entropy(counts),
                "is_recombination": n_distinct >= 2,
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    out["n_references_resolved"] = out["n_references_resolved"].astype("int64")
    out["n_distinct_topics"] = out["n_distinct_topics"].astype("int64")
    out["topic_entropy"] = out["topic_entropy"].astype("float64")
    out["is_recombination"] = out["is_recombination"].astype("bool")
    out = out.sort_values(
        ["n_distinct_topics", "topic_entropy"], ascending=[False, False]
    ).reset_index(drop=True)
    return out


def recombination_by_topic(
    events: pd.DataFrame,
    paper_topic: pd.DataFrame,
    *,
    citing_col: str = "citing_pmid",
    paper_key: str = "pmid",
    topic_col: str = "topic_id",
) -> pd.DataFrame:
    """Per-(focal) topic recombination summary from per-paper events.

    Joins the per-paper recombination scores (:func:`recombination_events`) to each
    citing paper's *own* focal topic, then summarises recombination breadth within
    each focal topic — i.e. "which topics are written by recombining references from
    many other topics?".

    Parameters
    ----------
    events :
        Output of :func:`recombination_events` (per citing paper).
    paper_topic :
        Frame mapping a paper to its focal topic — ``[paper_key, topic_col]``.
    citing_col :
        Citing-paper key in ``events``, default ``"citing_pmid"``.
    paper_key :
        Paper key in ``paper_topic``, default ``"pmid"``.
    topic_col :
        Focal-topic column in ``paper_topic``, default ``"topic_id"``.

    Returns
    -------
    pandas.DataFrame
        One row per focal ``topic_col`` with ``n_papers`` (papers in the topic with
        a recombination score), ``mean_distinct_topics``, ``mean_entropy``, and
        ``recombination_rate`` (share of the topic's papers with
        ``is_recombination``). Sorted by ``mean_distinct_topics`` descending. Both
        keys are cast to string for a robust join. Empty input → empty frame.

    Notes
    -----
    The focal-topic join is on ``pmid``; ``events`` keys on ``citing_pmid`` (the
    paper that *made* the references), so a paper appears once with its own topic.
    """
    import pandas as pd

    cols = [
        topic_col,
        "n_papers",
        "mean_distinct_topics",
        "mean_entropy",
        "recombination_rate",
    ]
    if events.empty or paper_topic.empty:
        return pd.DataFrame({c: [] for c in cols})

    ev = events.copy()
    ev[citing_col] = ev[citing_col].astype("string")
    pt = paper_topic.loc[:, [paper_key, topic_col]].copy()
    pt[paper_key] = pt[paper_key].astype("string")

    merged = ev.merge(pt, left_on=citing_col, right_on=paper_key, how="inner")
    if merged.empty:
        return pd.DataFrame({c: [] for c in cols})

    grouped = merged.groupby(topic_col, dropna=True)
    out = grouped.agg(
        n_papers=("n_distinct_topics", "size"),
        mean_distinct_topics=("n_distinct_topics", "mean"),
        mean_entropy=("topic_entropy", "mean"),
        recombination_rate=("is_recombination", "mean"),
    ).reset_index()
    out["n_papers"] = out["n_papers"].astype("int64")
    out = out.sort_values("mean_distinct_topics", ascending=False).reset_index(drop=True)
    return out[cols]
