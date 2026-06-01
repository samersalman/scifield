"""V1-S11 dual-novelty 2x2 archetype binning (semantic x structural).

Each corpus paper carries two *kinds* of novelty: a **semantic** score (how far
its embedding sits from prior same-field work -- :mod:`scifield.novelty.semantic`)
and a **structural** score (how much it disrupts the citation graph -- the CD
index in :mod:`scifield.novelty.cd_index`). Crossing one semantic axis with one
structural axis at their corpus-wide medians yields a 2x2 grid of qualitative
archetypes:

==================  =========================  ==========================
                    structural High (>= med)   structural Low (< med)
==================  =========================  ==========================
semantic High       ``disruptive-novel``       ``novel-consolidating``
semantic Low        ``conventional-disruptive``  ``incremental``
==================  =========================  ==========================

The semantic axis is always listed FIRST and the structural axis SECOND, both in
the quadrant code (``"HH"``, ``"HL"``, ``"LH"``, ``"LL"``) and in the prose label.

High/Low tie rule
-----------------
A value is **High** iff ``value >= threshold`` (greater-or-equal is High); it is
**Low** only when strictly below. See :func:`assign_quadrant`.

Null axes
---------
A paper missing EITHER axis value (NaN -- e.g. an earliest-in-field paper with no
semantic score, or a paper with no harvested forward citations and thus no CD)
gets a **null archetype label** (``pd.NA``) for any pairing that touches the
missing axis. Such rows are *excluded* from the 2x2 (not dropped from the frame),
so downstream cross-tabs should ignore null labels rather than impute them.

Emitted archetype columns
--------------------------
:func:`bin_archetypes` / :func:`compute_archetypes` emit exactly these four
label columns (semantic metric first, structural metric second):

* ``arch_mean_cd5``  -- ``sem_nov_mean`` x ``cd5``
* ``arch_mean_cd10`` -- ``sem_nov_mean`` x ``cd10``
* ``arch_min_cd5``   -- ``sem_nov_min``  x ``cd5``
* ``arch_min_cd10``  -- ``sem_nov_min``  x ``cd10``

The suffix is ``<short-semantic>_<structural>`` where the short semantic form
strips the ``sem_nov_`` prefix (``sem_nov_mean`` -> ``mean``) and the structural
metric is used verbatim (``cd5`` -> ``cd5``). See :func:`archetype_column_name`.

Thresholds
----------
:func:`compute_archetypes` builds a ``thresholds`` dict keyed by the four metric
column names (``"sem_nov_mean"``, ``"sem_nov_min"``, ``"cd5"``, ``"cd10"``), each
value the corpus-wide median over the **complete-case subset** -- the rows where
all four metric axes are simultaneously non-NaN (NOT a per-column NaN-skip). The
same subset defines ``info["n_complete_cases"]``, so the four medians and the
count are derived from one identical set of rows. The dict is returned inside
``info`` for the CLI sidecar.

Deviations from the plan
------------------------
None. Column names, label mapping, tie rule, and threshold keys all follow the
V1-S11 spec exactly. Real complete-case counts are COMPUTED, never hardcoded.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "QUADRANT_LABELS",
    "archetype_column_name",
    "assign_quadrant",
    "bin_archetypes",
    "compute_archetypes",
]

#: Fixed map from a two-letter quadrant code (semantic axis first, structural
#: axis second) to its prose archetype label.
QUADRANT_LABELS: dict[str, str] = {
    "HH": "disruptive-novel",
    "HL": "novel-consolidating",
    "LH": "conventional-disruptive",
    "LL": "incremental",
}


# --------------------------------------------------------------------------- #
# Pure logic core (no I/O -- unit-testable on in-memory data)
# --------------------------------------------------------------------------- #


def assign_quadrant(*, value: float, threshold: float) -> str:
    """Classify a single scalar onto the High/Low axis at ``threshold``.

    Parameters
    ----------
    value :
        The metric value. Assumed **non-NaN**; callers must screen NaNs before
        invoking this helper (see :func:`bin_archetypes`, which leaves the
        archetype label null for any NaN axis).
    threshold :
        The cut point (corpus-wide median, in the driver).

    Returns
    -------
    str
        ``"H"`` if ``value >= threshold`` (greater-or-equal is High), else
        ``"L"``.

    Notes
    -----
    The at-threshold tie goes to ``"H"``: a value exactly equal to the threshold
    is classified High. Passing a NaN ``value`` here yields ``"L"`` (since
    ``NaN >= x`` is ``False``), which is *not* the intended behaviour for null
    axes -- screen NaNs upstream instead.
    """
    return "H" if value >= threshold else "L"


def archetype_column_name(*, semantic_metric: str, structural_metric: str) -> str:
    """Derive the output column name for a (semantic, structural) pairing.

    Parameters
    ----------
    semantic_metric :
        A semantic metric column name, e.g. ``"sem_nov_mean"`` or
        ``"sem_nov_min"``. The leading ``"sem_nov_"`` prefix is stripped to form
        the short tag (``"mean"`` / ``"min"``); a name without that prefix is
        used verbatim.
    structural_metric :
        A structural metric column name, used verbatim, e.g. ``"cd5"`` / ``"cd10"``.

    Returns
    -------
    str
        ``"arch_<short-semantic>_<structural>"``, e.g.
        ``archetype_column_name(semantic_metric="sem_nov_mean",
        structural_metric="cd5")`` -> ``"arch_mean_cd5"``.

    Notes
    -----
    Deterministic and pure so the column-naming contract is testable in
    isolation from the binning logic.
    """
    prefix = "sem_nov_"
    has_prefix = semantic_metric.startswith(prefix)
    short = semantic_metric[len(prefix) :] if has_prefix else semantic_metric
    return f"arch_{short}_{structural_metric}"


def bin_archetypes(
    df: pd.DataFrame,
    *,
    pairings: Sequence[tuple[str, str]],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Add one 2x2 archetype label column per (semantic, structural) pairing.

    Pure and I/O-free: operates on an in-memory frame and returns a shallow copy
    with the new label columns appended. For each pairing, a row whose semantic
    *or* structural value is NaN receives a null label (``pd.NA``); it is NOT
    dropped and never raises.

    Parameters
    ----------
    df :
        Frame containing every metric column referenced by ``pairings`` and
        ``thresholds``.
    pairings :
        Sequence of ``(semantic_metric, structural_metric)`` column-name tuples.
        Each produces a label column named via :func:`archetype_column_name`.
    thresholds :
        Mapping from metric column name to its float cut point. Must contain a
        key for every metric appearing in ``pairings``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with one object-dtype label column per pairing. Labels
        are drawn from :data:`QUADRANT_LABELS`; rows with a NaN on either axis
        hold ``pd.NA``.

    Notes
    -----
    The semantic axis is mapped to the first letter of the quadrant code and the
    structural axis to the second, so e.g. semantic-High / structural-Low ->
    ``"HL"`` -> ``"novel-consolidating"``.
    """
    import numpy as np
    import pandas as pd

    out = df.copy()
    for semantic_metric, structural_metric in pairings:
        sem = out[semantic_metric]
        struct = out[structural_metric]
        sem_thr = thresholds[semantic_metric]
        struct_thr = thresholds[structural_metric]

        # Vectorised High/Low per axis; >= is High (tie -> High).
        sem_high = sem.to_numpy(dtype="float64") >= sem_thr
        struct_high = struct.to_numpy(dtype="float64") >= struct_thr

        sem_code = np.where(sem_high, "H", "L")
        struct_code = np.where(struct_high, "H", "L")
        codes = np.char.add(sem_code, struct_code)

        labels = pd.Series([QUADRANT_LABELS[c] for c in codes], index=out.index, dtype="object")
        # Any NaN axis -> null label (excluded from the 2x2, not dropped).
        null_mask = sem.isna().to_numpy() | struct.isna().to_numpy()
        labels[null_mask] = pd.NA

        out[
            archetype_column_name(
                semantic_metric=semantic_metric, structural_metric=structural_metric
            )
        ] = labels

    return out


# --------------------------------------------------------------------------- #
# I/O driver
# --------------------------------------------------------------------------- #


def compute_archetypes(
    *,
    duckdb_path: Path,
    semantic_path: Path,
    cd_index_path: Path,
    semantic_metrics: Sequence[str] = ("sem_nov_mean", "sem_nov_min"),
    structural_metrics: Sequence[str] = ("cd5", "cd10"),
) -> tuple[pd.DataFrame, dict]:
    """Merge semantic + structural novelty and bin every paper into 2x2 archetypes.

    Reads the semantic-novelty and CD-index parquets, left-joins them on ``pmid``
    (semantic frame is the paper universe), enriches with ``journal`` /
    ``cited_by_count`` from DuckDB, computes corpus-wide median thresholds over
    the complete-case subset (rows with all four metric axes non-NaN), and
    appends the four archetype label columns via :func:`bin_archetypes`.

    No network access. DuckDB is opened read-only; if ``papers_distinct`` is
    absent, it is created on a read-write fallback connection (mirrors
    :mod:`scifield.novelty.semantic`).

    Parameters
    ----------
    duckdb_path :
        Path to ``papers.duckdb`` providing the ``papers_distinct`` view
        (``journal`` per pmid) and ``openalex_works`` (``cited_by_count``).
    semantic_path :
        ``novelty_semantic.parquet`` with columns ``pmid (int64), topic_id,
        year, n_prior, sem_nov_mean, sem_nov_min``.
    cd_index_path :
        ``cd_index.parquet`` with columns ``pmid (Int64), openalex_id, cd5,
        cd10, n_citers, n_window_5, n_window_10``.
    semantic_metrics :
        Semantic axis metric column names to pair. Default
        ``("sem_nov_mean", "sem_nov_min")``.
    structural_metrics :
        Structural axis metric column names to pair. Default ``("cd5", "cd10")``.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        ``(df, info)``. ``df`` carries (at least, in this order): ``pmid, year,
        journal, topic_id, sem_nov_mean, sem_nov_min, cd5, cd10, cited_by_count,
        cited_by_pctile_within_year, arch_mean_cd5, arch_mean_cd10,
        arch_min_cd5, arch_min_cd10``. ``info`` holds ``{"thresholds": {...},
        "n_complete_cases": int, "n_total": int}`` where complete-cases counts
        rows with all four metric columns (``sem_nov_mean, sem_nov_min, cd5,
        cd10``) non-NaN.

    Notes
    -----
    Year is taken authoritatively from the semantic parquet; only ``journal`` and
    ``cited_by_count`` are pulled from DuckDB to avoid column collisions. The
    DuckDB ``pmid`` is ``VARCHAR`` while the parquets are int64, so the SQL casts
    ``CAST(pmid AS BIGINT) AS pmid`` on both joins.
    """
    import numpy as np
    import pandas as pd

    from scifield.thematic.dedup import ensure_papers_distinct_view

    duckdb_path = Path(duckdb_path)
    semantic_path = Path(semantic_path)
    cd_index_path = Path(cd_index_path)

    # --- Semantic frame is the paper universe (carries topic_id + year). ----
    sem = pd.read_parquet(semantic_path)
    sem["pmid"] = sem["pmid"].astype(np.int64)

    # --- CD index: drop null-pmid rows, cast to int64, keep cd columns. -----
    cd = pd.read_parquet(cd_index_path)
    cd = cd.dropna(subset=["pmid"]).copy()
    cd["pmid"] = cd["pmid"].astype(np.int64)
    cd_cols = ["pmid", "openalex_id", "cd5", "cd10"]
    cd = cd[[c for c in cd_cols if c in cd.columns]]

    df = sem.merge(cd, on="pmid", how="left")

    # --- journal (papers_distinct) + cited_by_count (openalex_works). -------
    journal_df = _load_lookup(
        duckdb_path,
        ensure_papers_distinct_view,
        "SELECT CAST(pmid AS BIGINT) AS pmid, journal FROM papers_distinct",
    )
    cites_df = _load_lookup(
        duckdb_path,
        ensure_papers_distinct_view,
        "SELECT CAST(pmid AS BIGINT) AS pmid, cited_by_count FROM openalex_works",
    )
    journal_df["pmid"] = journal_df["pmid"].astype(np.int64)
    cites_df["pmid"] = cites_df["pmid"].astype(np.int64)
    # A pmid may appear more than once in openalex_works; keep first occurrence.
    cites_df = cites_df.drop_duplicates(subset=["pmid"], keep="first")
    journal_df = journal_df.drop_duplicates(subset=["pmid"], keep="first")

    df = df.merge(journal_df, on="pmid", how="left")
    df = df.merge(cites_df, on="pmid", how="left")

    # --- Within-year citation percentile (0-1); null where no count. --------
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce")
    df["cited_by_pctile_within_year"] = df.groupby("year")["cited_by_count"].rank(pct=True)

    # --- Corpus-wide median thresholds over the COMPLETE-CASE subset. -------
    # Per V1-S11 S1: every threshold is the median over rows where ALL FOUR
    # metric columns are simultaneously non-NaN (not a per-column NaN-skip), so
    # the four medians and ``n_complete_cases`` derive from one identical subset.
    complete = df.dropna(subset=[*semantic_metrics, *structural_metrics])
    thresholds: dict[str, float] = {
        col: float(complete[col].median()) for col in (*semantic_metrics, *structural_metrics)
    }

    # --- Bin into the 2x2 archetypes. ---------------------------------------
    pairings = [(s, t) for s in semantic_metrics for t in structural_metrics]
    df = bin_archetypes(df, pairings=pairings, thresholds=thresholds)

    # --- Order the emitted columns; keep any extras after them. -------------
    arch_cols = [archetype_column_name(semantic_metric=s, structural_metric=t) for s, t in pairings]
    lead = [
        "pmid",
        "year",
        "journal",
        "topic_id",
        "sem_nov_mean",
        "sem_nov_min",
        "cd5",
        "cd10",
        "cited_by_count",
        "cited_by_pctile_within_year",
        *arch_cols,
    ]
    ordered = [c for c in lead if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    df = df[ordered + rest]

    info: dict = {
        "thresholds": thresholds,
        "n_complete_cases": int(len(complete)),
        "n_total": int(len(df)),
    }
    return df, info


def _load_lookup(duckdb_path: Path, ensure_view, query: str) -> pd.DataFrame:
    """Run ``query`` against DuckDB read-only, creating the view on fallback.

    Tries a read-only connection first; on a missing-catalog error opens a
    read-write connection, ensures ``papers_distinct`` exists, and retries
    (mirrors :func:`scifield.novelty.semantic._load_years`).
    """
    import duckdb

    try:
        con = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            return con.execute(query).df()
        finally:
            con.close()
    except duckdb.CatalogException:
        con = duckdb.connect(str(duckdb_path), read_only=False)
        try:
            ensure_view(con)
            return con.execute(query).df()
        finally:
            con.close()
