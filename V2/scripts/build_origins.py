"""Build the V2-S06 novelty-origin tables (sector, geography, recombination).

Runnable I/O driver for :mod:`scifield.cartography.origins`. This script owns *all*
the I/O the pure module deliberately avoids:

1. Load ``data/v1/archetypes.parquet`` (pmid, topic_id, novelty columns,
   openalex_id) — the noise topic (``topic_id == -1``) is dropped.
2. Load ``data/v1/enrichment/paper_institutions.parquet`` and
   ``data/v1/enrichment/institutions.parquet`` for the sector / geography axes.
3. For recombination, resolve references → topics via DuckDB:
   ``references_out.ref_openalex_id -> archetypes.openalex_id -> topic_id`` (per the
   T2 hand-off: ``ref_pmid_if_known`` is 100% empty, so ``ref_openalex_id`` is the
   only path). This is the one heavy join — done in DuckDB, then handed as an
   already-joined frame to the pure module.
4. Write ``V2/data/origins/{sector_novelty,geo_novelty,recombination,
   recombination_by_topic}.parquet``, each with a ``record_run`` provenance sidecar
   (task ``V2-S06``).
5. Print the headline tables (company vs non-company novelty; top/bottom countries;
   most-recombinant topics with ``top_words``).

Usage
-----
``.venv/bin/python V2/scripts/build_origins.py``

$0 / read-only: opens DuckDB with ``read_only`` ATTACH; no network, no GPU. The
OpenAlex ``grants`` (funding origin) and Semantic Scholar citation intents are
DEFERRED — see the V2 notebook coverage note (a schema change + re-harvest is
required and is out of scope for this $0 session).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from scifield.cartography.origins import (
    COMPANY_TYPE,
    novelty_by_country,
    novelty_by_institution_type,
    recombination_by_topic,
    recombination_events,
)
from scifield.repro import record_run

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHETYPES = REPO_ROOT / "data/v1/archetypes.parquet"
PAPER_INST = REPO_ROOT / "data/v1/enrichment/paper_institutions.parquet"
INSTITUTIONS = REPO_ROOT / "data/v1/enrichment/institutions.parquet"
REFERENCES_OUT = REPO_ROOT / "data/v1/enrichment/references_out.parquet"
TOPIC_HIERARCHY = REPO_ROOT / "data/v1/topic_hierarchy.parquet"
OUT_DIR = REPO_ROOT / "V2/data/origins"

NOISE_TOPIC_ID = -1
NOVELTY_COLS = ["sem_nov_mean", "sem_nov_min", "cd5", "cd10"]
LOW_N_THRESHOLD = 30


def load_paper_novelty() -> pd.DataFrame:
    """Per-paper novelty frame (noise topic dropped).

    Returns
    -------
    pandas.DataFrame
        ``["pmid"(str), "topic_id", "openalex_id"] + NOVELTY_COLS``. ``pmid`` is a
        string for joining to ``paper_institutions``; the noise topic is excluded.
    """
    cols = ["pmid", "topic_id", "openalex_id", *NOVELTY_COLS]
    df = pd.read_parquet(ARCHETYPES, columns=cols)
    df = df.loc[df["topic_id"] != NOISE_TOPIC_ID].copy()
    df["pmid"] = df["pmid"].astype("string")
    return df.reset_index(drop=True)


def load_references_topics() -> pd.DataFrame:
    """Resolve references to referenced-topic ids via DuckDB.

    Resolution path (T2 hand-off): ``references_out.ref_openalex_id ->
    archetypes.openalex_id -> topic_id``. Only references resolving to a non-noise
    corpus topic survive — unresolved / out-of-corpus references are dropped, so the
    recombination breadth is a panel-conditional lower bound.

    Returns
    -------
    pandas.DataFrame
        ``["citing_pmid"(str), "ref_topic_id"(int)]`` — one row per resolved
        reference.
    """
    con = duckdb.connect()
    query = f"""
        SELECT
            CAST(r.citing_pmid AS VARCHAR) AS citing_pmid,
            a.topic_id                     AS ref_topic_id
        FROM '{REFERENCES_OUT.as_posix()}' r
        JOIN '{ARCHETYPES.as_posix()}' a
            ON r.ref_openalex_id = a.openalex_id
        WHERE a.topic_id <> {NOISE_TOPIC_ID}
          AND r.ref_openalex_id IS NOT NULL
          AND r.ref_openalex_id <> ''
    """
    df = con.execute(query).df()
    con.close()
    df["citing_pmid"] = df["citing_pmid"].astype("string")
    df["ref_topic_id"] = df["ref_topic_id"].astype("int64")
    return df


def _write(df: pd.DataFrame, name: str, inputs: dict[str, Path], config: dict) -> Path:
    """Write one origin table + a record_run sidecar; return its path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    record_run(artifact_path=out_path, inputs=inputs, config=config)
    return out_path


def main() -> None:
    """Build + write all four origin tables and print the headline numbers."""
    print("Loading per-paper novelty (archetypes, noise topic dropped)...")
    novelty = load_paper_novelty()
    print(f"  {len(novelty):,} papers; novelty cols = {NOVELTY_COLS}")

    print("Loading paper_institutions + institutions...")
    paper_inst = pd.read_parquet(PAPER_INST, columns=["pmid", "institution_canonical_id"])
    paper_inst["pmid"] = paper_inst["pmid"].astype("string")
    institutions = pd.read_parquet(
        INSTITUTIONS, columns=["institution_canonical_id", "type", "country_code"]
    )
    print(f"  {len(paper_inst):,} paper-institution rows; {len(institutions):,} institutions")

    # --- Sector x novelty ---------------------------------------------------
    print("\nComputing sector x novelty...")
    sector = novelty_by_institution_type(
        novelty, paper_inst, institutions, novelty_cols=NOVELTY_COLS
    )
    sector_path = _write(
        sector,
        "sector_novelty",
        {"archetypes": ARCHETYPES, "paper_institutions": PAPER_INST, "institutions": INSTITUTIONS},
        {"task": "V2-S06", "axis": "sector", "novelty_cols": NOVELTY_COLS, "rule": "any-author"},
    )

    # --- Geography x novelty ------------------------------------------------
    print("Computing geography x novelty...")
    geo = novelty_by_country(
        novelty,
        paper_inst,
        institutions,
        novelty_cols=NOVELTY_COLS,
        low_n_threshold=LOW_N_THRESHOLD,
    )
    geo_path = _write(
        geo,
        "geo_novelty",
        {"archetypes": ARCHETYPES, "paper_institutions": PAPER_INST, "institutions": INSTITUTIONS},
        {
            "task": "V2-S06",
            "axis": "geography",
            "novelty_cols": NOVELTY_COLS,
            "low_n_threshold": LOW_N_THRESHOLD,
        },
    )

    # --- Topic recombination ------------------------------------------------
    print("Resolving references -> topics (DuckDB) for recombination...")
    refs_topics = load_references_topics()
    print(f"  {len(refs_topics):,} topic-resolved references")
    events = recombination_events(refs_topics)
    recomb_path = _write(
        events,
        "recombination",
        {"archetypes": ARCHETYPES, "references_out": REFERENCES_OUT},
        {"task": "V2-S06", "axis": "recombination", "metric": "distinct_topics+entropy"},
    )
    by_topic = recombination_by_topic(events, novelty.loc[:, ["pmid", "topic_id"]])
    # Attach top_words for the headline (focal topic interpretability).
    hierarchy = pd.read_parquet(TOPIC_HIERARCHY, columns=["topic_id", "top_words"])
    by_topic = by_topic.merge(hierarchy, on="topic_id", how="left")
    by_topic_path = _write(
        by_topic,
        "recombination_by_topic",
        {
            "archetypes": ARCHETYPES,
            "references_out": REFERENCES_OUT,
            "topic_hierarchy": TOPIC_HIERARCHY,
        },
        {"task": "V2-S06", "axis": "recombination_by_topic"},
    )

    # --- Headline prints ----------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTOR x NOVELTY  (any-author-affiliation rule; blank type excluded)")
    print(f"  path: {sector_path}")
    show = [
        "type",
        "n_papers",
        "sem_nov_mean_mean",
        "sem_nov_mean_median",
        "cd5_mean",
        "cd5_median",
    ]
    sect = sector.loc[:, show].sort_values("sem_nov_mean_mean", ascending=False)
    print(sect.to_string(index=False))
    if COMPANY_TYPE in set(sector["type"]):
        co = sector.set_index("type").loc[COMPANY_TYPE]
        noncompany = sector.loc[sector["type"] != COMPANY_TYPE]
        nc_mean = (noncompany["sem_nov_mean_mean"] * noncompany["n_papers"]).sum() / noncompany[
            "n_papers"
        ].sum()
        print(
            f"\n  COMPANY sem_nov_mean = {co['sem_nov_mean_mean']:.4f} (n={int(co['n_papers'])})"
            f"  vs  NON-COMPANY (paper-weighted) = {nc_mean:.4f}"
            f"  -> delta = {co['sem_nov_mean_mean'] - nc_mean:+.4f}"
        )
        co_cd = co["cd5_mean"]
        nc_cd = (noncompany["cd5_mean"] * noncompany["cd5_n"]).sum() / noncompany["cd5_n"].sum()
        print(
            f"  COMPANY cd5 = {co_cd:.4f}  vs  NON-COMPANY (disruption-weighted) = {nc_cd:.4f}"
            f"  -> delta = {co_cd - nc_cd:+.4f}"
        )

    print("\n" + "=" * 70)
    print("GEOGRAPHY x NOVELTY  (blank country excluded; low_n flagged < 30 papers)")
    print(f"  path: {geo_path}")
    well = geo.loc[~geo["low_n"]].sort_values("sem_nov_mean_mean", ascending=False)
    print(f"  {len(well)} well-sampled countries (>= {LOW_N_THRESHOLD} papers)")
    cols = ["country_code", "n_papers", "sem_nov_mean_mean", "cd5_mean", "low_n"]
    print("  -- TOP 8 by semantic novelty --")
    print(well.loc[:, cols].head(8).to_string(index=False))
    print("  -- BOTTOM 5 by semantic novelty --")
    print(well.loc[:, cols].tail(5).to_string(index=False))

    print("\n" + "=" * 70)
    print("TOPIC RECOMBINATION  (references span >= 2 distinct topics)")
    print(f"  paths: {recomb_path}\n         {by_topic_path}")
    n_recomb = int(events["is_recombination"].sum())
    print(
        f"  papers with >=1 resolved ref: {len(events):,}; "
        f"recombination events (>=2 topics): {n_recomb:,} "
        f"({100 * n_recomb / max(len(events), 1):.1f}%)"
    )
    print(
        f"  mean distinct referenced topics/paper: {events['n_distinct_topics'].mean():.2f}; "
        f"mean entropy: {events['topic_entropy'].mean():.3f}"
    )
    print("  -- TOP 10 most-recombinant focal topics (by mean distinct ref-topics) --")
    top = by_topic.head(10).copy()
    top["words"] = top["top_words"].apply(
        lambda w: ", ".join(list(w)[:5]) if w is not None and len(w) else ""
    )
    print(
        top.loc[
            :, ["topic_id", "n_papers", "mean_distinct_topics", "recombination_rate", "words"]
        ].to_string(index=False)
    )

    print("\nDEFERRED (coverage note): funding origin (OpenAlex grants — parsed but not")
    print("  persisted; needs schema change + re-harvest) and citation intents")
    print("  (Semantic Scholar; citation_intents.parquet empty). $0 session: no re-pull.")
    print("\nDone.")


if __name__ == "__main__":
    main()
