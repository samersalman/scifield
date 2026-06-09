"""Build the V2 canonical journal-flow tables (leaf + mid grain).

Runnable I/O driver for :mod:`scifield.cartography.flow`. This script owns *all* the
I/O the pure module deliberately avoids:

1. Load ``data/v1/archetypes.parquet`` (pmid, topic_id, year, novelty).
2. Resolve the **canonical** ``journal_slug`` by joining
   ``pmid -> papers_distinct.journal_slug`` from ``data/v1/papers.duckdb``
   (CAST archetypes.pmid BIGINT -> VARCHAR — see V2/CONTEXT.md §3). This collapses
   the jama_surg / Archives-of-surgery display-name split into the clean ten slugs.
3. Join ``data/v1/topic_hierarchy.parquet`` for ``mid_level_id`` (mid grain).
4. Pull corpus-internal ``CITES`` edges from the read-only Kuzu graph and feed them
   to :func:`scifield.cartography.flow.build_flow_table` to compute per-cell
   ``in_flow`` / ``out_flow``.
5. Write ``V2/data/flow/flow_leaf.parquet`` and ``V2/data/flow/flow_mid.parquet``,
   each with a ``record_run`` provenance sidecar.

The noise topic (``topic_id == -1``, ~21k rows) is dropped: it is not a real topic
and would pollute first-appearance anchors. The stray ``year == 2026`` (723 rows) is
kept — it is a real, if partial, harvest year; downstream layers that need a clean
held-out cut should filter on ``year`` themselves.

Usage
-----
``.venv/bin/python V2/scripts/build_flow.py``            # both grains (default)
``.venv/bin/python V2/scripts/build_flow.py --grain leaf``
``.venv/bin/python V2/scripts/build_flow.py --no-citations``   # skip the Kuzu pull

$0 / read-only: opens DuckDB and Kuzu with ``read_only=True``; no network, no GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from scifield.cartography.flow import assert_canonical_journals, build_flow_table
from scifield.repro import record_run

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHETYPES = REPO_ROOT / "data/v1/archetypes.parquet"
PAPERS_DUCKDB = REPO_ROOT / "data/v1/papers.duckdb"
TOPIC_HIERARCHY = REPO_ROOT / "data/v1/topic_hierarchy.parquet"
KUZU_GRAPH = REPO_ROOT / "data/v1/kuzu_graph"
OUT_DIR = REPO_ROOT / "V2/data/flow"

NOISE_TOPIC_ID = -1


def load_papers_topics() -> pd.DataFrame:
    """Load archetypes joined to canonical journal_slug and mid_level_id.

    Returns
    -------
    pandas.DataFrame
        One row per paper with ``["pmid", "topic_id", "mid_level_id",
        "journal_slug", "year", "sem_nov_mean"]``. ``pmid`` is a string; the noise
        topic is dropped. ``journal_slug`` comes from ``papers_distinct`` (canonical,
        ten slugs), never the archetypes display name.
    """
    con = duckdb.connect()
    con.execute(f"ATTACH '{PAPERS_DUCKDB.as_posix()}' AS pdb (READ_ONLY)")
    query = f"""
        SELECT
            CAST(a.pmid AS VARCHAR)  AS pmid,
            a.topic_id               AS topic_id,
            th.mid_level_id          AS mid_level_id,
            p.journal_slug           AS journal_slug,
            a.year                   AS year,
            a.sem_nov_mean           AS sem_nov_mean
        FROM '{ARCHETYPES.as_posix()}' a
        JOIN pdb.papers_distinct p
            ON CAST(a.pmid AS VARCHAR) = p.pmid
        LEFT JOIN '{TOPIC_HIERARCHY.as_posix()}' th
            ON a.topic_id = th.topic_id
        WHERE a.topic_id <> {NOISE_TOPIC_ID}
    """
    df = con.execute(query).df()
    con.close()
    df["pmid"] = df["pmid"].astype("string")
    # Fail loudly here too — the resolver must yield only the clean ten.
    assert_canonical_journals(df["journal_slug"])
    return df


def load_citation_edges() -> pd.DataFrame:
    """Load corpus-internal CITES edges (citing_pmid -> cited_pmid) from Kuzu.

    Returns
    -------
    pandas.DataFrame
        Columns ``["citing_pmid", "cited_pmid"]`` as strings. Read-only Kuzu access.
    """
    import kuzu

    db = kuzu.Database(KUZU_GRAPH.as_posix(), read_only=True)
    con = kuzu.Connection(db)
    cypher = """
        MATCH (a:Paper)-[c:CITES]->(b:Paper)
        RETURN a.pmid AS citing_pmid, b.pmid AS cited_pmid
    """
    edges = con.execute(cypher).get_as_df()
    edges["citing_pmid"] = edges["citing_pmid"].astype("string")
    edges["cited_pmid"] = edges["cited_pmid"].astype("string")
    return edges


def build_and_write(
    papers_topics: pd.DataFrame,
    *,
    grain: str,
    citation_edges: pd.DataFrame | None,
    min_papers: int,
) -> Path:
    """Build one grain's flow table, write parquet + sidecar, print an audit.

    Parameters
    ----------
    papers_topics :
        The per-paper frame from :func:`load_papers_topics`.
    grain :
        ``"leaf"`` or ``"mid"``.
    citation_edges :
        Edge list (or ``None`` to emit the not-computed sentinel columns).
    min_papers :
        Per-year first-appearance threshold.

    Returns
    -------
    pathlib.Path
        The written parquet path.
    """
    flow = build_flow_table(
        papers_topics,
        citation_edges=citation_edges,
        grain=grain,
        min_papers=min_papers,
    )
    topic_key = "topic_id" if grain == "leaf" else "mid_level_id"
    out_path = OUT_DIR / f"flow_{grain}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flow.to_parquet(out_path, index=False)

    inputs = {"archetypes": ARCHETYPES, "papers_duckdb": PAPERS_DUCKDB}
    if grain == "mid":
        inputs["topic_hierarchy"] = TOPIC_HIERARCHY
    if citation_edges is not None:
        inputs["kuzu_graph"] = KUZU_GRAPH
    record_run(
        artifact_path=out_path,
        inputs=inputs,
        config={
            "task": "V2-S01",
            "grain": grain,
            "min_papers": min_papers,
            "citations": citation_edges is not None,
            "noise_topic_dropped": NOISE_TOPIC_ID,
        },
    )

    n_topics = flow[topic_key].nunique()
    n_journals = flow["journal_slug"].nunique()
    print(f"\n=== flow_{grain}.parquet ===")
    print(f"  path:        {out_path}")
    print(f"  rows:        {len(flow):,}")
    print(f"  topics:      {n_topics} (key={topic_key})")
    print(f"  journals:    {n_journals}  -> {sorted(flow['journal_slug'].unique())}")
    print(f"  year range:  {int(flow['year'].min())}-{int(flow['year'].max())}")
    print(f"  total n_papers:        {int(flow['n_papers'].sum()):,}")
    print(f"  first-appearance rows: {int(flow['is_first_appearance'].sum()):,}")
    if citation_edges is not None:
        print(f"  total out_flow:        {int(flow['out_flow'].sum()):,}")
        print(f"  total in_flow:         {int(flow['in_flow'].sum()):,}")
    else:
        print("  citation flows:        NOT computed (out_flow=0, in_flow=NaN)")
    if n_journals > len(set(flow["journal_slug"])) or n_journals > 10:
        print(f"  !! WARNING: {n_journals} journals (>10) — jama_surg collapse FAILED")
    else:
        print("  OK: <=10 canonical journals (no phantom 11th journal)")
    print("  --- 5-row sample ---")
    print(flow.head(5).to_string(index=False))
    return out_path


def main() -> None:
    """CLI entry point: build the requested grain(s) and write the tables."""
    parser = argparse.ArgumentParser(description="Build V2 journal-flow tables.")
    parser.add_argument(
        "--grain",
        choices=["leaf", "mid", "both"],
        default="both",
        help="topic granularity to build (default: both)",
    )
    parser.add_argument(
        "--min-papers",
        type=int,
        default=1,
        help="per-year first-appearance threshold (default: 1)",
    )
    parser.add_argument(
        "--no-citations",
        action="store_true",
        help="skip the Kuzu CITES pull; emit the not-computed sentinel columns",
    )
    args = parser.parse_args()

    print("Loading papers + canonical journal_slug (pmid -> papers_distinct join)...")
    papers_topics = load_papers_topics()
    print(
        f"  loaded {len(papers_topics):,} papers across "
        f"{papers_topics['journal_slug'].nunique()} canonical journals, "
        f"{papers_topics['topic_id'].nunique()} leaf topics, "
        f"{papers_topics['mid_level_id'].nunique()} mid-level topics"
    )

    citation_edges: pd.DataFrame | None = None
    if not args.no_citations:
        print("Loading corpus-internal CITES edges from Kuzu (read-only)...")
        citation_edges = load_citation_edges()
        print(f"  loaded {len(citation_edges):,} CITES edges")

    grains = ["leaf", "mid"] if args.grain == "both" else [args.grain]
    for grain in grains:
        build_and_write(
            papers_topics,
            grain=grain,
            citation_edges=citation_edges,
            min_papers=args.min_papers,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
