"""Infrastructure benchmark for the SciField V2 literature-cartography map (V2-S02).

Times the 5--6 canonical map queries against the *current* 10-journal corpus on the
stack we run today: Kuzu 0.11.3 (citation graph), DuckDB 1.5.2 (columnar SQL over the
papers DB), and Parquet (zero-copy columnar scan via DuckDB). The goal is a *decision*
-- does the current stack hold for the map's query patterns? -- not a migration. The
companion ``V2/docs/cartography/infra_decision.md`` turns these numbers into a
go/stay verdict with explicit migration thresholds.

Canonical queries
------------------
Q1  Temporal first-appearance : first year a topic appears in each journal
    (DuckDB over archetypes + papers_distinct; journal derived via ``journal_slug``).
Q2  Cross-journal lag matrix  : pairwise first-appearance lead/lag across the 10
    journals over all topics (DuckDB self-join aggregate).
Q3  Citation-cascade traversal: multi-hop CITES from a seed paper -- implemented BOTH
    on Kuzu (native variable-length graph match) and as a DuckDB recursive self-join
    over references_out, to compare engine ergonomics + latency on the same workload.
Q4  Topic-recombination       : papers whose references span >=2 distinct topics
    (references_out -> archetypes.openalex_id -> topic_id; DuckDB).
Q5  Sector / geo rollups       : novelty (sem_nov_mean / cd5) by institution ``type``
    and by ``country_code`` (DuckDB join archetypes -> paper_institutions ->
    institutions).
Q6  Per-journal citational velocity: distribution of years from publication to inbound
    citation, per journal (Kuzu CITES with years; the corpus-internal citation graph).

Each query is run 3x and the *median* wall-clock latency (``time.perf_counter``,
monotonic) is reported. All engines are opened READ-ONLY. Total runtime is a few
minutes.

Run
---
``.venv/bin/python V2/scripts/infra_bench.py``

Constraints: $0, read-only on all data, no network, no migration.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import duckdb
import kuzu

# --------------------------------------------------------------------------------------
# Paths (resolve from repo root so the script runs from anywhere).
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "v1"
PAPERS_DUCKDB = DATA / "papers.duckdb"
KUZU_GRAPH = DATA / "kuzu_graph"

ARCHETYPES = DATA / "archetypes.parquet"
REFERENCES_OUT = DATA / "enrichment" / "references_out.parquet"
PAPER_INSTITUTIONS = DATA / "enrichment" / "paper_institutions.parquet"
INSTITUTIONS = DATA / "enrichment" / "institutions.parquet"

# Benchmark fixtures (chosen during a one-off corpus inspection, recorded in the
# decision doc). Topic 1 spans all 10 journals (2,395 papers); seed 33086352 has the
# highest CITES out-degree (257) in the corpus, so it stresses traversal.
SEED_TOPIC = 1
SEED_PMID = "33086352"
N_REPS = 3


class Result(NamedTuple):
    """One benchmarked query."""

    query: str
    engine: str
    median_ms: float
    rows: int
    notes: str


def _time_median(fn: Callable[[], int], reps: int = N_REPS) -> tuple[float, int]:
    """Run ``fn`` ``reps`` times, return (median latency in ms, row count of last run).

    Parameters
    ----------
    fn
        A zero-arg callable that executes the query and returns its row count.
    reps
        Number of repetitions; the median is reported to damp first-touch / cache noise.

    Returns
    -------
    tuple of (float, int)
        Median wall-clock latency in milliseconds and the row count.
    """
    timings: list[float] = []
    rows = 0
    for _ in range(reps):
        start = time.perf_counter()
        rows = fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings), rows


# --------------------------------------------------------------------------------------
# Q1 -- temporal first-appearance (DuckDB).
# --------------------------------------------------------------------------------------
def q1_first_appearance(con: duckdb.DuckDBPyConnection) -> int:
    """First year a given topic appears in each journal (canonical ``journal_slug``)."""
    sql = f"""
        SELECT p.journal_slug, min(p.year) AS first_year
        FROM '{ARCHETYPES}' a
        JOIN pdb.papers_distinct p ON CAST(a.pmid AS VARCHAR) = p.pmid
        WHERE a.topic_id = {SEED_TOPIC}
        GROUP BY p.journal_slug
        ORDER BY first_year
    """
    return len(con.execute(sql).fetchall())


# --------------------------------------------------------------------------------------
# Q2 -- cross-journal lag matrix (DuckDB).
# --------------------------------------------------------------------------------------
def q2_lag_matrix(con: duckdb.DuckDBPyConnection) -> int:
    """Pairwise first-appearance lead/lag across the 10 journals over all topics.

    For each topic, compute its first year in each journal; then for every ordered
    journal pair (A, B) average (first_year_B - first_year_A) across topics seen in
    both. Positive mean_lag => A leads B. The full 10x10 (minus diagonal) matrix.
    """
    sql = f"""
        WITH fa AS (
            SELECT a.topic_id, p.journal_slug, min(p.year) AS first_year
            FROM '{ARCHETYPES}' a
            JOIN pdb.papers_distinct p ON CAST(a.pmid AS VARCHAR) = p.pmid
            WHERE a.topic_id >= 0
            GROUP BY a.topic_id, p.journal_slug
        )
        SELECT x.journal_slug AS journal_a,
               y.journal_slug AS journal_b,
               count(*) AS n_shared_topics,
               avg(y.first_year - x.first_year) AS mean_lag
        FROM fa x
        JOIN fa y ON x.topic_id = y.topic_id AND x.journal_slug <> y.journal_slug
        GROUP BY x.journal_slug, y.journal_slug
        ORDER BY journal_a, journal_b
    """
    return len(con.execute(sql).fetchall())


# --------------------------------------------------------------------------------------
# Q3 -- citation-cascade traversal (Kuzu native vs. DuckDB recursive self-join).
# --------------------------------------------------------------------------------------
def q3_cascade_kuzu(con: kuzu.Connection) -> int:
    """Multi-hop (1..3) CITES descendants of a seed paper -- native graph match."""
    cypher = f"""
        MATCH (s:Paper {{pmid: '{SEED_PMID}'}})-[:CITES*1..3]->(d:Paper)
        RETURN count(DISTINCT d.pmid) AS n_reached
    """
    res = con.execute(cypher)
    n = int(res.get_next()[0])
    # one-row scalar; "rows" reported is the count of reached papers it summarizes
    return n


def q3_cascade_duckdb(con: duckdb.DuckDBPyConnection) -> int:
    """Same 1..3-hop reachable set via a recursive self-join over references_out.

    This is the relational expression of the graph traversal -- the comparison point
    for ergonomics + latency vs. Kuzu. Corpus-internal edges must be reconstructed by
    resolving each outbound reference's OpenAlex id back to an in-panel pmid
    (``ref_openalex_id -> archetypes.openalex_id -> pmid``). The convenient
    ``ref_pmid_if_known`` column is NOT usable: it is empty for 100% of references_out
    rows in this corpus (verified), so a self-join over it returns the empty set. The
    OpenAlex resolution recovers 541,493 of Kuzu's 574,478 CITES edges -- the gap is the
    ~33k corpus papers absent from archetypes (no topic/openalex row), so a handful of
    endpoints are unresolvable relationally. Reached-set sizes therefore differ slightly
    from Kuzu by design; the point is latency + ergonomics on the same workload.
    """
    sql = f"""
        WITH RECURSIVE oa2pmid AS (
            SELECT openalex_id, CAST(pmid AS VARCHAR) AS pmid
            FROM '{ARCHETYPES}'
            WHERE openalex_id IS NOT NULL
        ),
        edges AS (
            SELECT r.citing_pmid, o.pmid AS ref_pmid
            FROM '{REFERENCES_OUT}' r
            JOIN oa2pmid o ON r.ref_openalex_id = o.openalex_id
        ),
        reach AS (
            SELECT ref_pmid AS pmid, 1 AS depth
            FROM edges WHERE citing_pmid = '{SEED_PMID}'
            UNION
            SELECT e.ref_pmid, r.depth + 1
            FROM reach r
            JOIN edges e ON e.citing_pmid = r.pmid
            WHERE r.depth < 3
        )
        SELECT count(DISTINCT pmid) FROM reach
    """
    return int(con.execute(sql).fetchone()[0])


# --------------------------------------------------------------------------------------
# Q4 -- topic-recombination detection (DuckDB).
# --------------------------------------------------------------------------------------
def q4_recombination(con: duckdb.DuckDBPyConnection) -> int:
    """Papers whose reference list spans >=2 distinct topics.

    Resolve each outbound reference to a topic by joining references_out.ref_openalex_id
    to archetypes.openalex_id (which carries topic_id). A citing paper whose references
    map onto >=2 distinct (non-noise) topics is a recombination candidate.
    """
    sql = f"""
        WITH ref_topics AS (
            SELECT DISTINCT r.citing_pmid, a.topic_id
            FROM '{REFERENCES_OUT}' r
            JOIN '{ARCHETYPES}' a ON r.ref_openalex_id = a.openalex_id
            WHERE a.topic_id >= 0
        )
        SELECT citing_pmid, count(*) AS n_topics
        FROM ref_topics
        GROUP BY citing_pmid
        HAVING count(*) >= 2
    """
    return len(con.execute(sql).fetchall())


# --------------------------------------------------------------------------------------
# Q5 -- sector / geo novelty rollups (DuckDB).
# --------------------------------------------------------------------------------------
def q5_sector_geo(con: duckdb.DuckDBPyConnection) -> int:
    """Mean novelty (sem_nov_mean, cd5) by institution ``type`` and ``country_code``.

    Each paper is attributed to its first-author institution (author_position = 0) to
    avoid double counting; novelty is then averaged within (type, country_code) cells.
    """
    sql = f"""
        WITH first_inst AS (
            SELECT pi.pmid, pi.institution_canonical_id
            FROM '{PAPER_INSTITUTIONS}' pi
            WHERE pi.author_position = 0
        )
        SELECT i.type,
               i.country_code,
               count(*) AS n_papers,
               avg(a.sem_nov_mean) AS mean_sem_nov,
               avg(a.cd5) AS mean_cd5
        FROM '{ARCHETYPES}' a
        JOIN first_inst fi ON CAST(a.pmid AS VARCHAR) = fi.pmid
        JOIN '{INSTITUTIONS}' i
            ON fi.institution_canonical_id = i.institution_canonical_id
        WHERE a.topic_id >= 0
        GROUP BY i.type, i.country_code
        ORDER BY n_papers DESC
    """
    return len(con.execute(sql).fetchall())


# --------------------------------------------------------------------------------------
# Q6 -- per-journal citational velocity (Kuzu, corpus-internal CITES with years).
# --------------------------------------------------------------------------------------
def q6_velocity_kuzu(con: kuzu.Connection) -> int:
    """Per-journal distribution of citation lag (citing_year - cited_year).

    Uses the corpus-internal CITES graph: for each edge citing -> cited, the lag is the
    publication-year gap, grouped by the *cited* paper's journal. Returns one row per
    journal with mean lag and edge count -- a velocity summary the map can consume.
    """
    cypher = """
        MATCH (citing:Paper)-[:CITES]->(cited:Paper)
        WHERE citing.year >= cited.year
        RETURN cited.journal_slug AS journal,
               count(*) AS n_citations,
               avg(citing.year - cited.year) AS mean_lag_years
        ORDER BY n_citations DESC
    """
    return len(con.execute(cypher).get_as_df())


# --------------------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------------------
def main() -> None:
    """Run all canonical queries, time each (median of 3), print a markdown table."""
    # DuckDB: in-memory connection, attach the papers DB READ-ONLY, scan parquet direct.
    duck = duckdb.connect(":memory:")
    duck.execute(f"ATTACH '{PAPERS_DUCKDB}' AS pdb (READ_ONLY)")

    # Kuzu: open the materialized graph READ-ONLY.
    kdb = kuzu.Database(str(KUZU_GRAPH), read_only=True)
    kcon = kuzu.Connection(kdb)

    results: list[Result] = []

    # --- DuckDB-served queries -------------------------------------------------------
    med, rows = _time_median(lambda: q1_first_appearance(duck))
    results.append(
        Result("Q1 first-appearance", "DuckDB", med, rows, f"topic {SEED_TOPIC}; per-journal")
    )

    med, rows = _time_median(lambda: q2_lag_matrix(duck))
    results.append(
        Result("Q2 lag matrix", "DuckDB", med, rows, "10x10 pairs over all topics (self-join)")
    )

    med, rows = _time_median(lambda: q3_cascade_kuzu(kcon))
    results.append(
        Result(
            "Q3 cascade traversal",
            "Kuzu",
            med,
            rows,
            f"seed {SEED_PMID}; 1..3-hop reached papers (native *path)",
        )
    )

    med, rows = _time_median(lambda: q3_cascade_duckdb(duck))
    results.append(
        Result(
            "Q3 cascade traversal",
            "DuckDB",
            med,
            rows,
            "same 1..3-hop via RECURSIVE self-join on references_out",
        )
    )

    med, rows = _time_median(lambda: q4_recombination(duck))
    results.append(
        Result("Q4 recombination", "DuckDB", med, rows, "papers w/ refs spanning >=2 topics")
    )

    med, rows = _time_median(lambda: q5_sector_geo(duck))
    results.append(
        Result("Q5 sector/geo rollup", "DuckDB", med, rows, "novelty by inst type x country")
    )

    med, rows = _time_median(lambda: q6_velocity_kuzu(kcon))
    results.append(
        Result("Q6 citational velocity", "Kuzu", med, rows, "citation lag per journal (CITES)")
    )

    # --- Print markdown results table ------------------------------------------------
    print()
    print("# V2-S02 infrastructure benchmark — current corpus, read-only")
    print(f"# reps per query: {N_REPS} (median reported); clock: time.perf_counter (monotonic)")
    print()
    print("| query | engine | median latency | rows | notes |")
    print("|---|---|---:|---:|---|")
    for r in results:
        print(f"| {r.query} | {r.engine} | {r.median_ms:,.1f} ms | {r.rows:,} | {r.notes} |")
    print()

    kcon.close()
    duck.close()


if __name__ == "__main__":
    main()
