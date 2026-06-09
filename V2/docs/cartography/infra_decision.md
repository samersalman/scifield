# V2-S02 — Infrastructure Spike: Go / Stay Decision

**Task:** V2-S02 (Phase A). **Author:** T2 (infra spike). **Date:** 2026-06-09.
**Scope:** decide whether the current stack holds for the literature-cartography map's
query patterns — **not** to migrate or stand up a new DB.

**Stack under test:** Kùzu 0.11.3 (embedded citation graph) + DuckDB 1.5.2 (columnar SQL
over `papers.duckdb` + direct Parquet scan) + Parquet enrichment tables. All engines opened
**read-only**. Corpus = the current 10-journal panel.

**Corpus sizes (CONTEXT §3, re-verified):** 121,908 papers; 89,230 topic-assignments
(archetypes rows); 574,478 corpus-internal `CITES` edges (Kùzu); 2,982,356 outbound
references (`references_out`).

---

## 1. Benchmark results (real measured numbers)

Produced by `V2/scripts/infra_bench.py` on 2026-06-09 (median of 3 reps,
`time.perf_counter`, read-only). Numbers vary a few ms run-to-run; the magnitudes are
stable and that is what the decision turns on.

| query | engine | median latency | rows | notes |
|---|---|---:|---:|---|
| Q1 first-appearance | DuckDB | 41.6 ms | 10 | topic 1; per-journal |
| Q2 lag matrix | DuckDB | 41.7 ms | 90 | 10×10 pairs over all topics (self-join) |
| Q3 cascade traversal | Kùzu | 8.7 ms | 3,996 | seed 33086352; 1..3-hop reached papers (native `*path`) |
| Q3 cascade traversal | DuckDB | 36.8 ms | 3,523 | same 1..3-hop via RECURSIVE self-join on references_out |
| Q4 recombination | DuckDB | 35.8 ms | 33,916 | papers w/ refs spanning ≥2 topics |
| Q5 sector/geo rollup | DuckDB | 23.4 ms | 471 | novelty by inst type × country |
| Q6 citational velocity | Kùzu | 48.9 ms | 10 | citation lag per journal (CITES) |

**Headline:** every canonical map query completes in **< 50 ms** on the current corpus.
The whole panel of queries runs in well under a second of actual query time. Nothing is a
bottleneck at this scale.

---

## 2. Per-query ergonomics commentary

**Q1 — temporal first-appearance (DuckDB, 41.6 ms).** A `GROUP BY journal_slug, MIN(year)`
after the `archetypes → papers_distinct` cast-join (`CAST(a.pmid AS VARCHAR) = p.pmid`).
DuckDB is the natural home: it is a columnar `MIN` aggregate over a small pre-filtered set.
The journal **must** be derived from `journal_slug` (the cast-join), not the display
`journal` column — the display column splits `jama_surg` history (CONTEXT §3 caveat). The
~40 ms is dominated by the Parquet scan of `archetypes`, not the aggregation.

**Q2 — cross-journal lag matrix (DuckDB, 41.7 ms).** A self-join of the per-(topic, journal)
first-year CTE on `topic_id`, averaging `first_year_B − first_year_A` over the 90 ordered
journal pairs. This is a textbook relational aggregate — DuckDB expresses it cleanly and the
self-join over ~all-topics × 10 journals is trivially small. A graph DB would be the wrong
tool here; there is no traversal, just set algebra. **DuckDB is the clear winner.**

**Q3 — citation-cascade traversal (Kùzu 8.7 ms vs DuckDB 36.8 ms).** This is the one query
where the engines genuinely diverge, and it is the most instructive.
- **Kùzu** expresses it as `MATCH (s)-[:CITES*1..3]->(d)` — variable-length path is a
  first-class primitive, the query is one line, and it is the **fastest query in the suite**
  (8.7 ms) because the adjacency structure is materialized and indexed.
- **DuckDB** needs a `WITH RECURSIVE` CTE that re-materializes the corpus-internal edge set
  on every call (resolving `ref_openalex_id → archetypes.openalex_id → pmid`, then a
  recursive self-join). It works and is still fast at this scale (36.8 ms), but it is ~**4×
  slower** than Kùzu and the SQL is markedly less ergonomic. Crucially, the cost is a scan
  over the 2.98M-row `references_out` per traversal — that scan grows with corpus size while
  Kùzu's indexed adjacency does not. **Kùzu is the right tool for traversal**, and the gap
  will widen at scale.

**Q4 — topic-recombination (DuckDB, 35.8 ms).** Resolve each outbound reference to a topic
(`references_out.ref_openalex_id → archetypes.openalex_id → topic_id`), then count distinct
topics per citing paper and keep ≥2. This is a large hash join (2.98M references against
89k archetype rows) plus a grouped distinct-count — exactly DuckDB's strength. Found 33,916
recombination candidates. A graph DB could model it (refs → topic edges) but the relational
join is more natural and the columnar engine handles the 2.98M-row probe in 36 ms.
**DuckDB winner.**

**Q5 — sector/geo novelty rollups (DuckDB, 23.4 ms).** Three-way join
`archetypes → paper_institutions(first author) → institutions`, averaging `sem_nov_mean` /
`cd5` within (`type`, `country_code`) cells. Pure aggregate over joins; **DuckDB is ideal**
and this is the cheapest non-trivial query (471 cells). First-author attribution
(`author_position = 0`) avoids multi-affiliation double-counting.

**Q6 — per-journal citational velocity (Kùzu, 48.9 ms).** `MATCH (citing)-[:CITES]->(cited)`
with a year-gap aggregate grouped by the cited paper's `journal_slug`. This walks **all**
574,478 edges (not a seeded sub-traversal), so it is the **slowest query in the suite** —
the cost is the full edge scan + aggregation, not pathfinding. Kùzu expresses it cleanly
because `year` and `journal_slug` live on the `Paper` nodes; an equivalent DuckDB version
would require the same `openalex` edge reconstruction as Q3 plus two joins to recover years,
so Kùzu is both more ergonomic and competitive here. Still < 50 ms.

**Parquet-scan baseline.** Every DuckDB query above scans Parquet directly (no import step);
the `archetypes` (3.9 MB) and `references_out` (27 MB) scans are the dominant cost in Q1/Q2/Q4.
Direct Parquet scan is fast enough that there is no case for a separate import/materialization
step at this scale.

---

## 3. Surprise / gotcha found during the spike

**`references_out.ref_pmid_if_known` is empty for 100% of rows** (verified:
`COUNT(*) FILTER (ref_pmid_if_known IS NOT NULL AND <> '') = 0` over all 2,982,356 rows). The
intuitive way to express a relational cascade traversal — self-join on the "known PMID"
column — silently returns the **empty set**. The corpus-internal edge set must instead be
reconstructed by resolving `ref_openalex_id → archetypes.openalex_id → pmid`. That resolution
recovers **541,493 of Kùzu's 574,478** `CITES` edges; the ~6% gap is the ~33k corpus papers
absent from `archetypes` (no topic/openalex row), whose endpoints are unresolvable
relationally. **Downstream consequence (flagged for S07 and S03/S04):** if any task needs
corpus-internal citation edges in DuckDB/Parquet, it must go through `ref_openalex_id`, not
`ref_pmid_if_known`; and the relational edge set is ~6% smaller than Kùzu's. For exact
corpus-internal traversal, **use the Kùzu graph** — it has the complete, pre-resolved edge
set. This is captured in the `q3_cascade_duckdb` docstring in the bench script.

No other surprises: latencies were uniformly small and stable.

---

## 4. Recommendation — **STAY on Kùzu + DuckDB + Parquet for the prove phase**

The current hybrid stack already maps cleanly onto the query taxonomy, and at the current
corpus scale **every canonical query is sub-50-ms**:

- **DuckDB + direct Parquet scan** owns the columnar/temporal/relational patterns:
  first-appearance (Q1), lag matrix (Q2), recombination joins (Q4), sector/geo rollups (Q5).
  These are aggregates and joins, not traversals; a graph DB would be the wrong tool and
  DuckDB expresses them in idiomatic SQL.
- **Kùzu** owns multi-hop graph traversal (Q3) and whole-graph edge aggregates (Q6). It is
  the fastest engine for the seeded cascade (8.7 ms, ~4× faster than the DuckDB recursive
  equivalent) and has the complete pre-resolved corpus-internal edge set.

This is already the **hybrid** the plan anticipated ("the likely answer is hybrid"): use each
engine for what it is good at. There is **no case to migrate** to Memgraph / Neo4j or a
dedicated temporal columnar store now — it would add operational cost and a data-migration
step for zero measured benefit at this scale. **Re-evaluate at Phase F scale** (post-G6
expansion), per the explicit thresholds below.

This matches the plan's expected answer, and the measured numbers justify it: the slowest
query (Q6, 48.9 ms) leaves roughly **3 orders of magnitude** of headroom before interactive
latency (~1 s for a map UI; ~100 ms for "feels instant") becomes a concern.

---

## 5. Explicit migration thresholds (quantified against measured baselines)

Migration is justified only if a query pattern crosses a latency band that breaks the map's
interactivity, **and** the responsible engine cannot be tuned out of it. Targets and triggers,
anchored to the current baselines and corpus sizes (574,478 CITES edges; 121,908 papers;
89,230 topic-assignments):

**T1 — Graph traversal (Q3/Q6 → Memgraph or Neo4j).**
Baseline: seeded 1..3-hop traversal **8.7 ms** at 574,478 edges; whole-graph velocity
aggregate **48.9 ms**.
- **Trigger:** if **seeded cascade traversal exceeds ~250 ms**, or **whole-graph CITES
  aggregates (Q6) exceed ~1,000 ms**, at the expanded edge scale, evaluate a dedicated graph
  engine (Memgraph for in-memory traversal throughput; Neo4j if persistence/Cypher tooling is
  needed).
- **Scale anchor:** the Phase-F source layer (Nature/NEJM/Lancet/JAMA/PNAS) + deep specialty
  coverage will multiply edges roughly **10–50×** (→ ~6M–30M CITES edges). Kùzu's indexed
  traversal scales sub-linearly in seeded queries, so even ~10–30× edges likely keeps Q3
  inside ~100 ms — **re-benchmark before assuming a migration is needed.** The whole-graph
  Q6 scan scales roughly linearly with edges; it is the **first** query expected to approach
  the threshold (≈50 ms × 20 ≈ 1 s at 20× edges), so Q6 is the canary — watch it first.

**T2 — Temporal first-appearance / lag (Q1/Q2 → temporal columnar store, or DuckDB-only).**
Baseline: first-appearance **41.6 ms**, lag matrix **41.7 ms**, at 89,230 topic-assignments.
- **Trigger:** if **first-appearance or the lag matrix exceeds ~500 ms** at the expanded
  topic-assignment scale, consider materializing a per-(topic, journal, year) summary table
  (the `flow.py` artifact from T1b is exactly this) and/or a dedicated temporal store.
- **Scale anchor:** these costs are Parquet-scan-dominated and scale ~linearly with row count.
  Even at **10× topic-assignments (~890k rows)** and a wider journal panel (lag matrix grows
  with #journals²), DuckDB stays well under 500 ms. The cheap mitigation **before** any
  migration is to pre-aggregate first-appearance into a flow summary table (drops Q1/Q2 to
  reads of a tiny table) — try that first; a temporal columnar store is a last resort.

**T3 — Recombination / large reference joins (Q4 → stays DuckDB; pre-materialize if needed).**
Baseline: **35.8 ms** probing 2,982,356 references against 89,230 archetype rows.
- **Trigger:** if the recombination/reference-join pattern **exceeds ~1,000 ms** at the
  expanded reference scale, pre-materialize a `reference → topic` mapping table (one-time
  join) so the per-query work is a grouped count, not a 2.98M-row probe. **No engine change;**
  the fix is a materialized intermediate.
- **Scale anchor:** references grow with corpus size (≈3M now → tens of millions). DuckDB hash
  joins scale well; the materialized-mapping mitigation removes the repeated scan entirely.

**T4 — Interactivity ceiling (cross-cutting).**
- **Soft trigger:** any single map query routinely **> 1 s** end-to-end → it must be
  pre-computed offline into a served artifact (Parquet/JSON the map reads), regardless of
  engine. The map v0 (S07) should pre-compute heavy panels rather than query live.
- **Hard trigger:** if **two or more** of the patterns above breach their thresholds
  simultaneously at Phase-F scale, **adopt the hybrid explicitly**: Kùzu/Memgraph for the
  graph layer + DuckDB/Parquet (with materialized flow + reference-topic tables) for the
  analytic layer. The likely answer at scale is this hybrid, **not** a single-DB migration.

**Decision-review checkpoint:** re-run `V2/scripts/infra_bench.py` against the **expanded**
corpus immediately after the Phase-F harvest (V2-S08) and **before** committing to any DB
change. Migrate only on **measured** threshold breaches, never on anticipation.

---

## 6. One-line verdict

**STAY** on Kùzu 0.11.3 + DuckDB 1.5.2 + Parquet for the prove phase (Phases A–E). All
canonical map queries are < 50 ms; the stack is already the right hybrid (DuckDB for
aggregates/temporal/joins, Kùzu for traversal). The slowest query is **Q6 per-journal
citational velocity (48.9 ms, Kùzu, whole-edge scan)** — the canary for the graph-traversal
threshold at scale. Re-benchmark at Phase F before any migration; the likely answer at scale
remains a hybrid, not a single new DB.
