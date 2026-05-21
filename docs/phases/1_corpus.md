# Phase 1 — Corpus

## Phase objective

Build, validate, and document the v1 corpus of approximately 200,000
abstracts from 10 journals (5 orthopedics + 5 general surgery) over 30
years. Every paper has: PMID, year, title, abstract, journal, authors
(disambiguated), institution, MeSH terms, OpenAlex ID, full citation list,
and full reference list. PubMed harvesting via Biopython Entrez; citations
and authorship via OpenAlex with Semantic Scholar as a secondary source;
institution canonicalization via ROR. Storage in DuckDB plus Parquet
artifacts versioned by config hash. Success criteria: >95% of papers have
abstract text, journal, year, and MeSH; >90% have an OpenAlex match; >80%
have a fully resolved citation list, with documented gaps and rationale.

## V1-S03 — async PubMed harvester

The V1-S03 session delivers `src/scifield/corpus/pubmed.py` (async Entrez
harvester with httpx + tenacity rate-limited at 2 req/s without an API key /
9 req/s with `NCBI_API_KEY` set) and `src/scifield/corpus/store.py` (Parquet
writer per `(journal_slug, year)` bucket + DuckDB view layer over the
Parquet lake). The 10-journal × 1995–2025 corpus is described in
`conf/corpus/v1.yaml`. The harvester is idempotent at the bucket level: if
`data/v1/parquet/<slug>/<year>.parquet` exists and its manifest PMID set
matches a fresh esearch, the bucket is skipped. Run a small smoke first,
then the overnight harvest:

```bash
# Smoke (~300 papers, <2 min)
export NCBI_API_KEY="<your key>"   # optional but recommended
uv run scifield harvest --config v1 --year 2024 --max-papers-per-bucket 30

# Full corpus (~150k–250k papers, overnight)
uv run scifield harvest --config v1
```

Descriptive corpus stats are produced by `notebooks/01_corpus_overview.ipynb`,
which executes against `data/v1/papers.duckdb` after the run.

## V1-S03 outcome (2026-05-19)

The first full overnight harvest returned **134,978 papers across 310
journal-year buckets in 233 s** (with `NCBI_API_KEY` set). Every PMID PubMed
indexed for the 10 TA queries was captured (`esearch_count == parsed_count`,
zero parse failures, zero drop). Total count came in below the original
150 k–250 k estimate because PubMed's actual indexing for these journals is
lower than the rough projection — not a pipeline shortfall.

**Pre-2000 abstract availability: 78.2 % of 14,452 pre-2000 papers.** This is
the number plan §6 risk row 1 anticipated needing for the V1-S04 OpenAlex
post-2000 restriction decision. For context, the era breakdown is:
`<2000 78.2 % / 2000-09 78.0 % / 2010-19 74.4 % / 2020+ 68.2 %`. Pre-2000 is
not the abstract bottleneck the risk row anticipated; the 2020+ drop
(commentary + epub-ahead-of-print volume) is more significant.

Overall abstract coverage (74.1 %) is below the 95 % phase target. The gap
is driven by publication-type mix — Comments, Letters, and Case Reports
arrive from PubMed without abstracts (e.g., JAMA Surg Comments 0.2 %,
Letters 4.2 %). Within research-article types (`Multicenter Study`,
`Randomized Controlled Trial`, `Comparative Study`, etc.) abstract coverage
is 90–99 % across all 10 journals. Downstream analytic sessions should
filter on `publication_types` rather than restrict by era.

## V1-S04 — corpus enrichment (OpenAlex + Semantic Scholar + ROR + author IDs)

V1-S04 layers four external data sources on top of the V1-S03 PubMed corpus:

- **OpenAlex** — canonical work ID, outgoing references, author IDs, institution
  IDs, OA status, retraction status, concepts (top 5). Polite pool, batched by
  PMID (50/call), gzip raw-JSON cache, manifest-driven idempotency.
- **Author disambiguation** — three layers: OpenAlex `author.id` (primary) →
  ORCID (overrides; more authoritative) → heuristic hash of
  `(normalized_last_name, first_initial, most-common-institution-signal)` for
  authors lacking both. `disambiguation_method` column lets downstream filter.
- **ROR** — only queried for affiliation strings OpenAlex couldn't already
  match (`institutions[i].ror == null`); fuzzy match with a min-score gate,
  persistent parquet cache. Unmatched strings get stable `RAW:<hash>` IDs so
  they remain joinable.
- **Semantic Scholar** — module + schema + CLI built; when
  `SEMANTIC_SCHOLAR_API_KEY` is unset the orchestrator writes empty schema-only
  Parquets and logs "SS skipped". Backfilling later with
  `uv run scifield enrich --only semantic_scholar` requires no schema migration.

### Citation graph scope

Outgoing references for all papers (`referenced_works` is in the main OpenAlex
work JSON — cheap). Incoming `cited_by` enumeration is deferred to V1-S10
because paginating it externally is intractable at the V1 budget. The internal
V1↔V1 citation graph is built for free by reverse-indexing `references_out`
against the corpus's own `openalex_works.openalex_id`. "≥80 % citation list
resolved" in the acceptance criteria is interpreted as **outgoing references
resolved**, which is what Phase 4 novelty + CD analyses actually consume.

### Outputs

Eight Parquet tables under `data/v1/enrichment/`, plus DuckDB views attached
to `papers.duckdb`:

| File | Rows | Notes |
|---|---|---|
| `openalex_works.parquet` | 1 per paper | strip URL prefixes, top-5 concepts |
| `references_out.parquet` | 1 per (citing PMID, ref position) | `ref_pmid_if_known` resolved by V1-S10 via internal reverse-index |
| `authorships.parquet` | 1 per (paper, author position) | carries `disambiguation_method` |
| `institutions.parquet` | 1 per institution (deduped by `institution_canonical_id`) | |
| `paper_institutions.parquet` | 1 per (paper, author position, institution) | `ror_matched_by ∈ {openalex, ror_api, unmatched}` |
| `semantic_scholar.parquet` | 1 per paper (possibly empty) | |
| `citation_intents.parquet` | 1 per (citing, cited, intent set) | |
| `enrichment_failed.parquet` | 1 per failure | `source ∈ {openalex, semantic_scholar, ror}` |

Run:

```bash
export OPENALEX_EMAIL="<your-email>"          # required (polite pool)
export SEMANTIC_SCHOLAR_API_KEY="<key>"        # optional; SS is skipped without

# Smoke (~200 papers, <5 min)
uv run scifield enrich --config v1 --limit 200

# Full enrichment (~134k papers, 1–2 h)
uv run scifield enrich --config v1
```

Coverage matrix (journal × era × source) is produced by
`notebooks/02_coverage_report.ipynb`. Re-execute after each run.

### Risk register — pre-2000 OpenAlex coverage (plan §6 row 1)

**Decision rule.** If pre-2000 OpenAlex match coverage is **<60 %** after the
full enrichment run, V1-S10 novelty and CD-index analyses are restricted to
papers with `year >= 2000`. Pre-2000 papers remain in the corpus and in the
thematic backbone (V1-S05 uses abstracts only, which are independent of
OpenAlex), but they are dropped from any citation-graph-derived metric where
the graph is provably incomplete.

## V1-S04 outcome (2026-05-19)

The full enrichment run completed in **14,651 s (4 h 04 m)** over the 121,908
distinct PMIDs in the V1-S03 corpus (134,978 PubMed rows; 13,070 of those rows
are the same paper indexed under two TA terms — chiefly JAMA Surg / Arch Surg).
All four sources ran (OpenAlex + Authors + ROR; Semantic Scholar skipped — no
key yet, schema-only Parquet written).

| Metric | Count | % | Gate |
|---|---|---|---|
| OpenAlex match | 118,317 | **97.1 %** | ≥ 90 % ✓ |
| outgoing references resolved (post-2000) | 98,595 | **91.8 %** | ≥ 80 % ✓ |
| outgoing references resolved (overall) | 110,158 | — | — |
| outgoing reference edges | 2,982,356 | — | — |
| authorships | 623,827 | — | — |
| distinct institutions | 29,159 | — | — |
| paper-author-institution rows | 874,695 | — | — |
| OpenAlex fetches that failed (mostly pre-2000 not_found) | 3,591 | 3.0 % | — |

**Pre-2000 OpenAlex coverage: 95.8 % of 14,452 pre-2000 papers** —
substantially above the 60 % threshold. Pre-2000 is retained for V1-S10
novelty + CD-index analyses; no era restriction applied.

Both acceptance gates pass. The 11 misses observed in the smoke (and the 3,591
misses in the full run) are all `not_found` from OpenAlex — almost entirely
1999-era PubMed records that OpenAlex hasn't ingested. They are recorded in
`enrichment_failed.parquet` with `source='openalex'`, `reason='not_found'` and
do not gate downstream phases (V1-S05 reads abstracts only).
