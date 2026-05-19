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
