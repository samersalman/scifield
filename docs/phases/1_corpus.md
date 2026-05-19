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
