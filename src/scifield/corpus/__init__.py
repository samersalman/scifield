"""Phase 1 — Corpus harvesting (V1-S03 async PubMed pipeline)."""

from scifield.corpus.pubmed import (
    EntrezClient,
    EntrezConfig,
    HarvestConfig,
    HarvestReport,
    JournalSpec,
    OutputConfig,
    RateLimiter,
    harvest_corpus,
    harvest_journal_year,
    parse_pubmed_articles,
)
from scifield.corpus.store import (
    PAPER_SCHEMA,
    build_duckdb,
    read_manifest,
    write_bucket_parquet,
    write_manifest,
)

__all__ = [
    "EntrezClient",
    "EntrezConfig",
    "HarvestConfig",
    "HarvestReport",
    "JournalSpec",
    "OutputConfig",
    "PAPER_SCHEMA",
    "RateLimiter",
    "build_duckdb",
    "harvest_corpus",
    "harvest_journal_year",
    "parse_pubmed_articles",
    "read_manifest",
    "write_bucket_parquet",
    "write_manifest",
]
