"""Phase 1 — Corpus harvesting + enrichment (V1-S03 + V1-S04)."""

from scifield.corpus.authors import AuthorsConfig, disambiguate_authorships
from scifield.corpus.enrich_orchestrator import (
    ENRICH_SOURCES,
    EnrichmentPaths,
    EnrichmentReport,
    enrich_corpus,
    load_pmids_from_corpus,
)
from scifield.corpus.enrich_store import (
    ENRICH_SCHEMAS,
    read_enrichment_parquet,
    register_enrichment_views,
    write_enrichment_parquet,
)
from scifield.corpus.openalex import (
    OpenAlexClient,
    OpenAlexConfig,
    enrich_openalex,
    parse_authorships_staging,
    parse_openalex_work,
    parse_references_out,
)
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
from scifield.corpus.ror import RORConfig, RORMatcher, build_institution_tables
from scifield.corpus.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarConfig,
    enrich_semantic_scholar,
    parse_ss_paper,
)
from scifield.corpus.store import (
    PAPER_SCHEMA,
    build_duckdb,
    read_manifest,
    write_bucket_parquet,
    write_manifest,
)

__all__ = [
    "ENRICH_SCHEMAS",
    "ENRICH_SOURCES",
    "AuthorsConfig",
    "EnrichmentPaths",
    "EnrichmentReport",
    "EntrezClient",
    "EntrezConfig",
    "HarvestConfig",
    "HarvestReport",
    "JournalSpec",
    "OpenAlexClient",
    "OpenAlexConfig",
    "OutputConfig",
    "PAPER_SCHEMA",
    "RORConfig",
    "RORMatcher",
    "RateLimiter",
    "SemanticScholarClient",
    "SemanticScholarConfig",
    "build_duckdb",
    "build_institution_tables",
    "disambiguate_authorships",
    "enrich_corpus",
    "enrich_openalex",
    "enrich_semantic_scholar",
    "harvest_corpus",
    "harvest_journal_year",
    "load_pmids_from_corpus",
    "parse_authorships_staging",
    "parse_openalex_work",
    "parse_pubmed_articles",
    "parse_references_out",
    "parse_ss_paper",
    "read_enrichment_parquet",
    "read_manifest",
    "register_enrichment_views",
    "write_bucket_parquet",
    "write_enrichment_parquet",
    "write_manifest",
]
