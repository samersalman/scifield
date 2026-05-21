"""Enrichment storage primitives — Parquet schemas + DuckDB view registration.

V1-S04 produces seven new tables that live under ``data/v1/enrichment/`` and
attach to the V1-S03 ``papers.duckdb`` as views. Schemas are kept explicit so
nested-typed columns (concepts, intents) stay binary-compatible across
incremental writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

OPENALEX_WORKS_SCHEMA = pa.schema(
    [
        ("pmid", pa.string()),
        ("openalex_id", pa.string()),
        ("oa_doi", pa.string()),
        ("type", pa.string()),
        ("language", pa.string()),
        ("is_retracted", pa.bool_()),
        ("is_oa", pa.bool_()),
        ("oa_status", pa.string()),
        ("cited_by_count", pa.int64()),
        (
            "concepts",
            pa.list_(
                pa.struct(
                    [
                        ("id", pa.string()),
                        ("display_name", pa.string()),
                        ("score", pa.float64()),
                    ]
                )
            ),
        ),
        ("publication_year", pa.int32()),
        ("publication_date", pa.string()),
        ("fetched_at", pa.string()),
    ]
)

REFERENCES_OUT_SCHEMA = pa.schema(
    [
        ("citing_pmid", pa.string()),
        ("ref_openalex_id", pa.string()),
        ("ref_pmid_if_known", pa.string()),
        ("ref_year", pa.int32()),
        ("ref_position", pa.int32()),
    ]
)

AUTHORSHIPS_SCHEMA = pa.schema(
    [
        ("pmid", pa.string()),
        ("author_position", pa.int32()),
        ("author_position_label", pa.string()),
        ("author_oa_id", pa.string()),
        ("author_orcid", pa.string()),
        ("author_display_name", pa.string()),
        ("author_canonical_id", pa.string()),
        ("disambiguation_method", pa.string()),
        ("is_first", pa.bool_()),
        ("is_last", pa.bool_()),
    ]
)

INSTITUTIONS_SCHEMA = pa.schema(
    [
        ("institution_canonical_id", pa.string()),
        ("institution_oa_id", pa.string()),
        ("ror_id", pa.string()),
        ("display_name", pa.string()),
        ("country_code", pa.string()),
        ("type", pa.string()),
    ]
)

PAPER_INSTITUTIONS_SCHEMA = pa.schema(
    [
        ("pmid", pa.string()),
        ("author_position", pa.int32()),
        ("institution_canonical_id", pa.string()),
        ("raw_affiliation_string", pa.string()),
        ("ror_matched_by", pa.string()),
    ]
)

SEMANTIC_SCHOLAR_SCHEMA = pa.schema(
    [
        ("pmid", pa.string()),
        ("ss_id", pa.string()),
        ("ss_doi", pa.string()),
        ("citation_count", pa.int64()),
        ("references_with_intent_count", pa.int64()),
        ("fetched_at", pa.string()),
    ]
)

CITATION_INTENTS_SCHEMA = pa.schema(
    [
        ("citing_pmid", pa.string()),
        ("cited_id", pa.string()),
        ("intents", pa.list_(pa.string())),
        ("is_influential", pa.bool_()),
    ]
)

ENRICHMENT_FAILED_SCHEMA = pa.schema(
    [
        ("pmid", pa.string()),
        ("source", pa.string()),
        ("reason", pa.string()),
        ("attempted_at", pa.string()),
    ]
)


ENRICH_SCHEMAS: dict[str, pa.Schema] = {
    "openalex_works": OPENALEX_WORKS_SCHEMA,
    "references_out": REFERENCES_OUT_SCHEMA,
    "authorships": AUTHORSHIPS_SCHEMA,
    "institutions": INSTITUTIONS_SCHEMA,
    "paper_institutions": PAPER_INSTITUTIONS_SCHEMA,
    "semantic_scholar": SEMANTIC_SCHOLAR_SCHEMA,
    "citation_intents": CITATION_INTENTS_SCHEMA,
    "enrichment_failed": ENRICHMENT_FAILED_SCHEMA,
}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_enrichment_parquet(
    rows: list[dict[str, Any]],
    table_name: str,
    enrichment_dir: Path | str,
) -> Path:
    """Atomically write a list of dict rows to ``<enrichment_dir>/<table>.parquet``.

    Empty ``rows`` still materializes a schema-only Parquet so downstream
    notebooks + view registration always have a file to read.
    """
    if table_name not in ENRICH_SCHEMAS:
        raise ValueError(f"unknown enrichment table: {table_name}")
    schema = ENRICH_SCHEMAS[table_name]
    enrichment_dir = Path(enrichment_dir)
    enrichment_dir.mkdir(parents=True, exist_ok=True)
    out_path = enrichment_dir / f"{table_name}.parquet"
    table = pa.Table.from_pylist(rows, schema=schema)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(out_path)
    return out_path


def read_enrichment_parquet(table_name: str, enrichment_dir: Path | str) -> pa.Table | None:
    """Return the on-disk Parquet for ``table_name`` or ``None`` if missing."""
    if table_name not in ENRICH_SCHEMAS:
        raise ValueError(f"unknown enrichment table: {table_name}")
    path = Path(enrichment_dir) / f"{table_name}.parquet"
    if not path.exists():
        return None
    return pq.read_table(path)


# ---------------------------------------------------------------------------
# DuckDB view registration
# ---------------------------------------------------------------------------


def register_enrichment_views(
    duckdb_path: Path | str,
    enrichment_dir: Path | str,
) -> list[str]:
    """Create/replace views over the enrichment Parquet tables.

    Returns the list of view names that were registered (one per Parquet file
    that exists). Missing files are silently skipped — that's the safe default
    when ``semantic_scholar.parquet`` doesn't exist yet because no SS key was
    set.
    """
    duckdb_path = Path(duckdb_path)
    enrichment_dir = Path(enrichment_dir).resolve()
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    registered: list[str] = []
    conn = duckdb.connect(str(duckdb_path))
    try:
        for table_name in ENRICH_SCHEMAS:
            parquet_path = enrichment_dir / f"{table_name}.parquet"
            if not parquet_path.exists():
                continue
            conn.execute(
                f"CREATE OR REPLACE VIEW {table_name} AS "
                f"SELECT * FROM read_parquet('{parquet_path}');"
            )
            registered.append(table_name)
    finally:
        conn.close()
    return registered
