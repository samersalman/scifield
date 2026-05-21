"""Tests for enrichment Parquet schemas + DuckDB view registration (V1-S04)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq
import pytest

from scifield.corpus.enrich_store import (
    ENRICH_SCHEMAS,
    read_enrichment_parquet,
    register_enrichment_views,
    write_enrichment_parquet,
)


def _sample_row(table_name: str) -> dict[str, Any]:
    """Build a minimal-but-valid row matching every column of the named schema."""
    if table_name == "openalex_works":
        return {
            "pmid": "1",
            "openalex_id": "W1",
            "oa_doi": "10.x/1",
            "type": "article",
            "language": "en",
            "is_retracted": False,
            "is_oa": True,
            "oa_status": "green",
            "cited_by_count": 5,
            "concepts": [
                {"id": "C1", "display_name": "x", "score": 0.9},
            ],
            "publication_year": 2024,
            "publication_date": "2024-01-01",
            "fetched_at": "2024-01-01T00:00:00Z",
        }
    if table_name == "references_out":
        return {
            "citing_pmid": "1",
            "ref_openalex_id": "W2",
            "ref_pmid_if_known": "",
            "ref_year": 2020,
            "ref_position": 0,
        }
    if table_name == "authorships":
        return {
            "pmid": "1",
            "author_position": 0,
            "author_position_label": "first",
            "author_oa_id": "A1",
            "author_orcid": "",
            "author_display_name": "Jane",
            "author_canonical_id": "OA:A1",
            "disambiguation_method": "openalex",
            "is_first": True,
            "is_last": False,
        }
    if table_name == "institutions":
        return {
            "institution_canonical_id": "OA:I1",
            "institution_oa_id": "I1",
            "ror_id": "https://ror.org/abc",
            "display_name": "Stanford",
            "country_code": "US",
            "type": "education",
        }
    if table_name == "paper_institutions":
        return {
            "pmid": "1",
            "author_position": 0,
            "institution_canonical_id": "OA:I1",
            "raw_affiliation_string": "Stanford",
            "ror_matched_by": "openalex",
        }
    if table_name == "semantic_scholar":
        return {
            "pmid": "1",
            "ss_id": "SS1",
            "ss_doi": "10.x",
            "citation_count": 5,
            "references_with_intent_count": 2,
            "fetched_at": "2024-01-01T00:00:00Z",
        }
    if table_name == "citation_intents":
        return {
            "citing_pmid": "1",
            "cited_id": "PMID:99",
            "intents": ["background"],
            "is_influential": False,
        }
    if table_name == "enrichment_failed":
        return {
            "pmid": "1",
            "source": "openalex",
            "reason": "not_found",
            "attempted_at": "2024-01-01T00:00:00Z",
        }
    raise AssertionError(f"no sample row defined for {table_name}")


def test_each_schema_roundtrip(tmp_path: Path) -> None:
    """Every named schema should accept a sample row, round-trip on disk, and
    preserve its exact schema."""
    enrichment_dir = tmp_path / "enrichment"
    for table_name, schema in ENRICH_SCHEMAS.items():
        row = _sample_row(table_name)
        out_path = write_enrichment_parquet([row], table_name, enrichment_dir)
        assert out_path.exists()
        assert out_path == enrichment_dir / f"{table_name}.parquet"

        table = read_enrichment_parquet(table_name, enrichment_dir)
        assert table is not None
        assert table.schema.equals(schema), f"schema mismatch for {table_name}"
        assert table.num_rows == 1


def test_empty_write_creates_schema_only_parquet(tmp_path: Path) -> None:
    """Empty rows should still produce a 0-row Parquet whose schema matches."""
    enrichment_dir = tmp_path / "enrichment"
    out_path = write_enrichment_parquet([], "openalex_works", enrichment_dir)
    assert out_path.exists()

    table = pq.read_table(out_path)
    assert table.num_rows == 0
    assert table.schema.equals(ENRICH_SCHEMAS["openalex_works"])


def test_atomic_write(tmp_path: Path) -> None:
    """Writing twice should leave only the latest content + no .tmp leftover."""
    enrichment_dir = tmp_path / "enrichment"
    first = _sample_row("openalex_works")
    second = _sample_row("openalex_works")
    second["pmid"] = "2"
    second["openalex_id"] = "W2"

    write_enrichment_parquet([first], "openalex_works", enrichment_dir)
    out_path = write_enrichment_parquet([second], "openalex_works", enrichment_dir)

    # No .tmp file should be left behind.
    tmp_leftover = out_path.with_suffix(out_path.suffix + ".tmp")
    assert not tmp_leftover.exists()

    table = pq.read_table(out_path)
    assert table.num_rows == 1
    pmids_back = table.column("pmid").to_pylist()
    assert pmids_back == ["2"]


def test_unknown_table_raises(tmp_path: Path) -> None:
    """Unknown table names should raise ValueError on write."""
    enrichment_dir = tmp_path / "enrichment"
    with pytest.raises(ValueError):
        write_enrichment_parquet([], "not_a_table", enrichment_dir)


def test_register_views_creates_views(tmp_path: Path) -> None:
    """Writing several enrichment tables and registering views should create
    queryable DuckDB views with the right row counts."""
    enrichment_dir = tmp_path / "enrichment"
    duckdb_path = tmp_path / "papers.duckdb"

    # Write three tables with known row counts.
    write_enrichment_parquet([_sample_row("openalex_works")], "openalex_works", enrichment_dir)
    refs = [_sample_row("references_out") for _ in range(3)]
    for i, r in enumerate(refs):
        r["ref_position"] = i
    write_enrichment_parquet(refs, "references_out", enrichment_dir)
    write_enrichment_parquet([_sample_row("authorships")], "authorships", enrichment_dir)

    registered = register_enrichment_views(duckdb_path, enrichment_dir)
    assert set(registered) == {"openalex_works", "references_out", "authorships"}

    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        n_works = conn.execute("SELECT COUNT(*) FROM openalex_works").fetchone()[0]
        assert n_works == 1

        n_refs = conn.execute("SELECT COUNT(*) FROM references_out").fetchone()[0]
        assert n_refs == 3

        n_auth = conn.execute("SELECT COUNT(*) FROM authorships").fetchone()[0]
        assert n_auth == 1
    finally:
        conn.close()


def test_register_views_skips_missing(tmp_path: Path) -> None:
    """Tables without Parquet files on disk should be silently skipped."""
    enrichment_dir = tmp_path / "enrichment"
    duckdb_path = tmp_path / "papers.duckdb"

    written = ["openalex_works", "citation_intents"]
    for table_name in written:
        write_enrichment_parquet([_sample_row(table_name)], table_name, enrichment_dir)

    registered = register_enrichment_views(duckdb_path, enrichment_dir)
    assert set(registered) == set(written)
    # None of the un-written tables should show up.
    for table_name in ENRICH_SCHEMAS:
        if table_name not in written:
            assert table_name not in registered
