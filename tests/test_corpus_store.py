"""Tests for the corpus Parquet writer + DuckDB view builder (V1-S03 Task 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from scifield.corpus import (
    PAPER_SCHEMA,
    build_duckdb,
    write_bucket_parquet,
)


def _sample_row(pmid: str, slug: str, year: int, n_mesh: int) -> dict[str, Any]:
    """Build a row that exercises every PAPER_SCHEMA field including nested types."""
    mesh = [
        {
            "descriptor": f"Topic{i}",
            "descriptor_ui": f"D{i:07d}",
            "major_topic": (i == 0),
            "qualifiers": [
                {"name": f"qual{i}", "ui": f"Q{i:07d}", "major_topic": False},
            ],
        }
        for i in range(n_mesh)
    ]
    return {
        "pmid": pmid,
        "journal_slug": slug,
        "title": f"Title for {pmid}",
        "abstract": "Body abstract text.",
        "abstract_segments": [
            {"label": "", "nlm_category": "", "text": "Body abstract text."},
        ],
        "journal": "Sample Journal",
        "journal_ta": "Samp J",
        "year": year,
        "pub_date": f"{year}-01-01",
        "doi": f"10.1000/{pmid}",
        "publication_types": ["Journal Article"],
        "authors": [
            {
                "last_name": "Smith",
                "fore_name": "Jane",
                "initials": "J",
                "affiliation": "Stanford",
            },
            {
                "last_name": "Doe",
                "fore_name": "John",
                "initials": "J",
                "affiliation": "Mayo",
            },
        ],
        "mesh_headings": mesh,
        "has_abstract": True,
        "fetched_at": "2026-05-19T00:00:00+00:00",
        "source_ta_match": "Samp J",
    }


def test_write_bucket_parquet_roundtrip(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    row = _sample_row("12345", "arthroscopy", 2024, n_mesh=2)

    out_path = write_bucket_parquet([row], slug="arthroscopy", year=2024, parquet_dir=parquet_dir)
    assert out_path.exists()
    assert out_path == parquet_dir / "arthroscopy" / "2024.parquet"

    table = pq.read_table(out_path)
    assert table.schema.equals(PAPER_SCHEMA)
    assert table.num_rows == 1

    # Nested types survive the round-trip.
    authors_back = table.column("authors").to_pylist()[0]
    assert authors_back == row["authors"]
    mesh_back = table.column("mesh_headings").to_pylist()[0]
    assert mesh_back == row["mesh_headings"]
    segments_back = table.column("abstract_segments").to_pylist()[0]
    assert segments_back == row["abstract_segments"]


def test_build_duckdb_views(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    duckdb_path = tmp_path / "papers.duckdb"

    rows_a = [
        _sample_row("1", "arthroscopy", 2023, n_mesh=2),
        _sample_row("2", "arthroscopy", 2024, n_mesh=1),
    ]
    rows_b = [
        _sample_row("3", "ann_surg", 2024, n_mesh=3),
        _sample_row("4", "ann_surg", 2024, n_mesh=0),
    ]
    # Tweak journal/journal_ta on the second bucket so the journals view has
    # two distinct rows.
    for r in rows_b:
        r["journal"] = "Annals of Surgery"
        r["journal_ta"] = "Ann Surg"

    write_bucket_parquet(rows_a, slug="arthroscopy", year=2023, parquet_dir=parquet_dir)
    # Two arthroscopy years would land in the same Parquet path; keep one for
    # the first slug + one for the second slug.
    write_bucket_parquet(rows_b, slug="ann_surg", year=2024, parquet_dir=parquet_dir)

    build_duckdb(parquet_dir=parquet_dir, duckdb_path=duckdb_path)
    assert duckdb_path.exists()

    sidecar = Path(str(duckdb_path) + ".run.json")
    assert sidecar.exists()

    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        n_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        assert n_papers == 4

        n_journals = conn.execute("SELECT COUNT(*) FROM journals").fetchone()[0]
        # Two distinct (slug, journal, journal_ta) tuples.
        assert n_journals == 2

        n_mesh = conn.execute("SELECT COUNT(*) FROM mesh").fetchone()[0]
        expected_mesh = sum(len(r["mesh_headings"]) for r in rows_a + rows_b)
        assert n_mesh == expected_mesh
    finally:
        conn.close()
