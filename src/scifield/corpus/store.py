"""Corpus storage primitives — Parquet writer, manifest writer, DuckDB view builder."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from scifield.repro import record_run

# Stable PyArrow schema for paper rows. Kept explicit so nested-typed columns
# (authors / mesh_headings / abstract_segments) stay binary-compatible across
# (journal, year) buckets — `read_parquet(..., union_by_name=true)` relies on it.
PAPER_SCHEMA = pa.schema(
    [
        ("pmid", pa.string()),
        ("journal_slug", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        (
            "abstract_segments",
            pa.list_(
                pa.struct(
                    [
                        ("label", pa.string()),
                        ("nlm_category", pa.string()),
                        ("text", pa.string()),
                    ]
                )
            ),
        ),
        ("journal", pa.string()),
        ("journal_ta", pa.string()),
        ("year", pa.int32()),
        ("pub_date", pa.string()),
        ("doi", pa.string()),
        ("publication_types", pa.list_(pa.string())),
        (
            "authors",
            pa.list_(
                pa.struct(
                    [
                        ("last_name", pa.string()),
                        ("fore_name", pa.string()),
                        ("initials", pa.string()),
                        ("affiliation", pa.string()),
                    ]
                )
            ),
        ),
        (
            "mesh_headings",
            pa.list_(
                pa.struct(
                    [
                        ("descriptor", pa.string()),
                        ("descriptor_ui", pa.string()),
                        ("major_topic", pa.bool_()),
                        (
                            "qualifiers",
                            pa.list_(
                                pa.struct(
                                    [
                                        ("name", pa.string()),
                                        ("ui", pa.string()),
                                        ("major_topic", pa.bool_()),
                                    ]
                                )
                            ),
                        ),
                    ]
                )
            ),
        ),
        ("has_abstract", pa.bool_()),
        ("fetched_at", pa.string()),
        ("source_ta_match", pa.string()),
    ]
)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to path atomically via a sibling .tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_bucket_parquet(
    rows: list[dict[str, Any]],
    slug: str,
    year: int,
    parquet_dir: Path | str,
) -> Path:
    """Write a (journal, year) bucket of paper rows as a single Parquet file.

    The Parquet is written through `<path>.tmp` then `rename`d, so partial
    failures never leave a half-written file behind. An empty `rows` list still
    materializes a schema-only Parquet so downstream idempotency graphs see the
    partition exist (a minor cost; simplifies re-run logic).
    """
    parquet_dir = Path(parquet_dir)
    out_path = parquet_dir / slug / f"{year}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=PAPER_SCHEMA)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(out_path)
    return out_path


def _manifest_path(manifest_dir: Path | str, slug: str, year: int) -> Path:
    return Path(manifest_dir) / slug / f"{year}.json"


def read_manifest(slug: str, year: int, manifest_dir: Path | str) -> dict[str, Any] | None:
    """Return parsed manifest dict, or None if missing/unreadable."""
    path = _manifest_path(manifest_dir, slug, year)
    if not path.exists():
        return None
    try:
        loaded: dict[str, Any] = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return loaded


def write_manifest(
    manifest_dir: Path | str,
    slug: str,
    year: int,
    payload: dict[str, Any],
) -> Path:
    """Atomic-write the manifest JSON. Returns the manifest path."""
    path = _manifest_path(manifest_dir, slug, year)
    enriched = {**payload}
    enriched.setdefault("fetched_at", datetime.now(UTC).isoformat())
    _atomic_write_bytes(path, json.dumps(enriched, indent=2, sort_keys=True).encode())
    return path


def build_duckdb(
    parquet_dir: Path | str,
    duckdb_path: Path | str,
    config: dict[str, Any] | None = None,
) -> Path:
    """Rebuild the thin DuckDB view layer over the Parquet corpus.

    The DuckDB file is regenerable from Parquet — Parquet is the source of
    truth. We remove any pre-existing file so the resulting view definitions
    stay tidy (cheaper than dropping each view by name).
    """
    parquet_dir = Path(parquet_dir)
    duckdb_path = Path(duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    if duckdb_path.exists():
        duckdb_path.unlink()

    parquet_files = sorted(parquet_dir.glob("*/*.parquet"))
    conn = duckdb.connect(str(duckdb_path))
    try:
        if not parquet_files:
            print(
                f"[scifield.corpus.store] build_duckdb: no Parquet files under {parquet_dir}; "
                "writing empty DuckDB file without views.",
                file=sys.stderr,
            )
        else:
            # Resolve to an absolute path so the embedded view definitions
            # work regardless of the cwd from which the .duckdb file is later
            # opened (e.g., from notebooks/ via `jupyter execute`).
            glob_pattern = str(parquet_dir.resolve() / "*" / "*.parquet")
            conn.execute(
                "CREATE OR REPLACE VIEW papers AS "
                f"SELECT * FROM read_parquet('{glob_pattern}', union_by_name=true);"
            )
            conn.execute(
                "CREATE OR REPLACE VIEW journals AS "
                "SELECT journal_slug, journal, journal_ta, COUNT(*) AS n_papers "
                "FROM papers GROUP BY 1, 2, 3;"
            )
            conn.execute(
                "CREATE OR REPLACE VIEW mesh AS "
                "SELECT pmid, journal_slug, year, unnest(mesh_headings) AS heading "
                "FROM papers;"
            )
    finally:
        conn.close()

    record_run(
        artifact_path=duckdb_path,
        inputs={},
        config=config or {"parquet_dir": str(parquet_dir)},
    )
    return duckdb_path
