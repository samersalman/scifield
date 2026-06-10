"""Tests for :mod:`scifield.thematic.dedup`.

Synthetic fixtures only — no real ``papers.duckdb`` and no real
``embeddings.parquet`` are required. We plant duplicate rows by hand so
the dedup invariants are pinned to the helper's behavior, not to a
particular V1-S05 snapshot.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scifield.thematic.dedup import (
    ensure_papers_distinct_view,
    integrity_check_v1_carryover,
    load_deduped_embeddings,
)


def _make_papers_db(tmp_path: Path, *, with_fetched_at: bool) -> duckdb.DuckDBPyConnection:
    """Create a tiny in-memory ``papers`` table with planted duplicates."""
    con = duckdb.connect(str(tmp_path / "papers.duckdb"))
    if with_fetched_at:
        con.execute(
            "CREATE TABLE papers (pmid BIGINT, title VARCHAR, abstract VARCHAR, "
            "fetched_at TIMESTAMP)"
        )
        con.executemany(
            "INSERT INTO papers VALUES (?, ?, ?, ?)",
            [
                (1, "t1", "longer abstract", "2025-01-01 10:00:00"),
                (1, "t1", "short", "2024-12-31 09:00:00"),
                (2, "t2", "unique", "2025-01-02 10:00:00"),
                (3, "t3", "longer-abstract-3", "2025-01-03 10:00:00"),
                (3, "t3", "longer-abstract-3", "2025-01-03 11:00:00"),
            ],
        )
    else:
        con.execute("CREATE TABLE papers (pmid BIGINT, title VARCHAR, abstract VARCHAR)")
        con.executemany(
            "INSERT INTO papers VALUES (?, ?, ?)",
            [
                (1, "t1", "longer abstract"),
                (1, "t1", "short"),
                (2, "t2", "unique"),
                (3, "t3", "longer-abstract-3"),
                (3, "t3", "longer-abstract-3"),
            ],
        )
    return con


def test_ensure_papers_distinct_view_unique_per_pmid(tmp_path: Path) -> None:
    con = _make_papers_db(tmp_path, with_fetched_at=True)
    ensure_papers_distinct_view(con)
    total, distinct = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT pmid) FROM papers_distinct"
    ).fetchone()
    assert total == distinct == 3
    # Tiebreak: longest abstract wins.
    abstract = con.execute("SELECT abstract FROM papers_distinct WHERE pmid = 1").fetchone()[0]
    assert abstract == "longer abstract"


def test_ensure_papers_distinct_view_works_without_fetched_at(tmp_path: Path) -> None:
    con = _make_papers_db(tmp_path, with_fetched_at=False)
    ensure_papers_distinct_view(con)
    total, distinct = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT pmid) FROM papers_distinct"
    ).fetchone()
    assert total == distinct == 3


def test_cross_journal_collision_prefers_specialty(tmp_path: Path) -> None:
    """A PMID co-listed under a generalist + a specialty journal keeps the
    specialty row, regardless of abstract length or insertion order."""
    con = duckdb.connect(str(tmp_path / "papers.duckdb"))
    con.execute(
        "CREATE TABLE papers (pmid BIGINT, title VARCHAR, abstract VARCHAR, "
        "journal_slug VARCHAR, fetched_at TIMESTAMP)"
    )
    con.executemany(
        "INSERT INTO papers VALUES (?, ?, ?, ?, ?)",
        [
            # pmid 1: generalist row has the LONGER abstract + FRESHER fetch,
            # yet specialty must still win on the generalist-demotion term.
            (1, "t1", "a much longer abstract here", "nature", "2025-01-02 10:00:00"),
            (1, "t1", "short", "j_orthop_surg", "2025-01-01 10:00:00"),
            # pmid 2: two specialty journals (no generalist) — deterministic
            # final tiebreak is journal_slug ASC, so 'a_surg' beats 'z_surg'.
            (2, "t2", "same", "z_surg", "2025-01-03 10:00:00"),
            (2, "t2", "same", "a_surg", "2025-01-03 10:00:00"),
        ],
    )
    ensure_papers_distinct_view(con)
    total, distinct = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT pmid) FROM papers_distinct"
    ).fetchone()
    assert total == distinct == 2
    slug1 = con.execute("SELECT journal_slug FROM papers_distinct WHERE pmid = 1").fetchone()[0]
    assert slug1 == "j_orthop_surg"
    slug2 = con.execute("SELECT journal_slug FROM papers_distinct WHERE pmid = 2").fetchone()[0]
    assert slug2 == "a_surg"


def test_specialty_only_unaffected_by_generalist_term(tmp_path: Path) -> None:
    """v1-style specialty-only corpus: the longest-abstract winner is
    unchanged — the generalist-demotion term is a no-op when no row is a
    generalist (backward-compat guard for v1 outputs)."""
    con = duckdb.connect(str(tmp_path / "papers.duckdb"))
    con.execute(
        "CREATE TABLE papers (pmid BIGINT, title VARCHAR, abstract VARCHAR, journal_slug VARCHAR)"
    )
    con.executemany(
        "INSERT INTO papers VALUES (?, ?, ?, ?)",
        [
            (1, "t1", "longer abstract", "j_orthop_surg"),
            (1, "t1", "short", "j_orthop_surg"),
        ],
    )
    ensure_papers_distinct_view(con)
    abstract = con.execute("SELECT abstract FROM papers_distinct WHERE pmid = 1").fetchone()[0]
    assert abstract == "longer abstract"


def test_integrity_check_counts(tmp_path: Path) -> None:
    con = _make_papers_db(tmp_path, with_fetched_at=True)
    report = integrity_check_v1_carryover(con)
    assert report == {
        "papers_total": 5,
        "papers_distinct": 3,
        "papers_duplicate_pmids": 2,
    }


def _write_embeddings_parquet(
    path: Path,
    pmid_seq: list[int],
    rows: np.ndarray,
) -> None:
    """Write a ``pmid + embedding`` parquet matching V1-S05 schema (FixedSizeList<float16>)."""
    assert rows.dtype == np.float16
    dim = int(rows.shape[1])
    flat = pa.array(rows.reshape(-1).tolist(), type=pa.float16())
    embedding_arr = pa.FixedSizeListArray.from_arrays(flat, dim)
    table = pa.table(
        {
            "pmid": pa.array(pmid_seq, type=pa.int64()),
            "embedding": embedding_arr,
        }
    )
    pq.write_table(table, path)


def test_load_deduped_embeddings_byte_identical_duplicates(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=42)
    dim = 8
    unique_pmids = [10, 20, 30, 40, 50]
    unique_vecs = rng.standard_normal(size=(5, dim)).astype(np.float16)

    pmid_seq = [10, 20, 20, 30, 40, 40, 50]
    rows = np.stack(
        [
            unique_vecs[0],
            unique_vecs[1],
            unique_vecs[1],
            unique_vecs[2],
            unique_vecs[3],
            unique_vecs[3],
            unique_vecs[4],
        ]
    )
    path = tmp_path / "embeddings.parquet"
    _write_embeddings_parquet(path, pmid_seq, rows)

    pmids_out, emb_out = load_deduped_embeddings(path)
    assert pmids_out.tolist() == unique_pmids
    assert emb_out.shape == (5, dim)
    assert emb_out.dtype == np.float32
    np.testing.assert_allclose(emb_out, unique_vecs.astype(np.float32), rtol=1e-3, atol=1e-3)


def test_load_deduped_embeddings_divergent_duplicate_raises(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=7)
    dim = 8
    vec_a = rng.standard_normal(size=(dim,)).astype(np.float16)
    vec_a_alt = vec_a.copy()
    vec_a_alt[0] = np.float16(vec_a_alt[0] + 0.5)
    vec_b = rng.standard_normal(size=(dim,)).astype(np.float16)

    rows = np.stack([vec_a, vec_a_alt, vec_b])
    pmid_seq = [11, 11, 22]
    path = tmp_path / "embeddings.parquet"
    _write_embeddings_parquet(path, pmid_seq, rows)

    with pytest.raises(ValueError, match="divergent duplicate embeddings"):
        load_deduped_embeddings(path)
