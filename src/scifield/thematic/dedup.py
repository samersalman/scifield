"""V1-S05 carryover dedup utilities for the thematic backbone.

V1-S05 left two known duplication artifacts that must be handled at the
boundary of V1-S06 — not deeper inside UMAP/HDBSCAN, where identical rows
would silently inflate cluster density:

1. ``papers.duckdb`` has byte-identical duplicate PMIDs from overlapping
   eSearch pagination on OR'd TA-term queries. We expose a non-destructive
   ``papers_distinct`` VIEW that keeps the longest-abstract / freshest row
   per PMID. The VIEW is idempotent — re-running the helper is safe.
2. ``embeddings.parquet`` inherited the same duplication. Because every
   duplicate group was confirmed byte-identical at V1-S05 close, dedup is
   lossless; this module verifies that invariant on read and raises if it
   has been violated, rather than silently collapsing divergent vectors.

Heavy deps (pandas, bertopic, gensim, etc.) are intentionally NOT imported
here — dedup is a precondition for the rest of the thematic pipeline and
should be cheap.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

__all__ = [
    "ensure_papers_distinct_view",
    "load_deduped_embeddings",
    "integrity_check_v1_carryover",
]


def _papers_columns(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Return the lowercased column names of the ``papers`` table."""
    rows = con.execute("DESCRIBE papers").fetchall()
    # DuckDB DESCRIBE returns (column_name, column_type, null, key, default, extra)
    return {str(r[0]).lower() for r in rows}


def ensure_papers_distinct_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create or replace the ``papers_distinct`` VIEW with one row per PMID.

    Tiebreak prefers the longest abstract (NULLs last) and, when the
    ``fetched_at`` column is present, the freshest fetch. We fall back to
    the abstract-only ordering for older snapshots that pre-date the
    ``fetched_at`` column; this keeps the helper backwards-compatible
    without conditionals inside the SQL.

    The assertion at the end is the contract: callers can rely on
    ``COUNT(*) == COUNT(DISTINCT pmid)`` post-call.
    """
    has_fetched_at = "fetched_at" in _papers_columns(con)
    order_clause = "length(abstract) DESC NULLS LAST"
    if has_fetched_at:
        order_clause += ", fetched_at DESC"

    sql = f"""
        CREATE OR REPLACE VIEW papers_distinct AS
        SELECT * FROM papers
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY pmid
            ORDER BY {order_clause}
        ) = 1
    """
    con.execute(sql)

    total, distinct = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT pmid) FROM papers_distinct"
    ).fetchone()
    if total != distinct:
        raise AssertionError(
            f"papers_distinct view is not unique-per-pmid: total={total}, distinct={distinct}"
        )


def integrity_check_v1_carryover(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Report dedup-relevant counts for sidecar logging.

    Does not hard-code the V1-S05 numbers (13,070 dup PMIDs, etc.) — the
    point is to surface whatever the current snapshot is so the caller
    can compare against the recorded baseline.
    """
    ensure_papers_distinct_view(con)
    total = int(con.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
    distinct = int(con.execute("SELECT COUNT(*) FROM papers_distinct").fetchone()[0])
    return {
        "papers_total": total,
        "papers_distinct": distinct,
        "papers_duplicate_pmids": total - distinct,
    }


def _embedding_chunks_to_array(embedding_col) -> np.ndarray:
    """Decode the ``embedding`` column to a contiguous (n, dim) float32 array.

    Mirrors the fast-path / fallback split in ``scifield.cli.faiss_build``:
    a FixedSizeList of fp16/fp32 values flattens via ``values.to_numpy``,
    while a generic LIST<...> requires per-row materialisation.
    """
    chunks: list[np.ndarray] = []
    try:
        for chunk in embedding_col.chunks:
            values = chunk.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            list_size = chunk.type.list_size
            chunks.append(values.reshape(-1, list_size))
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    except (AttributeError, TypeError):
        py_lists = embedding_col.to_pylist()
        return np.asarray(py_lists, dtype=np.float32)


def load_deduped_embeddings(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pmids, embeddings)`` with one row per PMID.

    The parquet is read row-order-preserving; for each PMID we keep the
    first occurrence. We then verify that every other row for that PMID is
    elementwise close (``rtol=1e-3, atol=1e-3``, sized for fp16). A
    divergent duplicate group is treated as a corruption signal, not a
    pick-one situation — we raise ``ValueError`` with the offending PMID
    and the maximum elementwise difference so the caller can investigate.

    Parameters
    ----------
    parquet_path:
        Path to the V1-S05 ``embeddings.parquet`` (``pmid`` + ``embedding``
        FixedSizeList<float16>).
    """
    path = Path(parquet_path)
    table = pq.read_table(path)
    pmids_all = table.column("pmid").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    emb_all = _embedding_chunks_to_array(table.column("embedding"))

    if emb_all.ndim != 2:
        raise ValueError(f"embedding column must be 2-D; got shape {emb_all.shape}")
    if emb_all.shape[0] != pmids_all.shape[0]:
        raise ValueError(f"pmid/embedding row mismatch: {pmids_all.shape[0]} vs {emb_all.shape[0]}")

    # First-occurrence index per pmid preserves the natural row order from
    # the source parquet, which matches V1-S05 producer order.
    seen: dict[int, int] = {}
    first_idx: list[int] = []
    groups: dict[int, list[int]] = {}
    for i, pmid in enumerate(pmids_all.tolist()):
        if pmid not in seen:
            seen[pmid] = i
            first_idx.append(i)
        groups.setdefault(pmid, []).append(i)

    for pmid, idxs in groups.items():
        if len(idxs) <= 1:
            continue
        ref = emb_all[idxs[0]]
        for j in idxs[1:]:
            if not np.allclose(ref, emb_all[j], rtol=1e-3, atol=1e-3):
                max_diff = float(np.max(np.abs(ref - emb_all[j])))
                raise ValueError(
                    f"divergent duplicate embeddings for pmid={pmid}: max|Δ|={max_diff:.6f} "
                    f"(rows {idxs[0]} vs {j})"
                )

    keep_idx = np.asarray(first_idx, dtype=np.int64)
    pmids_out = pmids_all[keep_idx]
    emb_out = np.ascontiguousarray(emb_all[keep_idx], dtype=np.float32)
    return pmids_out, emb_out
