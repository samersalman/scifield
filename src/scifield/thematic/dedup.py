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

# Tier-1 generalist journals (conf/corpus/v2.yaml): the single source of truth is
# scifield.cartography.corpus_config. On a cross-journal PMID collision, the dedup
# tiebreak prefers the more-specific specialty journal, so generalists are sorted
# LAST. corpus_config imports only stdlib + yaml (no cartography/thematic siblings),
# so this import is cycle-free.
from scifield.cartography.corpus_config import get_generalist_slugs

__all__ = [
    "ensure_papers_distinct_view",
    "load_deduped_embeddings",
    "integrity_check_v1_carryover",
]


def _sql_quote_str_list(slugs: frozenset[str]) -> str:
    """Render ``slugs`` as a DuckDB string IN-list, single-quote escaped.

    Sorted for a stable, deterministic SQL string (the VIEW DDL is otherwise
    sensitive to frozenset iteration order). Single quotes are doubled per the
    SQL standard so the literal is injection-safe even if a slug ever contains
    one.
    """
    return ", ".join("'" + s.replace("'", "''") + "'" for s in sorted(slugs))


def _papers_columns(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Return the lowercased column names of the ``papers`` table."""
    rows = con.execute("DESCRIBE papers").fetchall()
    # DuckDB DESCRIBE returns (column_name, column_type, null, key, default, extra)
    return {str(r[0]).lower() for r in rows}


def ensure_papers_distinct_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create or replace the ``papers_distinct`` VIEW with one row per PMID.

    Tiebreak order, most-significant first:

    1. **Generalist demotion** — on a cross-journal PMID collision (a paper
       co-listed under a Tier-1 generalist and a specialty journal), the
       more-specific specialty row wins: generalists
       (:func:`scifield.cartography.corpus_config.get_generalist_slugs`) get
       ``CASE = 1`` and sort last, specialty/surgical rows get ``0``.
    2. **Longest abstract** (NULLs last).
    3. **Freshest fetch** (``fetched_at DESC``), when that column is present.
       We fall back to abstract-only ordering for older snapshots that
       pre-date ``fetched_at``; this keeps the helper backwards-compatible.
    4. **``journal_slug`` ASC** — a final total-order tiebreak so the row
       kept is fully deterministic even under exact ties on (1)–(3).

    For v1's specialty-only corpus no row is a generalist, so terms (1) and
    (4) never change which row wins versus the historical abstract/freshness
    ordering — (4) only decides ties that did not previously occur.

    Terms (1) and (4) are emitted only when the ``journal_slug`` column is
    present (it always is on the real store schema; this guard keeps the
    helper working on minimal/legacy snapshots that omit it, mirroring the
    ``fetched_at`` fallback).

    The assertion at the end is the contract: callers can rely on
    ``COUNT(*) == COUNT(DISTINCT pmid)`` post-call.
    """
    columns = _papers_columns(con)
    has_fetched_at = "fetched_at" in columns
    has_journal_slug = "journal_slug" in columns

    order_parts: list[str] = []
    if has_journal_slug:
        generalist_in_list = _sql_quote_str_list(get_generalist_slugs())
        order_parts.append(
            f"(CASE WHEN journal_slug IN ({generalist_in_list}) THEN 1 ELSE 0 END) ASC"
        )
    order_parts.append("length(abstract) DESC NULLS LAST")
    if has_fetched_at:
        order_parts.append("fetched_at DESC")
    if has_journal_slug:
        order_parts.append("journal_slug ASC")
    order_clause = ", ".join(order_parts)

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
    try:
        # combine_chunks() + flatten() respects per-chunk offsets. Iterating the
        # chunks and reading ``chunk.values`` returns the FULL shared values buffer
        # for every chunk, so a multi-chunk FixedSizeList column (any parquet with
        # > 131072 rows, e.g. the v2 corpus) yields n_chunks x rows.
        arr_col = embedding_col.combine_chunks()
        list_size = arr_col.type.list_size
        flat = arr_col.flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        return flat.reshape(len(arr_col), list_size)
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
