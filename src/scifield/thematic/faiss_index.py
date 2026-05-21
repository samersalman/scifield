"""FAISS HNSW index build, persist, and PMID-map I/O for the thematic backbone.

This module implements the persistence layer described in V1-S05 design
decision D5: an `IndexHNSWFlat` over L2-normalised embeddings using inner
product as the similarity metric (equivalent to cosine), together with a
sidecar Parquet that maps sequential FAISS `row_id` values to PubMed PMIDs.

The index is written to `data/v1/faiss.index` and the PMID map to
`data/v1/faiss_pmid_map.parquet`; both paths are configurable via the
caller.

Notes
-----
* HNSW does not require a training pass — `index.is_trained` is `True`
  immediately after construction, so :func:`build_faiss_hnsw` skips
  training entirely.
* Vectors must be L2-normalised before being passed in; this module does
  not normalise on the caller's behalf.
* Distances returned by ``index.search`` are inner products which, for
  unit vectors, equal cosine similarities (higher = more similar).
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "build_faiss_hnsw",
    "write_index",
    "read_index",
    "write_pmid_map",
    "read_pmid_map",
]


def build_faiss_hnsw(
    vectors: np.ndarray,
    *,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> faiss.Index:
    """Build an inner-product HNSW index over L2-normalised vectors.

    Parameters
    ----------
    vectors:
        Float32 array of shape ``(n, d)``; assumed to be L2-normalised so
        that inner product equals cosine similarity.
    M:
        HNSW out-degree per node (graph connectivity).
    ef_construction:
        Candidate-list size during graph construction (higher = better
        recall, slower build).
    ef_search:
        Candidate-list size at query time (higher = better recall,
        slower queries). Set on the returned index; can be overridden
        later via ``index.hnsw.efSearch``.

    Returns
    -------
    faiss.Index
        A populated ``IndexHNSWFlat`` with ``ntotal == n``.

    Raises
    ------
    TypeError
        If ``vectors`` is not float32.
    ValueError
        If ``vectors`` is not 2-D or has zero rows.
    """
    if not isinstance(vectors, np.ndarray):
        raise TypeError(f"vectors must be a numpy.ndarray, got {type(vectors).__name__}")
    if vectors.dtype != np.float32:
        raise TypeError(f"vectors must have dtype float32, got {vectors.dtype}")
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-D (n, d); got shape {vectors.shape}")
    n, d = vectors.shape
    if n < 1:
        raise ValueError("vectors must contain at least one row")

    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    # HNSW is graph-based: no training needed. Sanity-check the assumption
    # so a future faiss version that flips this default fails loudly.
    assert index.is_trained, "IndexHNSWFlat should not require training"

    # Configure efConstruction *before* add() so it controls the build,
    # not just future incremental inserts.
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    contig = np.ascontiguousarray(vectors, dtype=np.float32)
    index.add(contig)
    return index


def write_index(index: faiss.Index, path: Path | str) -> Path:
    """Persist a FAISS index to ``path`` and return it as a ``Path``."""
    out = Path(path)
    faiss.write_index(index, str(out))
    return out


def read_index(
    path: Path | str,
    *,
    ef_search: int | None = None,
) -> faiss.Index:
    """Load a FAISS index from disk, optionally overriding ``efSearch``.

    The ``ef_search`` override is applied after load and only affects
    query-time recall, not the on-disk graph.
    """
    index = faiss.read_index(str(path))
    if ef_search is not None:
        index.hnsw.efSearch = ef_search
    return index


def write_pmid_map(
    pmids: list[int] | np.ndarray,
    path: Path | str,
) -> Path:
    """Write a ``row_id INT32 / pmid INT64`` mapping to Parquet.

    ``row_id`` is generated as ``0..n-1`` to mirror the implicit FAISS
    row order; callers must therefore pass ``pmids`` in the same order
    they were added to the index.
    """
    pmid_arr = np.asarray(pmids, dtype=np.int64)
    if pmid_arr.ndim != 1:
        raise ValueError(f"pmids must be 1-D; got shape {pmid_arr.shape}")
    n = int(pmid_arr.shape[0])
    row_ids = np.arange(n, dtype=np.int32)

    table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.int32()),
            "pmid": pa.array(pmid_arr, type=pa.int64()),
        }
    )
    out = Path(path)
    pq.write_table(table, out)
    return out


def read_pmid_map(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read the PMID map written by :func:`write_pmid_map`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(row_ids, pmids)`` as ``int32`` and ``int64`` arrays
        respectively.
    """
    table = pq.read_table(str(path), columns=["row_id", "pmid"])
    row_ids = table.column("row_id").to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    pmids = table.column("pmid").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return row_ids, pmids
