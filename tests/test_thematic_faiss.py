"""Tests for :mod:`scifield.thematic.faiss_index`.

These tests use only synthetic data (no HuggingFace model, no DuckDB) so
they run fast in CI and on Samer's Mac. We plant 4 clusters of 25 unit
vectors in 16-D space and verify that the HNSW index recovers
within-cluster nearest neighbours, that on-disk roundtrips preserve
recall, and that the PMID map survives Parquet I/O exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from scifield.thematic.faiss_index import (
    build_faiss_hnsw,
    read_index,
    read_pmid_map,
    write_index,
    write_pmid_map,
)

# --- Fixtures ---------------------------------------------------------------

N_CLUSTERS = 4
PER_CLUSTER = 25
DIM = 16
N = N_CLUSTERS * PER_CLUSTER
NOISE_SIGMA = 0.05


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


@pytest.fixture()
def planted_clusters() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vectors, labels)`` with 4 planted clusters of 25 each."""
    rng = np.random.default_rng(seed=42)

    # Cluster centroids: random directions, L2-normalised.
    centroids = rng.standard_normal(size=(N_CLUSTERS, DIM)).astype(np.float32)
    centroids = _l2_normalise(centroids).astype(np.float32)

    vectors = np.empty((N, DIM), dtype=np.float32)
    labels = np.empty(N, dtype=np.int32)
    for k in range(N_CLUSTERS):
        noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=(PER_CLUSTER, DIM)).astype(np.float32)
        block = centroids[k][None, :] + noise
        block = _l2_normalise(block).astype(np.float32)
        start = k * PER_CLUSTER
        vectors[start : start + PER_CLUSTER] = block
        labels[start : start + PER_CLUSTER] = k

    return vectors, labels


def _top2_neighbour_labels(index, vectors: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    """Return ``(n_correct, n_total)`` where 'correct' = top-2 NN shares label."""
    # Search top-3 so we have a safety margin if a self-hit lands at rank 1 or 2.
    _, neigh = index.search(np.ascontiguousarray(vectors, dtype=np.float32), 3)
    n_correct = 0
    n_total = N
    for i in range(N):
        # Find the first hit that is not the query itself.
        nn = -1
        for j in range(neigh.shape[1]):
            cand = int(neigh[i, j])
            if cand != i and cand != -1:
                nn = cand
                break
        if nn >= 0 and labels[nn] == labels[i]:
            n_correct += 1
    return n_correct, n_total


# --- Tests ------------------------------------------------------------------


def test_build_hyperparams_set_and_index_populated(planted_clusters):
    vectors, _ = planted_clusters
    index = build_faiss_hnsw(vectors, M=8, ef_construction=64, ef_search=32)

    assert index.ntotal == N
    assert index.hnsw.efConstruction == 64
    assert index.hnsw.efSearch == 32


def test_in_memory_nn_recovers_planted_cluster(planted_clusters):
    vectors, labels = planted_clusters
    index = build_faiss_hnsw(vectors, M=8, ef_construction=64, ef_search=32)

    n_correct, n_total = _top2_neighbour_labels(index, vectors, labels)
    recall = n_correct / n_total
    assert recall >= 0.90, f"in-memory top-2 NN recall {recall:.3f} below 0.90 threshold"


def test_persistence_roundtrip_preserves_recall(planted_clusters, tmp_path):
    vectors, labels = planted_clusters
    index = build_faiss_hnsw(vectors, M=8, ef_construction=64, ef_search=32)
    n_correct_before, _ = _top2_neighbour_labels(index, vectors, labels)

    index_path = tmp_path / "faiss.index"
    returned = write_index(index, index_path)
    assert returned == index_path
    assert index_path.exists()

    loaded = read_index(index_path)
    # ef_search persists with the index, no need to override here.
    assert loaded.ntotal == N
    n_correct_after, _ = _top2_neighbour_labels(loaded, vectors, labels)
    assert n_correct_after == n_correct_before, (
        "recall changed after on-disk roundtrip: " f"{n_correct_before} -> {n_correct_after}"
    )

    # Override path works without raising and is reflected on the index.
    loaded2 = read_index(index_path, ef_search=128)
    assert loaded2.hnsw.efSearch == 128


def test_pmid_map_roundtrip(tmp_path):
    pmids = (np.arange(N, dtype=np.int64) + 100_000_000).tolist()
    map_path = tmp_path / "faiss_pmid_map.parquet"

    returned = write_pmid_map(pmids, map_path)
    assert returned == map_path
    assert map_path.exists()

    row_ids, loaded_pmids = read_pmid_map(map_path)
    assert row_ids.dtype == np.int32
    assert loaded_pmids.dtype == np.int64
    np.testing.assert_array_equal(row_ids, np.arange(N, dtype=np.int32))
    np.testing.assert_array_equal(loaded_pmids, np.asarray(pmids, dtype=np.int64))


# --- Input validation -------------------------------------------------------


def test_build_rejects_wrong_dtype():
    bad = np.zeros((10, 4), dtype=np.float64)
    with pytest.raises(TypeError):
        build_faiss_hnsw(bad)


def test_build_rejects_wrong_rank():
    bad = np.zeros((10,), dtype=np.float32)
    with pytest.raises(ValueError):
        build_faiss_hnsw(bad)


def test_build_rejects_empty():
    bad = np.zeros((0, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        build_faiss_hnsw(bad)
