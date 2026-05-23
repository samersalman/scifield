"""Tests for :mod:`scifield.thematic.topics`.

Plants 5 clusters of ~100 docs each in 64-D embedding space with a
controlled per-cluster vocabulary, then asserts that the BERTopic
pipeline recovers ≥4 of the 5 planted clusters and that
:func:`build_hierarchy` returns a well-shaped DataFrame.

Marked ``slow`` — the full BERTopic fit runs UMAP + HDBSCAN end-to-end on
500 vectors, which takes 10-30 seconds depending on BLAS thread count.
CI can opt out with ``pytest -m 'not slow'``; the default suite runs it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("bertopic")
pytest.importorskip("umap")
pytest.importorskip("hdbscan")

from scifield.thematic.topics import (  # noqa: E402 - importorskip pattern
    TopicConfig,
    build_hierarchy,
    fit_topics,
)

N_CLUSTERS = 5
PER_CLUSTER = 100
DIM = 64
NOISE_SIGMA = 0.05

_VOCAB: list[list[str]] = [
    ["alpha", "beta", "gamma", "delta", "epsilon"],
    ["foxtrot", "golf", "hotel", "india", "juliet"],
    ["kilo", "lima", "mike", "november", "oscar"],
    ["papa", "quebec", "romeo", "sierra", "tango"],
    ["uniform", "victor", "whiskey", "xray", "yankee"],
]


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _planted_dataset(seed: int = 42) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (embeddings (n, DIM), documents, planted-labels)."""
    rng = np.random.default_rng(seed=seed)
    centroids = _l2_normalise(rng.standard_normal(size=(N_CLUSTERS, DIM)).astype(np.float32))

    n = N_CLUSTERS * PER_CLUSTER
    embeddings = np.empty((n, DIM), dtype=np.float32)
    documents: list[str] = []
    labels = np.empty(n, dtype=np.int32)

    for k in range(N_CLUSTERS):
        noise = rng.normal(0.0, NOISE_SIGMA, size=(PER_CLUSTER, DIM)).astype(np.float32)
        block = _l2_normalise(centroids[k][None, :] + noise).astype(np.float32)
        start = k * PER_CLUSTER
        embeddings[start : start + PER_CLUSTER] = block
        labels[start : start + PER_CLUSTER] = k

        vocab = _VOCAB[k]
        for _ in range(PER_CLUSTER):
            order = rng.permutation(len(vocab))
            doc = " ".join(vocab[i] for i in order)
            # Pad each doc so c-TF-IDF has enough tokens; repeats are fine.
            documents.append(f"{doc} {doc}")
    return embeddings, documents, labels


@pytest.fixture(scope="module")
def planted() -> tuple[np.ndarray, list[str], np.ndarray]:
    return _planted_dataset()


@pytest.mark.slow
def test_fit_topics_recovers_planted_clusters(planted) -> None:
    embeddings, documents, _ = planted
    cfg = TopicConfig(
        umap_n_neighbors=10,
        umap_n_components=5,
        hdbscan_min_cluster_size=30,
        nr_topics="auto",
        vectorizer_min_df=1,
        vectorizer_ngram_max=1,
        random_state=42,
    )
    model = fit_topics(embeddings, documents, cfg)
    topics_arr = np.asarray(model.topics_)
    unique_non_noise = {int(t) for t in topics_arr.tolist() if int(t) != -1}
    assert (
        len(unique_non_noise) >= 4
    ), f"expected ≥4 of 5 clusters recovered; got {len(unique_non_noise)}: {unique_non_noise}"


@pytest.mark.slow
def test_build_hierarchy_returns_expected_columns(planted) -> None:
    embeddings, documents, _ = planted
    cfg = TopicConfig(
        umap_n_neighbors=10,
        umap_n_components=5,
        hdbscan_min_cluster_size=30,
        nr_topics="auto",
        vectorizer_min_df=1,
        vectorizer_ngram_max=1,
        random_state=42,
    )
    model = fit_topics(embeddings, documents, cfg)
    hier = build_hierarchy(model, documents, target_mid_levels=2, target_top_levels=1)

    assert isinstance(hier, pd.DataFrame)
    expected_cols = {
        "topic_id",
        "top_words",
        "size",
        "mid_level_id",
        "top_level_id",
        "representative_docs",
    }
    assert expected_cols.issubset(set(hier.columns))
    assert len(hier) >= 4, "hierarchy should have at least 4 leaf rows"
    # All leaf topic_ids should be non-negative (no noise in hierarchy output).
    assert (hier["topic_id"] >= 0).all()
    # Each row carries a list of top words (possibly truncated to 10).
    assert hier["top_words"].apply(lambda v: isinstance(v, list)).all()
    assert hier["top_words"].apply(lambda v: len(v) <= 10).all()
    # Size column is positive ints.
    assert (hier["size"] > 0).all()
    # mid/top group ids are dense non-negative integers.
    assert (hier["mid_level_id"] >= 0).all()
    assert (hier["top_level_id"] >= 0).all()
