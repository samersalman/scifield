"""Tests for :func:`scifield.thematic.topics.fit_subsample_assign_all` (V2-S08).

The V2 scaling fix fits UMAP+HDBSCAN on a deterministic subsample, then folds
the held-out remainder in via ``approximate_predict`` (through BERTopic's
``transform``). These tests plant ~300 well-separated 768-d Gaussian blobs,
fit on a 150-row subsample, and assert:

  (a) full-length assignments are returned (one per input row),
  (b) the path runs end-to-end without error and overwrites ``model.topics_``
      with the full-length, original-order assignments, and
  (c) determinism — the same seed reproduces identical assignments.

Marked ``slow``: even at 150 fit-rows / 768 dims this runs a real
UMAP + HDBSCAN + c-TF-IDF fit plus a transform over all 300 rows.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bertopic")
pytest.importorskip("umap")
pytest.importorskip("hdbscan")

from scifield.thematic.topics import (  # noqa: E402 - importorskip pattern
    TopicConfig,
    fit_subsample_assign_all,
)

N_BLOBS = 3
PER_BLOB = 100
DIM = 768
NOISE_SIGMA = 0.02
SEED = 20260609

_VOCAB: list[list[str]] = [
    ["alpha", "beta", "gamma", "delta", "epsilon"],
    ["foxtrot", "golf", "hotel", "india", "juliet"],
    ["kilo", "lima", "mike", "november", "oscar"],
]


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _planted_dataset(seed: int = 7) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings (n, DIM) fp32, documents) for N_BLOBS Gaussian blobs."""
    rng = np.random.default_rng(seed=seed)
    centroids = _l2_normalise(rng.standard_normal(size=(N_BLOBS, DIM)).astype(np.float32))

    n = N_BLOBS * PER_BLOB
    embeddings = np.empty((n, DIM), dtype=np.float32)
    documents: list[str] = []

    for k in range(N_BLOBS):
        noise = rng.normal(0.0, NOISE_SIGMA, size=(PER_BLOB, DIM)).astype(np.float32)
        block = _l2_normalise(centroids[k][None, :] + noise).astype(np.float32)
        embeddings[k * PER_BLOB : (k + 1) * PER_BLOB] = block

        vocab = _VOCAB[k]
        for _ in range(PER_BLOB):
            order = rng.permutation(len(vocab))
            doc = " ".join(vocab[i] for i in order)
            documents.append(f"{doc} {doc}")  # pad so c-TF-IDF has tokens
    return embeddings, documents


def _cfg() -> TopicConfig:
    return TopicConfig(
        umap_n_neighbors=10,
        umap_n_components=5,
        hdbscan_min_cluster_size=20,
        nr_topics=None,  # no reduction: keep tiny-corpus behavior simple/stable
        vectorizer_min_df=1,
        vectorizer_ngram_max=1,
        random_state=42,
    )


@pytest.fixture(scope="module")
def planted() -> tuple[np.ndarray, list[str]]:
    return _planted_dataset()


@pytest.mark.slow
def test_subsample_assign_returns_full_length(planted) -> None:
    embeddings, documents = planted
    model, topics, probabilities = fit_subsample_assign_all(
        embeddings, documents, _cfg(), subsample_n=150, seed=SEED
    )

    # (a) full-length assignments, one per input row.
    n = embeddings.shape[0]
    assert len(topics) == n, f"expected {n} assignments; got {len(topics)}"
    assert all(isinstance(t, int) for t in topics)

    # (b) model.topics_ overwritten with the full-length assignments and
    #     probabilities align to all rows; recovery is sane (a real cluster
    #     plus possible -1 noise, not everything collapsed to noise).
    assert len(np.asarray(model.topics_)) == n
    assert np.asarray(probabilities).shape[0] == n
    non_noise = {t for t in topics if t != -1}
    assert len(non_noise) >= 1, f"expected ≥1 real topic; got {sorted(set(topics))}"


@pytest.mark.slow
def test_subsample_assign_is_deterministic(planted) -> None:
    embeddings, documents = planted
    _, topics_a, _ = fit_subsample_assign_all(
        embeddings, documents, _cfg(), subsample_n=150, seed=SEED
    )
    _, topics_b, _ = fit_subsample_assign_all(
        embeddings, documents, _cfg(), subsample_n=150, seed=SEED
    )
    # (c) same seed → identical assignments.
    assert topics_a == topics_b


@pytest.mark.slow
def test_subsample_default_seed_is_random_state(planted) -> None:
    """Omitting ``seed`` falls back to ``cfg.random_state`` deterministically."""
    embeddings, documents = planted
    cfg = _cfg()  # random_state=42
    _, topics_default, _ = fit_subsample_assign_all(embeddings, documents, cfg, subsample_n=150)
    _, topics_explicit, _ = fit_subsample_assign_all(
        embeddings, documents, cfg, subsample_n=150, seed=cfg.random_state
    )
    assert topics_default == topics_explicit


def test_subsample_n_validation(planted) -> None:
    """Non-positive ``subsample_n`` is rejected before any heavy fit."""
    embeddings, documents = planted
    with pytest.raises(ValueError, match="subsample_n must be positive"):
        fit_subsample_assign_all(embeddings, documents, _cfg(), subsample_n=0, seed=SEED)
