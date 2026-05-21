"""Slow real-model tests for :mod:`scifield.thematic.embed`.

These tests download a small SentenceTransformer model from Hugging Face and
run a real CPU encode pass. They are gated by the ``SCIFIELD_RUN_SLOW_TESTS``
environment variable and skipped by default in CI.

Run with::

    SCIFIELD_RUN_SLOW_TESTS=1 uv run pytest tests/test_thematic_embed.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("SCIFIELD_RUN_SLOW_TESTS") != "1":
    pytest.skip(
        "Set SCIFIELD_RUN_SLOW_TESTS=1 to run real-model embedding tests.",
        allow_module_level=True,
    )

from scifield.thematic.embed import MpnetEmbedder  # noqa: E402

# Tiny stand-in model — keeps the slow test feasible on a laptop CPU.
_SMALL_HF_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
_SMALL_DIM = 384


def _small_embedder() -> MpnetEmbedder:
    emb = MpnetEmbedder()
    # Override metadata to point at a much smaller model. We keep using the
    # MpnetEmbedder shell so the prefix rules (none) and load path are
    # exercised end-to-end without inflating download size.
    emb.hf_id = _SMALL_HF_ID
    emb.dim = _SMALL_DIM
    emb.max_seq_length = 128
    return emb


@pytest.mark.slow
def test_real_model_encode_shape_dtype_and_norm() -> None:
    emb = _small_embedder()
    texts = [
        "Mitochondrial dysfunction in neurodegenerative disease.",
        "A randomized trial of vitamin D supplementation in adults.",
        "Graph neural networks for molecular property prediction.",
    ]
    vecs = emb.encode(texts, batch_size=4)
    assert vecs.shape == (3, _SMALL_DIM)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.slow
def test_real_model_encode_is_deterministic() -> None:
    emb1 = _small_embedder()
    emb2 = _small_embedder()
    texts = [
        "Deterministic embedding test sentence number one.",
        "Another canonical input for reproducibility.",
    ]
    v1 = emb1.encode(texts, batch_size=2)
    v2 = emb2.encode(texts, batch_size=2)
    assert v1.shape == v2.shape
    assert np.allclose(v1, v2, atol=1e-5)
