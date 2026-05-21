"""Offline tests for :mod:`scifield.thematic.embed` (no model downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from scifield.thematic.embed import (
    BgeLargeEmbedder,
    Embedder,
    MpnetEmbedder,
    NomicEmbedder,
    make_embedder,
)

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias, cls",
    [
        ("mpnet", MpnetEmbedder),
        ("all-mpnet-base-v2", MpnetEmbedder),
        ("bge", BgeLargeEmbedder),
        ("bge-large-en-v1.5", BgeLargeEmbedder),
        ("nomic", NomicEmbedder),
        ("nomic-embed-text-v1", NomicEmbedder),
    ],
)
def test_make_embedder_returns_expected_class(alias: str, cls: type) -> None:
    emb = make_embedder(alias)
    assert isinstance(emb, cls)


def test_make_embedder_unknown_raises_value_error_listing_allowed() -> None:
    with pytest.raises(ValueError) as excinfo:
        make_embedder("garbage")
    msg = str(excinfo.value)
    for allowed in [
        "mpnet",
        "all-mpnet-base-v2",
        "bge",
        "bge-large-en-v1.5",
        "nomic",
        "nomic-embed-text-v1",
    ]:
        assert allowed in msg


# ---------------------------------------------------------------------------
# Construction is cheap and offline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls, name, hf_id, dim",
    [
        (MpnetEmbedder, "all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2", 768),
        (BgeLargeEmbedder, "bge-large-en-v1.5", "BAAI/bge-large-en-v1.5", 1024),
        (NomicEmbedder, "nomic-embed-text-v1", "nomic-ai/nomic-embed-text-v1", 768),
    ],
)
def test_construction_sets_metadata_without_model_load(
    cls: type, name: str, hf_id: str, dim: int
) -> None:
    emb = cls()
    assert emb.name == name
    assert emb.hf_id == hf_id
    assert emb.dim == dim
    assert isinstance(emb.max_seq_length, int) and emb.max_seq_length > 0
    # Lazy load: model must not have been instantiated yet.
    assert emb._model is None
    # Each embedder satisfies the runtime-checkable Embedder Protocol.
    assert isinstance(emb, Embedder)


# ---------------------------------------------------------------------------
# Prefix logic via fake model
# ---------------------------------------------------------------------------


class _FakeModel:
    """Captures inputs to .encode() and returns L2-normalised fake vectors."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.captured_texts: list[str] | None = None
        self.captured_kwargs: dict[str, object] | None = None

    def encode(self, texts, **kwargs):  # noqa: ANN001 - mimics SentenceTransformer
        self.captured_texts = list(texts)
        self.captured_kwargs = dict(kwargs)
        n = len(self.captured_texts)
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((n, self.dim)).astype(np.float32)
        # L2-normalise rows to mimic normalize_embeddings=True.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def _encode_with_fake(emb, texts):
    """Inject a fake model and call .encode(), returning captured inputs."""
    fake = _FakeModel(emb.dim)
    emb._model = fake  # bypass lazy load
    out = emb.encode(texts)
    return out, fake


def test_mpnet_passes_texts_unchanged() -> None:
    emb = MpnetEmbedder()
    texts = ["hello world", "second doc"]
    out, fake = _encode_with_fake(emb, texts)
    assert fake.captured_texts == texts
    assert out.shape == (2, emb.dim)
    assert out.dtype == np.float32
    # Normalize_embeddings was forwarded.
    assert fake.captured_kwargs is not None
    assert fake.captured_kwargs.get("normalize_embeddings") is True
    assert fake.captured_kwargs.get("convert_to_numpy") is True
    assert fake.captured_kwargs.get("show_progress_bar") is False


def test_bge_passes_texts_unchanged_for_document_encoding() -> None:
    emb = BgeLargeEmbedder()
    texts = ["abstract one", "abstract two", "abstract three"]
    out, fake = _encode_with_fake(emb, texts)
    # BGE doc-side: no prefix applied.
    assert fake.captured_texts == texts
    assert out.shape == (3, emb.dim)


def test_nomic_applies_search_document_prefix_to_every_text() -> None:
    emb = NomicEmbedder()
    texts = ["alpha", "beta"]
    out, fake = _encode_with_fake(emb, texts)
    assert fake.captured_texts == [
        "search_document: alpha",
        "search_document: beta",
    ]
    assert out.shape == (2, emb.dim)


def test_encode_forwards_batch_size_kwarg() -> None:
    emb = MpnetEmbedder()
    _, fake = _encode_with_fake(emb, ["x", "y", "z"])
    assert fake.captured_kwargs is not None
    assert fake.captured_kwargs.get("batch_size") == 64

    emb2 = MpnetEmbedder()
    fake2 = _FakeModel(emb2.dim)
    emb2._model = fake2
    emb2.encode(["x"], batch_size=8)
    assert fake2.captured_kwargs is not None
    assert fake2.captured_kwargs.get("batch_size") == 8
