"""Pluggable sentence embedders for SciField's thematic backbone.

This module defines the :class:`Embedder` Protocol and three concrete
implementations wrapping :mod:`sentence_transformers` models:

* :class:`MpnetEmbedder`     — ``sentence-transformers/all-mpnet-base-v2`` (768d)
* :class:`BgeLargeEmbedder`  — ``BAAI/bge-large-en-v1.5`` (1024d)
* :class:`NomicEmbedder`     — ``nomic-ai/nomic-embed-text-v1`` (768d)

Each embedder is responsible for its own model-specific input prefix rules and
produces L2-normalised ``float32`` vectors, so downstream cosine similarity is
equivalent to plain inner product.

Models are loaded lazily on the first :meth:`Embedder.encode` call, so cheap
construction (e.g. in unit tests, factories, or CLI ``--help``) does not
require network or disk access.

The :func:`make_embedder` factory maps short slug names to the concrete class
so configs (Hydra/YAML) can declaratively select an embedder.
"""

from __future__ import annotations

import contextlib
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Embedder",
    "MpnetEmbedder",
    "BgeLargeEmbedder",
    "NomicEmbedder",
    "make_embedder",
]


# Short slug -> canonical name. The canonical name is also used as ``.name``
# on the embedder instance.
_ALIASES: dict[str, str] = {
    "mpnet": "all-mpnet-base-v2",
    "all-mpnet-base-v2": "all-mpnet-base-v2",
    "bge": "bge-large-en-v1.5",
    "bge-large-en-v1.5": "bge-large-en-v1.5",
    "nomic": "nomic-embed-text-v1",
    "nomic-embed-text-v1": "nomic-embed-text-v1",
}


@runtime_checkable
class Embedder(Protocol):
    """Protocol for sentence embedders used by the thematic backbone.

    Implementations must return L2-normalised ``float32`` vectors so cosine
    similarity reduces to inner product downstream.
    """

    name: str
    hf_id: str
    dim: int
    max_seq_length: int

    def encode(
        self, texts: list[str], batch_size: int = 64
    ) -> np.ndarray:  # pragma: no cover - Protocol stub
        ...


class _BaseEmbedder:
    """Shared lazy-loading + encode plumbing for concrete embedders."""

    # Subclasses set these as class attributes; they are re-assigned per instance
    # so that test code can monkeypatch (e.g. swap in a smaller model).
    name: str = ""
    hf_id: str = ""
    dim: int = 0
    max_seq_length: int = 0
    # Model-specific prefix applied to every input string before encoding.
    _doc_prefix: str = ""
    # Extra kwargs forwarded to SentenceTransformer (e.g. trust_remote_code).
    _st_kwargs: dict[str, object] = {}

    def __init__(
        self,
        hf_id: str | None = None,
        revision: str = "main",
        device: str | None = None,
    ) -> None:
        if hf_id is not None:
            self.hf_id = hf_id
        self.revision = revision
        self.device = device
        # Lazily loaded on first .encode() call.
        self._model: object | None = None

    def _load_model(self) -> object:
        """Import and instantiate the underlying SentenceTransformer."""
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, object] = dict(self._st_kwargs)
        if self.revision and self.revision != "main":
            kwargs["revision"] = self.revision
        if self.device is not None:
            kwargs["device"] = self.device
        model = SentenceTransformer(self.hf_id, **kwargs)
        # Pin the truncation length for deterministic behavior across versions.
        with contextlib.suppress(Exception):
            model.max_seq_length = self.max_seq_length
        return model

    def _ensure_model(self) -> object:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _apply_prefix(self, texts: list[str]) -> list[str]:
        if not self._doc_prefix:
            return list(texts)
        return [f"{self._doc_prefix}{t}" for t in texts]

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode ``texts`` to L2-normalised ``float32`` vectors."""
        model = self._ensure_model()
        prefixed = self._apply_prefix(texts)
        vectors = model.encode(  # type: ignore[attr-defined]
            prefixed,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr


class MpnetEmbedder(_BaseEmbedder):
    """``sentence-transformers/all-mpnet-base-v2`` — 768d, no prefix."""

    name = "all-mpnet-base-v2"
    hf_id = "sentence-transformers/all-mpnet-base-v2"
    dim = 768
    max_seq_length = 384
    _doc_prefix = ""
    _st_kwargs: dict[str, object] = {}


class BgeLargeEmbedder(_BaseEmbedder):
    """``BAAI/bge-large-en-v1.5`` — 1024d.

    BGE defines a *query-only* prefix
    (``"Represent this sentence for searching relevant passages: "``).
    For document/abstract encoding we use **no prefix**, matching BAAI's
    own usage guidance.
    """

    name = "bge-large-en-v1.5"
    hf_id = "BAAI/bge-large-en-v1.5"
    dim = 1024
    max_seq_length = 512
    _doc_prefix = ""
    _st_kwargs: dict[str, object] = {}


class NomicEmbedder(_BaseEmbedder):
    """``nomic-ai/nomic-embed-text-v1`` — 768d, ``search_document: `` prefix.

    Nomic requires task-specific prefixes; for indexing a corpus of documents
    every input is prepended with ``"search_document: "``. Loading the model
    requires ``trust_remote_code=True``.
    """

    name = "nomic-embed-text-v1"
    hf_id = "nomic-ai/nomic-embed-text-v1"
    dim = 768
    max_seq_length = 8192
    _doc_prefix = "search_document: "
    _st_kwargs: dict[str, object] = {"trust_remote_code": True}


def make_embedder(
    name: str,
    *,
    revision: str = "main",
    device: str | None = None,
) -> Embedder:
    """Construct an :class:`Embedder` by short slug.

    Parameters
    ----------
    name:
        One of ``"mpnet" | "all-mpnet-base-v2" | "bge" | "bge-large-en-v1.5" |
        "nomic" | "nomic-embed-text-v1"``.
    revision:
        Hugging Face model revision (commit SHA or branch); forwarded to
        :class:`sentence_transformers.SentenceTransformer`.
    device:
        Optional torch device override (e.g. ``"cuda"``, ``"mps"``, ``"cpu"``).
    """
    canonical = _ALIASES.get(name)
    if canonical is None:
        allowed = sorted(set(_ALIASES.keys()))
        raise ValueError(f"Unknown embedder name {name!r}. Allowed names: {allowed}")
    if canonical == "all-mpnet-base-v2":
        return MpnetEmbedder(revision=revision, device=device)
    if canonical == "bge-large-en-v1.5":
        return BgeLargeEmbedder(revision=revision, device=device)
    if canonical == "nomic-embed-text-v1":
        return NomicEmbedder(revision=revision, device=device)
    # Unreachable: every value in _ALIASES is handled above.
    raise ValueError(f"Unhandled canonical embedder name: {canonical}")
