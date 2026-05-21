"""Phase 2 — Thematic backbone: pluggable embedders + FAISS HNSW index."""

from scifield.thematic.embed import (
    BgeLargeEmbedder,
    Embedder,
    MpnetEmbedder,
    NomicEmbedder,
    make_embedder,
)
from scifield.thematic.faiss_index import (
    build_faiss_hnsw,
    read_index,
    read_pmid_map,
    write_index,
    write_pmid_map,
)

__all__ = [
    "Embedder",
    "MpnetEmbedder",
    "BgeLargeEmbedder",
    "NomicEmbedder",
    "make_embedder",
    "build_faiss_hnsw",
    "read_index",
    "read_pmid_map",
    "write_index",
    "write_pmid_map",
]
