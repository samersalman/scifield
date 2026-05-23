"""Phase 2 — Thematic backbone.

Pluggable embedders, FAISS HNSW index, V1-S05 carryover dedup, and the
V1-S06 BERTopic pipeline (UMAP → HDBSCAN → c-TF-IDF → hierarchical merge)
with coherence scoring.
"""

from scifield.thematic.coherence import (
    compute_coherence,
    tokenise_for_coherence,
)
from scifield.thematic.dedup import (
    ensure_papers_distinct_view,
    integrity_check_v1_carryover,
    load_deduped_embeddings,
)
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
from scifield.thematic.topics import (
    SweepRow,
    TopicConfig,
    build_hierarchy,
    fit_topics,
    make_bertopic_model,
    sweep,
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
    "ensure_papers_distinct_view",
    "integrity_check_v1_carryover",
    "load_deduped_embeddings",
    "compute_coherence",
    "tokenise_for_coherence",
    "TopicConfig",
    "SweepRow",
    "make_bertopic_model",
    "fit_topics",
    "build_hierarchy",
    "sweep",
]
