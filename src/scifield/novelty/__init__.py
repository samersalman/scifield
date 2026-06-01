"""Phase 4 — Semantic + structural novelty.

Semantic novelty against same-field priors (FAISS-backed), the Funk-Owen
consolidation/disruption (CD) index over the citation graph, a Kùzu
property-graph loader, and the OpenAlex `cited_by` harvest that supplies the
forward-citation edges the CD index needs.
"""

from scifield.novelty.cd_index import (
    CitationGraph,
    cd_index,
    classify_citers,
    compute_corpus_cd,
)
from scifield.novelty.cited_by import (
    CitedByConfig,
    harvest_cited_by,
    run_cited_by,
)
from scifield.novelty.kuzu_loader import build_kuzu_graph
from scifield.novelty.semantic import compute_semantic_novelty

__all__ = [
    "compute_semantic_novelty",
    "CitationGraph",
    "cd_index",
    "classify_citers",
    "compute_corpus_cd",
    "build_kuzu_graph",
    "CitedByConfig",
    "harvest_cited_by",
    "run_cited_by",
]
