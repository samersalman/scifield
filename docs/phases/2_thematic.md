# Phase 2 — Thematic

## Phase objective

Replicate and refine the *Arthroscopy* topic-modeling pipeline at the v1
corpus scale: sentence-transformer embeddings, UMAP, HDBSCAN, c-TF-IDF, and
hierarchical topic merging to produce a stable topic landscape across the
entire 10-journal corpus, plus per-journal sub-topic structures for
finer-grained analyses. Embedding-quality bake-off between `all-mpnet-base-v2`,
`BAAI/bge-large-en-v1.5`, and `nomic-embed-text-v1` on a labeled subset
before committing. Coherence cross-validated with NPMI and C_v via OCTIS or
Palmetto. Deliverables: a canonical topic hierarchy (100–200 leaf topics
organized into ~20 mid-level and 5–7 top-level domains), per-paper topic
assignments with probabilities, and per-journal-per-year topic distributions.
This phase is a gate: if the topic structure is not clinically interpretable
on a 20-topic spot-check, the rest of the framework is built on sand.
