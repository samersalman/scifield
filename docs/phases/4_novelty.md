# Phase 4 — Novelty

## Phase objective

Compute two complementary novelty scores for every paper and examine their
joint distribution. **Semantic novelty** is the mean and minimum cosine
distance from a paper's embedding to embeddings of all prior papers in the
same field, computed via FAISS HNSW indices. **Structural novelty** is the
Funk-Owen-Smith CD_n index computed on the OpenAlex citation graph stored
in Kùzu (an embedded columnar graph database that ships as a pip
dependency, no server needed); the implementation is validated against
published values on the Park et al. 2023 replication corpus. Deliverables:
per-paper semantic and structural novelty scores, 2×2 archetype assignment
(frontier work, methodological repackaging, isolated novelty, consolidation),
temporal trends in archetype distribution, and the dual-novelty figure —
potentially the manuscript's headline figure. Gate: if the 2×2 reveals no
interesting pattern, the F2 finding is dropped and the paper pivots to F1
and F3 only.
