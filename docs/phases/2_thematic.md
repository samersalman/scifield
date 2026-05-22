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

---

## V1-S06 results (2026-05-21)

**Status:** pipeline executed end-to-end; gate G1 awaiting Samer's 20-topic spot-check.

- **Chosen config:** `umap_n_neighbors=15`, `hdbscan_min_cluster_size=80`, `nr_topics=auto`, `random_state=42` (sidecar `config_hash=5a1ed67d…`, `git_sha=eebe89e7…`).
- **Hierarchy counts:** leaf=**89**, mid=58, top=58. Mid and top collapsed to the same value because BERTopic's `hierarchical_topics` linkage produced only ~31 merges before bottoming out — itself diagnostic of under-segmentation.
- **Coherence:** NPMI=**0.224** (≥0.18 baseline ✓), C_v=0.756 (top-10 words/topic, noise topic -1 excluded).
- **Noise fraction:** **29.5%** (above the 20% threshold).
- **Runtime:** 689.6 s total wall (sweep 581.7 s + final fit 63.4 s + hierarchy 6.4 s) on local Mac CPU, 14 threads. No Brev launches.
- **Sweep table:** 6 configs; **none satisfied the joint constraints** (100 ≤ n_leaf ≤ 200, noise ≤ 20%). CLI fell back to global `npmi_top10` argmax and stamped `constraints_unmet=true` in every sidecar. Closest near-miss: `(n_neighbors=10, min_cluster_size=120)` at 22.8% noise but only 40 leaves. Full grid in `data/v1/topic_sweep.parquet`.
- **Deviations from plan:** `constraints_unmet=true` recorded in `data/v1/topics.parquet.run.json` / `topic_hierarchy.parquet.run.json` / `bertopic_v1.run.json`. Pipeline produced its best attempt rather than aborting, per the plan-S06 risk-handling rule ("surface, recommend, stop — do not silently push to V1-S07"). No re-tune attempted in this session.
- **Gate G1 decision (2026-05-21):** `RETUNE_CLUSTERING` — confirmed by Samer after spot-check (17 coherent / 2 cross-domain-valid / 1 incoherent; t3 is the incoherent 4th-largest cluster mixing pancreatic / pancreatitis / burn / intestinal / molecular). Embedding is healthy (NPMI 0.224, 17/20 clean labels); failure is in post-embedding density estimation + c-TF-IDF merging. Report: `docs/gates/G1_topic_interpretability.md`. Next session is V1-S06b: focused 2×2 mini-sweep `nr_topics ∈ {150, auto}` × `hdbscan_min_samples ∈ {10, None}` at fixed `n_neighbors=15, min_cluster_size=80`; widen only if it fails the noise + leaf-count gates; escalate to RETUNE_EMBEDDING (bge-large) only as a last resort. V1-S07 is blocked until V1-S06b clears the gate.
- **Carryovers handled:** V1-S05 duplicate PMIDs (13,070 in `papers`, 10,708 in `embeddings.parquet`) deduped via `papers_distinct` view + `load_deduped_embeddings()`. Integrity check at run time confirmed papers_total=134,978 / papers_distinct=121,908 / duplicate_pmids=13,070 (exactly matches V1-S05 baseline). See `src/scifield/thematic/dedup.py`.
- **Software pinned:** bertopic 0.17.4, umap-learn 0.5.12, hdbscan 0.8.43, gensim 4.4.0, scikit-learn 1.8.0, numpy 2.4.6 (resolved cleanly against numpy 2.x — the gensim<2 risk from plan-S06 did not materialize).

---

## V1-S06b results (2026-05-21)

**Status:** clustering retune executed end-to-end (phase 1 + phase 2); gate G1 retune-results section appended; awaiting Samer's new 20-topic spot-check and gate decision.

- **Chosen config:** `umap_n_neighbors=15`, `hdbscan_min_cluster_size=90`, `hdbscan_min_samples=10`, `nr_topics=150`, `random_state=42` (sidecar `config_hash=33ee895d…`, `git_sha=eebe89e7…`). Picked by the V1-S06 sweep harness via global `npmi_top10` argmax on `constraints_unmet=true` (no config satisfied noise<20%).
- **Hierarchy counts:** leaf=**149** (V1-S06: 89), mid=**96**, top=**96**. Leaf count cleared the 100–200 gate; mid/top **still collapsed to the same value** — identical to the V1-S06 pattern (58/58 → 96/96), confirming the `build_hierarchy` mid/top defect is independent of leaf segmentation. **Reported, not blocking** in V1-S06b per plan-S06b §"Decisions baked in" §3; deferred to plan-S06c.
- **Coherence:** NPMI=**0.238** (V1-S06: 0.224; ≥0.18 baseline ✓), C_v=**0.782** (V1-S06: 0.756). Both metrics improved over V1-S06.
- **Noise fraction:** **24.0%** (V1-S06: 29.5%). Improved by 5.5 points but **still above the 20% threshold** — the primary V1-S06b gate.
- **Runtime:** phase 2 wall 1022.6 s (sweep 911.1 s + final fit 64.5 s + hierarchy 7.4 s); plus phase 1 ~6.5 min before its parquet write crashed. **Total session wall: ~23.5 min** on local Mac CPU (14 threads). No Brev launches.
- **Sweep table:** **9 configs** (phase 2 Cartesian widen at `n_neighbors=15`, `nr_topics=150`, `min_cluster_size ∈ {30, 60, 90}` × `min_samples ∈ {5, 10, 20}`); **all 9** hit `n_leaf=149` (passes leaf gate); **none cleared noise<20%**. Chosen row 7 `(min_cluster=90, min_samples=10)` at NPMI=0.238/noise=0.240 — lowest noise across the grid, highest NPMI. CLI stamped `constraints_unmet=true` in every sidecar. Full grid in `data/v1/topic_sweep.parquet`.
- **Deviations from plan:**
  1. **Phase 1 parquet-write defect.** The 4-config 2×2 mini-sweep prescribed in plan-S06b §2 mixed `nr_topics` types (`int(150)` and `str('auto')`) across grid rows. The V1-S06 sweep harness (`src/scifield/thematic/topics.py:sweep()`) builds the sweep dataframe with the per-config `dict` as a single object-dtype column; pyarrow cannot unify mixed int/str values in that dict's `nr_topics` key, so `sweep_df.to_parquet` raised `ArrowInvalid: Could not convert 'auto' with type str: tried to convert to int64`. All 4 phase-1 configs evaluated successfully before the crash (results recovered from stdout: `data/v1/topics_run_phase1.log.preserved`) and **no V1-S06 artifacts were overwritten by phase 1** — the crash preceded chosen-config selection/fit. Phase 1 decisively established `nr_topics=150 > auto` (the two `150` configs hit n_leaf=149 vs. `auto`'s 89-and-error); plan-S06b §"Risks" treats this as a code defect to surface (handled here and in the G1 append) without an in-session code fix per plan §"STAY IN SCOPE". Fix likely a 1-line JSON-serialize of the `config` column; routed to plan-S06c backlog. Phase 2's grid pins integer `nr_topics`, so the defect does not recur and the V1-S06b artifacts come entirely from phase 2.
  2. **`constraints_unmet=true` after phase 2.** Phase 2 ran cleanly but no config met noise<20% (best 24.0% at row 7). Per plan-S06b §6: end the session with a `RETUNE_EMBEDDING` recommendation in the G1 append; do **not** silently push to V1-S07. Notebook re-execute proceeded (figures + spot-check CSV regenerated from the V1-S06b chosen fit). V1-S07 remains blocked behind G1.
- **Carryovers handled:** V1-S05 dedup invariants re-verified at run time (papers_total=134,978 / papers_distinct=121,908 / duplicate_pmids=13,070 — exact match to V1-S05 baseline). Embeddings hash `595e7b2f5fafc58f7f251767d82164a3650cbaaa95b374a6b3fd896e8f7622c1` **matches** the V1-S06 sidecar — embeddings were not re-run, satisfying plan-S06b §"Reproducibility" + acceptance test #9. `papers_duckdb` byte-hash drifted (`a02dc34… → ed78e560…`); content is unchanged — DuckDB rewrites file metadata when a view is recreated read-write, so the file hash differs while the `papers` / `papers_distinct` rowcounts are identical.
- **Gate G1 decision (V1-S06b, 2026-05-21):** **`PROCEED to V1-S07`** — Samer's override of plan-S06b §6's procedural `RETUNE_EMBEDDING` mandate, after the V1-S06b 20-topic spot-check passed at **19 / 1 / 0** (coherent / cross-domain-valid / incoherent) — a clean lift over the V1-S06 17 / 2 / 1. Quantitative gates: NPMI 0.238 and leaf count 149 cleared; noise floored at 24.0% (4 pp above the 20% gate) — the saturation pattern across all 9 phase-2 configs is consistent with an embedding-structural ~24% sparse-region floor that no HDBSCAN parameterization can erode. RETUNE_EMBEDDING's expected gain is bounded by the V1-S05 bake-off Δ (+0.011 kNN, mpnet → bge-large), small relative to the cost of re-running V1-S05 + V1-S06. Full rationale + downstream caveats (noise-topic handling, topic-4 translational cluster, deferred mid/top hierarchy) in `docs/gates/G1_topic_interpretability.md` § Decision. **V1-S07 unblocked.**
