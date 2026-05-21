# V1-S05 — Embedding bake-off + full-corpus embeddings + FAISS HNSW index

**Session:** V1-S05 (Phase 2 — Thematic backbone) | **Plan ref:** `plans/Session-Objectives-MAP.md` §V1-S05 | **Effort:** ~1.5 days | **Depends on:** V1-S04 (complete but uncommitted; modules importable) | **Brev:** L40S 48 GB, 1–2 GPU-hr, ~$3–6

---

## Context

V1-S04 left the corpus enriched: 134,978 papers in `data/v1/papers.duckdb` (74 % with abstracts), eight enrichment Parquet tables mounted as DuckDB views, all sidecar JSONs present. The thematic subpackage (`src/scifield/thematic/`) is still a stub.

V1-S05 builds the **thematic backbone** that every later phase rides on:

- Choose an embedding model via a small, principled bake-off (not opinion).
- Embed the abstract-bearing subset of the corpus (~100 k papers) on a Brev L40S.
- Persist a FAISS HNSW index so V1-S06 (topic modeling), V1-S10 (semantic novelty), and V1-S15 (integration) can do fast kNN over the corpus without re-embedding.

This plan stays **strictly within V1-S05's "In scope"** list. Anything resembling UMAP, HDBSCAN, c-TF-IDF, coherence scoring, novelty, or topic-landscape narrative is V1-S06+ — not here.

---

## Scope (verbatim from Session-Objectives-MAP V1-S05)

### In scope
- `src/scifield/thematic/embed.py` — pluggable embedder (mpnet / bge-large / nomic).
- `notebooks/03_embedding_bakeoff.ipynb` — 500-abstract bake-off.
- `conf/thematic/embed.yaml` — chosen model + parameters.
- Full-corpus embedding on Brev L40S → `data/v1/embeddings.parquet` (PMID → vector).
- `src/scifield/thematic/faiss_index.py` — HNSW build + persist → `data/v1/faiss.index`.
- Brev launch + stop wrapping (this session: `scripts/brev_embed.sh`).

### Out of scope (defer)
- UMAP / HDBSCAN / c-TF-IDF topic modeling → V1-S06.
- Hierarchical merging → V1-S06.
- Coherence (NPMI / C_v) → V1-S06.
- `docs/phases/thematic.md` write-up → V1-S06.
- Semantic novelty computations → V1-S10.
- A proper `scifield brev launch/stop` CLI subcommand → first session that needs it (likely V1-S13).

---

## Locked decisions (from clarifying Qs)

1. **Labeled subset = MeSH-derived pseudo-labels.** Stratified random sample of 500 abstract-bearing papers across the top-K most frequent MeSH descriptors (K chosen so each cell has ≥20 papers; expect K ≈ 20–25, ~20 papers per label).
2. **Brev = full-corpus only.** Bake-off runs locally on Mac (CPU; ~10–20 min total for 500 × 3 models). Brev L40S spins up only for the full ~100 k-paper embedding.
3. **Embed scope = abstract-bearing only.** Filter to `abstract IS NOT NULL AND length(abstract) > 50`. Downstream phases that need every PMID will filter at their own boundary.
4. **Brev wrapping = script only.** New `scripts/brev_embed.sh` modeled on existing `scripts/brev_smoke.sh`. No `scifield brev` CLI subcommand this session.

---

## Design decisions

### D1. Embedder API (`src/scifield/thematic/embed.py`)

`Embedder` Protocol with three concrete classes:

```
class MpnetEmbedder    # sentence-transformers/all-mpnet-base-v2  (768d, no prefix)
class BgeLargeEmbedder # BAAI/bge-large-en-v1.5                   (1024d, query-prefix only)
class NomicEmbedder    # nomic-ai/nomic-embed-text-v1             (768d, prefix "search_document: ")
```

Each class encapsulates its prefix rules. All expose `encode(texts: list[str], batch_size: int) -> np.ndarray`. Output is L2-normalised at the embedder boundary so downstream cosine = inner-product. Factory: `make_embedder(name: str) -> Embedder` for Hydra-driven selection.

### D2. Input text construction

`text = (title or "") + ". " + (abstract or "")` then truncate at the model's `max_seq_length` via the tokenizer (no manual char limits). Empty abstracts excluded upstream (D3).

### D3. Input source

Query DuckDB view `papers`:

```sql
SELECT pmid, title, abstract
FROM papers
WHERE abstract IS NOT NULL AND length(abstract) > 50
ORDER BY pmid
```

Expected row count ≈ 100 k (74 % of 134 978). Exact count recorded in sidecar.

### D4. Output formats

- **`data/v1/embeddings.parquet`**: schema `pmid INT64`, `embedding LIST<FLOAT16>[D]`, `model_name STRING`. fp16 to halve disk (~210 MB for 100 k × 768 fp16). Reader upcasts to fp32 at use.
- **Sidecar JSON** via `record_run`: input hashes (papers.duckdb sha), config hash, plus extension fields in `config`: `{model_name, model_revision, sentence_transformers_version, torch_version, batch_size, max_seq_length, n_papers, total_runtime_s, gpu_model}`.

### D5. FAISS HNSW index (`src/scifield/thematic/faiss_index.py`)

- `faiss.IndexHNSWFlat(d, M=32)`; set `efConstruction=200`, `efSearch=64`.
- Metric: inner product on L2-normalised vectors (= cosine).
- Sequential row IDs; PMID mapping in a sidecar Parquet:
  - **`data/v1/faiss.index`** — binary FAISS index.
  - **`data/v1/faiss_pmid_map.parquet`** — `row_id INT32`, `pmid INT64`.
  - **`data/v1/faiss.index.run.json`** — sidecar.

### D6. Bake-off metrics (notebook)

For each of the 3 models, on the 500-paper MeSH-labelled sample:

| Metric | Definition |
|---|---|
| **Intra/inter cosine** | mean( cosine within label ) − mean( cosine across labels ); higher = better. |
| **kNN@10 precision** | per paper, fraction of 10 nearest neighbours (excluding self) sharing the paper's MeSH label; averaged across the 500. |
| **Runtime / 1 k** | wall-clock seconds to encode 1 000 abstracts on CPU (Mac). |

**Decision rule (encoded in notebook):**
1. Highest kNN@10 precision wins.
2. Tiebreaker (Δ ≤ 0.02): higher intra/inter separation.
3. **Stop-condition guard (from brief):** if winner does not beat `all-mpnet-base-v2` by ≥ 0.03 on kNN@10 precision, default to mpnet — document this in the recommendation cell.

### D7. CLI additions (`src/scifield/cli.py`)

Two new Typer subcommands, following the existing `harvest` / `enrich` pattern (Hydra config via `_load_config`):

- `scifield embed --config conf/thematic/embed.yaml [--limit N]` — reads abstracts from DuckDB, encodes, writes `embeddings.parquet` + sidecar.
- `scifield faiss-build --embeddings data/v1/embeddings.parquet --out data/v1/faiss.index` — builds HNSW + PMID map + sidecar.

Both reuse `scifield.repro.record_run` for sidecars.

### D8. Brev script (`scripts/brev_embed.sh`)

Bash script modelled on `scripts/brev_smoke.sh`:

```
1. brev credit balance (best-effort)
2. brev create <l40s-spec> --name scifield-embed-V1-S05
3. brev exec scifield-embed-V1-S05 -- <clone + uv sync + scifield embed>
4. brev cp scifield-embed-V1-S05:/repo/data/v1/embeddings.parquet ./data/v1/
5. brev cp scifield-embed-V1-S05:/repo/data/v1/embeddings.parquet.run.json ./data/v1/
6. brev stop scifield-embed-V1-S05  (always, in trap EXIT)
7. brev credit balance again; diff logged to docs/operations/brev.md
```

Same gating as smoke script: skip cleanly if `brev` CLI absent, document a manual fallback.

### D9. Config (`conf/thematic/embed.yaml`)

Initial committable version (pre-bake-off) defaults to `mpnet`; bake-off updates to winner:

```yaml
# Selected by V1-S05 bake-off (notebooks/03_embedding_bakeoff.ipynb).
model:
  name: all-mpnet-base-v2          # one of: all-mpnet-base-v2 | bge-large-en-v1.5 | nomic-embed-text-v1
  hf_id: sentence-transformers/all-mpnet-base-v2
  revision: main                   # pin to commit SHA after bake-off
  max_seq_length: 384
  normalize: true
batch_size: 64
input:
  duckdb_path: data/v1/papers.duckdb
  table: papers
  filter: "abstract IS NOT NULL AND length(abstract) > 50"
output:
  parquet_path: data/v1/embeddings.parquet
  dtype: float16
faiss:
  index_path: data/v1/faiss.index
  pmid_map_path: data/v1/faiss_pmid_map.parquet
  M: 32
  efConstruction: 200
  efSearch: 64
```

Wire into `conf/config.yaml` via `defaults: [- thematic: embed]`.

### D10. Dependencies (`pyproject.toml`)

Add to `[project] dependencies`:

- `sentence-transformers>=3.0`
- `faiss-cpu>=1.8` (HNSW is CPU-friendly; on Brev we still build the index on CPU after GPU encoding)
- `torch>=2.2` (transitive but pin explicitly so `uv lock` is reproducible across local Mac CPU build and Brev GPU build)
- `numpy>=1.26` (transitive but explicit)

Re-`uv lock && uv sync`.

### D11. Tests

- **`tests/test_thematic_embed.py`** — unit test on `MpnetEmbedder` using a stub model (e.g. `sentence-transformers/paraphrase-MiniLM-L3-v2`, 384d, downloaded once, cached). Asserts: output shape `(n, d)`, dtype, L2-norm ≈ 1, deterministic on rerun. Marked `@pytest.mark.slow` so CI can skip if needed; gated by `SCIFIELD_RUN_SLOW_TESTS=1`.
- **`tests/test_thematic_faiss.py`** — build HNSW on synthetic `100 × 16` normalised vectors with planted clusters; persist; reload via `faiss.read_index`; assert top-1 NN recovers the planted cluster ≥ 90 %. PMID map roundtrip checked.
- **`tests/test_cli_embed.py`** — `scifield embed --help` exit 0; `scifield faiss-build --help` exit 0. Standard CLI smoke.
- **`tests/test_thematic_embed_offline.py`** — pure-Python test of the prefix logic and Embedder factory (no model load). Always runs in CI.

### D12. Notebook outline (`notebooks/03_embedding_bakeoff.ipynb`)

1. **Setup** — connect to `papers.duckdb`, list MeSH coverage.
2. **Sample construction** — pick top-K MeSH descriptors with ≥ 20 abstract-bearing papers each, stratified-sample 500 papers, persist to `data/v1/embedding_bakeoff_sample.parquet`. Sidecar via `record_run`.
3. **Run 3 embedders on the 500** — record wall-clock per 1 k.
4. **Metrics table** — intra/inter separation, kNN@10 precision, runtime per 1 k.
5. **Recommendation cell** — explicit winner, quantitative basis, stop-condition guard (D6).
6. **FAISS spot-check** (run after the Brev embedding completes; cell prepared but not executed in the bake-off pass) — load `data/v1/faiss.index`, query 10 hand-chosen PMIDs, print top-5 NN PMID + title + journal for review.

---

## Files

### Create
- `src/scifield/thematic/embed.py`
- `src/scifield/thematic/faiss_index.py`
- `conf/thematic/embed.yaml`
- `notebooks/03_embedding_bakeoff.ipynb`
- `scripts/brev_embed.sh`
- `tests/test_thematic_embed.py`
- `tests/test_thematic_embed_offline.py`
- `tests/test_thematic_faiss.py`
- `tests/test_cli_embed.py`

### Modify
- `pyproject.toml` — add `sentence-transformers`, `faiss-cpu`, `torch`, `numpy`.
- `uv.lock` — regenerate.
- `conf/config.yaml` — add `thematic: embed` to `defaults`.
- `src/scifield/cli.py` — add `embed` and `faiss-build` subcommands.
- `src/scifield/thematic/__init__.py` — export `Embedder`, `make_embedder`, `build_faiss_hnsw`.

### Generated (NOT committed; tracked via sidecars + `.gitignore`)
- `data/v1/embedding_bakeoff_sample.parquet` (+ sidecar)
- `data/v1/embeddings.parquet` (~210 MB, fp16, ~100 k × 768) (+ sidecar)
- `data/v1/faiss.index` (+ sidecar)
- `data/v1/faiss_pmid_map.parquet` (+ sidecar)

---

## Reused existing code

- `scifield.repro.record_run` — for every artifact's sidecar; extend `config` dict per D4.
- `scifield.cli._load_config` — Hydra composition helper (see `src/scifield/cli.py:47–54`); reused for the two new subcommands.
- `scifield.corpus.store` DuckDB views — read `papers` directly; do not re-Parquet.
- `scripts/brev_smoke.sh` — copy and adapt structure (credit-balance probes, `trap EXIT` stop, manual-fallback gating) for `brev_embed.sh`.

---

## Verification (acceptance tests — run before declaring V1-S05 done)

1. **Static checks pass.** `uv run pre-commit run --all-files` green.
2. **Tests pass.** `uv run pytest` green, including the new offline tests. `SCIFIELD_RUN_SLOW_TESTS=1 uv run pytest -m slow` green locally (real model load).
3. **CLI smoke.** `uv run scifield embed --help` and `uv run scifield faiss-build --help` exit 0.
4. **Bake-off complete.** `notebooks/03_embedding_bakeoff.ipynb` renders end-to-end. Metrics table present; recommendation cell explicitly names the winner with its kNN@10 precision and intra/inter separation, and applies the stop-condition guard.
5. **`conf/thematic/embed.yaml`** updated to name the bake-off winner with its HF revision pinned to a commit SHA (not `main`).
6. **Embeddings exist.** `data/v1/embeddings.parquet` row count == abstract-bearing paper count from the same query (recorded in sidecar). Sidecar JSON contains `model_name`, `model_revision`, `sentence_transformers_version`, `torch_version`, `batch_size`, `max_seq_length`, `n_papers`, `total_runtime_s`, `gpu_model`.
7. **FAISS index loads.** `faiss.read_index("data/v1/faiss.index")` succeeds; index `ntotal` == embeddings row count.
8. **Spot-check.** Notebook section 6 queries 10 manually-chosen PMIDs (committed in the notebook), prints top-5 NN PMID + title + journal. Visually plausible nearest-neighbour clusters (judgement call by Samer).
9. **Brev hygiene.** Brev instance stopped (`brev list` shows no `scifield-embed-V1-S05`). Credit balance before/after logged in `docs/operations/brev.md`.
10. **Reproducibility.** All four generated artifacts (sample, embeddings, faiss.index, pmid_map) have sidecar JSONs with matching `git_sha` and a valid `config_hash`.
11. **No large binaries committed.** `git status` shows no `*.parquet` / `*.index` / `*.pt` staged. `.gitignore` already covers `data/`.

---

## Execution sequence (for `/awesome-execute`)

Where steps are independent they may parallelise; the order below preserves logical dependencies.

1. **Dependencies.** Edit `pyproject.toml` → add 4 deps → `uv lock && uv sync`. (Sets the stage; everything else depends on this.)
2. **Embedder module + offline tests.** Write `src/scifield/thematic/embed.py` and `tests/test_thematic_embed_offline.py`. Run the offline test.
3. **FAISS module + synthetic test.** Write `src/scifield/thematic/faiss_index.py` and `tests/test_thematic_faiss.py`. Run.
4. **CLI subcommands + smoke test.** Wire `embed` and `faiss-build` into `src/scifield/cli.py`; add `tests/test_cli_embed.py`. Run.
5. **Config files.** Write `conf/thematic/embed.yaml` (defaults to mpnet pre-bake-off) and add `thematic: embed` to `conf/config.yaml`.
6. **Slow embedder test.** Write `tests/test_thematic_embed.py` (real-model). Run with `SCIFIELD_RUN_SLOW_TESTS=1`.
7. **Bake-off notebook — sample construction.** Implement sections 1–2 of `notebooks/03_embedding_bakeoff.ipynb`; produce `data/v1/embedding_bakeoff_sample.parquet`.
8. **Bake-off notebook — run + recommend.** Implement sections 3–5; execute end-to-end locally on Mac CPU.
9. **Update config with winner.** Edit `conf/thematic/embed.yaml` per recommendation; pin HF revision SHA.
10. **Brev script.** Write `scripts/brev_embed.sh` (model on `brev_smoke.sh`).
11. **Brev run (external).** Execute `scripts/brev_embed.sh`. Watch logs; rsync `embeddings.parquet` + sidecar back to local `data/v1/`. Stop instance. Log credit balance.
12. **Local FAISS build.** `uv run scifield faiss-build --embeddings data/v1/embeddings.parquet --out data/v1/faiss.index`. Produces index, PMID map, sidecar.
13. **Spot-check notebook section 6.** Execute the FAISS spot-check cell; commit the 10-PMID NN inspection table.
14. **Final acceptance pass.** Run verification §1–§11 above.
15. **Commit.** `feat(thematic): V1-S05 embedding bake-off + full-corpus embeddings + FAISS HNSW`. Multi-line message lists artifacts produced and bake-off winner.
16. **Status update.** Append a `**Status: ✓ with notes (2026-05-20).**` block to V1-S05 in `plans/Session-Objectives-MAP.md` — same format Samer used for V1-S03.

---

## Stop conditions (encoded from brief)

- If no model beats `all-mpnet-base-v2` by ≥ 0.03 on kNN@10 precision, default to mpnet. **Document this** as the bake-off conclusion rather than silently picking mpnet.
- If Brev access is broken, document in `docs/operations/brev.md` (matching V1-S02's pattern) and run the embedding on the most powerful local hardware available; flag the deviation in `embeddings.parquet.run.json` under a `deviations` key.

---

## Risk register check (plan §6 mapping)

- **OpenAlex pre-2000 sparse** — irrelevant here; V1-S05 only uses PubMed-side fields (title, abstract, MeSH).
- **Compute insufficient** — first session that triggers this. Mitigation: L40S 48 GB is overspec for 100 k × 768d; 1–2 GPU-hr expected. If Brev unavailable, embedding is feasible on a recent Mac M-series in ~3–6 hr CPU.
