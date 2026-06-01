# Phase 4 — Novelty

## Phase objective

Compute two complementary novelty scores for every paper as the substrate for
the dual-novelty analysis. **Semantic novelty** is the mean and minimum cosine
distance from a paper's embedding to the embeddings of all prior papers in the
same field. **Structural novelty** is the Funk–Owen-Smith CD_t (disruption)
index computed on the citation graph stored in Kùzu (an embedded columnar graph
database that ships as a pip dependency, no server needed). The CD-index
implementation is validated **offline against the authoritative `cdindex` PyPI
package** (the Funk-lab port): it reproduces that package's documented toy
example exactly and is asserted to match it on a shared graph at Pearson
r ≥ 0.99. (This supersedes the earlier stub's "Park et al. 2023 replication
corpus" plan — no proprietary WoS/Park data is used; the offline `cdindex`
agreement is the comparable validation set.)

The downstream 2×2 archetype assignment (frontier work, methodological
repackaging, isolated novelty, consolidation), the temporal-trend analysis, the
dual-novelty figure, and Gate G3 are **out of scope for V1-S10** — they belong
to **V1-S11** (see the scope boundary below). V1-S10 delivers the per-paper
semantic + structural novelty machinery and the canonical Kùzu graph that the
2×2 analysis will consume.

---

## V1-S10 closeout (2026-05-29)

**Status:** semantic novelty + Kùzu graph shipped and verified; CD-index
machinery implemented and validated offline; the corpus CD application is
**gated** on the OpenAlex incoming-citation harvest (dry-run sized, stopped
pending Samer's go/no-go).

### What was built

- `src/scifield/novelty/semantic.py` — per-paper semantic novelty
  (`compute_semantic_novelty`).
- `src/scifield/novelty/cd_index.py` — from-scratch Funk–Owen-Smith CD_t
  (`CitationGraph`, `cd_index`, `classify_citers`, `compute_corpus_cd`).
- `src/scifield/novelty/cited_by.py` — async OpenAlex incoming-citation
  harvester (gated; see below).
- `src/scifield/novelty/kuzu_loader.py` — Kùzu schema + bulk loader
  (`build_kuzu_graph`).
- `conf/novelty/v1.yaml` — flat OmegaConf config (paths, field choice, CD
  windows, harvest tuning), read via `_load_novelty_config("v1")`.
- New `novelty` Typer sub-app in `src/scifield/cli.py`: `semantic`,
  `harvest-citedby [--dry-run]`, `cd`, `kuzu`.
- Dependencies added: `kuzu`, `cdindex`.
- Tests: `tests/test_novelty_{semantic,cd_index,kuzu}.py` (synthetic fixtures,
  no network) — all green.

### Semantic novelty

Artifact: `data/v1/novelty_semantic.parquet` — **89,230 deduped papers**, columns
`pmid, topic_id, year, n_prior, sem_nov_mean, sem_nov_min`.

- **Field partition = `topic_id`** (149 leaf topics + noise `-1`). A paper's
  "field" is its BERTopic leaf topic; noise papers (`topic_id == -1`) fall back
  to a corpus-wide prior.
- **`sem_nov_mean`** = distance to the centroid of all strictly-prior same-field
  vectors, `1 − f·centroid_prior`. Because every embedding is L2-normalised, the
  mean cosine collapses to a single dot product against a running prior-year
  centroid — **exact and O(N)**, no pairwise blowup.
- **`sem_nov_min`** = `1 − (nearest prior cosine)`. Computed brute-force within
  each leaf topic; for noise papers the global **FAISS HNSW** index is queried
  (`faiss_k = 256`) over the whole corpus, post-filtered to strict priors.
- **Strict-prior tie rule:** a paper is a prior of the focal iff `year <
  focal.year`. Same-year papers are NOT priors of each other (within-year order
  is unobservable at year granularity).
- **Earliest-in-field:** **2,367 papers** have `n_prior == 0` (no strictly-prior
  same-field paper) → null scores, flagged via `n_prior`.
- **Score ranges:** `sem_nov_mean` median 0.436 / mean 0.494 / range
  [0.060, 0.998]; `sem_nov_min` median 0.147 / mean 0.161 / range [≈0, 0.759].
- **FAISS recall** adequate: NaN among noise-papers-with-priors = 0.02% (the
  `faiss_k` fan-out is large enough that a strict prior survives the year filter
  for essentially every noise paper).

### Structural novelty — CD-index

Module: `src/scifield/novelty/cd_index.py`. CD_5 and CD_10.

- **Method (from scratch):** over citers `j` of focal `F` in the window
  `(y_F, y_F + t]`, classify `n_i` (cites `F` only), `n_j` (cites `F` **and** a
  reference of `F`), `n_k` (cites a reference of `F` but not `F`); then
  `CD_t = (n_i − n_j) / (n_i + n_j + n_k)`, bounded `[-1, 1]`, `NaN` when the
  window has no qualifying citers.
- **Validation** (`tests/test_novelty_cd_index.py`, no network): reproduces the
  `cdindex` documented toy example exactly (focal `4Z`, `t = 5` → `1/6`) and
  matches `cdindex` on a deterministic 120-node graph at **Pearson r = 1.0**
  (≥ 0.99 bar met). The time-window convention (`y_F < y_citer ≤ y_F + t`) was
  pinned empirically against the installed `cdindex == 1.0.20`.
- **Corpus application** `compute_corpus_cd` (artifact `data/v1/cd_index.parquet`,
  columns `pmid, openalex_id, cd5, cd10, n_citers, n_window_5, n_window_10`) is
  implemented but **GATED** on the OpenAlex incoming-citation harvest.

> **Documented limitation (agreed fallback).** The harvest enumerates only
> citers *of* the focal work, so the corpus CD path cannot observe `n_k` (works
> that cite a focal *reference* but not the focal). The corpus denominator is
> therefore `n_i + n_j` rather than `n_i + n_j + n_k`, making corpus CD a
> **slight upward-biased approximation** of the true CD_t. Rankings are largely
> preserved. The exact-`n_k` definition remains validated in the unit tests; the
> approximation applies only to the corpus driver. This is the plan's agreed
> fallback given the harvest scope.

### Citation graph — Kùzu

Artifact: `data/v1/kuzu_graph` (Kùzu **0.11.3** embedded DB). Built via
`build_kuzu_graph`, bulk-loaded DuckDB → temp Parquet → `COPY`. Idempotent
(drops and rebuilds tables on re-run). Verified counts:

| Node table   | Count   | | Edge table       | Count     |
|--------------|---------|-|------------------|-----------|
| Paper        | 121,908 | | CITES            | 574,478   |
| Author       | 238,119 | | AUTHORED_BY      | 622,294   |
| Institution  | 29,159  | | AFFILIATED_WITH  | 505,815   |
| Journal      | 10      | | PUBLISHED_IN     | 121,908   |
| Topic        | 149     | | ASSIGNED_TO      | 67,821    |

Totals: **390,345 nodes / 1,892,316 edges** (~2.3M graph objects). `CITES` is
corpus-internal only (a `references_out` row becomes an edge only when its
`ref_openalex_id` resolves to a corpus PMID). `AFFILIATED_WITH` is the
disambiguated Author → Institution edge, joined on `(pmid, author_position)`.

**Deviations from the plan's rough estimates** (noted):
- **Journal = 10** — the corpus is 10-journal scoped; the JAMA Surgery /
  Archives of Surgery rename collapses to a single `jama_surg` slug node.
- **ASSIGNED_TO = 67,821** — noise papers are excluded, so `Topic` has 149 nodes,
  not ~89k, and only non-noise papers carry a topic edge.
- **AFFILIATED_WITH = 505,815** distinct Author → Institution edges — the plan's
  ~875k figure was the raw pre-dedup `paper_institutions` row count, not distinct
  canonical author-institution pairs.

### Incoming-citation harvest (GATED)

`src/scifield/novelty/cited_by.py` — async OpenAlex harvester
(`cites:<id>` filter, cursor pagination, gzip disk cache, resumable manifest),
mirroring `corpus/openalex.py`. Output columns: `focal_oa_id, citing_oa_id,
citing_year, cites_focal_ref`.

- **Dry-run sizing (2026-05-29, no live calls):** 118,317 focal works,
  6,431,128 incoming citations, 127,700 API pages, ~4.4 h wall-time @ 8 req/s,
  ~491 MB.
- **STOPPED at the gate** pending Samer's explicit go/no-go before the full
  harvest (large pull; spend/effort discipline). Once approved:
  `scifield novelty harvest-citedby` then `scifield novelty cd`.

### Reproducibility & provenance

- Every artifact writes a `.run.json` sidecar via `scifield.repro.record_run`
  (git SHA + config hash + input hashes).
- Per repo convention the entire `data/` tree is gitignored, so artifacts and
  their sidecars live on the local filesystem only — identical handling to
  `embeddings.parquet` / `faiss.index`. **Only code and config are committed.**

### Scope boundary (V1-S10 vs V1-S11)

V1-S10 ships the per-paper **semantic** and **structural** novelty machinery and
the canonical **Kùzu** graph. It does **not** include the 2×2 archetype
assignment, temporal trends, the dual-novelty figure, or **Gate G3** — those are
**V1-S11**, which consumes `novelty_semantic.parquet`, `cd_index.parquet`, and
`kuzu_graph`. The CD corpus artifact must be unblocked (harvest go/no-go) before
the structural axis of the 2×2 is fully populated; until then the semantic axis
and the validated CD machinery are in hand.
