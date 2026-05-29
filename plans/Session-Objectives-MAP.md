# SciField — Session Objective Map (v0.1)

**Purpose.** Convert `plan/scifield_plan.md` into a flat, ordered list of Claude Code sessions that can be executed sequentially without scope drift. Each session entry below is a self-contained briefing — copy the entire `### V?-S??` block into a fresh Claude Code session and run.

**Author:** Samer Salman | **Date written:** 2026-05-18 | **Master plan:** `plan/scifield_plan.md`

---

## Context

You have a fully-formed v0.1 project plan for SciField (a multi-axis framework for monitoring scientific field health) in `plan/scifield_plan.md`. The plan covers Phases 0–9 over 50 weeks and is correctly scoped. What's missing is an execution layer: a way to translate "Phase 3: epistemic quality extraction (5–7 weeks)" into discrete, atomic Claude Code work units that each produce a verifiable deliverable, don't bleed into adjacent work, and respect the plan's five human-decision gates.

This document is that execution layer. It does not change the master plan — it indexes into it.

---

## Version split

- **Version 1 — Phases 0–6 (the validation run).** 15 sessions. Scaffolding through findings on the 10-journal corpus. All 5 decision gates live here. End state: a working framework with results on a manageable corpus; manuscript-grade analyses exist as draft artifacts.
- **Version 2 — Phase 7 only (the scaling run).** 3 sessions. Expand the corpus to 25–40 journals (500k–1M papers), re-run the validated pipeline, document scaling behavior, test v1 finding replication.
- **Version 3 — Phases 8–9 (release + publication).** 6 sessions. PyPI/Docker/Zenodo release, documentation site, demo app, manuscript draft, submission package.

Note: the existing `Version 1/` and `Version 2/` folders should be matched by a new `Version 3/` folder when V3 begins. Each version's source code, configs, and notebooks live in its own folder.

---

## How to use this document

1. Open this file alongside `plan/scifield_plan.md`.
2. Start a fresh Claude Code session in the SciField repo.
3. Copy the entire `### V?-S??` block for the next session into the prompt.
4. After Claude finishes, run the acceptance tests listed in the briefing.
5. If the session is followed by a **GATE**, stop. Read the gate criteria. Make the human decision. Do not start the next session until the gate is resolved (proceed, pivot, or stop).
6. Mark the session done in this file (add a `✓` next to its header) before moving on.

**Anti-drift rules.**
- Every session has an explicit **Out of scope** list. If you find yourself wanting to do something not listed in **In scope**, stop and ask whether it belongs in a later session.
- Every session names its deliverable file paths. If a session ends without producing those exact paths, it is not done.
- Decision gates are not Claude sessions. They are human reviews. A Claude session may produce a gate-review report; the decision is yours alone.

---

## Cross-cutting constraints (apply to every session)

- **Reproducibility.** Every artifact produced (DuckDB, Parquet, model checkpoint, figure) must be accompanied by a sidecar JSON recording: git SHA at time of run, Hydra config hash, input data hashes, software versions. Module: `scifield.repro` (built in V1-S01).
- **Pre-registration timing.** OSF pre-registration #1 (Phase 3 validation protocol) must be filed in **V1-S07 before any batch extraction**. OSF pre-registration #2 (Phase 5 forecasting protocol) must be filed in **V1-S12 before GNN training**. Non-negotiable per plan §5 Phase 3 and Phase 5.
- **Brev hygiene.** Any session that launches a Brev instance must (a) check the credit balance at start, (b) tag the instance with the session ID, (c) stop the instance as part of acceptance tests. Pattern: `scifield brev launch --session V1-S05` / `scifield brev stop --session V1-S05`. Plan §8.3.
- **Package-from-day-zero.** Every module written goes into `src/scifield/<axis>/` from the start. No notebooks-then-refactor. Notebooks live in `notebooks/` and import from `scifield`, never reimplement.
- **Risk register check.** Before starting any session that maps to a plan-level risk (see §6 of master plan), re-read the risk row.

---

## Decision gates (all in V1)

| # | After session | What's decided | Pass criterion | If fail |
|---|---|---|---|---|
| G1 | V1-S06 | Topic structure clinically interpretable? | 20-topic spot-check passes domain-expert read | Fix embedding/clustering — do NOT continue to Phase 3 |
| G2 | V1-S09 | Epistemic extraction reliable? | (v2, 2026-05-29; no human labels) ALL: cross-tool RCT agreement κ≥0.7 vs PubMed PublicationType MeSH; model-vs-model study_design agreement ≥80% on Claude-Code/DeepSeek overlap; six internal-validity priors all hold (see V1-S09 spec) | Document caveats in Limitations; F1 stays but reliability story qualified |
| G3 | V1-S11 | Dual-novelty 2×2 shows structure? | Semantic-structural correlation \|ρ\| < 0.4 AND ≥1 surprising-to-expert finding | Drop F2; paper pivots to F1+F3 |
| G4 | V1-S14 | GNN beats baselines? | >5 pp emergence AUC over best baseline at 3yr horizon, Wilcoxon significant | Frame F3 as null finding or drop it |
| G5 | V1-S16 | ≥2 of 3 findings hold? | At least 2 of {F1, F2, F3} have statistically supported, narratively coherent results | Downscope to methods + resource paper |

---

# Version 1 — Phases 0–6 (15 sessions)

---

### V1-S01 — Repo scaffolding, CI, CLI skeleton

**Phase:** 0 (Scaffolding) | **Plan ref:** §5 Phase 0 | **Effort:** ~1 day | **Depends on:** none

**Objective.** Stand up the `scifield` Python package with modern tooling so every subsequent session writes into a clean, reproducible repo. By end of session, `uv run scifield --help` works, CI is green, pre-commit blocks bad code.

**Preconditions.** Working directory `/Users/samersalman/Desktop/SciField/`. `uv` installed. A GitHub repo (private) named `scifield` exists or will be created in this session.

**In scope.**
- `pyproject.toml` with `scifield` package; `[project.scripts] scifield = "scifield.cli:main"`.
- `src/scifield/` package with module stubs: `corpus/`, `thematic/`, `epistemic/`, `novelty/`, `forecasting/`, `integration/`, `repro/`, `cli.py`. Each stub has one `__init__.py` and a single `# TODO` docstring naming its plan phase.
- `src/scifield/repro/__init__.py`: implement `record_run(artifact_path, inputs, config)` that writes a sidecar JSON with git SHA, config hash, input hashes, software versions.
- `src/scifield/cli.py`: Click or Typer CLI with `--help` and a `demo` subcommand stub that prints "demo not yet implemented".
- `.pre-commit-config.yaml`: ruff, black, mypy.
- `.github/workflows/ci.yml`: on PR — `uv sync`, ruff, black --check, mypy, pytest.
- `tests/test_cli.py`: one test that asserts `scifield --help` exits 0.
- `tests/test_repro.py`: one test that `record_run` writes the expected JSON.
- `LICENSE` Apache 2.0. `README.md` with one-paragraph project description and "see plan/" pointer. `.gitignore` for Python + uv + data artifacts.

**Out of scope (defer).**
- Hydra config system → V1-S02.
- Documentation site (mkdocs) → V1-S02.
- Any data harvesting or real pipeline logic → V1-S03+.
- DVC / data versioning beyond the sidecar JSON → consider in V1-S03 if needed.

**Acceptance tests.**
- `uv sync && uv run scifield --help` prints CLI usage.
- `uv run pytest` passes (2 tests).
- `uv run pre-commit run --all-files` passes.
- First push to GitHub → CI green.
- `git log` shows clean initial commit with conventional-commit style messages.

**Stop conditions.** If `mypy` is fighting you on the stub modules, ship with `# type: ignore` on the empty `__init__.py` files only — do not weaken the global mypy config.

---

### V1-S02 — Hydra configs, docs site, end-to-end demo on 100-paper toy corpus

**Phase:** 0 (Scaffolding) | **Plan ref:** §5 Phase 0, §3 | **Effort:** ~1 day | **Depends on:** V1-S01

**Objective.** Wire Hydra into the CLI, build the mkdocs site, and make `scifield demo` actually run end-to-end on a 100-paper toy corpus (any PubMed search). By end of session, a colleague could clone the repo and reproduce the demo in <10 minutes per plan §5 Phase 0 success criterion.

**Preconditions.** V1-S01 complete. PubMed access (no auth needed for Entrez but needs an email in the header).

**In scope.**
- `conf/` directory with Hydra layout: `conf/config.yaml`, `conf/corpus/`, `conf/thematic/`, `conf/epistemic/`, etc. Each has at least a `default.yaml` with a few placeholder keys.
- `conf/demo.yaml`: a tiny 100-paper config (e.g., `journal: "Arthroscopy"`, `year_range: [2024, 2024]`, `max_papers: 100`).
- `scifield.cli`: `demo` subcommand now: (1) reads `conf/demo.yaml` via Hydra, (2) pulls 100 abstracts via Biopython Entrez, (3) writes a Parquet at `data/demo/papers.parquet`, (4) writes a sidecar JSON via `repro.record_run`, (5) prints summary stats (n papers, mean abstract length).
- `docs/` mkdocs-material site: index page (project overview), `phases/` section with one page per phase (initially just the phase objective from the master plan), `api/` section auto-generated via `mkdocstrings`.
- `mkdocs.yml`.
- `.github/workflows/docs.yml`: build docs on push to main, deploy to GitHub Pages.
- `scripts/brev_smoke.sh`: launch smallest-tier instance, pull repo, run `uv sync`, run `scifield demo`, stop instance. Prints credit-balance before/after.

**Out of scope.**
- Full corpus harvest (10 journals × 30 years) → V1-S03.
- OpenAlex / Semantic Scholar / ROR enrichment → V1-S04.
- Anything beyond the demo's 100 papers.
- DVC.

**Acceptance tests.**
- `uv run scifield demo` produces `data/demo/papers.parquet` with 100 rows and a sidecar JSON.
- `mkdocs serve` builds locally without errors.
- GitHub Pages deployment URL works.
- `scripts/brev_smoke.sh` runs successfully on a real Brev instance and the instance is stopped at the end.
- CI still green.

**Stop conditions.** If Brev access has any issue, document it in `docs/operations/brev.md` and skip the smoke test — but do not delete the script. Flag for follow-up.

---

### V1-S03 — Corpus v1 harvesting: PubMed + DuckDB + Parquet

**Phase:** 1 (Corpus v1) | **Plan ref:** §4, §5 Phase 1 | **Effort:** ~1.5 days (mostly waiting on Entrez) | **Depends on:** V1-S02

**Objective.** Harvest abstracts and metadata for all 10 v1 journals across 1995–2025 from PubMed into a DuckDB database backed by Parquet. End state: `data/v1/papers.duckdb` exists with the expected ~150–250k papers, every paper has PMID, year, title, abstract, journal, MeSH; Parquet snapshots per journal per year for reproducibility.

**Preconditions.** V1-S02 complete. NCBI email + (optional) API key. Disk space ≥ 10 GB.

**In scope.**
- `src/scifield/corpus/pubmed.py`: async Entrez harvester using `httpx + tenacity`, with rate limiting (3 req/sec without API key, 10 with). Idempotent — re-runs only fetch missing PMIDs.
- `src/scifield/corpus/store.py`: DuckDB schema for papers + journals + mesh; Parquet writer per journal per year.
- `conf/corpus/v1.yaml`: list of 10 journals with PubMed Title abbreviations (JBJS Am, Arthroscopy, J Arthroplasty, Spine, Clin Orthop Relat Res, Ann Surg, JAMA Surg, J Am Coll Surg, Br J Surg, Surgery), year range 1995–2025.
- `scifield.cli`: `harvest` subcommand reading `conf/corpus/v1.yaml`.
- `notebooks/01_corpus_overview.ipynb`: read DuckDB, plot papers/year per journal, abstract length distribution, MeSH coverage.

**Out of scope.**
- OpenAlex / Semantic Scholar / ROR enrichment → V1-S04.
- Citation graph → V1-S04 (loaded into Kùzu in V1-S10).
- Embeddings → V1-S05.
- Any analysis beyond descriptive corpus stats.

**Acceptance tests.**
- `uv run scifield harvest --config conf/corpus/v1.yaml` completes (probably overnight).
- DuckDB row count is within 150k–250k; per-journal counts within ±20% of expected based on plan §4.
- `notebooks/01_corpus_overview.ipynb` renders.
- Sidecar JSON exists for every Parquet file.
- >95% of papers have non-empty abstract; flag and document any journal-era combination below 90%.

**Risk hooks.** Plan §6 risk row 1 (OpenAlex coverage pre-2000) — this session is PubMed only, so not yet triggered. But measure abstract availability pre-2000 here and write it into the corpus stats.

**Status: ✓ with notes (2026-05-19).**
- Harvest complete: **134,978 papers across 310 buckets (10 journals × 31 years), 233s wall-time** with API key.
- Pipeline: 0 parse failures; esearch_count == parsed_count == 134,978 (no drop between PubMed and Parquet).
- Acceptance gate notes:
  - **Total count came in below the 150k–250k band (134,978).** Not a harvest bug — esearch returned every PMID PubMed indexes for these TA queries; we captured all of them. The 150k–250k estimate appears to have over-counted vs. real PubMed indexing.
  - **Overall abstract coverage 74.1%, below the 95% gate.** Root cause is publication-type mix, not era sparseness: in general-surgery journals 30–35% of indexed items are Comments / Letters / Case Reports for which the publisher provides no abstract (e.g., JAMA Surg: Comments 0.2% abs, Letters 4.2% abs). Within research-article types — `Multicenter Study`, `Randomized Controlled Trial`, `Comparative Study` — abstract coverage runs 90–99% across all 10 journals. Downstream sessions that need a "high-abstract" analytic subset should filter on `publication_types`, not restrict by era.
  - Era breakdown (input to V1-S04 OpenAlex coverage decisions): **<2000 78.2% / 2000-09 78.0% / 2010-19 74.4% / 2020+ 68.2%.** Pre-2000 is NOT the abstract bottleneck the plan §6 risk row anticipated; the 2020+ drop reflects epub-ahead-of-print + commentary volume.
  - Sidecars: 310/310 Parquet, 310/310 manifest, DuckDB sidecar present.
- Notebook `notebooks/01_corpus_overview.ipynb` executes end-to-end against full corpus.

---

### V1-S04 — Corpus v1 enrichment: OpenAlex + Semantic Scholar + ROR + author disambiguation

**Phase:** 1 (Corpus v1) | **Plan ref:** §5 Phase 1 success criteria | **Effort:** ~2 days | **Depends on:** V1-S03

**Objective.** Enrich the v1 corpus with citation lists, authorship, institutions, and citation intents. End state: per-paper feature table has OpenAlex ID, full citation list (target ≥80% resolved), full reference list, disambiguated authors, ROR-canonical institutions, Semantic Scholar citation intents where available. Coverage report written and committed.

**Preconditions.** V1-S03 complete. OpenAlex polite-pool email (no key needed). Semantic Scholar API key (optional but improves rate limit).

**In scope.**
- `src/scifield/corpus/openalex.py`: async harvester for `/works/W…`, paginate `cited_by_api_url` for incoming citations and `referenced_works` for outgoing. Cache to disk aggressively.
- `src/scifield/corpus/semantic_scholar.py`: secondary citation source + intent labels (background/method/result).
- `src/scifield/corpus/ror.py`: ROR API lookups for institution canonicalization.
- `src/scifield/corpus/authors.py`: author disambiguation using OpenAlex author IDs + ORCID where available; fallback to name+institution heuristic.
- `src/scifield/cli`: `enrich` subcommand reading the same `conf/corpus/v1.yaml`.
- `notebooks/02_coverage_report.ipynb`: produces the §5 Phase 1 coverage report — % with OpenAlex match, % with full citation list, % with MeSH, % with disambiguated authors. Broken down by journal and by era (pre-2000, 2000–2010, 2010–2020, 2020+).
- `docs/phases/corpus.md`: written-up risk register entry per plan §5 Phase 1 risk register.

**Out of scope.**
- Loading the citation graph into Kùzu → V1-S10.
- Any topic / embedding / novelty / forecasting work.
- v2 expansion journals.

**Acceptance tests.**
- ≥90% of papers have OpenAlex match (plan success criterion).
- ≥80% of papers post-2000 have full citation list resolved.
- Coverage report committed as both notebook and `docs/phases/corpus.md`.
- Total enrichment runtime fits in the Brev budget for Phase 1 ($0, plan §8.2).

**Risk hooks.** Plan §6 row 1 (OpenAlex pre-2000) — measure and document explicitly. If coverage <60% pre-2000, restrict downstream novelty/CD analyses to post-2000 (decide in V1-S10).

---

### V1-S05 — Embedding bake-off + full-corpus embeddings + FAISS index

**Phase:** 2 (Thematic backbone) | **Plan ref:** §5 Phase 2 | **Effort:** ~1.5 days | **Depends on:** V1-S04 | **Brev:** L40S 48GB, 2–4 GPU-hours, ~$5–10

**Objective.** Decide on the embedding model via a small principled bake-off, then embed the full v1 corpus and build a FAISS HNSW index. End state: `data/v1/embeddings.npy` (or equivalent), `data/v1/faiss.index`, a one-page report justifying the model choice.

**Preconditions.** V1-S04 complete. Brev L40S access. Labeled subset for bake-off — use ~500 abstracts with rough topic labels (you can self-label or use Arthroscopy topics from prior work as ground truth where journals overlap).

**In scope.**
- `src/scifield/thematic/embed.py`: pluggable embedder supporting `all-mpnet-base-v2`, `BAAI/bge-large-en-v1.5`, `nomic-embed-text-v1`.
- `notebooks/03_embedding_bakeoff.ipynb`: on the labeled subset, compute intra-label vs. inter-label cosine separation, kNN retrieval precision@10, and runtime per 1k abstracts. Report winner.
- `conf/thematic/embed.yaml`: chosen model + parameters.
- Full-corpus embedding run on Brev L40S; output `data/v1/embeddings.parquet` (PMID → vector).
- `src/scifield/thematic/faiss_index.py`: build + persist HNSW index at `data/v1/faiss.index`.
- Brev launch + stop wrapped via `scifield brev`.

**Out of scope.**
- Topic modeling (UMAP/HDBSCAN/c-TF-IDF) → V1-S06.
- Hierarchical merging → V1-S06.
- Semantic novelty computations → V1-S10.

**Acceptance tests.**
- Bake-off report exists and recommends a model with quantitative justification.
- Embedding shape matches paper count.
- FAISS index loads and returns reasonable nearest neighbors on 10 spot-check papers.
- Brev instance stopped; credit balance recorded.
- Sidecar JSON for embedding artifact records model name + version + config hash.

**Stop conditions.** If no model materially outperforms `all-mpnet-base-v2` on your bake-off, default to mpnet (you know its behavior). Document this in the bake-off report.

**Status: ✓ with notes (2026-05-21).**
- Bake-off complete: all 3 candidates (mpnet, bge-large-en-v1.5, nomic-embed-text-v1) encoded the 500-paper MeSH-stratified sample on Mac CPU. kNN@10 precision: mpnet 0.149 / bge 0.161 / nomic 0.163. **Stop-condition guard fired:** best candidate (nomic) beat mpnet by Δ=0.013 < 0.03 threshold → defaulted to mpnet per plan D6. mpnet also had the highest intra/inter cosine separation (0.107 vs 0.035 / 0.033).
- `conf/thematic/embed.yaml` pinned to `revision: e8c3b32edf5434bc2275fc9bab85f82640a19130`.
- Full-corpus embeddings: **99,938 abstract-bearing papers × 768d (fp16)** at `data/v1/embeddings.parquet` (142 MB). FAISS HNSW index at `data/v1/faiss.index` (335 MB), `ntotal=99,938`, M=32, efConstruction=200, efSearch=64. Both sidecars contain all D4 fields + matching git_sha + config_hash.
- **Deviation: ran locally on Mac CPU rather than Brev L40S.** Bake-off showed mpnet was CPU-viable; chose to save ~$1 and skip first-run risk of unproven `scripts/brev_embed.sh`. Documented in `docs/operations/brev.md` "V1-S05 (2026-05-21): Brev deferred, ran locally" and in `embeddings.parquet.run.json` under `deviations`. Encoding took 6.3 hr wall-clock (CPU contended with other host workloads); `brev_embed.sh` is committed for future GPU sessions but never executed end-to-end.
- Spot-check (notebook §6): 10 hand-chosen PMIDs (one per journal) return semantically coherent top-5 NN. Highlights: posterior tibial slope → tibial slope papers (sim 0.86–0.91); disc nucleus-annulus → ISSLS disc-mechanics papers (sim 0.73–0.77); amputation disparities → racial/gender variation in amputation (sim 0.76–0.83); scapula tilt editorial → source paper at sim 0.857.
- Acceptance: full pytest 77 passed / 1 skipped, pre-commit all-green, every sidecar has matching git_sha + config_hash, no large binaries staged.
- Carryover for V1-S06: `data/v1/papers.duckdb` has duplicate PMIDs (same paper indexed in multiple journal-year buckets when journal TA-terms overlap). Spot-check exposed this via duplicate rows for PMID 29100772. V1-S06 / V1-S15 should dedupe at the boundary.

---

### V1-S06 ✓ (executed; gate G1 FAILED, retune queued as V1-S06b) — BERTopic pipeline + hierarchical merging + coherence + Gate G1 report

**Phase:** 2 (Thematic backbone) | **Plan ref:** §5 Phase 2 + Gate after Phase 2 | **Effort:** ~2 days | **Depends on:** V1-S05

**Objective.** Produce the canonical v1 topic landscape: 100–200 leaf topics organized hierarchically into ~20 mid-level and 5–7 top-level domains. Compute coherence. Generate the gate report for human review.

**Preconditions.** V1-S05 complete; FAISS index + embeddings exist.

**Carryover from V1-S05 (must address before BERTopic input).** `data/v1/papers.duckdb`'s `papers` view contains 13,070 PMIDs with duplicate byte-identical rows (root cause: V1-S03 harvest wrote duplicates inside single (journal, year) buckets — likely PubMed eSearch pagination overlap or OR'd TA-term clauses). The V1-S05 embeddings.parquet inherited this: 99,938 vector rows = ~89,230 distinct PMIDs + ~10,708 duplicate-PMID vector rows. Before feeding embeddings to UMAP/HDBSCAN, dedupe on PMID at one of: (a) `papers` view (preferred — fix once, all phases benefit), (b) FAISS row map filter, or (c) embeddings-parquet read step in `topics.py`. Verify duplicates are still byte-identical (`SELECT COUNT(*) FROM (...) HAVING COUNT(DISTINCT abstract) > 1` should return 0) before using `ANY_VALUE`/`SELECT DISTINCT`; if not, tiebreak by longest abstract.

**In scope.**
- `src/scifield/thematic/topics.py`: BERTopic pipeline (UMAP → HDBSCAN → c-TF-IDF), parameter sweep harness, hierarchical merging via BERTopic's `hierarchical_topics`.
- `src/scifield/thematic/coherence.py`: NPMI + C_v scoring via OCTIS or Palmetto wrapper.
- `conf/thematic/v1.yaml`: chosen hyperparameters.
- `data/v1/topics.parquet`: per-paper topic assignment + probability + hierarchy level.
- `notebooks/04_topic_landscape.ipynb`: intertopic distance map (Plotly), temporal heatmap of topic share, per-journal-per-year topic distribution.
- `docs/phases/thematic.md`: human-readable summary of the topic hierarchy with top-20 topic word lists.
- **`docs/gates/G1_topic_interpretability.md`**: gate report — coherence numbers vs. the *Arthroscopy* baseline (NPMI ~0.18), noise fraction, 20-topic spot-check table (topic ID, top words, your one-line clinical interpretation), recommendation (proceed / revisit clustering / revisit embedding).

**Out of scope.**
- Anything epistemic/novelty/forecasting.
- Re-running embeddings (only revisit if gate fails).
- v2 corpus.

**Acceptance tests.**
- NPMI ≥ 0.18 (plan baseline).
- Noise fraction <20%.
- 20-topic spot-check table is filled in by you (not Claude) before declaring done.
- Intertopic map renders.
- Gate report committed.

**STOP after this session.** Read `docs/gates/G1_topic_interpretability.md`. Make the gate decision. Only continue to V1-S07 if **G1 passes**.

---

#### 🚦 GATE G1 — Topic structure clinical interpretability (after V1-S06)

Pass = 20-topic spot-check reads as clinically coherent to you (or a co-author with the relevant domain expertise). Coherence within range. If fail: do not start V1-S07. Tune embedding (re-run V1-S05) or clustering (re-run V1-S06) until topics make sense. The entire framework rests on this.

---

### V1-S07 — Epistemic schema, prompt engineering, hand-labeling tool, OSF pre-registration #1

**Phase:** 3 (Epistemic quality) | **Plan ref:** §5 Phase 3 (the hardest single phase) | **Effort:** ~2 days | **Depends on:** V1-S06 + G1 passed

**Objective.** Prepare every piece needed to run batch epistemic extraction at scale. Lock down the schema, prompt, validation protocol, and OSF pre-registration *before* spending API budget. End state: pre-registration is submitted, hand-labeling tool is ready, 500-sample stratification is selected.

**Preconditions.** G1 passed. Anthropic API key with batch access. OSF account.

**In scope.**
- `src/scifield/epistemic/schema.py`: Pydantic model for extracted fields per plan §5 Phase 3 — study_design (RCT / cohort / case-control / case-series / review / other), sample_size, has_control, effect_direction (positive / null / negative / mixed / na), statistical_claim_present, coi_disclosed_in_abstract.
- `src/scifield/epistemic/prompt.py`: system prompt + few-shot examples. Versioned (`v0.1`).
- `src/scifield/epistemic/labeling_tool/`: minimal Streamlit or CLI tool for double-coding 500 abstracts by you + Rohan/other co-author. Stores annotations in `data/v1/epistemic_handlabel.parquet`.
- `notebooks/05_handlabel_sampling.ipynb`: stratified random sample of 500 abstracts across journal × era × MeSH cluster. Selection committed to `data/v1/handlabel_sample.parquet`.
- `src/scifield/epistemic/kappa.py`: Cohen's kappa + Krippendorff's alpha for inter-rater reliability.
- Small pilot: run the LLM on 50 abstracts via the regular API (not batch) — eyeball outputs, refine prompt, iterate. Document iterations in `docs/phases/epistemic.md`.
- **OSF pre-registration #1** (`docs/preregistrations/PR1_epistemic_extraction.md`, also submitted on OSF): the validation protocol — sample size, κ targets (≥0.7 design / ≥0.8 controls / ≥0.6 effect direction), comparison to LLM, pivot conditions.

**Out of scope.**
- Running the full 200k-abstract batch extraction → V1-S08.
- κ computation on real data → V1-S08.
- Trialstreamer/RobotReviewer cross-validation → V1-S09.
- Any analysis using the extracted fields → Phase 6.

**Acceptance tests.**
- Pydantic schema validates on the 50-abstract pilot output.
- Labeling tool successfully records a few test annotations.
- 500-sample selection is finalized (stratification visible in notebook).
- OSF pre-registration is submitted and has a public link; the link is in `docs/preregistrations/PR1_epistemic_extraction.md`.
- Pilot prompt iteration log committed.

**Stop conditions.** Do not start V1-S08 until OSF pre-registration is **submitted** (not just drafted). This is the most important pre-condition in V1.

---

### V1-S08 — Hand-labeling execution + batch LLM extraction at scale

**Phase:** 3 (Epistemic quality) | **Plan ref:** §5 Phase 3 | **Effort:** ~2 weeks elapsed (hand-labeling) + 2 days compute (batch) | **Depends on:** V1-S07 | **LLM API:** $50–150 per plan §8.5

**Objective.** Get the ground-truth labels and the full LLM extraction done. End state: 500 abstracts double-coded with arbitrated final labels; full 200k corpus extracted via Claude Batch API; both tables in DuckDB.

**Preconditions.** V1-S07 complete, pre-registration submitted, hand-labeling tool tested.

**In scope.**
- Hand-labeling sprint: you + co-author label all 500 abstracts independently, then meet for arbitration. Final labels stored in `data/v1/epistemic_handlabel_final.parquet`.
- `src/scifield/epistemic/batch.py`: Claude Batch API client — submit jobs, poll, write results to `data/v1/epistemic_extracted.parquet`. Resumable; failed records written to `data/v1/epistemic_failed.parquet` for retry.
- `scifield.cli`: `epistemic extract --batch` subcommand.
- API cost tracking — every batch invocation logs token counts + cost to `docs/operations/api_costs.md`.

**Out of scope.**
- Validation analysis (LLM vs. hand-labels) → V1-S09.
- Trialstreamer/RobotReviewer comparison → V1-S09.
- Error analysis → V1-S09.

**Acceptance tests.**
- 500/500 hand-labeled records final-arbitrated.
- ≥99% of corpus successfully extracted by batch API (retry failures explicitly).
- Total API spend logged and within plan §8.5 budget.
- Pre-registration link is included in the extraction sidecar JSON.

**Risk hooks.** Plan §6 row 2 (epistemic extraction unreliable) — outcome of V1-S09 determines whether this risk fires.

---

### V1-S09 — Epistemic validation v2 (no human labels), Gate G2 report

**Phase:** 3 (Epistemic quality) | **Plan ref:** §5 Phase 3 + Gate after Phase 3 (v2 — redefined 2026-05-29) | **Effort:** ~1 day | **Depends on:** V1-S08

**Objective.** Validate the 87,268 DeepSeek extractions without human labels, via three reliability lenses: cross-tool agreement, model-vs-model agreement, and internal-validity priors. Produce the G2 gate report.

**Scope rationale (2026-05-29 redefinition).** The original G2 was hand-label κ on a 500-abstract sample. Samer explicitly dropped this — the goal of SciField is not to re-establish LLM extraction quality (already well-established in the meta-research literature). Hand-labeling 500 abstracts is high effort for a finding that is not the paper's contribution. The redefinition substitutes three cheaper-but-still-defensible reliability lenses.

**Preconditions.** V1-S08 complete; `data/v1/epistemic_extracted.parquet` exists with both `deepseek-v4-flash` and `claude-via-claude-code` rows; DuckDB `papers` table carries PubMed `PublicationType` MeSH headings (verify with V1-S03 harvester output schema).

**In scope.**

- `notebooks/06_epistemic_validation.ipynb` — runs all three lenses, emits the gate-report tables and figures.
- `src/scifield/epistemic/validate.py` — pure functions for each lens (cross-tool agreement, model-vs-model agreement, internal-validity checks), tested via unit tests with synthetic fixtures.
- `docs/gates/G2_epistemic_reliability.md` — the gate report, structured around C1/C2/C3 below with explicit pass/fail per criterion.
- `docs/phases/epistemic.md` — closeout addendum referencing the gate report.

**Out of scope.**

- Hand-labeling, inter-rater κ, arbitration workbooks. The infrastructure for these stays in the codebase (V1-S07/S08 `arbitrate.py`, `sampling.py`) as optional future work, but is not used in this gate.
- BERT fine-tuning fallback path — eliminated; if G2 fails, F1 stays in the manuscript with a qualified reliability section.
- RobotReviewer integration (risk-of-bias) — defer; not a field in our schema, would require running RobotReviewer on full-texts which we don't have.
- v2 anything.

**Three-lens gate — ALL must hold:**

**C1. Cross-tool agreement: RCT detection vs PubMed PublicationType MeSH**

PubMed already classifies RCTs via the MeSH PublicationType `Randomized Controlled Trial`. This is a free, high-coverage cross-tool comparator: our DeepSeek `study_design == "RCT"` extraction can be benchmarked against it for the entire corpus.

- Pass: Cohen's κ ≥ 0.7 on PMIDs with non-null PublicationType, OR simple agreement ≥ 85%.
- Sample size N = all DeepSeek-extracted PMIDs with at least one PubMed PublicationType assigned (expected ~87k).
- Stretch (optional): if time permits, pull Trialstreamer's downloadable RCT corpus and run a second cross-tool agreement table for the PMID intersection. Not required for gate pass.

**C2. Model-vs-model agreement: DeepSeek vs Claude Code**

19 PMIDs have both a `claude-via-claude-code` row and a `deepseek-v4-flash` row from the V1-S08 dual-run. Report exact-match agreement per field on this overlap. Optionally expand to ~100–200 paired observations by re-running DeepSeek on a sample of the 1,981 Claude-Code-only PMIDs (cost ~$0.01–0.02, requires Samer's spend approval per `feedback-deepseek-spend-gating`).

- Pass: study_design exact-match agreement ≥ 80%; has_control exact-match agreement ≥ 80%; sample_size Spearman ρ ≥ 0.75 on PMIDs where both report a non-null value.
- N: report whatever N is available. If N < 50, note as "underpowered model-vs-model lens" and weight C1+C3 accordingly.

**C3. Internal-validity checks: corpus-wide priors**

Cheap distributional sanity checks on the full 87,268 DeepSeek extractions. Each must hold:

- (a) RCT prevalence (study_design == "RCT") ∈ [3%, 15%]. Biomedical prior.
- (b) statistical_claim_present rate ∈ [60%, 95%]. Most published biomedical work makes a statistical claim; well below 60% suggests prompt failure, above 95% suggests over-eager labeling.
- (c) coi_disclosed_in_abstract rate < 15%. Most journals put COI in full-text footers, not abstracts.
- (d) Conditional: among PMIDs with study_design == "RCT", has_control == True ≥ 90%. RCTs have controls by definition; failures here flag a definitional drift.
- (e) sample_size sanity: max < 10,000,000 (human studies); median ∈ [10, 5000].
- (f) effect_direction distribution: "na" share ≥ 5% (presence of reviews / methods papers); "positive" share ≤ 70% (some null and mixed results expected).

**Acceptance tests.**

- All three lenses computed and tabled in `docs/gates/G2_epistemic_reliability.md`.
- Each pass/fail decision per criterion explicit and reproducible from `notebooks/06_epistemic_validation.ipynb`.
- Unit tests for `validate.py` functions pass (synthetic fixtures only — no API calls).
- Gate report committed.

**STOP after this session. Resolve gate G2 (pass / qualified-pass / fail) before V1-S10.**

---

#### 🚦 GATE G2 v2 — Epistemic extraction reliability without human labels (after V1-S09)

**Pass** = C1 cross-tool κ ≥ 0.7 OR agreement ≥ 85% AND C2 study_design and has_control agreement ≥ 80% on whatever overlap N is available AND all six C3 internal-validity priors hold.

**Qualified pass** = C1 passes AND C3 passes BUT C2 fails or N too small. F1 stays in the manuscript with an explicit caveat that model-vs-model agreement is underpowered.

**Fail** = C1 fails OR ≥2 of the C3 priors fail. Document the failure in the Limitations section, qualify any F1 claims accordingly, do NOT pivot to BERT fine-tuning (out of scope per the 2026-05-29 redefinition).

---

### V1-S10 — Novelty: semantic + Kùzu citation graph + CD index implementation

**Phase:** 4 (Novelty) | **Plan ref:** §5 Phase 4 | **Effort:** ~2 days | **Depends on:** V1-S09 + G2 resolved

**Objective.** Build both novelty pipelines and validate the CD-index implementation against published values. End state: per-paper semantic novelty scores; citation graph loaded into Kùzu; CD index computed for every paper; cdindex validation against Park et al. 2023 corpus passes.

**Preconditions.** V1-S09 complete. Park et al. 2023 replication corpus or a comparable validation set with published CD values.

**In scope.**
- `src/scifield/novelty/semantic.py`: for each paper, compute mean + minimum cosine distance to all papers published before it in the same field. Chosen variant documented (e.g., Foster et al. 2015 style). FAISS-backed.
- `src/scifield/novelty/kuzu_loader.py`: Kùzu schema per plan §5 Phase 4 — Paper, Author, Journal, Institution, Topic nodes; CITES, AUTHORED_BY, AFFILIATED_WITH, PUBLISHED_IN, ASSIGNED_TO edges. Load from DuckDB + OpenAlex citation data.
- `src/scifield/novelty/cd_index.py`: implementation + validation harness against the Park 2023 corpus.
- `conf/novelty/v1.yaml`: window sizes (CD_5, CD_10), normalization choices.
- `notebooks/07_novelty_compute.ipynb`: produces both scores for all v1 papers; spot-checks ranking sanity (high-CD seminal papers should be recognizable).

**Out of scope.**
- 2×2 archetype analysis + temporal trends → V1-S11.
- Joint distribution figure → V1-S11.
- Forecasting → Phase 5.

**Acceptance tests.**
- Kùzu graph contains expected node/edge counts (within order of magnitude of plan §4 expectation).
- CD-index validation: implementation reproduces Park 2023 published values within tolerance (Pearson r ≥ 0.95 on overlapping papers).
- Semantic novelty scores look reasonable on spot-check (review papers should score low; novel methods papers high).
- Sidecar JSONs committed.

**Risk hooks.** Plan §6 row 1 — if pre-2000 OpenAlex coverage is too sparse, restrict primary novelty analyses to post-2000 and document explicitly.

---

### V1-S11 — Dual-novelty 2×2 archetype, temporal trends, headline figure, Gate G3 report

**Phase:** 4 (Novelty) | **Plan ref:** §5 Phase 4 + Gate after Phase 4 | **Effort:** ~1.5 days | **Depends on:** V1-S10

**Objective.** Test whether semantic and structural novelty are meaningfully decorrelated; assign each paper to a 2×2 archetype; trend the archetype distribution over 30 years; produce the dual-novelty figure (potential headline figure of the paper).

**Preconditions.** V1-S10 complete; novelty scores exist for all papers.

**In scope.**
- `src/scifield/novelty/archetypes.py`: 2×2 binning logic; per-paper archetype assignment stored in `data/v1/archetypes.parquet`.
- `notebooks/08_dual_novelty.ipynb`: scatter of semantic × structural novelty; archetype distribution by year and by journal; archetype × citation-impact analysis.
- `docs/figures/F2_dual_novelty.png`: publication-grade figure draft.
- **`docs/gates/G3_dual_novelty.md`**: gate report — Pearson/Spearman correlation between novelty axes (target \|ρ\| < 0.4); archetype distribution interpretation; at least one finding flagged "surprising" by a domain reader (you); recommendation.

**Out of scope.**
- F2 narrative for manuscript → Phase 6.
- Any forecasting.
- Reconciling F2 with F1 or F3 → Phase 6.

**Acceptance tests.**
- \|ρ\| computed and reported.
- 2×2 distribution table exists.
- Headline figure renders cleanly.
- Gate report committed.

**STOP. Resolve gate G3 before V1-S12.**

---

#### 🚦 GATE G3 — Dual-novelty structure (after V1-S11)

Pass = \|ρ\| < 0.4 AND ≥1 surprising-to-expert finding in the archetype distribution or its temporal evolution. Fail = drop F2 from the manuscript; pivot the paper architecture to F1+F3 only. If F2 dropped, the headline figure becomes one of the F1 or F3 figures and Phase 6 work re-prioritizes.

---

### V1-S12 — Forecasting: OSF pre-registration #2, temporal CV, all baselines

**Phase:** 5 (Forecasting) | **Plan ref:** §5 Phase 5 | **Effort:** ~2 days | **Depends on:** V1-S11 + G3 resolved

**Objective.** Lock the forecasting protocol on OSF *before* training anything. Implement the temporal CV split with leakage tests. Implement all 4 baselines so the GNN has something to beat.

**Preconditions.** G3 resolved (proceeding with F3 regardless of F2 outcome). Topic time series available from V1-S06.

**In scope.**
- **OSF pre-registration #2** (`docs/preregistrations/PR2_forecasting.md`, submitted): architecture choice (HGT first), evaluation metrics (emergence AUC, share MAPE), baselines, statistical comparison (Wilcoxon signed-rank across topics), train/val/test split (1995–2017 / 2018–2020 / 2021–2025), pivot conditions.
- `src/scifield/forecasting/data.py`: build topic-level time-series features from V1-S06 output; temporal CV split with explicit leakage tests.
- `src/scifield/forecasting/baselines/`: `naive.py` (3yr moving average), `arima.py` (statsmodels), `mlp.py` (simple MLP on topic-level features), `no_graph.py` (same features as the GNN but no graph).
- `notebooks/09_baselines.ipynb`: baseline performance table on validation set.
- `conf/forecasting/v1.yaml`: split definition + baseline hyperparameters.

**Out of scope.**
- HGT/TGN architecture → V1-S13.
- Training the GNN → V1-S13.
- Final evaluation + Wilcoxon → V1-S14.

**Acceptance tests.**
- OSF pre-registration #2 **submitted** with public link.
- Leakage tests pass — no future data leaks into training.
- All 4 baselines produce metrics on the validation set.
- Pre-registration link in the baseline-run sidecar JSONs.

**Stop conditions.** Do not start V1-S13 until pre-registration #2 is submitted.

---

### V1-S13 — HGT/TGN architecture, training loop, Brev spot hyperparameter sweep

**Phase:** 5 (Forecasting) | **Plan ref:** §5 Phase 5 | **Effort:** ~1 week elapsed (most is compute wait) | **Depends on:** V1-S12 | **Brev:** A100 80GB, 30–60 GPU-hours, ~$50–120

**Objective.** Implement the HGT (preferred) heterogeneous graph transformer, train with proper checkpointing for spot reclaims, run a hyperparameter sweep on Brev spot A100s, identify the best model.

**Preconditions.** V1-S12 complete. PyTorch Geometric installed. Kùzu graph from V1-S10 available; convert relevant subgraphs to PyG format.

**In scope.**
- `src/scifield/forecasting/gnn/hgt.py`: HGT architecture per plan §5 Phase 5 (Paper/Author/Topic node types; HBM-friendly batching).
- `src/scifield/forecasting/train.py`: training loop with checkpoint-resume so spot reclaims are non-fatal. Logs per-epoch metrics + cost.
- `src/scifield/forecasting/sweep.py`: Optuna or Ray Tune sweep over learning rate, hidden dim, n_layers, dropout.
- `scripts/brev_train.sh`: launch A100 spot, pull repo, run sweep, push checkpoints to local + S3 (or institutional storage), stop instance.
- Sweep results in `data/v1/forecasting_sweep.parquet`.
- Best model checkpoint at `models/v1/hgt_best.pt`.

**Out of scope.**
- Final evaluation vs. baselines → V1-S14.
- Wilcoxon statistical comparison → V1-S14.
- TGN as a fallback — only if HGT badly underperforms (would be a V1-S13b).

**Acceptance tests.**
- Sweep completes; best config recorded.
- Checkpoint loads cleanly outside Brev.
- All Brev instances stopped; total spend logged against plan §8.2 budget.
- Training sidecar JSON references pre-registration #2 link.

**Risk hooks.** Plan §6 row 4 (GNN fails to beat baselines) — outcome determined in V1-S14.

---

### V1-S14 — Forecasting evaluation, calibration, Wilcoxon, Gate G4 report

**Phase:** 5 (Forecasting) | **Plan ref:** §5 Phase 5 + Gate after Phase 5 | **Effort:** ~1 day | **Depends on:** V1-S13

**Objective.** Run the pre-registered evaluation on the held-out 2021–2025 test set. Compare HGT against all 4 baselines using the pre-registered Wilcoxon test. Produce calibration plots. Write the G4 gate report honestly.

**Preconditions.** V1-S13 complete; best model checkpoint exists; 2021–2025 test data has not been touched.

**In scope.**
- `notebooks/10_forecasting_eval.ipynb`: emergence AUC + share MAPE per topic for HGT and all baselines; Wilcoxon signed-rank across topics; calibration plots; ablation isolating the contribution of graph structure (HGT vs. no-graph baseline).
- `docs/figures/F3_forecasting.png`: publication-grade figure draft.
- **`docs/gates/G4_forecasting.md`**: gate report — full evaluation table, Wilcoxon p-values, ablation result, recommendation. If GNN beats best baseline by >5 pp on emergence AUC with p<0.05, pass. Otherwise, frame F3 as null finding.

**Out of scope.**
- F3 narrative for manuscript → Phase 6.
- Retraining anything (test data was touched once, in this session, per pre-registration).

**Acceptance tests.**
- All pre-registered metrics computed.
- Ablation table exists.
- Gate report committed.

**STOP. Resolve G4 before V1-S15.**

---

#### 🚦 GATE G4 — Forecasting beats baselines (after V1-S14)

Pass = HGT > best baseline by >5 pp emergence AUC at 3yr horizon, Wilcoxon p<0.05. Fail = F3 is framed as a null finding ("graph structure does not add forecasting value beyond temporal features") — itself publishable per plan §6 row 4. Do not retrain on test data, do not p-hack, do not amend the pre-registration except via a clearly-logged deviation note.

---

### V1-S15 — Integration: F1 + F2 + F3 analyses, bonus cross-journal/institutional, draft results + figures, Gate G5 report

**Phase:** 6 (Integration) | **Plan ref:** §5 Phase 6 + Gate after Phase 6 | **Effort:** ~1.5 weeks | **Depends on:** V1-S14 + G4 resolved

**Objective.** Synthesize the four axes into the three target findings, plus bonus analyses. Produce a draft results section, the final figure set, and the G5 gate report deciding whether the V1 narrative holds up enough to scale to V2.

**Preconditions.** V1-S14 complete; gates G1–G4 all resolved (passed or pivoted).

**In scope.**
- `notebooks/11_F1_epistemic_cascade.ipynb`: for each topic with sufficient sample size, plot evidence-quality trajectory vs. topic-volume trajectory; lead-lag analysis via cross-correlation and Granger causality; significance threshold pre-specified.
- `notebooks/12_F2_dual_novelty_trends.ipynb` (or skipped if G3 dropped F2): archetype distribution by year + by topic; archetype × citation impact.
- `notebooks/13_F3_forecasting_narrative.ipynb`: re-frame V1-S14 results in the broader narrative.
- `notebooks/14_bonus_cross_journal.ipynb`: cross-journal seeding networks (which journals lead vs. follow on a given topic); per-specialty differences; institutional/geographic patterns.
- `docs/figures/`: full publication-grade figure set drafts (F1, F2 if applicable, F3, plus 1–2 bonus).
- `docs/results_draft.md`: draft results section per plan §5 Phase 6 deliverable.
- **`docs/gates/G5_v1_findings.md`**: gate report — which of {F1, F2, F3} held, narrative coherence assessment, recommendation (proceed to V2 / downscope to methods+resource paper).

**Out of scope.**
- v2 corpus expansion → V2-S01.
- OSS release / docs polish → V3.
- Manuscript drafting (intro/methods/discussion) → V3-S05.
- Re-running anything from earlier phases.

**Acceptance tests.**
- Each notebook renders end-to-end.
- Figures committed.
- Results draft is comprehensible to a co-author who has not seen the pipeline.
- Gate report committed.

**STOP. Resolve G5 before V2 begins.**

---

#### 🚦 GATE G5 — V1 narrative coherence (after V1-S15)

Pass = ≥2 of {F1, F2, F3} held with statistical support; co-authors agree the narrative is coherent. Fail = downscope to a methods + resource paper (still publishable per plan §3); do not proceed to V2 scaling until you understand why the narrative didn't hold on the 10-journal corpus.

---

# Version 2 — Phase 7 (3 sessions)

V2 starts only after G5 passes. V2 deliverables can be merged into the V1 manuscript or split into a separate scaling paper; that decision happens in V2-S03.

---

### V2-S01 — Corpus v2 expansion: 25–40 journals harvest + enrich

**Phase:** 7 (Scaling v2) | **Plan ref:** §4 v2 expansion + §5 Phase 7 | **Effort:** ~1 week | **Depends on:** V1-S15 + G5 passed

**Objective.** Harvest and enrich the v2 corpus: add neurosurgery (Neurosurgery, J Neurosurg), CT surgery (Ann Thorac Surg, J Thorac Cardiovasc Surg), plastic (PRS), vascular (J Vasc Surg), plus 3–5 non-surgical specialties (cardiology, hematology, oncology, neurology, internal medicine). Target: 500k–1M papers.

**In scope.** Same modules as V1-S03 and V1-S04, driven by `conf/corpus/v2.yaml`. Coverage report comparing v1 vs. v2.

**Out of scope.** Re-running the pipeline → V2-S02. Any analysis → V2-S03.

**Acceptance tests.** Corpus size within target range; coverage metrics ≥ v1 levels on the new journals; v2 coverage report committed.

---

### V2-S02 — Re-run full pipeline on v2 + scaling benchmark

**Phase:** 7 (Scaling v2) | **Plan ref:** §5 Phase 7 | **Effort:** ~2 weeks (mostly compute) | **Depends on:** V2-S01 | **Brev:** A100 80GB on-demand, 20–40 GPU-hours, ~$40–80

**Objective.** Run the validated pipeline (embed, topic-model, epistemic extract, novelty, forecast) on v2. Produce a scaling benchmark report (runtime / memory / cost vs. corpus size). Fix any awkwardness exposed by 5× scale.

**In scope.** Re-run V1-S05 / V1-S06 / V1-S08 / V1-S10 / V1-S13 logic on v2 data; record runtime + memory + cost per phase; identify and fix any scaling bottleneck.

**Out of scope.** Re-validating epistemic extraction (validation is from V1; only re-run if pre-registration explicitly required it). Replication analysis → V2-S03.

**Acceptance tests.** Pipeline completes end-to-end on v2; benchmark table committed; no Brev instance left running.

---

### V2-S03 — v1↔v2 replication, scaling figures, decision on scaling paper

**Phase:** 7 (Scaling v2) | **Plan ref:** §5 Phase 7 success criteria | **Effort:** ~1 week | **Depends on:** V2-S02

**Objective.** Test whether v1 findings replicate in v2 — including the specialty-dependence story (does F1 hold differently in surgical vs. medical specialties?). Update figures. Decide whether v2 results merge into the V1 manuscript or constitute their own paper.

**In scope.** Replication analyses for F1, F2 (if alive), F3 on v2; specialty-comparative breakdowns; updated figure set; decision memo on manuscript architecture.

**Out of scope.** Manuscript drafting → V3.

**Acceptance tests.** Replication table committed; updated figures committed; decision memo on whether v2 produces a separate paper.

---

# Version 3 — Phases 8–9 (6 sessions)

V3 is the "ship it" phase. Lighter session detail because V3 is months away and the manuscript shape depends on V1+V2 outcomes.

---

### V3-S01 — PyPI packaging, Apache 2.0, CITATION.cff, Docker image

**Phase:** 8 (OSS release) | **Plan ref:** §5 Phase 8 | **Effort:** ~3 days

**Objective.** `pip install scifield` works. Docker image builds and runs. Citation metadata correct.

**In scope.** Finalize `pyproject.toml` metadata; build and test PyPI release on TestPyPI; CITATION.cff; Dockerfile + multi-arch build; tagged v1.0.0-rc1 release on GitHub.

**Out of scope.** Documentation polish → V3-S02. Demo app → V3-S03.

---

### V3-S02 — Documentation site polish: quickstart, tutorial, API reference, contribution guidelines

**Phase:** 8 (OSS release) | **Effort:** ~3 days | **Depends on:** V3-S01

**Objective.** A researcher unfamiliar with the project can install the tool, run the tutorial, and apply it to their own corpus within an afternoon (plan §5 Phase 8 success criterion).

**In scope.** Polish mkdocs site: quickstart, full tutorial walkthrough of v1 results, complete API reference (autogenerated), contribution guidelines, hosting on GitHub Pages with custom domain optional.

---

### V3-S03 — Streamlit/Dash demo app deployed on Hugging Face Spaces

**Phase:** 8 (OSS release) | **Effort:** ~3 days | **Depends on:** V3-S02

**Objective.** Reviewers and readers can explore v1 results interactively without installing anything.

**In scope.** Interactive topic explorer; dual-novelty 2×2 viewer; forecasting trajectories per topic; deploy to HF Spaces.

---

### V3-S04 — Zenodo deposit + DOI + cross-field example notebooks

**Phase:** 8 (OSS release) | **Effort:** ~2 days | **Depends on:** V3-S03

**Objective.** Pre-computed v1 and v2 artifacts released via Zenodo with citable DOI. Example notebooks for 2–3 non-surgical fields (e.g., radiology, oncology) demonstrate generality.

**In scope.** Zenodo upload of corpus indices + embeddings + topic assignments + extracted features + model checkpoints; example notebooks running the framework on a radiology or oncology corpus (small; for demonstration).

---

### V3-S05 — Manuscript draft: intro, methods, results, discussion

**Phase:** 9 (Manuscript) | **Plan ref:** §5 Phase 9 | **Effort:** ~3 weeks elapsed

**Objective.** A submission-ready manuscript draft.

**In scope.** Use the `manuscript-pipeline` skill (per your existing tooling) starting from the V1+V2 results notebooks and draft results section. Target structure per plan §5 Phase 9. Anticipate reviewer concerns re: no senior science-of-science author (plan §11) in Methods and Supplementary Methods.

**Out of scope.** Submission logistics → V3-S06.

---

### V3-S06 — Internal review, external review, bioRxiv preprint, submission

**Phase:** 9 (Manuscript) | **Effort:** ~2 weeks

**Objective.** Get the paper out the door.

**In scope.** Co-author review round; 2–3 external senior reviews; bioRxiv preprint deposit simultaneous with journal submission; target Nature Communications first per plan §5 Phase 9; backup journals identified; cover letter; suggested reviewers list.

**Acceptance tests.** Preprint live with DOI; manuscript submitted; submission ID logged.

---

# Risk register cross-reference

Mirrors plan §6, mapped to sessions where each risk first becomes actionable.

| Plan risk | First actionable session | Mitigation in this map |
|---|---|---|
| OpenAlex pre-2000 sparse | V1-S04 | Measure and document in `docs/phases/corpus.md`; restrict novelty analyses if needed in V1-S10 |
| Epistemic LLM unreliable | V1-S09 | Gate G2; fallback BERT path inserted as V1-S09b if triggered |
| Dual-novelty 2×2 boring | V1-S11 | Gate G3; F2 dropped, paper rescopes |
| GNN fails to beat baselines | V1-S14 | Gate G4; F3 framed as null finding, still publishable |
| All findings underwhelming | V1-S15 | Gate G5; downscope to methods+resource paper |
| Compute insufficient | V1-S05, V1-S13, V2-S02 | Brev hygiene; institutional HPC fallback documented in `docs/operations/brev.md` |
| Timeline slips | every gate | Gates allow pause-and-resume; phases independently valuable |
| Pre-registration delay | V1-S07, V1-S12 | Pre-reg blocks downstream sessions but is <1 day of work |

---

# What's NOT in this map (intentionally)

- **Senior collaborator outreach.** Per plan §11 decision, project proceeds without one. If you change your mind, that's a workstream alongside this map, not a session in it.
- **Authorship decisions.** Plan §12 open question. Resolve outside this map before V3-S05.
- **Project rename.** Plan §12 open question. Resolve before V3-S01 (the PyPI release name is hard to change).
- **First-sprint Phase 0 logistics.** Plan §13 lists 6 small tasks; those are absorbed into V1-S01 and V1-S02.
- **Stretch ideas** (plan §10). Explicitly out of scope for V1+V2+V3. Future papers.

---

# Maintenance

- Mark sessions done with `✓` next to the header as you finish them.
- If a session's scope changes during execution, edit its `In scope` / `Out of scope` lists *in this file* — don't let scope drift go unrecorded.
- If a gate fails and a pivot session is inserted (e.g., V1-S09b for the BERT fallback), document it inline below the failing gate.
- Treat this file as living: update after each gate, never silently.
