# Gate G1 — Topic structure clinical interpretability

**Session:** V1-S06  **Date generated:** 2026-05-21  **Plan ref:** `plans/plan-S06.md`, `plans/Session-Objectives-MAP.md` §V1-S06

This is the first of five human-decision gates in the SciField execution plan.
The downstream framework — epistemic cascade (F1), dual novelty (F2),
forecasting (F3) — all rests on the topic landscape being clinically
interpretable. If the 20-topic spot-check below does not read as coherent
to Samer (or a domain co-author), **stop and re-tune** before proceeding to
V1-S07.

---

## Pass criteria (from `plans/Session-Objectives-MAP.md` §Gate G1)

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| NPMI (top-10 words/topic, noise excluded) | ≥ 0.18 (Arthroscopy baseline) | **0.224** | **PASS** |
| C_v (top-10 words/topic, noise excluded) | report | 0.756 | — |
| Noise fraction | < 20% | **29.5%** | **FAIL** |
| Leaf topic count | 100 ≤ n ≤ 200 | **89** | **FAIL** |
| Mid-level count | ≈ 20 | 58 | **FAIL** |
| Top-level count | 5–7 | 58 | **FAIL** |
| 20-topic spot-check (clinical coherence) | qualitative pass | 17 coherent / 2 cross-domain-valid / 1 incoherent (`docs/gates/G1_spotcheck.csv` filled 2026-05-21) | **MIXED** |

> The *Arthroscopy* NPMI baseline (~0.18) comes from prior single-journal work
> referenced in V1-S05 design notes. It is the conservative lower bound for a
> coherent biomedical topic model on this corpus.

> **Hierarchy degeneracy note.** Mid and top counts collapsed to the same value
> (58) because BERTopic's `hierarchical_topics` linkage on this fit produced
> only ~31 merges before bottoming out — `build_hierarchy`'s union-find
> exhausted the linkage rows before reaching the target counts of 20 / 6.
> This is itself diagnostic: the leaf set is under-segmented, so there is
> little structure for the linkage to compress.

## Chosen configuration

Sidecar: `data/v1/topics.parquet.run.json` — `config_hash=5a1ed67d…`, `git_sha=eebe89e7…`.

| Hyperparameter | Value |
|---|---|
| umap_n_neighbors | 15 |
| umap_n_components | 5 |
| umap_min_dist | 0.0 |
| umap_metric | cosine |
| hdbscan_min_cluster_size | 80 |
| hdbscan_min_samples | (None — defaults to min_cluster_size) |
| hdbscan_cluster_selection_method | eom |
| nr_topics | auto |
| random_state | 42 |
| vectorizer_min_df | 10 |
| vectorizer_ngram_max | 2 |

Encoder: `sentence-transformers/all-mpnet-base-v2` @ revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` (768-d, V1-S05 bake-off winner). Embeddings cast fp16 → fp32 on load.

Runtime (Mac CPU, 14 threads): sweep 581.7 s, final fit 63.4 s, hierarchy 6.4 s, **total 689.6 s (≈11.5 min)**.

Software pinned at run time: bertopic 0.17.4, umap-learn 0.5.12, hdbscan 0.8.43, gensim 4.4.0, scikit-learn 1.8.0, numpy 2.4.6.

## Sweep summary

Sweep over the 3×2 grid of `(umap_n_neighbors, hdbscan_min_cluster_size)`;
selector = `npmi_top10` among configs satisfying the constraints
(100 ≤ n_leaf_topics ≤ 200, noise_fraction ≤ 0.20). **No config satisfied
the constraints**, so the CLI fell back to the global argmax of
`npmi_top10` (`constraints_unmet=true` recorded in every sidecar).

| # | umap_n_neighbors | hdbscan_min_cluster_size | n_leaf | noise | NPMI@10 | C_v@10 | wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 50 | 34 | 0.316 | 0.095 | 0.591 | 109.4 |
| **1** (chosen) | **15** | **80** | **89** | **0.295** | **0.224** | **0.756** | **103.6** |
| 2 | 15 | 120 | 70 | 0.282 | 0.223 | 0.763 | 99.6 |
| 3 | 10 | 50 | 26 | 0.314 | 0.083 | 0.571 | 84.7 |
| 4 | 10 | 80 | 96 | 0.304 | 0.220 | 0.755 | 96.6 |
| 5 | 10 | 120 | 40 | 0.228 | 0.191 | 0.718 | 87.6 |

Source: `data/v1/topic_sweep.parquet`.

Pattern across the grid: NPMI clusters tightly at 0.22 for the four configs in the `min_cluster_size ∈ {80, 120}` band; the two `min_cluster_size=50` configs under-segment to 26–34 topics and tank NPMI. **No grid cell crossed the 20% noise threshold** — `min_cluster_size=120, n_neighbors=10` came closest at 22.8% but only produced 40 leaves.

## Top-20 word lists

Auto-rendered from `data/v1/topic_hierarchy.parquet`. Sorted by topic size descending; noise topic (-1) excluded.

| topic_id | size | top-10 words |
|---:|---:|---|
| 0 | 10,493 | hip, arthroplasty, femoral, total, knee, revision, acetabular, total hip, tha, hip arthroplasty |
| 1 | 8,241 | fusion, scoliosis, spinal, lumbar, spine, pain, bone, cervical, patients, thoracic |
| 2 | 4,300 | shoulder, cuff, rotator, rotator cuff, arthroscopic, repair, elbow, instability, humeral, tendon |
| 3 | 3,933 | pancreatic, pancreatitis, injury, intestinal, survival, patients, burn, expression, resection, levels |
| 4 | 3,772 | resection, liver, survival, cancer, colorectal, rectal, patients, laparoscopic, cholecystectomy, rectal cancer |
| 5 | 2,308 | acl, cruciate, ligament, cruciate ligament, reconstruction, anterior cruciate, anterior, tunnel, acl reconstruction, graft |
| 6 | 2,241 | infection, pji, periprosthetic, joint, joint infection, periprosthetic joint, antibiotic, infections, arthroplasty, revision |
| 7 | 1,451 | breast, breast cancer, cancer, axillary, mastectomy, node, women, biopsy, sentinel, lymph |
| 8 | 1,237 | esophageal, esophagectomy, fundoplication, reflux, esophageal cancer, cancer, survival, patients, oesophageal, laparoscopic |
| 9 | 1,149 | transplantation, donor, liver, recipients, transplant, liver transplantation, donors, graft, kidney, survival |
| 10 | 1,097 | cartilage, osteochondral, articular, articular cartilage, defects, osteoarthritis, chondrocyte, knee, lesions, autologous |
| 11 | 958 | hernia, mesh, hernia repair, repair, inguinal, inguinal hernia, hernias, ventral, ventral hernia, recurrence |
| 12 | 943 | training, residents, skills, orthopaedic, surgical, residency, performance, faculty, women, surgeons |
| 13 | 912 | thyroid, thyroid cancer, papillary, cancer, nodules, carcinoma, lymph, patients, node, neck |
| 14 | 865 | opioid, opioid use, pain, opioids, use, morphine, postoperative, consumption, analgesia, prescribing |
| 15 | 808 | bariatric, bariatric surgery, gastric, bypass, gastric bypass, weight, weight loss, obesity, rouxeny, gastric rouxeny |
| 16 | 802 | hospitals, surgical, care, mortality, hospital, quality, surgery, procedures, medicare, emergency |
| 17 | 785 | trauma, injury, mortality, trauma centers, care, centers, trauma center, trauma patients, center, injury severity |
| 18 | 767 | cancer, cells, expression, tumor, cell, colorectal, growth, colorectal cancer, gene, pancreatic |
| 19 | 719 | hyperparathyroidism, parathyroid, parathyroid hormone, hormone, calcium, primary, glands, exploration, patients, localization |

The full per-topic word table is available in `data/v1/topic_hierarchy.parquet`.

## 20-topic spot-check — results

| Category | Count | Topics |
|---|---:|---|
| Coherent (single clinical theme) | 17 | 0, 1, 2, 4, 22, 23, 24, 25, 26, 44, 45, 46, 47, 48, 65, 68, 69 |
| Cross-domain but valid | 2 | 66 (OR sterility: TJA + general surgery), 67 (laryngeal nerve: thyroid + anterior cervical spine) |
| Incoherent | 1 | 3 (n=3,933): pancreatic + pancreatitis + burn + intestinal injury + molecular expression — no single clinical theme |

The one incoherent topic is the 4th-largest cluster (~9% of corpus membership). Cross-referenced with topic 18 (molecular oncology bleed across tissue types), this indicates the molecular/injury-response sub-corpus is being smeared across two large topics rather than forming its own coherent clusters — consistent with over-aggressive c-TF-IDF merging under `nr_topics='auto'`. Numerical gates (noise 29.5%, leaf count 89, hierarchy collapse) confirm the same failure mode.

Filled column: `docs/gates/G1_spotcheck.csv`.

## Figures

Notebook execution produced:
- `docs/figures/topic_landscape.html` — BERTopic intertopic distance map (Plotly, 4.7 MB).
- `docs/figures/topic_share_by_year.png` — top-level topic share heatmap by year.
- `docs/figures/topic_share_by_journal_year.png` — per-journal stacked-area small multiples.

## Recommendation

**`RETUNE_CLUSTERING`** — confirmed by Samer 2026-05-21 after reviewing the spot-check.

Three constraints are unmet despite a healthy NPMI:

1. **Noise fraction 29.5% > 20%.** All six sweep configs are above the threshold; closest is 22.8% at `(n_neighbors=10, min_cluster_size=120)` which only yields 40 leaves.
2. **Leaf count 89 < 100.** Under-segmented; the *Arthroscopy* baseline corpus had ~150 leaves at similar settings, suggesting the 10-journal mix is producing larger heterogeneous clusters that BERTopic's c-TF-IDF then merges further via `nr_topics='auto'`.
3. **Hierarchy collapse.** Mid and top reduce to the same 58 because the linkage tree is shallow — itself a symptom of (1) and (2).

Suggested next-tune levers (in order of expected impact):

1. **Drop `nr_topics='auto'` and pass an explicit `nr_topics=150`.** The current run lets BERTopic auto-reduce post-HDBSCAN, which is collapsing real clinical sub-topics. Forcing a target count gives the linkage tree more room.
2. **Try `hdbscan_min_samples=10` (smaller than `min_cluster_size`).** This lets HDBSCAN's density estimation reach into sparser regions and re-classify points currently labeled noise (-1).
3. **Widen the grid to `min_cluster_size ∈ {30, 60, 90}` × `min_samples ∈ {5, 10, 20}`** at the chosen `n_neighbors=15`. Re-evaluate.
4. **If the retune doesn't push noise below 20%:** `RETUNE_EMBEDDING` — re-run V1-S05 with `bge-large-en-v1.5` (was Δ=+0.011 kNN over mpnet in the bake-off; the noise problem may be embedding-related, not clustering-related).

NPMI 0.224 (well above the 0.18 baseline) and the visible coherence of the top-20 word lists are strong signals that the *winner* topics are real — the problem is everything below the top 20, where the 29.5% noise sits.

## Decision

- [ ] PROCEED to V1-S07
- [x] **RETUNE_CLUSTERING (re-run V1-S06 with new params)**
- [ ] RETUNE_EMBEDDING (re-run V1-S05 + V1-S06)

**Rationale.** NPMI 0.224 and 17/20 coherent spot-check labels indicate the embedding is healthy; the failure is in post-embedding density estimation and merging. The single incoherent topic being the 4th-largest cluster makes this a structural problem, not an edge case. RETUNE_EMBEDDING is premature.

**Retune plan (revised lever order):**

1. **Joint 2×2 retune** at fixed `umap_n_neighbors=15`: `nr_topics ∈ {150, auto}` × `hdbscan_min_samples ∈ {10, None}`, holding `min_cluster_size=80`. Run as a focused mini-sweep before widening.
2. If noise still ≥ 20% or leaf count < 100, widen to `min_cluster_size ∈ {30, 60, 90}` × `min_samples ∈ {5, 10, 20}` at `n_neighbors=15`.
3. If both fail, escalate to RETUNE_EMBEDDING with `bge-large-en-v1.5`.

Signed: _Samer Salman_  Date: 2026-05-21

---

## V1-S06b retune results

**Session:** V1-S06b  **Date generated:** 2026-05-21  **Plan ref:** `plans/plan-S06b.md`  **Predecessor:** V1-S06 (`config_hash=5a1ed67d…`, `git_sha=eebe89e7…`)

Append-only follow-up to the V1-S06 sections above, per the `RETUNE_CLUSTERING` decision recorded on 2026-05-21. The V1-S06 artifacts were overwritten in place (recoverable via git history + the pinned hashes in the V1-S06 section above); the V1-S06b chosen-config sidecar is `data/v1/topics.parquet.run.json` with `config_hash=33ee895d…`.

### Pass-criteria table (V1-S06 vs. V1-S06b chosen config)

| Criterion | Threshold | V1-S06 result | V1-S06b result | V1-S06b status |
|---|---|---|---|---|
| NPMI (top-10 words/topic, noise excluded) | ≥ 0.18 (Arthroscopy baseline) | 0.224 | **0.238** | **PASS** |
| C_v (top-10 words/topic, noise excluded) | report | 0.756 | **0.782** | — |
| Noise fraction | < 20% | 29.5% (FAIL) | **24.0%** | **FAIL** (gap 4.0 pp) |
| Leaf topic count | 100 ≤ n ≤ 200 | 89 (FAIL) | **149** | **PASS** |
| Mid-level count | ≈ 20 | 58 (FAIL) | 96 | **reported, not blocking** (deferred to plan-S06c) |
| Top-level count | 5–7 | 58 (FAIL) | 96 | **reported, not blocking** (deferred to plan-S06c) |
| 20-topic spot-check (clinical coherence) | qualitative pass | 17 / 2 / 1 | **19 / 1 / 0** (`docs/gates/G1_spotcheck.csv` filled 2026-05-21) | **PASS** |

> **NPMI and leaf count cleared; noise gate did not.** All 9 phase-2 configs hit n_leaf=149 (the `nr_topics=150` pin acts as a hard target post-HDBSCAN), and noise saturated in a narrow band around 24–31% across the grid, with the chosen row 7 at the floor.

### Phase summary

**Phase 1 — focused 2×2 mini-sweep (plan-S06b §2-§3, `conf/thematic/topics_retune_phase1.yaml`).**
Configs evaluated: 4 (sweep wall ≈ 6.5 min before crash). The V1-S06 sweep parquet writer crashed with `ArrowInvalid: Could not convert 'auto' with type str: tried to convert to int64` because the grid mixed `nr_topics` types (`int(150)` and `str('auto')`) across rows and the `config` column is serialized as a single pyarrow type. **No V1-S06 artifacts were overwritten by phase 1** — the crash preceded chosen-config selection and final fit. Results recovered from stdout (`data/v1/topics_run_phase1.log.preserved`):

| min_samples | nr_topics | n_leaf | noise | NPMI@10 | wall (s) | note |
|---|---|---:|---:|---:|---:|---|
| 10 | 150 | 149 | 0.249 | 0.234 | 118.5 | closest to constraints in phase 1 |
| 10 | auto | 0 | — | — | 61.6 | errored (`ValueError: max_df < min_df` in vectorizer) |
| None | 150 | 149 | 0.295 | 0.245 | 105.8 | best NPMI in phase 1 |
| None | auto | 89 | 0.295 | 0.224 | 101.0 | reproduces V1-S06 baseline exactly |

**Phase 1 findings:**
1. `nr_topics=150` dominates `nr_topics=auto` on every passing dimension — the two `150` rows both reach n_leaf=149, the two `auto` rows either error out or reproduce the V1-S06 baseline (n_leaf=89). Plan-S06b §"Risks" §1 swap *not* required; phase 2 pins integer `nr_topics=150` as originally specified in plan-S06b §5.
2. **No phase-1 config cleared noise<20%.** Per plan-S06b §4 decision branch, auto-widen to phase 2.

**Code-defect surfacing (per plan-S06b §"Risks", *"treat as a code defect, surface to Samer, do not silently mark passed"*):** the V1-S06 sweep harness (`src/scifield/thematic/topics.py:sweep()`) returns `pd.DataFrame([asdict(r) for r in rows])`, and the `config` column ends up as a `dict[str, int|str]` whose pyarrow inference fails when the grid mixes types. Likely fix is a 1-line JSON-serialize of the `config` column before `to_parquet` (or pin `nr_topics` to a single dtype across the grid at the YAML level, as phase 2 does). Plan-S06b §"STAY IN SCOPE" prohibits the code change in this session; routed to **plan-S06c backlog**.

**Phase 2 — Cartesian widen at fixed `n_neighbors=15`, `nr_topics=150` (plan-S06b §5-§6, `conf/thematic/topics_retune_phase2.yaml`).**
Configs evaluated: **9** (3×3 grid over `min_cluster_size ∈ {30, 60, 90}` × `min_samples ∈ {5, 10, 20}`). Sweep wall **911.1 s**, final fit 64.5 s, hierarchy 7.4 s — **total phase-2 wall 1022.6 s (≈ 17.0 min)**. CLI fell back to global `npmi_top10` argmax (no config met constraints) and stamped `constraints_unmet=true` in every sidecar (`data/v1/topics.parquet.run.json`, `topic_hierarchy.parquet.run.json`, `topic_sweep.parquet.run.json`, `models/v1/bertopic_v1.run.json`).

**Total session wall (phase 1 + phase 2):** ~23.5 min on local Mac CPU, 14 threads. No Brev launches; embeddings not re-run (`embeddings_parquet` input hash `595e7b2f5f…` matches V1-S06 sidecar exactly — acceptance test #9 ✓).

### Chosen configuration (V1-S06b)

Sidecar: `data/v1/topics.parquet.run.json` — `config_hash=33ee895d20ee70b60c437413f41c69ca1f4a4c4c8585d240d22f481adb7effac`, `git_sha=eebe89e76fc7835fd3e7b02ab8ddb92920b150c7` (= V1-S06 git_sha; no commits this session), `git_dirty=true` (working-tree changes only — configs, docs, notebook, data outputs).

| Hyperparameter | V1-S06 | V1-S06b chosen (row 7) |
|---|---|---|
| umap_n_neighbors | 15 | 15 |
| umap_n_components | 5 | 5 |
| umap_min_dist | 0.0 | 0.0 |
| umap_metric | cosine | cosine |
| hdbscan_min_cluster_size | 80 | **90** |
| hdbscan_min_samples | None (= min_cluster_size) | **10** |
| hdbscan_cluster_selection_method | eom | eom |
| nr_topics | auto | **150** |
| random_state | 42 | 42 |
| vectorizer_min_df | 10 | 10 |
| vectorizer_ngram_max | 2 | 2 |

Encoder unchanged: `sentence-transformers/all-mpnet-base-v2` @ rev `e8c3b32edf5434bc2275fc9bab85f82640a19130` (768-d, V1-S05 bake-off winner). Embeddings cast fp16 → fp32 on load. **No re-embed** in V1-S06b.

Software pinned at run time (unchanged from V1-S06): bertopic 0.17.4, umap-learn 0.5.12, hdbscan 0.8.43, gensim 4.4.0, scikit-learn 1.8.0, numpy 2.4.6.

### Phase-2 sweep summary

Selector = `npmi_top10` among configs satisfying constraints (100 ≤ n_leaf ≤ 200, noise ≤ 0.20). **No config satisfied the noise constraint**, so the CLI fell back to the global argmax of `npmi_top10` per the V1-S06 sweep harness's documented fallback (`constraints_unmet=true` stamped in all sidecars).

| # | min_cluster_size | min_samples | n_leaf | noise | NPMI@10 | C_v@10 | wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30 | 5 | 149 | 0.275 | 0.214 | 0.748 | 113.3 |
| 1 | 30 | 10 | 149 | 0.295 | 0.218 | 0.754 | 98.9 |
| 2 | 30 | 20 | 149 | 0.314 | 0.217 | 0.754 | 99.4 |
| 3 | 60 | 5 | 149 | 0.274 | 0.229 | 0.771 | 100.4 |
| 4 | 60 | 10 | 149 | 0.282 | 0.230 | 0.774 | 99.4 |
| 5 | 60 | 20 | 149 | 0.287 | 0.234 | 0.777 | 100.4 |
| 6 | 90 | 5 | 149 | 0.247 | 0.235 | 0.778 | 99.3 |
| **7** (chosen) | **90** | **10** | **149** | **0.240** | **0.238** | **0.782** | **98.8** |
| 8 | 90 | 20 | 149 | 0.265 | 0.237 | 0.778 | 100.7 |

Source: `data/v1/topic_sweep.parquet`.

**Pattern across the phase-2 grid.** All 9 configs land at n_leaf=149 — the `nr_topics=150` pin acts as a hard cap post-HDBSCAN regardless of raw cluster count, so leaf-count is essentially uninformative as a discriminator in this grid. Noise has a clear monotone-decreasing trend with `min_cluster_size` (best 0.240 at `min_cluster_size=90, min_samples=10`; worst 0.314 at `min_cluster_size=30, min_samples=20`) but the floor is **24.0%, still 4.0 percentage points above the 20% gate**. NPMI rises with `min_cluster_size` (0.21 at MC=30 → 0.24 at MC=90), consistent with larger clusters producing more semantically coherent top-10 word lists. C_v tracks NPMI tightly.

### Top-20 word lists (V1-S06b)

Auto-rendered from `data/v1/topic_hierarchy.parquet`. Sorted by topic size descending; noise topic (-1) excluded.

| topic_id | size | top-10 words |
|---:|---:|---|
| 0 | 2,607 | acetabular, hip, femoral, total hip, stem, hip arthroplasty, revision, hips, component, cup |
| 1 | 2,395 | infection, pji, periprosthetic, joint infection, periprosthetic joint, joint, antibiotic, infections, arthroplasty, aureus |
| 2 | 2,369 | acl, cruciate, ligament, cruciate ligament, anterior cruciate, reconstruction, anterior, tunnel, acl reconstruction, graft |
| 3 | 1,897 | liver, resection, hepatic, hcc, hepatectomy, hepatocellular, hepatocellular carcinoma, liver resection, survival, metastases |
| 4 | 1,493 | burn, injury, sepsis, intestinal, protein, reperfusion, mice, expression, hours, levels |
| 5 | 1,454 | scoliosis, idiopathic scoliosis, idiopathic, curve, adolescent, correction, adolescent idiopathic, thoracic, ais, curves |
| 6 | 1,451 | breast, breast cancer, cancer, axillary, mastectomy, node, women, sentinel, biopsy, lymph |
| 7 | 1,444 | arthroplasty, tka, total, medicare, tha, care, readmission, costs, discharge, hospital |
| 8 | 1,444 | shoulder, glenoid, instability, arthroscopic, glenohumeral, shoulder instability, shoulders, shoulder arthroplasty, repair, anterior |
| 9 | 1,408 | knee, knee arthroplasty, total knee, alignment, tka, knees, flexion, arthroplasty, tibial, uka |
| 10 | 1,317 | residents, training, skills, performance, resident, surgical, surgeons, simulation, education, trainees |
| 11 | 1,194 | cartilage, osteochondral, articular, articular cartilage, prp, chondrocyte, defects, osteoarthritis, chondrocytes, lesions |
| 12 | 1,185 | wear, polyethylene, metal, metalonmetal, hip, ceramic, corrosion, crosslinked, total hip, head |
| 13 | 1,143 | cuff, rotator, rotator cuff, repair, tears, cuff repair, cuff tears, tear, arthroscopic, shoulder |
| 14 | 1,092 | lumbar, fusion, spondylolisthesis, spine, stenosis, interbody, interbody fusion, spinal stenosis, spinal, degenerative |
| 15 | 1,072 | pain, low pain, low, lbp, disability, chronic, work, questionnaire, physical, neck pain |
| 16 | 1,050 | hip, labral, hip arthroscopy, impingement, arthroscopy, femoroacetabular, femoroacetabular impingement, fai, hips, acetabular |
| 17 | 1,033 | trauma, injury, injuries, care, trauma centers, centers, trauma center, mortality, trauma patients, injured |
| 18 | 968 | hospitals, care, surgical, mortality, hospital, quality, medicare, surgery, readmission, procedures |
| 19 | 958 | hernia, mesh, hernia repair, repair, inguinal, inguinal hernia, hernias, ventral, ventral hernia, incisional |

The full per-topic word table (149 leaves) is available in `data/v1/topic_hierarchy.parquet`.

> **Qualitative shift vs. V1-S06.** The V1-S06 incoherent 4th-largest topic (t3 — "pancreatic + pancreatitis + burn + intestinal + molecular expression", n=3,933) is no longer present in its previous form. The closest V1-S06b analogue is t4 (n=1,493: "burn, injury, sepsis, intestinal, protein, reperfusion, mice, expression"), which reads as a tighter "burn-induced sepsis + visceral injury" molecular cluster — the pancreatic and pancreatitis terms have left this cluster (likely re-clustered into lower-rank leaves with the wider grid). t0 "hip arthroplasty" splits more cleanly into acetabular/femoral/stem terms; t12 isolates "polyethylene wear" as its own cluster (was inside t0 in V1-S06). These shifts are consistent with `nr_topics=150` + `min_cluster_size=90` producing finer, more coherent partitions, even though the overall noise fraction did not clear 20%.

### Hierarchy disclosure

`hierarchy_parquet` reports **mid=96, top=96** — both collapsed to the same value (V1-S06 had 58/58). Identical structural pattern: `build_hierarchy`'s union-find linkage exhausts merges before reaching the target counts of 20 / 6 even with more leaves. This is the same defect documented in the V1-S06 "Hierarchy degeneracy note" above and is **independent of leaf segmentation** — adding 60 leaves did not change the structural behaviour.

**Per plan-S06b §"Decisions baked in" §3, this is reported, not blocking** in V1-S06b. The mid/top-collapse bug is deferred to **plan-S06c**. Acceptance test #4 explicitly calls this out as a known gap.

### 20-topic spot-check task for Samer

`docs/gates/G1_spotcheck.csv` has been regenerated against the V1-S06b chosen-config artifacts (notebook `notebooks/04_topic_landscape.ipynb` re-executed in place 2026-05-21). Sampling rule (unchanged): 5 topics per size-quartile, top 20 by size within sample.

Samer's task (mirrors the V1-S06 step that produced the 17/2/1 result):
1. Open `docs/gates/G1_spotcheck.csv` — 20 leaf topics with `top_words` + 3 sample titles each.
2. Fill the `clinical_interpretation` column with a short clinical label per topic.
3. Re-count coherent / cross-domain-valid / incoherent.
4. Use the count + this section's quantitative table to choose one of the decision checkboxes below.

The previous V1-S06 fill is preserved in git history (was committed at HEAD); the file at HEAD has been overwritten by the notebook re-execute.

### Recommendation

**`RETUNE_EMBEDDING`** — per plan-S06b §6: *"If `constraints_unmet == True` [in phase 2]: end the session with a `RETUNE_EMBEDDING` recommendation surfaced to Samer. Do not silently push to V1-S07."*

**Quantitative case.**
- NPMI improved 0.224 → 0.238 (+0.014), well above the 0.18 *Arthroscopy* baseline.
- Leaf count moved 89 → 149, comfortably inside the 100–200 gate.
- C_v improved 0.756 → 0.782.
- **Noise floored at 24.0%** across the full 9-config widen grid. The monotone-decreasing trend with `min_cluster_size` (0.275 at MC=30 → 0.240 at MC=90) suggests further widening might shave another point or two — but only at the cost of n_leaf re-collapsing once `min_cluster_size` exceeds the `nr_topics=150` cap.
- The pattern "noise saturates ~24% as density estimation is tightened" is consistent with an embedding-level density floor: the 768-d mpnet representation places ~24% of biomedical abstracts in genuinely sparse regions of the embedding space that no HDBSCAN parameterization can recover.

**Suggested escape-hatch path** (plan-S05 → plan-S06 re-fit):
1. Re-run V1-S05 with `BAAI/bge-large-en-v1.5` (1024-d, was Δ=+0.011 kNN over mpnet in the V1-S05 bake-off). The bake-off Δ was small, but the noise problem here is structural and may be more sensitive to embedding dimensionality / pre-training corpus than the kNN bake-off captured.
2. Re-fit V1-S06 using the V1-S06b chosen config (`min_cluster_size=90, min_samples=10, nr_topics=150`) as the starting point, since that combination was clearly the best across phase 2.
3. If `bge-large` does not push noise below 20%, the next lever is the `hdbscan_cluster_selection_method` swap (`eom → leaf`) — not pursued in V1-S06b because it was outside the plan-S06b §"Decisions baked in" lever set.

**Qualitative caveat.** The top-20 word lists read substantively cleaner than V1-S06 (purified hip arthroplasty, isolated polyethylene wear, tighter burn-sepsis cluster — see the qualitative-shift note above). Samer's spot-check fill-in may overrule the recommendation if 18+/20 read coherent and the noise miss is judged acceptable for V1-S07 (i.e. accept-with-caveat `PROCEED`). The quantitative case for `RETUNE_EMBEDDING` is conservative; the qualitative top-of-stack is the strongest evidence for `PROCEED`.

### Decision

- [x] **PROCEED to V1-S07 (override of plan-S06b §6 procedural mandate)**
- [ ] RE-RETUNE (further clustering work)
- [ ] RETUNE_EMBEDDING (re-run V1-S05 with bge-large-en-v1.5)

**Rationale for override.** Gate G1's purpose is to test clinical interpretability of the topic landscape. The V1-S06b spot-check (19 coherent / 1 cross-domain-valid / 0 incoherent) is a clear pass on that criterion. NPMI 0.238 and leaf count 149 also cleared. The remaining noise miss (24.0% vs 20%) appears embedding-structural — noise saturates across all 9 phase-2 configs in a narrow band around 24–31%, consistent with ~24% of abstracts sitting in genuinely sparse mpnet regions. RETUNE_EMBEDDING's expected gain is bounded by the V1-S05 bake-off Δ (+0.011 kNN, mpnet → bge-large), which is small relative to the cost of re-running V1-S05 + V1-S06.

**Caveats carried forward to V1-S07.**
1. **Noise topic (-1) handling.** ~24% of the corpus is in topic -1. Downstream F1/F2/F3 must either include -1 documents as an explicit "uncategorized" bucket or document their exclusion. Do not silently drop them.
2. **Topic 4 (translational animal/injury research, n=1,493).** Coherent but structurally distinct from clinical topics. Tag for either separate handling or exclusion in F1/F2/F3.
3. **Hierarchy collapse (mid=top=96).** Known defect, deferred to plan-S06c. Downstream work that depends on the mid/top hierarchy (if any) is blocked until plan-S06c.

Signed: _Samer Salman_  Date: 2026-05-21
