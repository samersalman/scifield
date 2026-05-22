# V1-S06b — BERTopic clustering retune (post-G1 `RETUNE_CLUSTERING`)

**Date:** 2026-05-21  **Phase:** 2 (Thematic backbone)  **Master plan:** `plans/Session-Objectives-MAP.md` § V1-S06  **Predecessor:** `plans/plan-S06.md`  **Gate:** `docs/gates/G1_topic_interpretability.md` (decision = `RETUNE_CLUSTERING`, 2026-05-21)

---

## Context

V1-S06 executed the BERTopic pipeline end-to-end on 89,230 deduped abstracts and produced the artifacts in `data/v1/topics.parquet`, `topic_hierarchy.parquet`, `topic_sweep.parquet`, and `models/v1/bertopic_v1/`. Coherence cleared the *Arthroscopy*-anchored baseline (NPMI 0.224 ≥ 0.18) and 17/20 spot-check topics read clinically coherent to Samer. However, three of the numerical gates in `docs/gates/G1_topic_interpretability.md` failed:

| Gate | Threshold | V1-S06 result |
|---|---|---|
| Noise fraction | < 20% | 29.5% |
| Leaf topic count | 100 ≤ n ≤ 200 | 89 |
| Mid / top hierarchy | ~20 mid / 5–7 top | both collapsed to 58 |

Samer's gate review (2026-05-21) declared `RETUNE_CLUSTERING` (not `RETUNE_EMBEDDING`) on the strength of the NPMI and spot-check signal: the embedding is healthy, the problem is post-embedding density estimation and merging. This session's job is to **re-tune HDBSCAN + post-hoc reduction** until the noise / leaf gates clear, **without** touching `topics.py` / `coherence.py` / `dedup.py` / `cli.py` and **without** re-embedding.

**Why this matters.** V1-S07 (epistemic backbone) is blocked behind G1. The hierarchy bug (mid/top collapsed to leaf count) is a separate structural defect in `build_hierarchy` that Samer has chosen to defer to plan-S06c; V1-S06b is scoped narrowly to the clustering retune so the gate can clear (or escalate cleanly to `RETUNE_EMBEDDING`) in one session.

**Decisions baked in from the G1 review (2026-05-21):**

1. **Overwrite the V1-S06 artifact paths**: `data/v1/topics.parquet`, `data/v1/topic_hierarchy.parquet`, `data/v1/topic_sweep.parquet`, `models/v1/bertopic_v1/`. The previous run is recoverable via git history + the pinned `config_hash=5a1ed67d…` / `git_sha=eebe89e7…` in the V1-S06 sidecars.
2. **Append-only** to `docs/gates/G1_topic_interpretability.md` — add a `## V1-S06b retune results` section; do **not** create a new `G1b_topic_interpretability.md`.
3. **`build_hierarchy` mid/top-collapse bug is out of scope.** Hierarchy counts in V1-S06b are reported, not blocking. Defer to plan-S06c.
4. **Auto-widen in one session.** If the focused 2×2 mini-sweep fails the noise/leaf gates, Claude proceeds to the 9-config widen grid in the same session without surfacing back to Samer in between. The gate decision (PROCEED / RE-RETUNE / RETUNE_EMBEDDING) is still Samer's at the end.
5. **Encoder is fixed** at the V1-S05-chosen `all-mpnet-base-v2` 768-d fp16 embedding. RETUNE_EMBEDDING is an *escape hatch* if both phases fail, not a lever inside V1-S06b.

**User scope directive:** *"STAY IN SCOPE."* No new modules, no new code, no refactors. Reuse the V1-S06 implementation as-is. The only writeables are configs, configs, data outputs (overwritten), the notebook (re-executed), and two append sections in two docs.

---

## Scope

### In scope
- `conf/thematic/topics_retune_phase1.yaml` — new YAML, 4-row 2×2 mini-sweep grid.
- `conf/thematic/topics_retune_phase2.yaml` — new YAML, 9-config widen grid (**only written if phase 1 fails the noise + leaf gates**).
- `data/v1/topics.parquet`, `topic_hierarchy.parquet`, `topic_sweep.parquet` (overwritten) + their `.run.json` sidecars (auto-stamped with new `config_hash` + `git_sha`).
- `models/v1/bertopic_v1/` (overwritten) + `bertopic_v1.run.json`.
- `notebooks/04_topic_landscape.ipynb` re-executed against the overwritten V1-S06b artifacts (via `jupyter nbconvert --to notebook --execute --inplace`).
- `docs/figures/topic_landscape.html`, `topic_share_by_year.png`, `topic_share_by_journal_year.png` (overwritten by the notebook re-execute).
- `docs/gates/G1_spotcheck.csv` (overwritten by the notebook; Samer re-fills `clinical_interpretation` in the gate-decision step).
- `docs/gates/G1_topic_interpretability.md` — append a `## V1-S06b retune results` section; do not modify the existing V1-S06 sections.
- `docs/phases/2_thematic.md` — append a `## V1-S06b results (2026-05-21)` sub-section mirroring the V1-S06 summary block.

### Out of scope (defer)
- Any code change to `src/scifield/thematic/topics.py`, `coherence.py`, `dedup.py`, or `src/scifield/cli.py`. `TopicConfig` already exposes `hdbscan_min_samples: int | None = None` and `nr_topics: str | int = "auto"`; the CLI already accepts `--config <path>`.
- Re-embedding (V1-S05) — RETUNE_EMBEDDING is the escape hatch in §9, not a lever inside this plan.
- FAISS rebuild — same.
- The `build_hierarchy` mid/top-collapse bug → plan-S06c.
- V1-S07 scaffolding — blocked behind this gate.
- v2 corpus, OCTIS, Brev. No Brev launches this session.

---

## Decisions baked in from G1 review

(Recap, since these are load-bearing for the gates below.)

| # | Decision | Source |
|---|---|---|
| 1 | Overwrite V1-S06 artifact paths; recover prior run via git + sidecar hashes. | Samer, G1 review 2026-05-21 |
| 2 | Append `## V1-S06b retune results` to existing G1 doc; no new G1b file. | Samer, G1 review 2026-05-21 |
| 3 | `build_hierarchy` bug is out of scope for V1-S06b; defer to plan-S06c. | Samer, G1 review 2026-05-21 |
| 4 | Auto-widen 2×2 → 9-config in one session if phase 1 fails noise/leaf gates. | Samer, G1 review 2026-05-21 |
| 5 | Encoder fixed at V1-S05's `all-mpnet-base-v2`; RETUNE_EMBEDDING is escape hatch only. | G1 doc § Decision |

---

## Step-by-step plan

### 1. Pre-flight sanity (~5 min)

Confirm the V1-S06 baseline is intact before overwriting:

```bash
# V1-S06 sidecar pins
uv run python -c "import json; j=json.load(open('data/v1/topics.parquet.run.json')); \
    print('config_hash:', j['config_hash']); print('git_sha:', j['git_sha'])"
# expect: 5a1ed67d1eb42cf382c2998814774daa58a2bc2c7877cdf1f9e3377a9af0c815 / eebe89e76fc7835fd3e7b02ab8ddb92920b150c7

# Existing test suite still green
uv run pytest -q                                  # expect 89 passed, 1 skipped

# Embedding input still byte-identical to V1-S06
uv run python -c "import hashlib, pathlib; \
    print(hashlib.sha256(pathlib.Path('data/v1/embeddings.parquet').read_bytes()).hexdigest())"
# Cross-check against j['input_hashes']['data/v1/embeddings.parquet']
```

If any of those drift, **stop and surface** — the V1-S06 baseline has changed underneath us and the retune isn't meaningful.

### 2. Write `conf/thematic/topics_retune_phase1.yaml`

Copy `conf/thematic/topics.yaml`. Keep `input:`, `output:`, `hierarchy:`, `coherence:`, `software:`, `sweep.selector`, and `sweep.constraints` **identical** to V1-S06 (so the gate criteria are unchanged). Override:

- `defaults_config.hdbscan_min_cluster_size: 80` (already the V1-S06 chosen value — keep)
- `defaults_config.hdbscan_min_samples: 10` (was `null` in V1-S06)
- `defaults_config.nr_topics: 150` (was `auto`)
- Replace `sweep.grid` with the 4-row 2×2:

```yaml
sweep:
  enabled: true
  grid:
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 80, hdbscan_min_samples: 10,   nr_topics: 150}
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 80, hdbscan_min_samples: 10,   nr_topics: auto}
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 80, hdbscan_min_samples: null, nr_topics: 150}
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 80, hdbscan_min_samples: null, nr_topics: auto}
  selector: npmi_top10
  constraints:
    n_leaf_topics_min: 100
    n_leaf_topics_max: 200
    noise_fraction_max: 0.20
```

Header comment in the YAML should reference `plans/plan-S06b.md` and the G1 retune lever order.

### 3. Run phase 1

```bash
uv run scifield topics --config conf/thematic/topics_retune_phase1.yaml
```

Expected wall: ~7–8 min (4 configs ≈ 100 s each + final fit). All four V1-S06 sidecars are overwritten with a new `config_hash` / `git_sha` and the new sweep table.

Inspect the sidecar:

```bash
uv run python -c "import json; j=json.load(open('data/v1/topics.parquet.run.json')); \
    print('constraints_unmet:', j['config'].get('constraints_unmet')); \
    print('noise:', j['config'].get('noise_fraction')); \
    print('n_leaf:', j['config'].get('n_leaf_topics')); \
    print('npmi:', j['config'].get('npmi_top10'))"
```

### 4. Decision branch (in-session, not a human gate)

- **If `constraints_unmet == False`** (noise < 20% AND 100 ≤ n_leaf ≤ 200): **stop the retune**. Skip to step 7 (re-execute notebook + write append sections + propose `PROCEED`).
- **If `constraints_unmet == True`**: proceed to step 5 (widen grid).

Record in plan execution log which branch was taken and why (which constraint failed by how much).

### 5. Write `conf/thematic/topics_retune_phase2.yaml` (only if phase 1 failed)

Cartesian 9-grid at `n_neighbors=15`, `nr_topics=150` pinned:

```yaml
sweep:
  enabled: true
  grid:
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 30, hdbscan_min_samples: 5,  nr_topics: 150}
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 30, hdbscan_min_samples: 10, nr_topics: 150}
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 30, hdbscan_min_samples: 20, nr_topics: 150}
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 60, hdbscan_min_samples: 5,  nr_topics: 150}
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 60, hdbscan_min_samples: 10, nr_topics: 150}
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 60, hdbscan_min_samples: 20, nr_topics: 150}
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 90, hdbscan_min_samples: 5,  nr_topics: 150}
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 90, hdbscan_min_samples: 10, nr_topics: 150}
    - {umap_n_neighbors: 15, hdbscan_min_cluster_size: 90, hdbscan_min_samples: 20, nr_topics: 150}
  selector: npmi_top10
  constraints:
    n_leaf_topics_min: 100
    n_leaf_topics_max: 200
    noise_fraction_max: 0.20
```

**Pin rationale:** phase 1 will have already answered `150 vs auto`; phase 2 isolates HDBSCAN density. **Override if** phase 1 surfaced `auto > 150` on the constraint-satisfying tiebreak — in that case, swap `nr_topics: 150` → `nr_topics: auto` across the grid and document the swap in the gate append section.

`defaults_config` for phase 2: copy the *best* `defaults_config` produced by phase 1 (best per the `npmi_top10` selector among configs that came closest to the constraints), so that the final-fit step after the sweep uses a sensible default.

### 6. Run phase 2 (only if phase 1 failed)

```bash
uv run scifield topics --config conf/thematic/topics_retune_phase2.yaml
```

Expected wall: ~16–18 min. Same sidecar inspection as step 3. Same decision rule as step 4:

- **If `constraints_unmet == False`**: proceed to step 7 with `PROCEED` recommendation.
- **If `constraints_unmet == True`**: end the session with a `RETUNE_EMBEDDING` recommendation surfaced to Samer. Do **not** silently push to V1-S07. Notebook re-execute (step 7) still runs so the figures and spot-check CSV reflect whichever sweep's chosen config was last fit, but the gate doc append explicitly recommends RETUNE_EMBEDDING.

### 7. Re-execute the notebook

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/04_topic_landscape.ipynb
```

The notebook derives all paths from `REPO_ROOT` and reads the overwritten V1-S06b artifacts. Output overwrites:
- `docs/figures/topic_landscape.html`
- `docs/figures/topic_share_by_year.png`
- `docs/figures/topic_share_by_journal_year.png`
- `docs/gates/G1_spotcheck.csv` (the V1-S06 fill-in is preserved in git history; Samer re-fills against the new top-words + representative abstracts in the gate-decision step)

### 8. Append `## V1-S06b retune results` to `docs/gates/G1_topic_interpretability.md`

**Append only.** Do not touch the existing V1-S06 sections. The new section must include:

1. **Pass-criteria table** — same columns as the V1-S06 version, rows = the V1-S06b chosen config. Columns: gate, threshold, V1-S06 result, V1-S06b result, pass/fail.
2. **Phase summary** — which phase(s) ran, wall-time, configs evaluated, best constraint-satisfying NPMI.
3. **Chosen-config table** — full `TopicConfig` fields for the V1-S06b winner.
4. **Top-20 word lists** — re-rendered from the new model.
5. **20-topic spot-check task for Samer** — new `G1_spotcheck.csv` is in place; Samer fills `clinical_interpretation` and re-counts coherent/incoherent.
6. **Hierarchy disclosure** — explicit note that mid/top counts are reported but **not blocking** in V1-S06b, with a pointer to plan-S06c.
7. **Recommendation block** — one of:
   - `PROCEED` (noise + leaf cleared, regardless of hierarchy)
   - `RE-RETUNE` (only if a third lever Samer wants is surfaced — e.g., bge-large hybrid)
   - `RETUNE_EMBEDDING` (both phases failed noise/leaf)
8. **New decision checkboxes** for Samer, mirroring the V1-S06 block:
   ```
   - [ ] PROCEED to V1-S07
   - [ ] RE-RETUNE (further clustering work)
   - [ ] RETUNE_EMBEDDING (re-run V1-S05 with bge-large-en-v1.5)
   ```

### 9. Append `## V1-S06b results (2026-05-21)` sub-section to `docs/phases/2_thematic.md`

Mirror the V1-S06 summary block: chosen config, hierarchy counts (with the "reported, not blocking" caveat), coherence numbers, runtime, sweep table summary (one row per evaluated config), and deviations from plan (if any — e.g., phase 1 sufficed, or phase 2 also failed → escalation note).

### 10. Final smoke

```bash
uv run pytest -q                  # expect unchanged: 89 passed, 1 skipped
uv run pre-commit run --all-files
git status                        # confirm only the files in §"Deliverable file paths" changed
```

---

## Deliverable file paths (must exist before declaring done)

- `conf/thematic/topics_retune_phase1.yaml` — **always**.
- `conf/thematic/topics_retune_phase2.yaml` — **only if phase 1 failed the noise/leaf gates**.
- `data/v1/topics.parquet` (overwritten) + `data/v1/topics.parquet.run.json` (new `config_hash` ≠ `5a1ed67d…`).
- `data/v1/topic_hierarchy.parquet` (overwritten) + `.run.json`.
- `data/v1/topic_sweep.parquet` (overwritten, contains phase-1 OR phase-2 grid rows) + `.run.json`.
- `models/v1/bertopic_v1/` (overwritten) + `bertopic_v1.run.json`.
- `notebooks/04_topic_landscape.ipynb` (re-executed in place).
- `docs/figures/topic_landscape.html`, `docs/figures/topic_share_by_year.png`, `docs/figures/topic_share_by_journal_year.png` (overwritten by notebook).
- `docs/gates/G1_spotcheck.csv` (overwritten by notebook; Samer fills `clinical_interpretation` post-session).
- `docs/gates/G1_topic_interpretability.md` (append-only `## V1-S06b retune results` section).
- `docs/phases/2_thematic.md` (append-only `## V1-S06b results (2026-05-21)` sub-section).

---

## Acceptance tests (run before declaring done)

1. **NPMI ≥ 0.18** on the V1-S06b chosen config (sanity — V1-S06 was already 0.224; the retune should not regress coherence).
2. **Noise fraction < 20%** on the chosen config — **the primary V1-S06b gate**.
3. **100 ≤ n_leaf_topics ≤ 200** on the chosen config — **the secondary V1-S06b gate**.
4. **Hierarchy mid≈20 / top∈[5,7]** — *reported, not blocking* (`build_hierarchy` bug deferred to plan-S06c). The append section must explicitly call out this gap.
5. `notebooks/04_topic_landscape.ipynb` executes end-to-end (intertopic map + temporal heatmap + per-journal-per-year all render).
6. `uv run pytest -q` = unchanged baseline (89 passed, 1 skipped). **No new tests required** — no new code.
7. `uv run pre-commit run --all-files` clean.
8. Every overwritten parquet has a sidecar JSON whose `git_sha`, `config_hash`, and `input_hashes` are non-null. `config_hash` ≠ `5a1ed67d1eb42cf382c2998814774daa58a2bc2c7877cdf1f9e3377a9af0c815`.
9. `input_hashes['data/v1/embeddings.parquet']` in the new sidecar **matches** the V1-S06 sidecar (embeddings are not re-run).
10. `git status` shows only the files in the §Deliverable list as changed.
11. `constraints_unmet=true` recorded in sidecars *iff* phase 2 also failed; in that case the gate-doc append must recommend `RETUNE_EMBEDDING`.

The 20-topic spot-check fill-in is **not a Claude step** — Samer fills `G1_spotcheck.csv:clinical_interpretation` after the session.

---

## Reuse / cross-cutting (no new code; every module reused as-is)

| Need | Existing function / asset | File |
|---|---|---|
| Dedup view + V1-S05 carryover integrity check | `ensure_papers_distinct_view`, `integrity_check_v1_carryover` | `src/scifield/thematic/dedup.py` |
| Load deduped embeddings (fp16 → fp32) | `load_deduped_embeddings` | `src/scifield/thematic/dedup.py` |
| Topic pipeline + sweep harness | `TopicConfig` (already exposes `hdbscan_min_samples`, `nr_topics`), `fit_topics`, `build_hierarchy`, `sweep` | `src/scifield/thematic/topics.py` |
| NPMI + C_v coherence | `compute_coherence`, `tokenise_for_coherence` | `src/scifield/thematic/coherence.py` |
| CLI entrypoint | `scifield topics --config <path>` (already supports override) | `src/scifield/cli.py` |
| Reproducibility sidecar | `record_run` (auto-stamps `git_sha`, `config_hash`, `input_hashes`, `software_versions`) | `src/scifield/repro/__init__.py` |
| Spot-check + figures | notebook cells 1–11 (paths via `REPO_ROOT`) | `notebooks/04_topic_landscape.ipynb` |
| YAML template | full V1-S06 config to copy + override | `conf/thematic/topics.yaml` |

Critical read-only files for the implementer (do **not** modify):

- `conf/thematic/topics.yaml` — template only; copy and edit `defaults_config` + `sweep.grid`.
- `src/scifield/cli.py` (`topics` subcommand, lines ~620–940) — reads `defaults_config`, `sweep.grid`, `sweep.selector`, `sweep.constraints`; accepts `--config <path>`.
- `src/scifield/thematic/topics.py` — `TopicConfig` frozen dataclass, already exposes all needed levers.
- `notebooks/04_topic_landscape.ipynb` — re-execute only; do not edit cells.
- `docs/gates/G1_topic_interpretability.md` — **append only**; do not modify the existing V1-S06 sections.

---

## Reproducibility / cross-cutting

- `defaults_config.random_state=42`, UMAP `random_state=42`, HDBSCAN `core_dist_n_jobs=1` — inherited from V1-S06; do not change.
- `record_run` captures: `git_sha`, `config_hash` (over the *full* chosen TopicConfig + paths), `input_hashes` (embeddings parquet + `papers.duckdb`), `software_versions` (bertopic, umap-learn, hdbscan, gensim, numpy). The CLI does this automatically.
- No Brev launches. Note in `docs/operations/brev.md`: "V1-S06b (2026-05-21): no Brev needed; ran locally on Mac CPU." *(Append, do not overwrite existing notes.)*
- UMAP non-determinism: thread count + library versions captured in sidecar (unchanged from V1-S06).

---

## Risks / stop conditions

- **Phase 1 chosen config picks `auto > 150` on the constraint-satisfying tiebreak.** Swap `nr_topics` in the phase-2 grid and document the swap in the gate append. Do not silently keep `150`.
- **Phase 2 still fails noise < 20%.** Surface, recommend `RETUNE_EMBEDDING` with `bge-large-en-v1.5` (per V1-S05 bake-off Δ=+0.011 kNN over mpnet), stop. Write the recommendation into the G1 append and end the session — do not silently push to V1-S07.
- **A single config wall-time exceeds 15 min** (unlikely at this corpus scale): document and proceed with whatever ran; do not retry. If the *full* phase 2 wall exceeds 30 min, abort, reduce to a 3-config diagonal (`min_cluster_size ∈ {30, 60, 90}` at `min_samples=10`), document.
- **UMAP non-determinism across thread counts.** Already captured in sidecar; if topic *content* differs across reviewer reruns but constraints still pass, that is acceptable. If constraints flip, document and re-pin thread count in the YAML's `software:` block.
- **`nr_topics=150` pushes too many small clusters into named topics → n_leaf >> 200.** The constraint check will catch this (`constraints_unmet=true`) and the auto-widen branch handles it.
- **`build_hierarchy` mid/top still collapses to n_leaf** (the V1-S06 bug). Expected; reported in the append as "deferred to plan-S06c". Does **not** block the gate.
- **Sidecar `constraints_unmet` field is missing on the run.** Means the CLI did not emit it on the chosen-config sidecar; treat as a code defect, surface to Samer, do not silently mark passed.

The V1-S06b session ends with the new G1 retune-results section committed and Samer reviewing it. **The gate decision is Samer's, not Claude's.**

---

## Verification (end-to-end smoke before declaring done)

From the repo root:

```bash
# Tests + lint
uv run pytest -q
uv run pre-commit run --all-files

# Sidecar pins are new (config_hash != V1-S06's 5a1ed67d…)
ls -la data/v1/topics.parquet.run.json
uv run python -c "import json; j=json.load(open('data/v1/topics.parquet.run.json')); \
    print('config_hash:', j['config_hash']); \
    print('constraints_unmet:', j['config'].get('constraints_unmet')); \
    print('noise:', j['config'].get('noise_fraction')); \
    print('n_leaf:', j['config'].get('n_leaf_topics')); \
    print('npmi:', j['config'].get('npmi_top10'))"

# Embeddings input unchanged from V1-S06
uv run python -c "import json; j=json.load(open('data/v1/topics.parquet.run.json')); \
    print('emb_hash:', j['input_hashes'].get('data/v1/embeddings.parquet'))"

# Diff scope
git status                        # only files in §Deliverable should appear
```

Then open `docs/gates/G1_topic_interpretability.md` and confirm the new `## V1-S06b retune results` section renders below the existing V1-S06 content. **Hand off to Samer for the 20-topic spot-check fill-in and gate decision — that step is not Claude's.**
