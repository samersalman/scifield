# V1-S07 — Epistemic schema, prompt engineering, Excel labeling workflow, OSF pre-registration #1

**Date:** 2026-05-22
**Author plan ref:** `plans/Session-Objectives-MAP.md` § V1-S07
**Master plan ref:** `plan/scifield_plan.md` §5 Phase 3
**Gate clearance:** G1 PROCEED (override) — `docs/gates/G1_topic_interpretability.md`

---

## Context

V1-S07 is the pre-flight session for the hardest phase of the project (epistemic extraction). The goal is to build **every artifact V1-S08 will need** — schema, prompt, sampling, labeling assets, κ module, 50-abstract pilot — and **submit OSF pre-registration #1** before any batch run is launched.

Critical scope discipline:
- We do NOT run the full 200k batch (that is V1-S08).
- We do NOT execute the 500-abstract hand-labeling sprint (that is V1-S08).
- We do NOT compute κ on real labels (V1-S08), do Trialstreamer/RobotReviewer triangulation (V1-S09), or use any extracted field downstream (Phase 6).

**Key user decisions confirmed:**
1. **Labeling tool = Excel workbook**, not Streamlit. Rationale: co-rater (likely a non-coder) gets a familiar `.xlsx` with data-validated enum cells; round-trip via small `export_labels` / `import_labels` CLI helpers; no UI to maintain.
2. **Pilot extractor = Claude Code subprocess** (`claude --print`), not Anthropic API. No API key spend in this session. V1-S08 can later flip to Batch API or stay on Claude Code; the `extract_one()` interface is model-agnostic.
3. **Stratification = journal × era, topic balance-checked.** 41 cells (10 journals × 4 eras), all ≥689 abstract-bearing papers; proportional allocation to n=500; verify ≥80/149 topics represented.

---

## Deliverables (exact paths)

**Code (new):**
- `src/scifield/epistemic/schema.py` — Pydantic `EpistemicLabel` (6 fields) + `EpistemicExtraction` wrapper; `LABEL_SCHEMA_VERSION = "v0.1"`
- `src/scifield/epistemic/prompt.py` — versioned system prompt `SYSTEM_PROMPT_V0_1` + 5–8 few-shot examples + `build_prompt(abstract: str) -> str`
- `src/scifield/epistemic/kappa.py` — `cohens_kappa(...)`, `krippendorffs_alpha(...)`, per-field summary table
- `src/scifield/epistemic/sampling.py` — `stratified_sample(con, cfg) -> pd.DataFrame`; journal × era proportional allocation with largest-remainder rounding; seeded
- `src/scifield/epistemic/labeling.py` — `export_to_xlsx(...)` (with `openpyxl.DataValidation` enum cells) + `import_from_xlsx(...)` Pydantic-validated; long-form parquet append
- `src/scifield/epistemic/extract.py` — `extract_one(abstract) -> EpistemicExtraction` via `subprocess.run(["claude","--print", ...])`, JSON parse + Pydantic validate, 1 retry on parse failure
- `src/scifield/epistemic/pilot.py` — `run_pilot(sample_path, n=50, prompt_version="v0.1")`; writes `data/v1/epistemic_pilot.parquet` + failures parquet
- `src/scifield/epistemic/__init__.py` — replace V1-S01 stub; re-export public API

**Code (modified):**
- `src/scifield/cli.py` — add `epistemic` Typer sub-app with commands: `sample`, `export-labels`, `import-labels`, `pilot`
- `pyproject.toml` — add `pydantic>=2.0`, `openpyxl>=3.1`, `krippendorff>=0.6.0`

**Config:**
- `conf/epistemic/v1.yaml` — corpus DB path, topics parquet path, sample output path, pilot output path, n_sample=500, n_pilot=50, seed=20260522, model_version, prompt_version

**Data artifacts (with `.run.json` sidecars):**
- `data/v1/handlabel_sample.parquet` — 500 rows: `pmid, journal, year, era, topic_id, title, abstract`
- `data/v1/epistemic_pilot.parquet` — 50 rows from pilot extraction
- `data/v1/epistemic_pilot_failed.parquet` — any retry-exhausted rows
- `data/v1/labels_<rater>.xlsx` — template per rater (generated, not committed)
- `data/v1/epistemic_handlabel.parquet` — appended by `import-labels` (empty for now; populated in V1-S08)

**Notebook:**
- `notebooks/05_handlabel_sampling.ipynb` — drives sampling, renders per-cell counts + topic coverage tables

**Docs:**
- `docs/preregistrations/PR1_epistemic_extraction.md` — full OSF pre-registration draft with hypotheses, operational definitions, sampling plan, model + prompt v0.1, κ targets, pivot conditions, OSF DOI placeholder
- `docs/phases/epistemic.md` — prompt iteration log (≥3 iterations: v0.1.0 baseline → revisions after pilot)

**Tests (new):**
- `tests/test_epistemic_schema.py`, `tests/test_epistemic_prompt.py`, `tests/test_epistemic_kappa.py`, `tests/test_epistemic_sampling.py`, `tests/test_epistemic_labeling.py`, `tests/test_epistemic_extract.py`

---

## Implementation phases

### A. Schema, prompt, κ (module foundations)

Pydantic schema mirrors the master plan §5 Phase 3 fields exactly:

| Field | Type | Enum / range |
|---|---|---|
| `study_design` | str enum | `RCT`, `cohort`, `case_control`, `case_series`, `review`, `other` |
| `sample_size` | int \| None | ≥1 or null if not stated |
| `has_control` | bool \| None | null if not applicable (e.g., review) |
| `effect_direction` | str enum \| None | `positive`, `null`, `negative`, `mixed`, `na` |
| `statistical_claim_present` | bool | |
| `coi_disclosed_in_abstract` | bool | |

The prompt module ships `v0.1` prompt + few-shots spanning RCT / observational cohort / case series / review / negative trial. `build_prompt(abstract)` returns the system+user composition Claude Code receives via stdin. `kappa.py` wraps `sklearn.metrics.cohen_kappa_score` and `krippendorff.alpha`; returns a per-field DataFrame with both metrics and N.

Tests: round-trip JSON, enum rejection, prompt contains required directives, κ/α match hand-computed values on synthetic 10-pair data.

### B. Stratified sampling

`sampling.py` reads `papers_distinct` filtered to `has_abstract = TRUE` (89,244-paper pool), left-joins `data/v1/topics.parquet` (papers missing topic kept with `topic_id = NULL`), and assigns each paper to one of 4 era buckets (`<2000`, `2000-2009`, `2010-2019`, `2020+`). Proportional allocation across 41 (journal, era) cells targeting 500 rows total, with largest-remainder rounding to hit exactly 500. Within each cell, uniform random sample with `seed = cfg.seed`. Returns columns: `pmid, journal, year, era, topic_id, title, abstract`.

Post-sample assertions:
- Exactly 500 rows
- All 41 (journal × era) cells represented (or recorded as exhausted)
- ≥80 of 149 topics covered (spot-check for thematic diversity)
- Sample is deterministic under fixed seed

The notebook runs sampling, writes parquet + sidecar JSON via `scifield.repro.record_run`, and renders three tables: per-cell counts, era counts, topic-coverage histogram.

Reuses: `scifield.thematic.dedup.ensure_papers_distinct_view` (idempotently confirms one-row-per-PMID before sampling).

### C. Excel labeling workflow

`labeling.export_to_xlsx(sample_path, out_path, rater_name)`:
- Creates a workbook with two sheets: **Instructions** (operational definitions + how to fill) and **Labels** (one row per abstract).
- **Labels** columns: `pmid, journal, year, title, abstract` (locked) + 6 empty rater-fill columns.
- Each rater-fill column gets `openpyxl.worksheet.datavalidation.DataValidation` restricting input to the enum values from the Pydantic schema (study_design dropdown, effect_direction dropdown, booleans as `TRUE`/`FALSE` dropdown).
- Abstract column wrapped, width 80.

`labeling.import_from_xlsx(xlsx_path, rater_name)`:
- Reads workbook, validates each row through `EpistemicLabel` (Pydantic raises on bad enum; row-level errors collected).
- Appends to `data/v1/epistemic_handlabel.parquet` in long form: `pmid, rater, field, value, imported_at`.
- Idempotent: re-importing same `(pmid, rater)` overwrites.

CLI:
- `scifield epistemic sample` → runs sampling, writes `handlabel_sample.parquet`
- `scifield epistemic export-labels --rater samer` → `data/v1/labels_samer.xlsx`
- `scifield epistemic import-labels --rater samer --file path.xlsx` → appends to handlabel parquet

Tests round-trip a synthetic 5-row sample through export → programmatic fill → import; assert Pydantic rejects bad enums; assert re-import is idempotent.

### D. 50-abstract pilot via Claude Code subprocess

`extract.extract_one(abstract, claude_cmd=("claude","--print"))`:
- Builds prompt via `prompt.build_prompt(abstract)`.
- Runs subprocess, captures stdout, strips fenced JSON if present, parses, validates via `EpistemicExtraction`.
- On JSON parse failure: 1 retry with stricter "respond with valid JSON only — no prose" suffix.
- On persistent failure: returns sentinel + error message for failures parquet.

`pilot.run_pilot(sample_path, n=50, prompt_version="v0.1")`:
- Reads `handlabel_sample.parquet`, takes first n by deterministic order (sorted by pmid for reproducibility — sampling already random).
- Iterates `extract_one` per abstract, prints per-row outcome (`ok` / `parse-fail-retry-ok` / `failed`).
- Writes `data/v1/epistemic_pilot.parquet` (successful) + `data/v1/epistemic_pilot_failed.parquet` (failures) + sidecar JSON for each.

CLI: `scifield epistemic pilot --n 50` (config defaults to 50).

Tests mock `subprocess.run` to return canned JSON; assert Pydantic validates; assert retry path fires on bad JSON; assert failure recorded.

**Prompt iteration log** (`docs/phases/epistemic.md`) — after pilot run, document at least 3 iterations:
- v0.1.0 — initial prompt
- v0.1.1 — adjustments after observing parse failures or systematic miscategorization
- v0.1.2 — adjustments after edge cases (e.g., reviews with no effect direction, case series with implicit sample size)

Bump `prompt.SYSTEM_PROMPT_V0_1` constant only after iteration is complete; final version that ships into the pilot parquet must match the `prompt_version` field stored on every row.

### E. OSF pre-registration #1

Draft `docs/preregistrations/PR1_epistemic_extraction.md` with these sections (per OSF pre-registration template + plan §5 Phase 3 + §6 risk row 2):

1. **Title / authors / date** — front matter, with OSF DOI placeholder to fill after submission
2. **Background and rationale** — 1 paragraph linking to SciField plan & V1-S06 topic landscape
3. **Hypotheses** — H1: epistemic-quality features are extractable from PubMed abstracts with κ exceeding pre-specified thresholds; H2: LLM-vs-human agreement is within 10% of human-vs-human agreement
4. **Operational definitions** — one paragraph per field (study_design, sample_size, has_control, effect_direction, statistical_claim_present, coi_disclosed_in_abstract) with explicit edge-case rules
5. **Sampling plan** — 500 abstracts; eligibility (`has_abstract = TRUE` from `papers_distinct`); journal × era stratification (41 cells); proportional allocation; seed 20260522; sample written to `data/v1/handlabel_sample.parquet` at commit SHA `<filled at submission>`
6. **Hand-labeling protocol** — 2 raters, independent labeling via Excel workbook, arbitration meeting for disagreements, final labels at `data/v1/epistemic_handlabel_final.parquet` (V1-S08)
7. **LLM extraction protocol** — model + version (Claude via Claude Code subprocess, model id captured per-call); prompt v0.1 (full text in appendix); JSON output validated via Pydantic schema v0.1
8. **Primary analysis** — Cohen's κ for inter-rater (per field) and LLM-vs-arbitrated (per field); Krippendorff's α as secondary; confusion matrices
9. **Pre-registered pass/fail criteria** — κ ≥ 0.7 (study_design), ≥ 0.8 (has_control), ≥ 0.6 (effect_direction); LLM-vs-human within 10pp of inter-rater across all fields
10. **Pivot conditions** — if κ targets miss: (a) fine-tune BERT classifier on the 500 labels (insert V1-S09b), OR (b) drop F1 from manuscript and proceed with F2+F3
11. **Data and code availability** — GitHub URL + commit SHA at submission
12. **Appendix A** — full system prompt + few-shot examples
13. **Appendix B** — Pydantic schema source

Workflow: Claude drafts the markdown; user uploads to osf.io; user pastes the resulting OSF DOI/URL into the front matter; the file is then committed with the link populated. **No V1-S08 work begins until that link is in the file.**

---

## Acceptance tests (run in order)

1. `uv sync` — picks up new deps cleanly
2. `uv run pytest tests/test_epistemic_*.py -v` — all 6 test files green
3. `uv run pre-commit run --all-files` — green
4. `uv run scifield epistemic sample` — writes `data/v1/handlabel_sample.parquet` (exactly 500 rows) + sidecar JSON; assertions in sampling module pass (topic coverage ≥80)
5. `uv run jupyter nbconvert --to notebook --execute notebooks/05_handlabel_sampling.ipynb --inplace` — runs clean, renders tables
6. `uv run scifield epistemic export-labels --rater samer` — produces `data/v1/labels_samer.xlsx`; open in Excel/Numbers and verify enum dropdowns work; fill 3 rows; `uv run scifield epistemic import-labels --rater samer --file data/v1/labels_samer.xlsx` — appends 3 rows to handlabel parquet
7. `uv run scifield epistemic pilot --n 50` — completes; ≥45/50 rows valid (90% threshold); per-field failure count reported; iteration log updated in `docs/phases/epistemic.md`
8. `docs/preregistrations/PR1_epistemic_extraction.md` exists with all 13 sections drafted; **manual step:** user submits to OSF, pastes link back, commits; `grep -q "osf.io" docs/preregistrations/PR1_epistemic_extraction.md` passes

**Hard stop:** V1-S08 must not start until step 8 grep passes (OSF link is in the file).

---

## Out of scope (deferred to later sessions)

- Running the full 200k-abstract batch (V1-S08)
- The actual 500-abstract hand-labeling sprint (V1-S08)
- κ on real labels (V1-S08) / Gate G2 report (V1-S09)
- Trialstreamer / RobotReviewer comparison (V1-S09)
- Any analysis using epistemic features (Phase 6 / V1-S15)
- BERT fine-tune fallback (V1-S09b, only if Gate G2 fails)
- Touching the topics, novelty, or forecasting pipelines

---

## Reuse hooks (existing utilities — do not reimplement)

- `scifield.repro.record_run(artifact_path, inputs, config)` — for every artifact (`handlabel_sample.parquet`, pilot outputs, label imports)
- `scifield.thematic.dedup.ensure_papers_distinct_view(con)` — invoke once at the top of `sampling.stratified_sample`
- Typer + Hydra `_load_config(name)` pattern in `src/scifield/cli.py` (see `harvest`, `topics` commands lines 54–69)
- Frozen-dataclass config style from `src/scifield/thematic/topics.py` (no Hydra inside modules)
- `tests/test_repro.py` `tmp_path` pattern for any test that writes artifacts

---

## Critical files (paths the implementer will touch)

**Create:**
- `src/scifield/epistemic/schema.py`
- `src/scifield/epistemic/prompt.py`
- `src/scifield/epistemic/kappa.py`
- `src/scifield/epistemic/sampling.py`
- `src/scifield/epistemic/labeling.py`
- `src/scifield/epistemic/extract.py`
- `src/scifield/epistemic/pilot.py`
- `conf/epistemic/v1.yaml`
- `notebooks/05_handlabel_sampling.ipynb`
- `docs/preregistrations/PR1_epistemic_extraction.md`
- `docs/phases/epistemic.md`
- `tests/test_epistemic_{schema,prompt,kappa,sampling,labeling,extract}.py`

**Modify:**
- `src/scifield/epistemic/__init__.py` (replace stub, re-export public API)
- `src/scifield/cli.py` (add `epistemic` Typer sub-app + 4 commands)
- `pyproject.toml` (add `pydantic`, `openpyxl`, `krippendorff`)

---

## Risk + stop-condition notes

- **Pre-reg blocks V1-S08, not V1-S07.** If OSF submission has any friction, that's a real-world delay measured in days; V1-S07's code work proceeds to completion regardless.
- **Pilot quality threshold (≥90% valid JSON):** if pilot returns <45/50 valid rows after prompt iteration, escalate before declaring done — possible signals: prompt under-specified, model swap needed for V1-S08, or schema too strict. Document outcome either way.
- **No κ computation on real data.** If you find yourself running `kappa.py` on anything except synthetic test data in this session, you've drifted into V1-S08 — stop.
- **Topic coverage <80 of 149.** Sampling assertion will fail; investigate cell allocation before adjusting seed or stratification.
- **Excel data validation across platforms:** openpyxl `DataValidation` works in Excel + LibreOffice but may degrade in Numbers (no native dropdown). If your co-rater is on Mac and refuses to install Excel/LibreOffice, the labeling tool falls back to instructions-sheet enforcement + import-time Pydantic rejection (still safe, just less user-friendly).

---

## End-to-end verification command sequence

```bash
uv sync
uv run pytest tests/test_epistemic_*.py -v
uv run pre-commit run --all-files
uv run scifield epistemic sample
uv run jupyter nbconvert --to notebook --execute notebooks/05_handlabel_sampling.ipynb --inplace
uv run scifield epistemic export-labels --rater samer
# (manual: fill 3 rows in data/v1/labels_samer.xlsx, save)
uv run scifield epistemic import-labels --rater samer --file data/v1/labels_samer.xlsx
uv run scifield epistemic pilot --n 50
# (manual: submit docs/preregistrations/PR1_epistemic_extraction.md to OSF, paste link, commit)
grep -q "osf.io" docs/preregistrations/PR1_epistemic_extraction.md && echo "OSF link present ✓"
```
