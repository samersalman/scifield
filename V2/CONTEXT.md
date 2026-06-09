# V2 Literature Cartography — Shared Ground Truth (read this first)

This file is the single source of truth for every V2 subagent. It was compiled from a
verified inspection of the live repo on 2026-06-09. **Do not re-explore the corpus; trust
these numbers** (but you may re-verify a specific value with a one-liner if a result looks
wrong). If you discover something a *downstream* task must know, append a note to
`V2/EXECUTION_LOG.md` under your task heading.

---

## 0. Mission & scope (why we are here)

V1 was confirmatory (pre-registered hypotheses, gates G1–G5; two honest nulls, one finding).
**V2 is exploratory cartography**: build a *usable map of the literature* — where topics
originate, how they cascade across journals, which journals are sources/bridges/terminals,
where novelty comes from (sector/geography/funding), and a trajectory layer.

**This session executes Phases A–E only ($0, current 10-journal corpus), stopping at the
human Gate G6.** No corpus expansion, no DeepSeek, no GPU, no paid API calls. Phases F–G
(spend) are gated behind Samer's G6 decision and are OUT of scope.

**Framing commitments (carry these into every doc):**
1. Exploratory ≠ unrigorous. No hypothesis gates, but heavy rigor: held-out years,
   null/permutation models, drop-one-journal jackknife, topic-granularity sensitivity.
2. Cascades are **panel-conditional**: "topic T originated in journal A" really means
   "first *in our 10-journal set*." The true origin may be a journal we never harvested.
   Every cascade/origin claim must carry this caveat.
3. The journal cascade is **orthogonal to the null F1** (F1 = does evidence *quality* lead
   *volume* within a topic — null). The cascade asks *which journal publishes topic T first*
   and *do some journals systematically lead*. Different question; not a revival of F1.

---

## 1. Path conventions (IMPORTANT — read carefully)

The user instruction is "all V2 outputs go in `/Users/samersalman/Desktop/SciField/V2`."
We honor that for **deliverables**, but Python modules and tests follow repo convention so
notebooks can `import scifield` and pytest/pre-commit work:

- **Python logic** → `src/scifield/cartography/` (importable as `scifield.cartography.*`).
  The plan mandates this verbatim ("logic lives in `src/scifield/cartography/`").
- **Tests** → `tests/test_cartography_<module>.py` (FLAT, matching `tests/test_findings_seeding.py`).
  `tests/__init__.py` already exists. Do NOT create `tests/cartography/`.
- **All generated deliverables** → under `V2/`:
  - `V2/docs/cartography/` — charter, protocol, decision docs, validation verdicts.
  - `V2/notebooks/` — `v2_*.ipynb` (these `import scifield.cartography`).
  - `V2/scripts/` — small runnable builders that produce the data tables.
  - `V2/data/` — output parquet tables + their `.run.json` sidecars.
  - `V2/map_v0/` — the proof-of-concept map (static HTML).

Repo root is `/Users/samersalman/Desktop/SciField`. Use paths relative to it in scripts, or
resolve from a `REPO_ROOT` constant. Run python via `.venv/bin/python` (the project venv;
`scifield` is installed editable, so `import scifield` works).

---

## 2. Coding standards (pre-commit will enforce these)

- Python 3.11; `from __future__ import annotations` at top of every module.
- ruff (`select = E,F,I,B,UP,SIM`) + black, **line-length = 100**. mypy non-strict.
- **numpy-style docstrings**, matching the house style in `src/scifield/findings/seeding.py`
  (read it — it is the gold-standard template for a cartography module: module docstring
  explaining the method, pure I/O-free functions, real counts COMPUTED not hardcoded).
- Functions should be **pure and I/O-free where possible** (take a DataFrame, return a
  DataFrame). I/O (reading parquet, writing outputs, drawing) lives in the `V2/scripts/`
  builder or the notebook, not in the `scifield.cartography` module.
- `.pre-commit-config.yaml` exists at repo root. Before claiming done, run:
  `.venv/bin/ruff check <files> && .venv/bin/black --check <files> && .venv/bin/pytest tests/test_cartography_<x>.py -q`.

---

## 3. Verified data inventory (paths relative to repo root)

### Canonical journal panel (10 journals, 1995–2025)
Slugs (the canonical entity — ALWAYS key journals on these, never display names):
`ann_surg, arthroscopy, br_j_surg, clin_orthop_relat_res, j_am_coll_surg, j_arthroplasty,
j_bone_joint_surg_am, jama_surg, spine, surgery`.
Defined in `conf/corpus/v1.yaml`. Specialty split (locked 5-vs-5) lives in
`scifield.findings.seeding.SPECIALTY_GROUPS`: orthopedic = {j_bone_joint_surg_am, arthroscopy,
j_arthroplasty, spine, clin_orthop_relat_res}; general_surgery = {surgery, ann_surg, br_j_surg,
j_am_coll_surg, jama_surg}.

### ⚠️ CRITICAL: canonical journal mapping
`data/v1/archetypes.parquet` has a `journal` column with **11 distinct DISPLAY names**
because "Archives of surgery (Chicago...)" appears separately from "JAMA surgery" — but both
are the single slug `jama_surg` (the journal renamed in 2013). **Treating display names as
journals is a bug** (it invents an 11th journal and splits jama_surg's history).
**Rule:** derive the canonical journal by joining `pmid → journal_slug` from the papers
parquet / DuckDB `papers_distinct`, NOT by parsing the display string.
- `archetypes.parquet.pmid` is **BIGINT**; `papers.parquet.pmid` / `papers_distinct.pmid` is
  **VARCHAR** → cast on join (`CAST(a.pmid AS VARCHAR) = p.pmid`).
- `flow.py` (T1b) owns this resolver and exposes a clean `(pmid → journal_slug)` mapping;
  downstream tasks consume `journal_slug` from the flow table and must not re-derive it.

### Key tables
| Table | Path | Rows | Key columns |
|---|---|---|---|
| papers (by journal/year) | `data/v1/parquet/{slug}/{year}.parquet` | 134,978 | `pmid`(VARCHAR), `journal_slug`, `year`(INT32), title, abstract, mesh_headings |
| papers_distinct (DuckDB) | `data/v1/papers.duckdb` table `papers_distinct` | 121,908 | `pmid`(VARCHAR), `journal_slug`, `journal`, `year`(INT) |
| topics | `data/v1/topics.parquet` | 89,230 | `pmid`(INT64), `topic_id`(INT64), `is_noise`(BOOL) |
| topic_hierarchy | `data/v1/topic_hierarchy.parquet` | 149 | `topic_id`, `top_words`(LIST), `size`, `mid_level_id`, `top_level_id` |
| **archetypes (master)** | `data/v1/archetypes.parquet` | 89,230 | `pmid`(INT64), `year`(INT64), `journal`(display!), `topic_id`, `sem_nov_mean`, `sem_nov_min`, `cd5`, `cd10`, `cited_by_count`, `cited_by_pctile_within_year`, `arch_*`, `n_prior`, `openalex_id` |
| novelty_semantic | `data/v1/novelty_semantic.parquet` | 89,230 | `pmid`, `topic_id`, `year`, `n_prior`, `sem_nov_mean`, `sem_nov_min` |
| cd_index | `data/v1/cd_index.parquet` | 104,523 | `pmid`, `openalex_id`, `cd5`, `cd10`, `n_citers` |
| openalex_works | `data/v1/enrichment/openalex_works.parquet` | 118,317 | `pmid`(VARCHAR), `openalex_id`, `cited_by_count`, `concepts`(LIST), `publication_year`, `publication_date` |
| references_out | `data/v1/enrichment/references_out.parquet` | 2,982,356 | `citing_pmid`, `ref_openalex_id`, `ref_pmid_if_known`, `ref_year` |
| cited_by | `data/v1/enrichment/cited_by.parquet` | 5,036,306 | `focal_oa_id`, `citing_oa_id`, `citing_year`, `cites_focal_ref` |
| paper_institutions | `data/v1/enrichment/paper_institutions.parquet` | 874,695 | `pmid`(VARCHAR), `author_position`, `institution_canonical_id`, `raw_affiliation_string` |
| institutions | `data/v1/enrichment/institutions.parquet` | 29,159 | `institution_canonical_id`, `display_name`, `country_code`, `type` |
| citation_intents | `data/v1/enrichment/citation_intents.parquet` | **0 (EMPTY)** | schema only; defer with coverage note |

`institutions.type` distribution: blank≈10,073; healthcare 7,887; education 4,663;
**company 1,871**; facility 1,307; nonprofit 1,256; government 747; other 629; funder 585;
archive 141. `country_code` present for all rows (180 unique). `type=='company'` is the
"tech/industry sector" proxy for the novelty-by-sector question.

### Topic granularity (for sensitivity analysis)
Leaf topics: 149 (`topic_id`). Mid-level: 96 (`mid_level_id` in topic_hierarchy). Use
leaf-vs-mid as the two granularities. Join `archetypes.topic_id → topic_hierarchy.topic_id`
to get `mid_level_id`.

### Kuzu graph (citation traversal + velocity)
Path `data/v1/kuzu_graph`. kuzu **0.11.3**. Open READ-ONLY:
```python
import kuzu
db = kuzu.Database("data/v1/kuzu_graph", read_only=True)
con = kuzu.Connection(db)
res = con.execute("MATCH ()-[c:CITES]->() RETURN count(c)")
print(res.get_next()[0])          # scalar
df = con.execute("MATCH (p:Paper) RETURN p.pmid, p.year").get_as_df()   # to pandas
```
Node tables: `Paper(pmid STRING PK, title, year INT64, journal_slug)`, `Journal(journal_slug PK, journal)`,
`Topic(topic_id INT64 PK)`, `Author(author_canonical_id PK, display_name)`,
`Institution(institution_canonical_id PK, display_name, country_code)`.
Rel tables: `CITES(Paper→Paper)` 574,478 edges (corpus-internal only); `PUBLISHED_IN(Paper→Journal)`
121,908; `ASSIGNED_TO(Paper→Topic)` 67,821; `AUTHORED_BY`, `AFFILIATED_WITH`.
Note: `CITES` is corpus-internal (both endpoints in corpus). For citational velocity you may
also use `cited_by.parquet` (external inbound citations with `citing_year`) joined via
`openalex_id`. Paper.pmid is STRING in Kuzu.

---

## 4. Reproducibility sidecars (mandatory for every output artifact)

Use `scifield.repro.record_run`:
```python
from pathlib import Path
from scifield.repro import record_run
record_run(
    artifact_path=Path("V2/data/flow/flow.parquet"),
    inputs={"archetypes": Path("data/v1/archetypes.parquet"),
            "papers_duckdb": Path("data/v1/papers.duckdb")},
    config={"task": "V2-S01", "grain": "leaf", "min_papers": 1},
)  # writes V2/data/flow/flow.parquet.run.json
```
It records git_sha, git_dirty, config_hash, input file sha256s, software versions, timestamp.
**Every parquet/figure/map artifact gets a `.run.json` sidecar.** `V2/data/` is under the
`data/` .gitignore rule on most paths — but `V2/` is a new tree; assume sidecars are written
next to artifacts regardless. Notebooks must `import scifield.cartography` (logic) and only
do I/O + plotting themselves.

---

## 5. Reusable existing code (DO NOT reinvent)

`src/scifield/findings/seeding.py` (pure pandas, no networkx) — the lead/follow prototype to
**generalize**, not replace:
- `first_publication_year(df, *, entity_col="journal", min_papers=1)` → `[topic_id, entity, first_year]`.
- `seeding_score(df, *, entity_col, min_papers=1)` → `[entity, seeding_score(0–1), n_topics]`
  (mean normalized-lead; earliest=1.0, latest=0.0).
- `directed_seeding_network(df, *, entity_col, min_papers=1)` → edge list
  `[src, dst, n_precedes, n_shared, weight]`, weight = n_precedes/n_shared, src leads dst.
- `seeding_by_group(df, *, group_col, min_papers=1)` → grain-agnostic, skips blank entities.
- `specialty_of(journal)` → "orthopedic"|"general_surgery"|None; `normalize_journal(journal)`.
- `SPECIALTY_GROUPS` keyed by BOTH slug and display name.
Its tests: `tests/test_findings_seeding.py` (read for the testing idiom). Notebook
`notebooks/14_bonus_cross_journal.ipynb` shows the I/O + matplotlib drawing pattern.

`docs/figures/topic_landscape.html` (built by `notebooks/04_topic_landscape.ipynb`, Plotly)
is the base layer for the map v0 — reuse the Plotly/static-HTML approach (streamlit & dash are
NOT installed; do not add a server dependency — emit standalone HTML like topic_landscape.html).

---

## 6. Environment quick facts
- Run python: `.venv/bin/python`. `scifield` importable (editable install).
- Installed: pandas 3.0.3, duckdb 1.5.2, kuzu 0.11.3, pyarrow 24.0.0, plotly 6.7.0,
  networkx 3.6.1, statsmodels 0.14.6, scipy 1.17.1, scikit-learn 1.8.0, nbconvert 7.16+.
  NOT installed: streamlit, dash.
- Execute a notebook headless:
  `.venv/bin/jupyter nbconvert --to notebook --execute --inplace V2/notebooks/<nb>.ipynb`
  (set a sane timeout; keep cells fast — these run on the current corpus, seconds–minutes).
- `$0 rule`: no DeepSeek, no GPU, no paid APIs this session. Polite-pool re-harvests are
  technically allowed but defer heavy re-pulls with a coverage note (see S06 guidance).

---

## 7. Cross-task hand-off protocol
If your task surfaces something a later task needs (a column name, a caveat, a gotcha, a
decision), append a dated bullet to `V2/EXECUTION_LOG.md` under your task's heading. Later
tasks read that log before starting.
