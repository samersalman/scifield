# V1-S03 — Corpus v1 harvesting: PubMed + DuckDB + Parquet

**Date:** 2026-05-19
**Briefing:** `plan/Session-Objectives-MAP.md` lines 139–169
**Depends on:** V1-S02 (complete — `scifield demo` works on 100-paper Arthroscopy slice)

---

## Context

SciField is in Phase 1 of a 9-phase plan to build a multi-axis framework for monitoring scientific field health. V1-S01 stood up the package skeleton (Typer CLI, Hydra, repro sidecars, CI). V1-S02 wired a 100-paper PubMed demo end-to-end. V1-S03 is the first session that produces real research data: harvest abstracts + metadata for the 10 v1 journals × 1995–2025 from PubMed into a DuckDB-queryable corpus.

Every downstream session (embeddings, topic modeling, epistemic LLM extraction, novelty, forecasting) reads from this corpus. Getting it clean, reproducible, and resumable matters more than getting it fast: it runs overnight, once.

**Why now.** V1-S02 proved Entrez access works. The 10-journal list is locked. No earlier session is blocking. V1-S04 (OpenAlex enrichment) needs PMIDs from this session.

**Intended outcome.** `data/v1/papers.duckdb` (a thin view layer) over `data/v1/parquet/<journal_slug>/<year>.parquet` partitions, ~150k–250k papers total, each row carrying PMID / year / title / abstract / journal / journal-TA / DOI / authors / MeSH / publication-types. Every Parquet has a `.run.json` sidecar. The harvester is idempotent at the (journal, year) bucket level and rate-limit-honest.

**Out-of-session boundary.** Claude delivers the harvester + a smoke run (~500 papers) that exercises every code path. The user runs the full overnight harvest themselves; row-count and per-journal-coverage acceptance checks live in `notebooks/01_corpus_overview.ipynb` which re-renders after the overnight job.

---

## Decisions baked in (from user dialogue)

- **API key handling.** Code reads `NCBI_API_KEY` env var. If present → concurrency 10 req/s. If absent → 3 req/s. User has a key; the recommended invocation in docs sets it.
- **Dual-TA per journal.** Config supports a list of TA terms per journal. JAMA Surgery maps to `["JAMA Surg", "Arch Surg"]` so the 1995–2012 *Archives of Surgery* era is captured. Pattern is general — works for any future rename.
- **DuckDB shape.** Views over Parquet (Parquet is source of truth). DuckDB file is regenerable, ~KB-sized, holds `CREATE OR REPLACE VIEW papers / journals / mesh` statements that scan the Parquet lake.
- **Idempotency granularity.** (journal_slug, year) bucket. If the Parquet for a bucket exists and matches the manifest's PMID set, skip. `--refresh <slug>` or `--refresh-year <YYYY>` forces a rebuild of matching buckets.

---

## In scope (literal V1-S03 brief, restated)

- `src/scifield/corpus/pubmed.py` — async Entrez harvester (httpx + tenacity); rate-limited; idempotent.
- `src/scifield/corpus/store.py` — DuckDB view writer; Parquet writer per (journal, year).
- `conf/corpus/v1.yaml` — 10 journals, year range 1995–2025.
- `scifield.cli` — `harvest` subcommand.
- `notebooks/01_corpus_overview.ipynb` — descriptive corpus stats.

## Out of scope (defer)

- OpenAlex / Semantic Scholar / ROR enrichment → V1-S04.
- Citation graph / Kùzu loading → V1-S10.
- Embeddings → V1-S05.
- Author disambiguation → V1-S04.
- Any analysis beyond descriptive corpus stats.

---

## Reuse — existing patterns to follow, not reinvent

| Need | Existing pattern | Where |
|---|---|---|
| CLI subcommand wiring | `@app.command()` + `_load_config(name)` | `src/scifield/cli.py:36-67` |
| Hydra config composition | `hydra.initialize_config_dir(...)` in `_load_config` | `src/scifield/cli.py:26-33` |
| Sidecar metadata | `record_run(artifact_path, inputs, config)` | `src/scifield/repro/__init__.py:55` |
| Parquet write | `pa.Table.from_pylist(rows); pq.write_table(table, out_path)` | `src/scifield/cli.py:60-61` |
| Test isolation | pytest `tmp_path` + `CliRunner` | `tests/test_cli.py`, `tests/test_repro.py` |

The Biopython-based `pubmed_demo.py` stays in place (V1-S02 deliverable). The new async harvester lives alongside it; the `demo` command continues to use Biopython.

---

## Implementation steps

### 1. Dependencies

Add to `pyproject.toml` `[project] dependencies`:
- `httpx>=0.27` — async HTTP client.
- `tenacity>=9.0` — retry decorators.
- `duckdb>=1.1` — DuckDB Python bindings.

Add to `[dependency-groups] dev`:
- `pytest-asyncio>=0.24` — async test support.
- `respx>=0.21` — httpx mocking for tests.
- `jupyter>=1.1` — for notebook execution in CI/local.

Run `uv sync`.

### 2. Config — `conf/corpus/v1.yaml`

Schema (replaces the V1-S02 stub):

```yaml
journals:
  - slug: j_bone_joint_surg_am
    display: "J. Bone & Joint Surgery (Am)"
    ta_terms: ["J Bone Joint Surg Am"]
  - slug: arthroscopy
    display: "Arthroscopy"
    ta_terms: ["Arthroscopy"]
  - slug: j_arthroplasty
    display: "Journal of Arthroplasty"
    ta_terms: ["J Arthroplasty"]
  - slug: spine
    display: "Spine"
    ta_terms: ["Spine (Phila Pa 1976)"]
  - slug: clin_orthop_relat_res
    display: "Clinical Orthopaedics & Related Research"
    ta_terms: ["Clin Orthop Relat Res"]
  - slug: ann_surg
    display: "Annals of Surgery"
    ta_terms: ["Ann Surg"]
  - slug: jama_surg
    display: "JAMA Surgery (incl. Arch Surg pre-2013)"
    ta_terms: ["JAMA Surg", "Arch Surg"]
  - slug: j_am_coll_surg
    display: "Journal of the American College of Surgeons"
    ta_terms: ["J Am Coll Surg"]
  - slug: br_j_surg
    display: "British Journal of Surgery"
    ta_terms: ["Br J Surg"]
  - slug: surgery
    display: "Surgery"
    ta_terms: ["Surgery"]

year_range: [1995, 2025]

entrez:
  email: "samer.salman2021@gmail.com"
  # api_key is read from NCBI_API_KEY env var at runtime; not stored in config.
  base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
  request_timeout_s: 60
  max_retries: 5

harvest:
  batch_size: 200          # PMIDs per efetch call
  rate_limit_with_key: 9   # req/s when NCBI_API_KEY set (NCBI ceiling is 10; leave headroom)
  rate_limit_no_key: 2     # req/s without key (NCBI ceiling is 3; leave headroom)

output:
  parquet_dir: "data/v1/parquet"
  duckdb_path: "data/v1/papers.duckdb"
  manifest_dir: "data/v1/manifests"
  log_dir: "data/v1/logs"
```

Also update `conf/config.yaml` to add `corpus: v1` to `defaults` so a top-level compose can resolve it. (Not strictly required since the harvest command will load `corpus/v1` directly, but keeps the composition root tidy.)

### 3. Harvester — `src/scifield/corpus/pubmed.py`

Public surface:

```python
async def harvest_journal_year(
    *, slug: str, ta_terms: list[str], year: int,
    entrez: EntrezClient, batch_size: int,
) -> list[dict]: ...

async def harvest_corpus(
    *, journals: list[JournalSpec], year_range: tuple[int, int],
    entrez: EntrezClient, output: OutputSpec, harvest_cfg: HarvestCfg,
    progress_cb: Callable | None = None,
) -> HarvestReport: ...
```

Internals:

- `EntrezClient` — wraps `httpx.AsyncClient`. Holds the email + optional api_key + base_url. Exposes `esearch(term, retstart, retmax) -> list[str]` and `efetch(pmids: list[str]) -> bytes` (returns raw XML). Uses tenacity with `wait_exponential` + `retry_if_exception_type((httpx.HTTPError, httpx.ReadTimeout))`, max 5 attempts.
- **Rate limiter** — a simple async token-bucket (or `asyncio.Semaphore` + `asyncio.sleep(1/rate)` interleave). Apply globally across all journal/year workers, not per-bucket.
- **esearch loop** — paginate with `retstart`/`retmax=1000` up to esearch's 9999 cap; if a (TA, year) returns >9999 PMIDs, split by month. This shouldn't happen for our journals (largest is ~25k papers / 30 yr ≈ 833/yr), but the safety branch belongs in the harvester.
- **Multi-TA query** — for each (journal, year), query each TA term in `ta_terms` separately; union the PMID sets. Log per-TA counts in the bucket manifest.
- **efetch batching** — chunk PMIDs to `batch_size` (200). Each chunk → `efetch.fcgi?db=pubmed&id=…&rettype=xml&retmode=xml`.
- **XML parsing** — stdlib `xml.etree.ElementTree.iterparse` over the response bytes. Per-`<PubmedArticle>`:
  - `pmid` — `MedlineCitation/PMID`
  - `title` — `Article/ArticleTitle` (text + children, stripped)
  - `abstract` — concatenated `Article/Abstract/AbstractText` segments; each segment kept separately in `abstract_segments` (list of `{label, nlm_category, text}`) for downstream epistemic work
  - `journal` — `Article/Journal/Title`
  - `journal_ta` — `Article/Journal/ISOAbbreviation` (or `MedlineCitation/MedlineJournalInfo/MedlineTA`)
  - `year` — `Article/Journal/JournalIssue/PubDate/Year` (fall back to `MedlineDate` parsing for ranges like "1995-1996")
  - `pub_date` — ISO date if Year+Month+Day available, else `null`
  - `doi` — `Article/ELocationID[@EIdType="doi"]`
  - `publication_types` — `Article/PublicationTypeList/PublicationType` (list of strings)
  - `authors` — `Article/AuthorList/Author` as list of `{last_name, fore_name, initials, affiliation}` (first listed affiliation only — full affiliation parsing is V1-S04)
  - `mesh_headings` — `MedlineCitation/MeshHeadingList/MeshHeading` as list of `{descriptor, descriptor_ui, major_topic, qualifiers: [{name, ui, major_topic}]}`
  - `has_abstract` — derived bool
  - `fetched_at` — UTC ISO timestamp
  - `source_ta_match` — which `ta_terms` entry produced this PMID
- **Robustness** — wrap each `<PubmedArticle>` parse in try/except; route parse failures to `data/v1/parquet/<slug>/<year>.failed.jsonl` for inspection. Don't drop the whole batch.
- **Idempotency** — at bucket start: if `data/v1/parquet/<slug>/<year>.parquet` exists AND `data/v1/manifests/<slug>/<year>.json` PMID set matches the current esearch result, skip the bucket. Otherwise: fetch the diff PMIDs, merge with existing rows, atomically rewrite Parquet + manifest + sidecar.

### 4. Store — `src/scifield/corpus/store.py`

- `write_bucket_parquet(rows, slug, year, parquet_dir) -> Path` — schema-stable PyArrow table write; atomic rename through `<slug>/<year>.parquet.tmp`. Uses an explicit `pa.schema` so MeSH/authors/abstract_segments nested types are stable across buckets.
- `read_manifest(slug, year, manifest_dir) -> ManifestData | None` and `write_manifest(...)` — manifest JSON includes the PMID list per TA, esearch query string, count, run timestamp.
- `build_duckdb(parquet_dir, duckdb_path)` — opens the DuckDB file and runs:
  ```sql
  CREATE OR REPLACE VIEW papers AS
    SELECT * FROM read_parquet('<parquet_dir>/*/*.parquet', union_by_name=true);
  CREATE OR REPLACE VIEW journals AS
    SELECT journal_slug, journal, journal_ta, COUNT(*) AS n_papers
    FROM papers GROUP BY 1, 2, 3;
  CREATE OR REPLACE VIEW mesh AS
    SELECT pmid, journal_slug, year, unnest(mesh_headings) AS heading FROM papers;
  ```
  Then writes a sidecar via `record_run(duckdb_path, inputs={"parquet_dir": parquet_dir}, config=...)`.

### 5. CLI — extend `src/scifield/cli.py`

Add a `harvest` subcommand:

```python
@app.command()
def harvest(
    config: str = typer.Option("v1", "--config", "-c",
                               help="Hydra config name under conf/corpus/"),
    journal: str | None = typer.Option(None, help="Limit to one journal slug"),
    year: int | None = typer.Option(None, help="Limit to one year"),
    refresh: bool = typer.Option(False, help="Re-fetch even if bucket Parquet exists"),
    max_papers_per_bucket: int | None = typer.Option(None,
        help="Smoke-test cap; harvests at most N PMIDs per (journal, year)."),
) -> None: ...
```

Loads `conf/corpus/<config>.yaml` via Hydra. Reads `NCBI_API_KEY` from env. Constructs `EntrezClient`, `OutputSpec`, `HarvestCfg`. Invokes `asyncio.run(harvest_corpus(...))`. Calls `store.build_duckdb(...)`. Echoes a one-line summary (n papers, n journal-year buckets, elapsed).

### 6. Tests — `tests/`

All tests must run without network. Fixtures live in `tests/fixtures/pubmed_xml/`.

- `tests/fixtures/pubmed_xml/esearch_arthroscopy_2024.xml` — captured esearch response (~10 PMIDs).
- `tests/fixtures/pubmed_xml/efetch_arthroscopy_2024.xml` — captured efetch response covering those PMIDs (multiple `<PubmedArticle>` blocks, including one with structured abstract, one with no abstract, one with empty MeSH list).
- `tests/test_corpus_pubmed.py`:
  - `test_parse_record_full` — feed one well-formed `<PubmedArticle>` to the parser; assert all 13 fields populated correctly.
  - `test_parse_record_no_abstract` — paper without abstract → `has_abstract=False`, `abstract=""`.
  - `test_parse_record_structured_abstract` — multi-segment abstract → `abstract_segments` has labels preserved.
  - `test_parse_record_arch_surg_legacy` — record from 1998 with `MedlineTA = "Arch Surg"` parses with correct year + TA.
  - `test_harvest_journal_year_dedup_dual_ta` (uses respx) — two TA terms returning overlapping PMID lists yield a deduped row set.
  - `test_idempotent_skip_when_manifest_matches` (uses respx + tmp_path) — second invocation makes zero esearch calls if manifest matches.
- `tests/test_corpus_store.py`:
  - `test_write_bucket_parquet_roundtrip` — write rows → read back → schema preserved including nested types.
  - `test_build_duckdb_views` — after building, querying `SELECT COUNT(*) FROM papers` returns expected count; `SELECT COUNT(*) FROM mesh` matches unnested count.
- `tests/test_cli_harvest.py`:
  - `test_harvest_help` — `runner.invoke(app, ["harvest", "--help"])` exits 0.
  - `test_harvest_dry_smoke` — invoke with `--max-papers-per-bucket 2 --year 2024 --journal arthroscopy` against a respx-mocked NCBI; assert Parquet + manifest + DuckDB + sidecar all written.

### 7. Notebook — `notebooks/01_corpus_overview.ipynb`

Authored as an executed `.ipynb`. Cells (top to bottom):

1. Setup: open `data/v1/papers.duckdb`, import duckdb + matplotlib.
2. Total counts: `SELECT COUNT(*) FROM papers`; print total + by journal table.
3. Papers/year per journal (line plot, faceted or 10-line single axis).
4. Abstract length distribution (histogram of `len(abstract)` for `has_abstract=True`).
5. Abstract availability by journal × era — heatmap, rows=journal, columns=`{pre-2000, 2000-2009, 2010-2019, 2020+}`, value=% with non-empty abstract. **Flag (red text) any cell <0.90.**
6. MeSH coverage — bar chart, % of papers with ≥1 MeSH heading per journal.
7. Pre-2000 abstract availability — explicit single-number readout per plan §6 risk row 1 ("OpenAlex pre-2000 sparse"). Written to `docs/phases/1_corpus.md` as a one-paragraph note at the bottom (manual append, not auto-generated).

Notebook will be executed against whatever data exists at the time of execution: first against the smoke run (~500 papers), later by the user after the overnight harvest.

### 8. Documentation touches

- Append to `docs/phases/1_corpus.md` (existing stub): one paragraph naming the harvester module, the config file, and the smoke-run command. Don't write the full corpus report — that's V1-S04.
- No new top-level docs page.

---

## Critical files to be modified or created

**Created:**
- `src/scifield/corpus/pubmed.py`
- `src/scifield/corpus/store.py`
- `conf/corpus/v1.yaml`
- `notebooks/01_corpus_overview.ipynb`
- `tests/fixtures/pubmed_xml/esearch_arthroscopy_2024.xml`
- `tests/fixtures/pubmed_xml/efetch_arthroscopy_2024.xml`
- `tests/test_corpus_pubmed.py`
- `tests/test_corpus_store.py`
- `tests/test_cli_harvest.py`

**Modified:**
- `src/scifield/cli.py` — add `harvest` command.
- `src/scifield/corpus/__init__.py` — export `harvest_corpus`, `build_duckdb`.
- `pyproject.toml` — add httpx, tenacity, duckdb, pytest-asyncio, respx, jupyter.
- `conf/corpus/default.yaml` → replaced by `conf/corpus/v1.yaml` (delete the stub, the new file becomes the convention).
- `conf/config.yaml` — optional: add `corpus: v1` to defaults.
- `docs/phases/1_corpus.md` — append harvester-module paragraph.

**Untouched (out of scope):**
- `src/scifield/corpus/pubmed_demo.py` — V1-S02 demo path stays in place.
- All other modules (`thematic/`, `epistemic/`, etc.).

---

## Verification

### Local (Claude session, no overnight wait)

1. `uv sync` — installs new deps cleanly.
2. `uv run pytest` — all old tests still pass; new corpus/store/cli-harvest tests pass.
3. `uv run pre-commit run --all-files` — ruff/black/mypy green.
4. `uv run scifield harvest --help` — prints usage including `--config / --journal / --year / --refresh / --max-papers-per-bucket`.
5. **Smoke run hitting live PubMed (small N, no overnight):**
   ```bash
   export NCBI_EMAIL="samer.salman2021@gmail.com"
   export NCBI_API_KEY="<your key>"   # if set; harvester reads it
   uv run scifield harvest --config v1 --year 2024 --max-papers-per-bucket 30
   ```
   Expected: ~300 papers across the 10 journals in <2 min; `data/v1/papers.duckdb` exists; sidecar JSONs on every Parquet; rerunning the same command produces zero new fetches (idempotency check).
6. `uv run jupyter execute notebooks/01_corpus_overview.ipynb` — notebook executes end-to-end on the smoke dataset and renders. (Numbers will be small — that's expected.)
7. `git status` — check that exactly the planned files changed; no incidental edits.

### Acceptance-test simulation summary (for the gate)

After step 6, the Claude session is **done**. Print a one-screen summary including:

- Smoke-run total paper count + per-journal breakdown.
- Confirmation that idempotency rerun was zero-fetch.
- Notebook execution status.
- Recommended next command for the user:
  ```bash
  uv run scifield harvest --config v1   # the overnight run
  ```

### After-session (user, overnight)

User runs the full harvest. After completion:

1. Open `data/v1/papers.duckdb`; `SELECT COUNT(*) FROM papers` is within 150k–250k.
2. Per-journal counts within ±20% of the expected ranges in `plan/scifield_plan.md` §4.
3. Re-execute `notebooks/01_corpus_overview.ipynb`. Confirm: >95% of papers have non-empty abstract overall; every journal-era cell ≥90% (or documented as a known sparse cell).
4. `git status` shows sidecar JSONs for every Parquet under `data/v1/parquet/**/`.

Only after these four checks does V1-S03 get its `✓` in `plan/Session-Objectives-MAP.md`.

---

## Risk hooks (per V1-S03 brief)

- **Plan §6 row 1 — OpenAlex pre-2000.** Not yet triggered in this session (no OpenAlex calls). But step 7 of the notebook produces the pre-2000 abstract-availability number, which V1-S04 will need when deciding whether to restrict novelty analyses to post-2000.
- **NCBI rate-limit ban.** Mitigated by: token-bucket rate limiter set at 2 req/s (no key) / 9 req/s (with key) — both below NCBI's ceilings; tenacity backoff on 429; email always set; api_key always sent when present.
- **XML parse drift.** Mitigated by: bucket-level parse failures routed to `<slug>/<year>.failed.jsonl` rather than aborting the bucket; manifest records esearch count vs. parse count delta so the user can spot the divergence in the overnight log.

---

## Anti-drift checks (the V1-S03 brief insists)

If during execution something tempts scope creep, stop and ask whether it belongs in a later session. Specifically:

- DOI resolution / DOI→OpenAlex ID lookups → V1-S04. We extract the DOI from PubMed XML if present, but **do not** call OpenAlex.
- Citation lists / referenced_works → V1-S04.
- Author disambiguation beyond raw `<Author>` blocks → V1-S04.
- Full affiliation parsing → V1-S04.
- Loading anything into Kùzu → V1-S10.
- Any embedding, topic modeling, novelty score, or forecasting → V1-S05+.
- Notebook analyses beyond descriptive stats — no inter-journal comparisons, no clustering, no temporal trend modeling.
