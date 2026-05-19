# V1-S02 — Hydra configs, docs site, end-to-end demo on 100-paper toy corpus

**Phase:** 0 (Scaffolding wrap-up) | **Repo:** `/Users/samersalman/Desktop/SciField/`
**Briefing:** `plan/Session-Objectives-MAP.md` lines 105–135 | **Depends on:** V1-S01 (complete)

---

## Context

V1-S01 produced a clean Python package (Typer CLI stub, `scifield.repro.record_run`, stub modules per phase, ruff/black/mypy/pre-commit, green CI). What's still missing before any real pipeline work begins is the execution-and-reproducibility plumbing:

1. **Hydra** — every analysis must be config-driven so reruns are deterministic by config hash (plan §3, §5 Phase 0).
2. **mkdocs site** — written from day zero so the eventual OSS release "requires zero refactoring" (plan §3).
3. **A working end-to-end `scifield demo`** — the Phase 0 success criterion is "a colleague clones the repo and reproduces the demo in <10 minutes" (plan §5 Phase 0).
4. **Brev smoke test** — proves the launch/sync/run/stop harness on a cheap CPU instance now, so V1-S05's first GPU launch (~$5–10) is not also the first time the harness has been tested.

This session lays that plumbing. No real pipeline logic — that's V1-S03+.

---

## Decisions confirmed with user

- **GH Pages deploy:** GitHub Actions Pages (modern `actions/deploy-pages@v4`). One-time manual step: Repo Settings → Pages → Source = "GitHub Actions".
- **Brev smoke test:** Live run on smallest CPU tier. User does not yet have `brev` CLI; install before execute. Smoke script is defensive (checks `brev --version`, exits cleanly with a documented note if absent — matches the briefing's stop condition).
- **Demo target:** PubMed query `journal=Arthroscopy`, `year_range=[2024, 2024]`, `max_papers=100` (first journal in V1's 10-journal list).
- **CLI framework:** Keep Typer (already installed in V1-S01). Use Hydra's programmatic `compose` API — not the `@hydra.main` decorator — for Typer compatibility.

---

## Preconditions for execute

Run these before `/awesome-execute`:

1. `brev` CLI installed and authenticated:
   - `brew install brevdev/homebrew-brev/brev` (or the curl installer)
   - `brev login` (browser SSO)
   - `brev --version` should print a version
2. NCBI Entrez email — already known: `samer.salman2021@gmail.com` (no API key needed for 100-paper demo).
3. Repo Settings → Pages → Source = "GitHub Actions" (one-time, browser).

---

## Files to create / modify

### Dependencies — `pyproject.toml`
Add to `[project] dependencies`:
- `hydra-core>=1.3`
- `omegaconf>=2.3`
- `biopython>=1.84`
- `pyarrow>=17`

Add a new dependency group `[dependency-groups] docs` (so docs deps don't bloat the dev install):
- `mkdocs>=1.6`
- `mkdocs-material>=9.5`
- `mkdocstrings[python]>=0.26`

Keep mypy's `ignore_missing_imports = true` — biopython and hydra have weak stubs.

### Hydra configs — new `conf/` directory
- `conf/config.yaml` — root with empty `defaults: []` plus comment noting it's the future composition root.
- `conf/corpus/default.yaml`, `conf/thematic/default.yaml`, `conf/epistemic/default.yaml`, `conf/novelty/default.yaml`, `conf/forecasting/default.yaml`, `conf/integration/default.yaml` — each has 1–2 placeholder keys with a `# TODO` comment naming its plan phase (mirrors the stub `__init__.py` pattern from V1-S01).
- `conf/demo.yaml` — concrete:
  ```yaml
  demo:
    journal: "Arthroscopy"        # PubMed Title abbreviation
    year_range: [2024, 2024]
    max_papers: 100
    email: "samer.salman2021@gmail.com"
    output_path: "data/demo/papers.parquet"
  ```

### CLI — `src/scifield/cli.py` (modify)
- Add helper `_load_config(name: str) -> DictConfig` using `hydra.initialize_config_dir(version_base="1.3", config_dir=<absolute conf path>)` + `hydra.compose(config_name=name)`. Resolve `conf/` via `Path(__file__).resolve().parents[2] / "conf"` so it works under `uv run` (editable install).
- Rewrite `demo()` command to:
  1. `cfg = _load_config("demo")`
  2. Build Entrez query: `f'"{cfg.demo.journal}"[Journal] AND {y0}:{y1}[PDAT]'`
  3. `Entrez.email = cfg.demo.email; Entrez.esearch(db="pubmed", term=..., retmax=cfg.demo.max_papers)` → PMID list
  4. `Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")` → `Entrez.read(handle)`
  5. Extract per record: `pmid`, `title`, `abstract` (joined `AbstractText` fragments), `journal`, `year`. Tolerate missing abstract (empty string + flag).
  6. Write Parquet via `pyarrow.Table.from_pylist(...).write_table(out_path)`; ensure `data/demo/` exists.
  7. `record_run(artifact_path=out_path, inputs={}, config=OmegaConf.to_container(cfg, resolve=True))` — note `inputs={}` because the demo has no local file inputs (PubMed is a remote query).
  8. Print: `f"n_papers={len(rows)}  mean_abstract_chars={mean_len:.0f}"`.

### Corpus module — `src/scifield/corpus/pubmed_demo.py` (new, minimal)
Pull the Entrez calls out of `cli.py` into `scifield.corpus.pubmed_demo` so the demo doesn't reimplement what V1-S03 will replace. One function: `fetch_demo_papers(journal: str, year_range: tuple[int,int], max_papers: int, email: str) -> list[dict]`. V1-S03 will fully replace this with the async harvester; this is intentionally throwaway and labeled as such in its docstring.

### Tests — `tests/`
- `tests/test_demo_config.py` (new) — `hydra.compose("demo")` returns expected keys (`journal`, `year_range`, `max_papers`, `email`, `output_path`). No network.
- `tests/test_repro.py` (modify) — extend `expected_keys` to include `git_dirty` (currently missing per V1-S01 oversight).
- Keep `tests/test_cli.py` as-is.

### Docs — new `docs/` + `mkdocs.yml`
- `mkdocs.yml` — `site_name: SciField`, `theme: name: material` (palette + features), `nav` listing index, phases, api. `plugins: [search, mkdocstrings]`. `repo_url` pointing to `github.com/samersalman/scifield`.
- `docs/index.md` — overview paragraph from README + a "see plan/" pointer + one-line summary of each Phase.
- `docs/phases/0_scaffolding.md` through `docs/phases/9_manuscript.md` — ten files. Each contains the **Phase Objective** seeded from `plan/scifield_plan.md` §5 (one paragraph per phase). No deep content yet; just enough for the site to feel populated and for future sessions to fill in.
- `docs/api/scifield.md` — `# API` heading + `::: scifield` (mkdocstrings autogen marker). Sub-pages per submodule can be added when modules have content.
- `docs/operations/brev.md` — Brev hygiene + cost table + smoke-test instructions. If the live smoke fails during execute, append a "Smoke test deferred — re-run in V1-S05 prerequisites" note.

### CI — `.github/workflows/`
- `.github/workflows/docs.yml` (new):
  ```yaml
  name: Docs
  on:
    push:
      branches: [main]
  permissions:
    contents: read
    pages: write
    id-token: write
  concurrency:
    group: pages
    cancel-in-progress: false
  jobs:
    build-deploy:
      runs-on: ubuntu-latest
      environment:
        name: github-pages
        url: ${{ steps.deployment.outputs.page_url }}
      steps:
        - uses: actions/checkout@v4
        - uses: astral-sh/setup-uv@v3
          with: { enable-cache: true }
        - run: uv python install 3.11
        - run: uv sync --all-groups --group docs
        - run: uv run mkdocs build --strict
        - uses: actions/configure-pages@v5
        - uses: actions/upload-pages-artifact@v3
          with: { path: site }
        - id: deployment
          uses: actions/deploy-pages@v4
  ```
- `.github/workflows/ci.yml` (modify) — add `uv sync --group docs` step before mypy so `mkdocs build --strict` can run on PRs too (catch broken doc builds early). Add `uv run mkdocs build --strict` as a CI step.

### Brev smoke — `scripts/brev_smoke.sh` (new)
Defensive bash, `set -euo pipefail`:
1. `command -v brev >/dev/null || { echo "brev CLI not installed; see docs/operations/brev.md" >&2; exit 0; }` (exit 0 — non-fatal per stop condition).
2. `brev --version` echo.
3. Record credit balance before (best-effort: `brev org` or `brev profile` — the exact subcommand depends on current brev CLI; the script will tolerate failures and just log raw output).
4. Launch smallest CPU instance, tagged `scifield-smoke-v1s02`: `brev create scifield-smoke-v1s02 --instance-type <smallest-cpu> --git https://github.com/samersalman/scifield.git` (exact flag set verified at execute time via `brev create --help`).
5. Wait for ready: `brev refresh` poll loop.
6. SSH-exec the smoke commands: `brev shell scifield-smoke-v1s02 -- 'cd scifield && uv sync && uv run scifield demo'`.
7. `brev stop scifield-smoke-v1s02` (guaranteed via `trap` on EXIT).
8. Record credit balance after; print delta.
9. Optional `brev delete scifield-smoke-v1s02` to clean up — leave commented unless user opts in.

### .gitignore (modify)
Add: `/site/` (mkdocs build output).

---

## Critical files to read before editing

- `src/scifield/cli.py` — pattern to extend (Typer app + commands)
- `src/scifield/repro/__init__.py` — exact `record_run` signature (`inputs: dict[str, Path]`, `config: dict[str, Any]`)
- `pyproject.toml` — dep groups + tool configs
- `.github/workflows/ci.yml` — uv-based job pattern to mirror in docs.yml
- `plan/scifield_plan.md` §5 — phase objectives to seed `docs/phases/*.md`
- `.gitignore` — confirm `data/` and `*.parquet` already ignored (they are)

## Existing utilities to reuse

- `scifield.repro.record_run(artifact_path, inputs, config)` — call with `inputs={}` since the demo has no local file inputs; pass `OmegaConf.to_container(cfg, resolve=True)` as `config`.
- `scifield.__version__` — already wired into `record_run`'s sidecar.

---

## Verification (end-to-end test plan)

Run in order, on the local machine, after execute:

1. **Sync:** `uv sync --all-groups --group docs` — exits 0.
2. **Lint/format/types/tests:**
   - `uv run ruff check .`
   - `uv run ruff format --check .`
   - `uv run black --check .`
   - `uv run mypy src tests`
   - `uv run pytest -v` — all tests pass (3 tests: cli help, repro, demo-config).
3. **Demo end-to-end (no network mock):**
   - `uv run scifield demo`
   - Should complete in <60s, print `n_papers≈100 mean_abstract_chars=…`.
   - Assert: `data/demo/papers.parquet` exists with 100 rows (or close — Entrez sometimes returns fewer for rare queries).
   - Assert: `data/demo/papers.parquet.run.json` exists, contains `git_sha`, `git_dirty`, `config_hash`, `software_versions`.
4. **Docs build:**
   - `uv run mkdocs serve` — opens at localhost:8000, all pages render, no broken links/strict errors.
   - `uv run mkdocs build --strict` — exits 0.
5. **Brev smoke (live, on cheap CPU tier):**
   - `bash scripts/brev_smoke.sh` — full launch → sync → demo → stop cycle. Credit-balance delta is logged.
   - If `brev` not installed or auth fails: script exits 0 with the documented warning; record this in `docs/operations/brev.md`.
6. **Push + CI:**
   - Commit + push. CI workflow (ci.yml) green. Docs workflow (docs.yml) green, GH Pages URL live.
   - Open the GH Pages URL — index, phases, api all render.
7. **Reproducibility timing test (Phase 0 success criterion):**
   - On a fresh clone: `time (uv sync && uv run scifield demo)` — under 10 minutes wall-clock (per plan §5 Phase 0).

---

## Out of scope (do not drift)

These are tempting but belong to later sessions:

- Full 10-journal harvest (`conf/corpus/v1.yaml`) → V1-S03.
- OpenAlex / Semantic Scholar / ROR / authors → V1-S04.
- DVC or any data-versioning beyond sidecar JSON → V1-S03 decision.
- `scifield brev launch/stop` CLI subcommand (per cross-cutting constraint) → defer; for V1-S02 the bash script is sufficient.
- MeSH parsing → V1-S03.
- Async Entrez harvester with tenacity → V1-S03; today's demo uses the synchronous Biopython `Entrez` module, deliberately throwaway.
- Async / multi-page Entrez pagination → not needed for 100 papers.

---

## Commit plan (suggested at execute time)

Three logical commits:
1. `feat(config): wire Hydra and add demo config (V1-S02)` — pyproject deps, `conf/`, `_load_config` helper.
2. `feat(demo): pull 100 PubMed papers end-to-end with sidecar (V1-S02)` — `corpus/pubmed_demo.py`, `cli.demo` rewrite, demo test.
3. `feat(docs): mkdocs-material site + GH Pages workflow + Brev smoke (V1-S02)` — `docs/`, `mkdocs.yml`, `docs.yml` workflow, `scripts/brev_smoke.sh`, `.gitignore` update.

---

## Stop conditions (per briefing)

- If Brev access has any issue, document in `docs/operations/brev.md` and **do not delete** `scripts/brev_smoke.sh`. Continue with everything else.
- Do not start V1-S03 until acceptance tests above are all green and CI is passing.
