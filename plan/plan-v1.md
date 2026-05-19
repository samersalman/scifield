# V1-S01 — Repo scaffolding, CI, CLI skeleton

**Session brief:** `plan/Session-Objectives-MAP.md` lines 69–102.
**Working dir:** `/Users/samersalman/Desktop/SciField/`
**Effort:** ~1 day | **Depends on:** none | **Plan ref:** §5 Phase 0

---

## Context

SciField is a 50-week, multi-axis scientific-field-health framework spanning Phases 0–9 across three "versions" (10-journal validation → 25–40-journal scaling → release+publication). The full execution roadmap lives in `plan/Session-Objectives-MAP.md` and the substance in `plan/scifield_plan.md`.

This is **session 1 of 24** — the very first execution session. Its only job is to stand up the `scifield` Python package with modern tooling so every subsequent session writes into a clean, reproducible repo. Nothing about real pipeline logic. Nothing about Hydra (deferred to V1-S02). Nothing about data harvesting (V1-S03+).

By end of session, three things must be true:
1. `uv run scifield --help` prints CLI usage.
2. `uv run pytest` passes (2 tests).
3. CI is green on the first GitHub push.

User has decided (this session):
- Package lives at **repo root** in `src/scifield/`; empty `Version 1/` and `Version 2/` folders get deleted (vestigial).
- Install `uv` and `gh` via Homebrew.
- Create new private GitHub repo via `gh repo create scifield --private --source=. --push`.
- CLI framework: **Typer**.

---

## Step 0 — Prerequisites

```bash
brew install uv gh
gh auth login   # interactive; user runs if not already authed
```

Then in `/Users/samersalman/Desktop/SciField/`:
```bash
rm -rf "Version 1" "Version 2"   # confirmed empty
rm -f .DS_Store
git init -b main
```

---

## Step 1 — `pyproject.toml`

Hatchling build backend, PEP 621 metadata, `src/` layout, Python ≥3.11. Single runtime dep (`typer`); dev deps via `[dependency-groups]` (PEP 735) so `uv sync` picks them up.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "scifield"
version = "0.0.1"
description = "Multi-axis framework for monitoring scientific field health."
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.11"
authors = [{ name = "Samer Salman" }]
dependencies = [
    "typer>=0.12",
]

[project.scripts]
scifield = "scifield.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/scifield"]

[dependency-groups]
dev = [
    "pytest>=8",
    "ruff>=0.6",
    "black>=24.8",
    "mypy>=1.11",
    "pre-commit>=3.8",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.mypy]
python_version = "3.11"
strict = false
warn_unused_ignores = true
warn_return_any = true
no_implicit_optional = true
ignore_missing_imports = true   # ok for stub-heavy day-zero scaffolding; tighten later

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Note on `strict = false`:** the session's stop condition explicitly says "do not weaken the global mypy config" if stubs fight mypy. We're not weakening — we're starting moderate, which is the right baseline for a project that doesn't have real code yet. V1-S02 can tighten as real code lands.

---

## Step 2 — Package skeleton

Create `src/scifield/__init__.py`:
```python
"""SciField — multi-axis framework for monitoring scientific field health."""

__version__ = "0.0.1"
```

Create one `__init__.py` per axis, each with a single-line docstring naming its plan phase (per V1-S01 brief):

| File | Docstring |
|---|---|
| `src/scifield/corpus/__init__.py` | `"""Phase 1 — Corpus harvesting + enrichment. TODO."""` |
| `src/scifield/thematic/__init__.py` | `"""Phase 2 — Thematic backbone (embeddings + topics). TODO."""` |
| `src/scifield/epistemic/__init__.py` | `"""Phase 3 — Epistemic quality extraction. TODO."""` |
| `src/scifield/novelty/__init__.py` | `"""Phase 4 — Semantic + structural novelty. TODO."""` |
| `src/scifield/forecasting/__init__.py` | `"""Phase 5 — Forecasting (GNN). TODO."""` |
| `src/scifield/integration/__init__.py` | `"""Phase 6 — Integration of findings F1/F2/F3. TODO."""` |

---

## Step 3 — `repro` module (the only real code this session)

`src/scifield/repro/__init__.py` implements `record_run(artifact_path, inputs, config)` per brief. Writes a sidecar JSON at `<artifact_path>.run.json` capturing:

- **`git_sha`** — `git rev-parse HEAD` (subprocess); also `git_dirty: bool` from `git status --porcelain`. If not in a git repo, record `git_sha: null`.
- **`config_hash`** — sha256 of `json.dumps(config, sort_keys=True)`.
- **`input_hashes`** — `{name: sha256_of_file_contents}` for each path in `inputs`.
- **`software_versions`** — `python`, `scifield`, `platform`.
- **`timestamp`** — ISO 8601 UTC.

Function signature:
```python
def record_run(
    artifact_path: Path,
    inputs: dict[str, Path],
    config: dict[str, Any],
) -> Path:
    """Write a sidecar JSON next to artifact_path; return the sidecar path."""
```

Returns the sidecar path so callers can log it.

---

## Step 4 — Typer CLI

`src/scifield/cli.py`:
```python
"""Command-line interface for scifield."""

import typer

app = typer.Typer(
    name="scifield",
    help="SciField — multi-axis framework for monitoring scientific field health.",
    no_args_is_help=True,
)


@app.command()
def demo() -> None:
    """Run the end-to-end demo on a toy corpus (placeholder; implemented in V1-S02)."""
    typer.echo("demo not yet implemented")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
```

---

## Step 5 — Pre-commit (`.pre-commit-config.yaml`)

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=1000]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies: [typer>=0.12]
```

Then run `uv run pre-commit install`.

---

## Step 6 — GitHub Actions CI (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - name: Set up Python
        run: uv python install 3.11
      - name: Sync dependencies
        run: uv sync --all-groups
      - name: Ruff lint
        run: uv run ruff check .
      - name: Ruff format check
        run: uv run ruff format --check .
      - name: Black check
        run: uv run black --check .
      - name: Mypy
        run: uv run mypy src tests
      - name: Pytest
        run: uv run pytest -v
```

---

## Step 7 — Tests

`tests/__init__.py`: empty.

`tests/test_cli.py`:
```python
from typer.testing import CliRunner

from scifield.cli import app

runner = CliRunner()


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "scifield" in result.stdout.lower()
```

`tests/test_repro.py`:
```python
import json
from pathlib import Path

from scifield.repro import record_run


def test_record_run_writes_sidecar(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.parquet"
    artifact.write_bytes(b"fake parquet bytes")

    input_file = tmp_path / "input.csv"
    input_file.write_text("col\n1\n")

    sidecar = record_run(
        artifact_path=artifact,
        inputs={"input": input_file},
        config={"k": "v"},
    )

    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert {"git_sha", "config_hash", "input_hashes", "software_versions", "timestamp"} <= payload.keys()
    assert "input" in payload["input_hashes"]
```

---

## Step 8 — Repo-level files

**`LICENSE`** — Apache 2.0 standard text (https://www.apache.org/licenses/LICENSE-2.0.txt), with copyright line `Copyright 2026 Samer Salman`.

**`README.md`** — one paragraph describing SciField + a "See `plan/` for the full project plan and session-execution roadmap" pointer.

**`.gitignore`** — Python defaults + uv + data artifacts:
```
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.uv/
uv.lock         # NOT — see note below
.mypy_cache/
.ruff_cache/
.pytest_cache/
.DS_Store
data/
models/
*.duckdb
*.parquet
```

**Decision on `uv.lock`:** Commit `uv.lock` to the repo for reproducibility (uv recommends this for application repos, which this is). Will remove that line from `.gitignore` above; only `.venv/`, `.uv/`, caches, and data artifacts are ignored.

---

## Step 9 — Initial commit + GitHub push

Conventional-commit message style (brief calls this out):

```bash
git add -A
git commit -m "chore: initial repo scaffolding (V1-S01)

- pyproject.toml with hatchling build, src/ layout, typer CLI entrypoint
- src/scifield/ package skeleton + axis stubs
- scifield.repro.record_run for sidecar JSON provenance
- pre-commit (ruff, black, mypy) + GitHub Actions CI
- tests for CLI --help and record_run
- Apache 2.0 LICENSE, README, .gitignore"

gh repo create scifield --private --source=. --push
```

---

## Step 10 — Acceptance tests (verification before declaring done)

Per V1-S01 brief, all five must pass:

| # | Test | How |
|---|---|---|
| 1 | `uv run scifield --help` prints usage | `uv sync && uv run scifield --help`; expect exit 0 with help text |
| 2 | `uv run pytest` passes (2 tests) | `uv run pytest -v`; expect 2 passed |
| 3 | `uv run pre-commit run --all-files` passes | run command; expect all hooks pass |
| 4 | CI green on first GitHub push | watch with `gh run watch` after push; expect green |
| 5 | Clean git log with conventional-commit messages | `git log --oneline`; one or more `chore:` / `feat:` commits |

Use the `superpowers:verification-before-completion` skill before claiming done — no asserting success without running the commands.

---

## Critical files (exhaustive list to create)

```
/Users/samersalman/Desktop/SciField/
├── .github/workflows/ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── pyproject.toml
├── src/scifield/
│   ├── __init__.py
│   ├── cli.py
│   ├── corpus/__init__.py
│   ├── epistemic/__init__.py
│   ├── forecasting/__init__.py
│   ├── integration/__init__.py
│   ├── novelty/__init__.py
│   ├── repro/__init__.py
│   └── thematic/__init__.py
└── tests/
    ├── __init__.py
    ├── test_cli.py
    └── test_repro.py
```

Also generated (not authored): `uv.lock` (committed).

Deleted (cleanup): `Version 1/`, `Version 2/`, `.DS_Store`.

---

## Out of scope (per brief — do NOT do)

- Hydra config system → V1-S02
- mkdocs documentation site → V1-S02
- Any data harvesting, real pipeline logic → V1-S03+
- DVC / data versioning beyond sidecar JSON
- Brev scripts → V1-S02

---

## Risks & stop conditions

- **mypy fights stubs.** Brief: ship `# type: ignore` on empty `__init__.py` files only; don't weaken global config. With `ignore_missing_imports = true` and stubs containing only docstrings, this likely won't trigger.
- **First CI run fails.** Most likely cause: ruff/black formatting mismatch between local and CI versions. Fix by running `uv run pre-commit run --all-files` locally, recommitting, repushing. Do not skip CI checks.
- **`gh auth login` requires interactive browser flow.** If user isn't already authed, I'll pause and prompt them to run `! gh auth login` in the session before continuing.
- **Homebrew install of uv/gh fails.** Fall back to official `uv` installer (`curl -LsSf https://astral.sh/uv/install.sh | sh`); for `gh`, pause and ask user.
