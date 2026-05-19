# Phase 0 — Scaffolding

## Phase objective

Establish the package structure, config system, data layout, and
reproducibility infrastructure that every subsequent phase will build into.
This includes the `scifield` package skeleton, a Hydra-driven config layer,
documentation site, pre-commit hooks (ruff, black, mypy), CI from day one,
and a working end-to-end `scifield demo` on a 100-paper toy corpus. The
deliverable is a repo where `uv run scifield --help` works, the docs site
builds, CI passes, and a colleague with Python experience can clone the
repo, run `uv sync && uv run scifield demo`, and see a result in under ten
minutes.
