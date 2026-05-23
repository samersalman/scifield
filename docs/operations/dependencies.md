# Dependency log

This file records dependency-resolution decisions and resolved versions for SciField, especially when adding non-trivial third-party stacks (BERTopic, UMAP, HDBSCAN, gensim, etc.) that have historically had numpy / scipy / numba ABI churn.

## 2026-05-21 — plan-S06 T1: BERTopic stack added

### Why
Plan-S06 introduces the thematic-topic pipeline (BERTopic + UMAP + HDBSCAN), the temporal smoothness analysis (gensim coherence + dynamic topic helpers), interactive topic visualizations (plotly), and an executed-notebook smoke check (nbconvert). `scikit-learn` is now pinned explicitly so the CountVectorizer / clustering version BERTopic relies on does not drift silently behind a transitive bound.

### Resolved versions

| Package        | Version    |
| -------------- | ---------- |
| bertopic       | 0.17.4     |
| umap-learn     | 0.5.12     |
| hdbscan        | 0.8.43     |
| gensim         | 4.4.0      |
| plotly         | 6.7.0      |
| scikit-learn   | 1.8.0      |
| nbconvert      | 7.17.1     |
| numpy          | 2.4.6      |

Python: `3.11.15` (via `uv run python --version`).

### numpy<2 constraint
**Not applied.** The plan-S06 risk note flagged that gensim / hdbscan wheels historically pinned `numpy<2`, which would have forced us to constrain `numpy>=1.26,<2.0` in `pyproject.toml`. With the current resolved wheels (gensim 4.4.0, hdbscan 0.8.43, numba 0.65.1, llvmlite 0.47.0) `uv sync` resolved cleanly against numpy 2.4.6, so no constraint was needed. If a future `uv sync` regresses (e.g. numba publishes a wheel that re-pins numpy<2), revisit this and add the constraint back.

### Transitive additions worth knowing
`uv sync` also pulled in: `llvmlite==0.47.0`, `numba==0.65.1`, `pynndescent==0.6.0`, `smart-open==7.6.1`, `wrapt==2.2.0`, `narwhals==2.21.2`. These are required by UMAP / HDBSCAN / gensim / plotly respectively and were resolved without conflict.

### Verification
Import smoke test passed for all 8 packages (bertopic, umap, hdbscan, gensim, plotly, sklearn, nbconvert, numpy). `hdbscan` does not expose `__version__` on the module; use `importlib.metadata.version("hdbscan")` instead.
