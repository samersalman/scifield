# SciField V2, Literature Cartography

**Date:** 2026-06-09 | **Status:** Phase A (prove-phase, current corpus, $0)

V2 is the **exploratory cartography** program for SciField: building a *usable map of the
surgical literature*, where topics originate, how they cascade across journals, which journals
act as sources / bridges / terminals, where novelty arises (sector / geography / funding), and a
descriptive trajectory layer. It deliberately departs from V1's confirmatory, pre-registered-gate
science: V1 (which ships as a methods-and-resource paper) tested three hypotheses and closed with
two honest nulls (F1, F3) and one finding (F2). V2 instead *characterizes structure* and earns
trust through rigor, held-out years, null/permutation models, drop-one-journal jackknife, and
topic-granularity sensitivity, rather than through hypothesis gates. **This session executes
Phases A–E only, on the current 10-journal corpus at $0, and stops at the human Gate G6**, which
decides whether the cartographic method is real enough to justify spending money on corpus
expansion (Phases F–G).

Start here: read [`CONTEXT.md`](CONTEXT.md) (the shared ground truth, scope, data inventory,
conventions) and [`EXECUTION_LOG.md`](EXECUTION_LOG.md) (the cross-task hand-off log). The
governing commitments are in [`docs/cartography/charter.md`](docs/cartography/charter.md) and
[`docs/cartography/validation_protocol.md`](docs/cartography/validation_protocol.md).

---

## Path conventions (why deliverables and code live apart)

Per `CONTEXT.md` §1: the user instruction is "all V2 outputs go in `V2/`." We honor that for
**deliverables**, but Python modules and tests follow the repo's existing convention so notebooks
can `import scifield`, and so `pytest` / `pre-commit` / the editable install keep working
unchanged. Splitting them this way is deliberate, not an oversight:

- **Python logic** → `src/scifield/cartography/`, importable as `scifield.cartography.*`. The
  plan mandates this verbatim ("logic lives in `src/scifield/cartography/`"). Living inside the
  installed package is what makes `import scifield.cartography` work from any notebook.
- **Tests** → `tests/test_cartography_<module>.py`, **flat**, matching the existing
  `tests/test_findings_seeding.py`. `tests/__init__.py` already exists; do **not** create a
  `tests/cartography/` subdirectory (it would break the flat test discovery the repo uses).
- **All generated deliverables** → under `V2/`, docs, notebooks, scripts, data tables, and the
  map. These are the human-facing artifacts the user asked to keep in one place.

Repo root is `/Users/samersalman/Desktop/SciField`. In scripts, resolve paths from a `REPO_ROOT`
constant rather than hard-coding absolutes. Run Python via the project venv: `.venv/bin/python`
(`scifield` is installed editable, so `import scifield` works).

---

## Directory map of `V2/` (as it will be once all phases land)

```
V2/
├── README.md                         # this file, program orientation
├── CONTEXT.md                        # shared ground truth (read first)
├── EXECUTION_LOG.md                  # append-only cross-task hand-off log
├── docs/
│   └── cartography/
│       ├── charter.md                # the map's questions + exploratory stance (LOCKED)
│       ├── validation_protocol.md    # the rigor that replaces the gates (LOCKED)
│       ├── infra_decision.md         # V2-S02 infra spike: go/stay decision + thresholds
│       ├── cascade_validation.md     # V2-S04 verdict: real / artifact / partial
│       └── map_v0/                   # V2-S07 proof-of-concept map notes
├── notebooks/                        # v2_*.ipynb, import scifield.cartography; do I/O + plotting
│   ├── v2_01_cascades.ipynb          # V2-S03 cascade visualizations + lag matrix
│   ├── v2_02_roles_velocity.ipynb    # V2-S05 source/bridge/terminal + velocity
│   ├── v2_03_novelty_origins.ipynb   # V2-S06 novelty by sector/geo/funding
│   ├── v2_04_scaled_cartography.ipynb# V2-S09 (Phase F, gated, not this session)
│   └── v2_05_trajectories.ipynb      # V2-S10 (Phase G, gated, not this session)
├── scripts/                          # small runnable builders that produce the data tables
├── data/                             # output parquet tables + their .run.json sidecars
│   └── flow/                         # V2-S01 journal-flow data model output
└── map_v0/                           # the proof-of-concept map (standalone static HTML)
```

The Python logic these notebooks import lives outside `V2/`:

```
src/scifield/cartography/
├── flow.py        # V2-S01, canonical (pmid → journal_slug) resolver + journal-flow data model
├── cascade.py     # V2-S03, first-appearance, inter-journal lead-lag, diffusion, origin attribution
└── roles.py       # V2-S05, source / bridge / terminal role scores + citational velocity
tests/
├── test_cartography_flow.py
├── test_cartography_cascade.py
└── test_cartography_roles.py
```

---

## Running a notebook headless

Notebooks run on the current corpus and should stay fast (seconds–minutes). Execute in place with
nbconvert (installed), setting a sane timeout:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace V2/notebooks/v2_01_cascades.ipynb
```

Before claiming a code task done, run the repo's quality gates on the touched files:

```bash
.venv/bin/ruff check <files> \
  && .venv/bin/black --check <files> \
  && .venv/bin/pytest tests/test_cartography_<x>.py -q
```

Every parquet / figure / map artifact must be written with a `scifield.repro.record_run` sidecar
JSON (git SHA, config hash, input hashes, software versions), see `CONTEXT.md` §4.

## Reproducing the data tables (`V2/data/` is intentionally git-ignored)

The repo's blanket `.gitignore` rules (`data/`, `*.parquet`) cover `V2/data/**`, so **none of the
14 cartography parquet tables or their `.run.json` sidecars are tracked in git**, by design, matching
the V1 convention (large/derived data stays out of git; provenance lives in the sidecars + the builder
scripts). They are **deterministically regenerable at $0** from the current corpus:

```bash
.venv/bin/python V2/scripts/build_flow.py            # V2/data/flow/*
.venv/bin/python V2/scripts/build_cascade.py         # V2/data/cascade/{lag_matrix,origin_attribution,diffusion_curves}
.venv/bin/python V2/scripts/run_cascade_validation.py# V2/data/cascade/{null,jackknife,granularity}_results
.venv/bin/python V2/scripts/build_roles.py           # V2/data/roles/*
.venv/bin/python V2/scripts/build_origins.py         # V2/data/origins/{sector,geo,recombination}*
.venv/bin/python V2/scripts/build_origins_robustness.py  # V2/data/origins/{geo,sector}_robustness
.venv/bin/python V2/scripts/build_map.py             # V2/map_v0/index.html
```

Note the asymmetry: `V2/map_v0/index.html` and `V2/notebooks/figures/*.png` are **not** ignored
(they are end deliverables, not intermediate tables) and will be committed if you `git add V2/`.

---

## Pointers

- **Shared ground truth (read first):** [`CONTEXT.md`](CONTEXT.md)
- **Cross-task hand-off log:** [`EXECUTION_LOG.md`](EXECUTION_LOG.md)
- **Charter (the map's questions + stance, LOCKED):** [`docs/cartography/charter.md`](docs/cartography/charter.md)
- **Validation protocol (the rigor that replaces the gates, LOCKED):** [`docs/cartography/validation_protocol.md`](docs/cartography/validation_protocol.md)
