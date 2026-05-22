# Brev operations

## Why Brev hygiene matters

NVIDIA Brev bills by the minute and continues billing until an instance is
explicitly stopped. The single largest source of wasted credits is a
forgotten running machine — a GPU instance left on overnight can burn
through more credits than a full week of intentional work. The rules below
are non-negotiable: always run `brev stop <name>` the moment a job finishes,
prefer snapshots over keeping instances alive between sessions, and use the
smallest instance class that meets the workload's needs (CPU-only for
smoke tests, L40S for embeddings, A100 only for GNN training). See
`plan/scifield_plan.md` §8 for the full allocation strategy and the per-phase
GPU budget.

## Cost table (placeholder — verify current pricing before running)

| Instance class | Use case | Approximate rate |
|---|---|---|
| Smallest CPU (e.g. 2 vCPU) | Smoke tests, `brev_smoke.sh`, CPU-only utilities | ~$0.05/hr |
| Medium CPU (8+ vCPU) | Corpus harvesting, novelty graph computations | ~$0.15/hr |
| L40S 48GB (on-demand) | Phase 2 embedding of ~200k abstracts | ~$0.50/hr |
| A100 80GB (spot) | Phase 5 GNN hyperparameter sweeps (checkpoint + resume) | ~$1.00/hr |
| A100 80GB (on-demand) | Phase 5/7 final GNN training runs | ~$2.00/hr |

> **Verify current pricing before running.** Brev is a meta-broker over
> Lambda, GCP, and other clouds; rates fluctuate. Check the Brev console for
> the live rate in your pinned region before launching, and update this
> table when prices shift materially.

## Smoke test

The repository ships a defensive shell script at `scripts/brev_smoke.sh`
that exercises the full launch → sync → demo → stop cycle on the smallest
CPU instance. Running it once before any real GPU work catches credential,
networking, and CLI-version issues while the cost is still pennies.

**Prerequisites**

- Brev CLI installed: `brew install brevdev/homebrew-brev/brev`
  (or use the curl installer at <https://brev.dev>).
- Authenticated: `brev login` (opens a browser for SSO).
- Sanity check: `brev --version` prints a version.

**How to run**

```bash
bash scripts/brev_smoke.sh
```

**What it does**

1. **Launch** the smallest CPU instance tagged `scifield-smoke-v1s02`,
   cloning this repository onto it.
2. **Sync** the Python environment via `uv sync` on the remote.
3. **Demo:** runs `uv run scifield demo` to pull 100 PubMed abstracts and
   confirm the pipeline executes end-to-end.
4. **Stop:** unconditionally stops the instance via a `trap` on EXIT so a
   crashed step never leaves a machine billing in the background.

The script is defensive — if the `brev` CLI is not installed or not
authenticated, it exits cleanly with a documented message rather than
failing the build, and the smoke run is deferred to a later session.

## Smoke run log

**2026-05-19 (V1-S02 execute) — first attempt.** Live smoke on
`n2d-highcpu-2` ($0.05/hr, GCP). `brev create` succeeded, instance
reached `Ready`, `brev exec` reached the SSH layer, but the original
implementation did the repo clone in `--startup-script` and raced
against `brev exec`: Brev marks an instance `Ready` as soon as SSH is
up, *not* when the startup script finishes, so `brev exec` found
`$HOME/scifield` missing and bailed. The trap-based `brev stop` fired
cleanly on EXIT (no orphan instance). Fix landed in commit
`fix(brev): clone inline inside brev exec, not in --startup-script`.

**2026-05-19 (V1-S02 execute) — retry with inline clone.** Fresh
`n2d-highcpu-2`. Full cycle completed end-to-end:

```
brev create → Ready → brev exec (git clone + uv install + uv sync +
uv run scifield demo) → n_papers=100 mean_abstract_chars=1682 →
trap brev stop on EXIT → brev delete
```

Output matched the local-machine run exactly (100 papers, 1682 mean
abstract chars). Total wall time ~10 minutes, total spend ~$0.01. The
launch / clone / sync / run / stop harness is now validated against
the V1-S02 demo; V1-S05's first real GPU launch will be the second
time this harness has been exercised, not the first.

## V1-S05 (2026-05-21): Brev deferred, ran locally

The V1-S05 full-corpus embedding (99,938 abstract-bearing papers, mpnet
768d) was run on **local Mac CPU** rather than the Brev L40S originally
planned. This is a deliberate deviation, not a Brev failure.

**Why deferred:**
1. The bake-off (notebooks/03_embedding_bakeoff.ipynb) established that
   mpnet encodes ~500 abstracts in ~28 s on Mac CPU (CPU-forced, 2 torch
   threads). Extrapolating gives ~1.5 hr for 100 k papers; the actual run
   took 6.3 hr (CPU contention from other workloads on the host).
2. Brev's L40S would have completed in ~25 min wall-clock at ~$0.50–$1,
   but the local path costs $0 and skips first-run risk of any Brev
   script bugs (`scripts/brev_embed.sh` has not yet been executed
   end-to-end against a billed GPU).
3. Recorded in `data/v1/embeddings.parquet.run.json` via the sidecar's
   `config.gpu_model = "cpu"` and `config.device = "cpu"` fields.

**Credit balance:** no Brev minutes were spent during V1-S05.

**Carryover for V1-S06+:** `scripts/brev_embed.sh` is staged and
syntax-checked but **never executed**. The first session that needs
multi-hour GPU compute (likely V1-S08 attention / V1-S15 GNN training)
should run `brev_embed.sh` first as a paid smoke against a real L40S
before depending on it.

---

## V1-S06 (2026-05-21): no Brev needed

The topic-modelling step (UMAP + HDBSCAN + BERTopic + Gensim coherence) ran
locally on the Mac CPU. The 3×2 sweep + final fit landed inside ~1 hr per
plan-S06; UMAP is single-process by design and HDBSCAN benefits little
from GPU on a corpus of ~89k vectors. No Brev launch this session — the
GPU credit budget reserved for Phase 2 was unused.

---

## V1-S06b (2026-05-21): no Brev needed; ran locally on Mac CPU

The post-G1 `RETUNE_CLUSTERING` session (plan-S06b) re-fit the BERTopic
pipeline against the same V1-S05 embeddings (no re-embed), so it had the
same cost profile as V1-S06: pure CPU, no GPU benefit. Phase 1 mini-sweep
(4 configs, ~6.5 min before the V1-S06 sweep-parquet serialization defect
surfaced and aborted the run; see G1 retune-results append) plus phase 2
widen sweep (9 configs + final fit + hierarchy = 17.0 min) totalled
~23.5 min wall on the same 14-thread Mac host. No Brev launch this
session. Credit balance unchanged.
