# V1-S12 — Forecasting: OSF pre-registration #2, temporal CV, all baselines

## Context

SciField's Phase 5 (Forecasting) asks whether a graph neural network can predict topic
emergence 3 years ahead of conventional detection, beating classical baselines. Per the
project's anti-drift discipline (`plans/Session-Objectives-MAP.md`), **the forecasting
protocol must be pre-registered on OSF *before* any model is trained**, and a full set of
baselines must exist so the GNN (built next session, V1-S13) has something to beat.

This session (V1-S12) is the "lock the protocol + build the floor" step. It produces:
the OSF pre-registration #2 document, a leakage-safe topic-level time-series feature
pipeline with an explicit temporal train/val/test split, and the four baselines
evaluated **on the validation set only** (the 2021–2025 test set stays sealed until V1-S14).
It is **CPU/local only** — no GPU (Brev A100 is V1-S13), no paid API, no DeepSeek
(so the DeepSeek spend-gate does not apply here). G1/G2/G3 are all resolved, so the only
gating dependency is OSF pre-reg #2 submission, which blocks V1-S13.

### Decisions locked (with Samer, 2026-06-01)

- **Emergence label = multiplicative share growth.** A `(topic, origin_year=t)` sample is
  *emergent* iff `forward_3yr_mean_share ≥ 1.5 × trailing_3yr_mean_share` AND
  `trailing_3yr_volume ≥ 30 papers`. Topics below the volume guard are *excluded* from that
  origin's labeled set (recorded, never silently dropped). `GAMMA=1.5`, `V_min=30`,
  `H=3` live in config and are tunable; the executor reports train/val class balance so
  Samer can adjust before OSF submission. Rationale: topic-birth is degenerate on this
  corpus (only 1/149 leaf topics first appears after 2017), so emergence must be a
  share-acceleration event on existing topics. Additive-jump and count-surge variants are
  computed and reported as **sensitivity**, not the primary.
- **Split denotes the forecast-ORIGIN year** (feature cutoff), the only convention that
  makes "no test leakage" enforceable per-row. Features use only `year ≤ t`; the label uses
  only `year ∈ {t+1,t+2,t+3}`.
- **OSF = code-complete now, human submits after.** Claude drafts PR2 and builds/runs
  everything with `osf_url: PENDING_OSF_SUBMISSION`; all runnable tests pass. Samer uploads
  to OSF, pastes the DOI into PR2 + config; a cheap CPU re-run stamps the DOI into the
  baseline sidecars. V1-S13 stays blocked until that re-run is committed.
- **Noise (topic_id = -1, 24% of corpus):** share denominator = **leaf papers only** (noise
  excluded from numerator and denominator), per the signed G1 override that noise must be
  explicitly handled. The include-noise denominator is reported as sensitivity. Sidecars
  record `n_noise_total`, `noise_frac`, denominator policy.

## In scope (exact deliverable paths)

- `docs/preregistrations/PR2_forecasting.md` — drafted, mirrors PR1's 11-section structure.
- `src/scifield/forecasting/data.py` — topic-level time-series features + temporal split + leakage asserts.
- `src/scifield/forecasting/baselines/{base,naive,arima,mlp,no_graph,evaluate}.py`.
- `conf/forecasting/v1.yaml` — split, label thresholds, baseline hyperparams, preregistration block.
- `notebooks/09_baselines.ipynb` — validation performance table + class-balance/noise diagnostics.
- `tests/test_forecasting_data.py`, `tests/test_forecasting_baselines.py`, `tests/test_cli_forecasting.py`.
- `pyproject.toml` — add `statsmodels>=0.14`.
- CLI: `forecasting features` + `forecasting baselines` subcommands in `src/scifield/cli.py`.

## Out of scope (defer)

- HGT/TGN architecture, GNN training, the Brev A100 sweep → **V1-S13**.
- Final test-set evaluation, calibration, the Wilcoxon test, Gate G4 → **V1-S14**.
  (S12 *stages* the per-topic score arrays so the Wilcoxon is a later drop-in, but does not run it.)
- `torch-geometric` dependency — not needed until V1-S13; do **not** add it now.
- Reading or materializing any 2021–2025 (test-origin) rows.

## Data & conventions (verified)

- `data/v1/archetypes.parquet` (89,230 rows): `pmid, year(1995–2026; 2026 partial), journal,
  topic_id(-1=noise,0..148), sem_nov_mean, sem_nov_min, cd5, cd10, cited_by_count,
  cited_by_pctile_within_year, n_prior, openalex_id`. This is the convenient join (year+topic attached).
- `data/v1/topics.parquet` (V1-S06b canonical): `pmid, topic_id, is_noise`.
- `data/v1/papers.duckdb` `papers` table: `publication_types VARCHAR[]` → cheap RCT/review-share
  features (the LLM epistemic layer is incomplete and is **not** used here). NB: `pmid` is VARCHAR
  in duckdb, int64 in parquets — cast on join.
- `repro.record_run(artifact_path, inputs: dict[str,Path], config: dict) -> Path` → writes
  `{path}.run.json`. Call after writing each artifact.
- CLI is **Typer**; add `forecasting_app` and `app.add_typer(forecasting_app, name="forecasting")`,
  plus `_load_forecasting_config(name="v1")` mirroring `_load_novelty_config` in `cli.py`.
- Config is flat YAML via `OmegaConf.load`; mirror `conf/novelty/v1.yaml`.
- Tests: pure functions, hand-built numpy/pandas fixtures, `pytest.approx`, no real I/O
  (mirror `tests/test_novelty_semantic.py`).

## Implementation (bottom-up build order)

**1. `pyproject.toml`** — add `"statsmodels>=0.14"` to `dependencies`; `uv sync`. (torch, scikit-learn, pandas already present.)

**2. `conf/forecasting/v1.yaml`** — flat YAML:
`input` (archetypes_parquet, topics_parquet, duckdb_path); `output` (features_parquet,
metrics_parquet); `split` (train=1998–2017, val=2018–2020, test=2021–2022, horizon=3,
trailing_window=3); `label` (gamma=1.5, v_min=30, mode=multiplicative, delta=0.002);
`noise` (denominator=leaf_only, noise_topic_id=-1); `features` (mlp_columns, node_columns lists);
`baselines.{naive,arima,mlp,no_graph}` hyperparams; `preregistration` (osf_url:
"PENDING_OSF_SUBMISSION", pr_doc: docs/preregistrations/PR2_forecasting.md).

**3. `src/scifield/forecasting/data.py`** — the testable core (pure functions + one I/O entrypoint):
```python
MLP_FEATURES: tuple[str, ...]      # Block A: count/share/growth/momentum/accel/volatility/age/share_of_max
NODE_FEATURES: tuple[str, ...]     # Block B ⊇ A: + trailing novelty (sem_nov, cd5/cd10, cited_by_pctile)
                                   #   + rct_share_3yr/review_share_3yr (from publication_types)
                                   #   + n_journals_3yr  ← the future GNN's node-feature contract
def compute_yearly_topic_counts(papers, noise_topic_id=-1) -> DataFrame   # topic_id,year,n,N,share
def build_features(counts, novelty, epistemic, trailing_window=3) -> DataFrame
def build_labels(counts, horizon=3, gamma=1.5, v_min=30, mode="multiplicative", delta=0.002) -> DataFrame
def assign_split(origin_year, split_cfg) -> Series
def assert_no_leakage(features, labels, split, *, horizon, allowed_origins) -> None
def materialize(cfg, *, allow_test=False) -> tuple[DataFrame, dict]   # real I/O; allow_test default False
```
Features strictly trailing (`year ≤ t`); labels strictly leading (`t+1..t+3`); rows whose forward
window exceeds corpus max are `label_complete=False` and excluded from labeled sets. `materialize`
refuses test rows unless `allow_test=True` (S12 never passes it).

**4. `src/scifield/forecasting/baselines/base.py`** — common interface so all 4 baselines + the
future GNN are interchangeable:
```python
class ForecastPrediction(NamedTuple): emergence_score: np.ndarray; share_forecast: np.ndarray
class Predictor(Protocol):
    name: str
    def fit(self, features: DataFrame, labels: DataFrame) -> "Predictor": ...
    def predict(self, features: DataFrame) -> ForecastPrediction: ...
def emergence_auc(scores, labels) -> float
def share_mape(forecast, target, eps) -> float   # computed only where forward_share > 0 (pre-registered guard)
```

**5–8. baselines** (each implements `Predictor`, returns BOTH an emergence score for AUC and a
share forecast for MAPE):
- `naive.py` `NaiveMovingAverage`: share_forecast = trailing 3yr mean; emergence_score = logistic of `share_growth_3yr`.
- `arima.py` `ArimaPerTopic` (statsmodels): per-topic ARIMA on the trailing share series; 3-step forecast;
  emergence_score from forecast mean vs `GAMMA×trailing` via predictive-std normal CDF. **Short/degenerate/
  non-converging series → naive fallback** (try/except, `fallback=True` diagnostic); never crash the run.
- `mlp.py` `MlpForecaster` (torch CPU): consumes `MLP_FEATURES`; shared MLP with sigmoid (emergence)
  + linear (log-share) heads; scaler fit on **train only**; deterministic seed.
- `no_graph.py` `NoGraphForecaster`: identical trainer to `mlp.py` but consumes `NODE_FEATURES`
  (the GNN node features, no edges) — the honest "GNN minus graph" ablation. Share the trainer;
  differ only by config-driven `feature_columns`.

**9. `src/scifield/forecasting/baselines/evaluate.py`** — `run_baselines(features, cfg) -> DataFrame`
(metrics table: emergence AUC + share MAPE per baseline on validation) and `per_topic_scores(...)`
(arrays staged for the later Wilcoxon).

**10. `src/scifield/cli.py`** — `_load_forecasting_config`, `forecasting_app`, `app.add_typer(...)`,
and two commands: `forecasting features` (calls `materialize`, writes features_parquet + `record_run`)
and `forecasting baselines` (loads features, runs `run_baselines`, writes metrics_parquet + `record_run`
**with `cfg.preregistration` folded into the config dict** so the OSF link lands in every sidecar).

**11. tests** — synthetic fixtures only. Cover: the 7 leakage assertions (feature causality, label
horizon, split disjointness, boundary values, label completeness, no-test-read default,
`sum_g share=1` per year), the label math (known closed-form), ARIMA→naive fallback on a constant
series, MLP forward/shape + AUC/MAPE plumbing on a tiny frame (1–2 epochs), and a CLI smoke test.

**12. `notebooks/09_baselines.ipynb`** — repo-root sniff header; load metrics_parquet; render the
validation AUC+MAPE table, per-split positive rates / class balance, noise diagnostics. **No test reads.**

**13. `docs/preregistrations/PR2_forecasting.md`** — mirror PR1's structure: background; hypotheses;
operational definitions (emergence = multiplicative share growth GAMMA=1.5/V_min=30/H=3, the origin-year
split, the feature schema, leaf-only share denominator); primary analysis (emergence AUC + share MAPE);
baselines spec; statistical comparison (Wilcoxon signed-rank across topics — run in V1-S14); pre-registered
pass/fail (Gate G4: GNN > best baseline by >5pp emergence AUC, Wilcoxon p<0.05); pivot conditions
(F3 as null finding; additive-slope label as pre-registered sensitivity); data/code availability;
OSF workflow (front-matter `OSF DOI / URL` line filled by Samer post-submission, mirroring PR1). Flag
GAMMA/V_min as the human's scientific call.

## Verification (acceptance tests)

Runnable by Claude this session:
1. `uv run scifield forecasting features` → `data/v1/forecasting_features.parquet` + sidecar; runtime
   asserts pass (no leakage; test rows absent).
2. `uv run scifield forecasting baselines` → `data/v1/forecasting_baselines.parquet` + sidecar; all
   **4 baselines produce emergence-AUC + share-MAPE on the validation set**.
3. `uv run pytest` green (incl. the new leakage + baseline + CLI tests).
4. `uv run pre-commit run --all-files` green.
5. `notebooks/09_baselines.ipynb` executes end-to-end; renders the baseline table + class balance;
   no 2021–2025 reads. Class-balance (train/val positive rate) reported for Samer to sanity-check
   before OSF submission.
6. PR2 drafted; baseline sidecars contain the `preregistration` block (with `PENDING_OSF_SUBMISSION`).

Completed by Samer (the brief's "submitted with public link" test — human action):
7. Upload `PR2_forecasting.md` to OSF → DOI; paste DOI into PR2 front matter + `conf/forecasting/v1.yaml`;
   re-run `scifield forecasting baselines` (cheap, CPU) so sidecars carry the real OSF URL; commit.
   **Only then is V1-S13 unblocked.**

## Stop conditions / guardrails

- Do **not** read or materialize test-origin (2021–2025) rows. Do not start V1-S13.
- If train positive rate is degenerate (<5% or >95%) at GAMMA=1.5/V_min=30, surface it and let
  Samer retune via config before he submits to OSF — do not silently change the thresholds.
- Do not add `torch-geometric`. Do not call any paid API.
- Mark V1-S12 `✓` in `plans/Session-Objectives-MAP.md` only after the human OSF re-run is committed
  (note "code-complete; pending OSF DOI" in the interim).
