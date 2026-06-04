# Pre-registration #2: Forecasting topic emergence in the orthopedic-surgery literature

**Authors:** Samer G. Salman, Baylor College of Medicine (ORCID: [0009-0007-9897-4071](https://orcid.org/0009-0007-9897-4071))
**Date drafted:** 2026-06-01
**OSF DOI / URL:** [10.17605/OSF.IO/XP94F](https://doi.org/10.17605/OSF.IO/XP94F) (registration page: https://osf.io/xp94f/)
**OSF parent project:** [10.17605/OSF.IO/RXYZ4](https://doi.org/10.17605/OSF.IO/RXYZ4) (project page: https://osf.io/rxyz4/)
**Project commit SHA at submission:** `670a74b`
**Codebase:** https://github.com/samersalman/scifield  (Zenodo DOI minted at release v0.1.0 — 10.5281/zenodo.20360023 | concept DOI: https://doi.org/10.5281/zenodo.20360022)
**Plan reference:** `plans/plan-v1-s12.md`
**Master plan reference:** `plan/scifield_plan.md` §5 Phase 5

---

## 1. Background and rationale

SciField is a multi-axis framework for monitoring the health of the orthopedic-surgery research field. Across phases, it layers (a) a deduplicated 10-journal PubMed corpus (V1-S03/S04), (b) a topic landscape over abstract embeddings (V1-S05/S06), (c) an LLM-assisted epistemic-quality layer over abstracts (Phase 3), (d) a novelty signal (Phase 4), and (e) a forecasting layer (Phase 5), with downstream analyses in Phase 6. Phase 5 asks a single, sharply scoped question: **can a graph neural network predict topic emergence three years ahead of conventional detection, beating classical time-series baselines?**

This pre-registration scopes the forecasting protocol — the emergence label, the temporal cross-validation split, the feature schema, the baseline floor, and the primary pass/fail gate — *before any forecasting model is trained*. Per the project's anti-drift discipline (`plans/Session-Objectives-MAP.md`), the forecasting protocol must be registered prior to model fitting so that the choice of label threshold, evaluation horizon, and comparison statistic cannot be tuned post hoc to a favorable result. The session that this document accompanies (V1-S12) is the "lock the protocol + build the floor" step: it commits this pre-registration, the leakage-safe feature pipeline, and a full set of classical baselines evaluated **on the validation set only**, so that the graph neural network built in the next session (V1-S13) has a fixed, pre-declared bar to clear.

The corpus operated on is the 10-journal orthopedic-and-surgery subset of PubMed. The forecasting layer reads the V1-S06b canonical topic assignment (149 leaf topics plus a noise class, `topic_id = -1`) joined to per-paper year and journal via `data/v1/archetypes.parquet` (89,230 rows spanning 1995–2026, with 2026 partial). The topic landscape's G1 PROCEED override clearance is documented in `docs/gates/G1_topic_interpretability.md`, and the requirement that the noise class be handled explicitly (rather than silently folded into a topic) is carried forward from that signed override.

A pre-registration is necessary here for a second, corpus-specific reason. The intuitive operationalization of "emergence" — a topic that is *born* (first appears) within the forecast window — is **degenerate on this corpus**: only 1 of the 149 leaf topics first appears after 2017. Topic-birth therefore yields a near-empty positive class and cannot support a forecasting study. We instead operationalize emergence as a **share-acceleration event** on already-existing topics (defined in §3). Because that operationalization involves a human-chosen growth multiplier and volume floor, pre-registering those choices — and pre-declaring the additive-jump and count-surge variants as *sensitivity* analyses rather than as escape hatches — is what gives the eventual result its credibility.

---

## 2. Hypotheses

- **H1 (graph lift).** A heterogeneous graph neural network (HGT, built in V1-S13) achieves emergence AUC at the 3-year forecast horizon that exceeds the best classical baseline's emergence AUC by **more than 5 percentage points**.
- **H2 (significance).** That AUC margin is statistically significant across topics by a **Wilcoxon signed-rank test on per-topic scores (p < 0.05)**.

These hypotheses are pre-registered **before any forecasting model is trained** — before the GNN exists and before the classical baselines have been evaluated on anything but the validation set. Any deviation from the pre-registered analysis plan will be reported transparently in the registered report. The graph neural network referenced in H1 is built and trained in V1-S13; the Wilcoxon test in H2 is computed in V1-S14 (this session only stages the per-topic score arrays that test will consume — see §7 and §8).

---

## 3. Operational definitions

The unit of analysis is a `(topic, origin_year)` pair, defined and sampled in §4. The definitions below specify how each such sample is labeled, how the temporal split is assigned, what features are computed, and how the share denominator is constructed. The canonical machine-readable source for every threshold and column list is `conf/forecasting/v1.yaml`; the operational rules below are what the pipeline applies.

### 3.1 Emergence label (primary): multiplicative share growth

A `(topic, origin_year = t)` sample is labeled **emergent** iff **both** of the following hold:

1. `forward_3yr_mean_share ≥ GAMMA × trailing_3yr_mean_share`, and
2. `trailing_3yr_volume ≥ V_min`,

with **`GAMMA = 1.5`**, **`V_min = 30`** papers, and forecast horizon **`H = 3`** years. Here `trailing_3yr_mean_share` is the topic's mean leaf-only share over the trailing window `[t-2, t]`, `forward_3yr_mean_share` is its mean leaf-only share over the forward window `[t+1, t+3]`, and `trailing_3yr_volume` is the topic's paper count summed over `[t-2, t]`.

**`GAMMA` and `V_min` are the human investigator's scientific call, not a tuned hyperparameter.** They are set deliberately — a 50% relative rise in publication share, sustained on a topic that already carried at least 30 papers over its trailing window, is the chosen operationalization of "this topic is taking off." Both live in `conf/forecasting/v1.yaml` (`label.gamma`, `label.v_min`) and are tunable; the executor reports the resulting train/validation class balance so that the investigator can adjust them **before** OSF submission if the positive class is degenerate. The threshold values frozen in this document are the values submitted to OSF.

The multiplicative-share-growth label is the **primary** emergence definition. Two alternative labels are computed on every row and reported only as **sensitivity** analyses (§9): an **additive-jump** label (`forward_3yr_mean_share − trailing_3yr_mean_share ≥ delta`, with `delta = 0.002`) and a **count-surge** label (`forward_3yr_volume ≥ GAMMA × trailing_3yr_volume`). Neither alternative is part of the primary hypothesis test.

### 3.2 Origin-year split convention

The split label denotes the **forecast-origin year** `t` (the feature cutoff), which is the only convention under which "no test leakage" is enforceable on a per-row basis. For every sample:

- **Features** use only data from years **`≤ t`** (strictly trailing).
- **Labels** use only data from years **`{t+1, t+2, t+3}`** (strictly leading).

### 3.3 Temporal split

Samples are partitioned by origin year into three disjoint, time-ordered blocks (ranges inclusive):

| Split | Origin years `t` | Status |
|---|---|---|
| **train** | 1998–2017 | used to fit baselines (and, in V1-S13, the GNN) |
| **val** | 2018–2020 | used for all model selection and the validation metrics in this session |
| **test** | 2021–2022 | **SEALED** — never read or materialized until V1-S14 |

The test block is sealed: the V1-S12 materializer refuses to emit test-origin rows (§5), and no analysis in this session touches origin years 2021–2022. Final test-set evaluation occurs in V1-S14.

### 3.4 Feature schema

Two feature blocks are computed per `(topic, origin_year)`. Block B is a strict superset of Block A (Block A's nine columns come first, in order). The MLP baseline consumes Block A; the no-graph ablation and (in V1-S13) the GNN node features consume Block B. Column names are fixed; they are byte-identical to `data.MLP_FEATURES` / `data.NODE_FEATURES` and to `conf/forecasting/v1.yaml` (`features.mlp_columns`, `features.node_columns`).

**Block A — "MLP" features (9):** `count_3yr`, `share_3yr_mean`, `share_last`, `share_growth_3yr`, `share_momentum`, `share_accel`, `share_volatility_3yr`, `topic_age`, `share_of_max`.

**Block B — "node" features (16, ⊇ Block A):** all nine Block A columns, plus `sem_nov_mean_3yr`, `cd5_3yr`, `cd10_3yr`, `cited_by_pctile_3yr` (trailing novelty / impact aggregates, NaN-skipping over the topic's papers in `[t-2, t]`), `rct_share_3yr` and `review_share_3yr` (trailing publication-type shares derived from `publication_types` in `data/v1/papers.duckdb`), and `n_journals_3yr` (distinct journals among the topic's papers in `[t-2, t]`). Block B is the future GNN's node-feature contract.

All trailing aggregations skip missing values (novelty columns carry NaNs: `cd5` ≈ 94% non-null, `sem_nov` ≈ 97%, `cited_by_pctile` ≈ 98%). The LLM epistemic layer (Phase 3) is **not** used in this forecasting layer; the only epistemic-adjacent signals are the cheap, fully observed `rct_share_3yr` / `review_share_3yr` derived from PubMed publication-type tags.

### 3.5 Leaf-only share denominator

For a given year `y`, the share denominator `N(y)` is the count of papers assigned to a **leaf topic** in year `y` — i.e., the noise class (`topic_id = -1`, 24% of the corpus, 21,409 rows) is excluded from **both** the numerator and the denominator. A topic's share is `share(g, y) = n(g, y) / N(y)`, so leaf-only shares sum to 1 within every year (enforced as a leakage assertion; see §5). This implements the signed G1 override that the noise class be handled explicitly. The **include-noise** denominator (noise counted in `N(y)`) is reported as a **sensitivity** analysis; sidecars record `n_noise_total`, `noise_frac`, and the denominator policy (`leaf_only`).

---

## 4. Sampling plan / units of analysis

- **Unit of analysis.** One row per `(topic_id, origin_year = t)`, restricted to **leaf topics** (`topic_id ≠ -1`). Origin years are drawn from the train and validation blocks only in this session (1998–2020); test-block origins (2021–2022) are sealed.
- **Population.** All `(leaf topic, origin year)` combinations derivable from `data/v1/archetypes.parquet` for which a trailing window is computable. The corpus carries 149 leaf topics; the noise class is excluded from the unit set (and from the share denominator, §3.5).
- **Volume guard.** A sample enters the **labeled set** only if `trailing_3yr_volume ≥ V_min` (`volume_ok = True`) **and** the forward window lies within the corpus (`label_complete = True`, i.e., `t + H ≤` the corpus's maximum year). Samples failing the volume guard are **recorded, not silently dropped**: they are kept in the materialized frame with `volume_ok = False` and excluded from the labeled set used for fitting and metrics. The `emergent` flag is computed for every row but consumed only on the labeled set.
- **Class balance reporting.** Per-split positive rate and sample count (train / val) are written into the materializer's `info` dict and the baseline-metrics sidecar, and rendered in `notebooks/09_baselines.ipynb`, so the investigator can sanity-check the labeled-set balance before OSF submission. If the train positive rate is degenerate (below 5% or above 95%) at `GAMMA = 1.5` / `V_min = 30`, that condition is surfaced for the investigator to retune the thresholds via config prior to submission rather than being silently accepted.
- **Output artifact.** `data/v1/forecasting_features.parquet` (one row per `(topic_id, origin_year)` with Block B features, label columns, and a `split` column), written by `scifield forecasting features`. A sidecar `.run.json` captures git SHA + the full config + class-balance / noise diagnostics, enabling exact replay.

---

## 5. Feature pipeline and leakage controls

The forecasting feature pipeline is built so that **no information from year `> t` can enter a sample whose origin year is `t`**, and so that test-origin rows can never be evaluated in this session.

- **Strictly trailing features.** Every feature in Blocks A and B is computed from data in `[…, t]` only (the trailing 3-year window is `[t-2, t]`; longer-lookback features such as `share_growth_3yr`, `topic_age`, and `share_of_max` still read only years `≤ t`).
- **Strictly leading labels.** Every label component (`forward_3yr_mean_share`, the emergent flag, and the sensitivity variants) reads only data from `{t+1, t+2, t+3}`. The trailing-side label components (`trailing_3yr_mean_share`, `trailing_3yr_volume`) read only `[t-2, t]`.
- **Leakage assertions.** `assert_no_leakage(...)` enforces **seven** checks and raises `AssertionError` with a short keyword on any violation:
  1. every materialized `origin_year` is within the allowed set (`leakage:allowed_origins`);
  2. split labels are a subset of the allowed names and the materialized set excludes `test` (`leakage:test_present`);
  3. train/val/test origin-year ranges are disjoint and ordered, `max(train) < min(val) < min(test)` (`leakage:disjoint`);
  4. boundary origins land in the correct split — 2017 → train, 2018 and 2020 → val, 2021 → test (`leakage:boundary`);
  5. every labeled-set row has `label_complete = True`, i.e., `t + H` lies within the corpus (`leakage:label_complete`);
  6. features and labels share identical `(topic_id, origin_year)` keys — no orphan future rows (`leakage:key_align`);
  7. per-year leaf shares sum to ≈ 1 within tolerance 1e-6 (`leakage:share_sum`).
- **No-test-read guard.** `materialize(cfg, *, allow_test=False)` is the single I/O entrypoint; it drops `split == "test"` rows unless `allow_test=True`, and **V1-S12 never passes `True`**. The 2021–2022 test-origin rows are therefore not read into any artifact or analysis in this session.

---

## 6. Models (baseline floor)

Four classical baselines are implemented and evaluated **on the validation set** in this session. All four expose a common interface (`Predictor`): each `fit(features, labels)` on the labeled training rows and `predict(features)` returns, for each input row, **both** an emergence score (consumed by emergence AUC) and a share forecast (consumed by share MAPE). All four are interchangeable with the future GNN under the same interface.

- **`NaiveMovingAverage`** (`name = "naive"`). A no-op fit; predicts `share_forecast = share_3yr_mean` (the trailing 3-year mean share) and an `emergence_score` that is a monotonic function of `share_growth_3yr`. AUC is rank-based, so any monotonic transform of the trailing growth ratio is admissible.
- **`ArimaPerTopic`** (`name = "arima"`, default order `(1, 1, 0)`). Fits a per-topic ARIMA (statsmodels) on the topic's trailing share series, forecasts 3 steps ahead, sets `share_forecast` to the mean of the three forecast steps, and derives `emergence_score` as the probability that the forecast mean reaches `GAMMA × trailing_mean` under a normal CDF using the forecast predictive standard deviation. **Short, degenerate, or non-converging series fall back to the naive predictor** (via `try`/`except`), and the fallback rate is exposed as a diagnostic; the run never crashes on an unfittable topic.
- **`MlpForecaster`** (`name = "mlp"`). A torch (CPU) multilayer perceptron consuming **Block A** (`MLP_FEATURES`): a shared trunk with a sigmoid head (emergence) and a linear head (log-share). The feature scaler is fit on **training rows only**; the seed is fixed for determinism.
- **`NoGraphForecaster`** (`name = "no_graph"`). The **same trainer** as `MlpForecaster`, differing only in that it consumes **Block B** (`NODE_FEATURES`, the GNN node features with no edges). This is the honest **"GNN minus graph"** ablation: it isolates the contribution of the graph structure by giving a non-graph model the exact node-feature inputs the GNN will receive.

The graph neural network itself (HGT) is **not** part of this session; it is built and trained in V1-S13. No GPU, no paid API, and no `torch-geometric` dependency are used here.

---

## 7. Primary analysis

Two metrics are computed for every model, on the **validation set** in this session (and, in V1-S14, on the sealed test set):

- **Emergence AUC.** The ROC-AUC of the model's emergence score against the binary `emergent` label, over labeled validation rows (`split == "val"` & `volume_ok` & `label_complete`). If the labeled validation set is single-class, AUC is returned as `NaN` (guarded, not raised). This is the metric named in hypotheses H1 and H2.
- **Share MAPE.** The mean absolute percentage error of `share_forecast` against `forward_share` (`= forward_3yr_mean_share`), **computed only over rows where `forward_share > 0`** — a pre-registered guard so that zero-forward-share rows cannot make the percentage error undefined or unbounded. Returned as `NaN` if no row qualifies.

The validation metrics table (one row per baseline: `baseline`, `emergence_auc`, `share_mape`, `n_train`, `n_val`, `n_val_pos`, `fallback_frac`) is written to `data/v1/forecasting_baselines.parquet` and rendered in `notebooks/09_baselines.ipynb`. In addition, **per-topic score arrays are staged** (`per_topic_scores(...)`) so that the V1-S14 Wilcoxon signed-rank test (H2) is a later drop-in; the staged arrays carry, per baseline, the per-row `topic_id`, `emergence_score`, `emergent`, `share_forecast`, and `forward_share`. The Wilcoxon test itself is **not** run in this session.

---

## 8. Pre-registered pass/fail criteria

**Gate G4 (the primary forecasting gate).** The forecasting feature (F3) is declared a success iff **both** conditions hold on the **sealed test set** in V1-S14:

1. the graph neural network (HGT) beats the **best classical baseline** by **more than 5 percentage points of emergence AUC at the 3-year horizon**, and
2. the **per-topic Wilcoxon signed-rank test** comparing the GNN's per-topic scores against the best baseline's is **significant at p < 0.05**.

These two conditions correspond to H1 and H2 respectively.

**Timing.** Gate G4 is *adjudicated* in V1-S14, not in this session. V1-S12 (this session) only (a) freezes the label, split, feature schema, metrics, and the >5pp + p<0.05 bar in this document, and (b) stages the per-topic score arrays that the Wilcoxon will consume. The Wilcoxon signed-rank test is **run in V1-S14**, not now. The GNN that G4 evaluates is trained in V1-S13.

---

## 9. Pivot conditions

**If Gate G4 fails** — the GNN does not clear the best baseline by >5pp emergence AUC, or the per-topic Wilcoxon is not significant at p < 0.05 — then the forecasting feature **F3 is reported as a null finding** (the graph structure did not buy a forecastable lift over classical time-series baselines) **or dropped** from the manuscript, rather than being rescued by re-tuning the label or the horizon after the fact.

**Pre-registered sensitivity analyses (not the primary).** The following are registered here as sensitivity checks and are *not* substitutes for the primary multiplicative-share-growth label under G4:

- the **additive-jump** ("additive-slope") label variant (`forward − trailing ≥ delta`, `delta = 0.002`);
- the **count-surge** label variant (`forward_3yr_volume ≥ GAMMA × trailing_3yr_volume`);
- the **include-noise** share denominator (noise counted in `N(y)`, contrasted with the primary leaf-only denominator).

Agreement (or disagreement) of the primary result with these variants is reported descriptively; a favorable sensitivity variant cannot convert a failed primary G4 into a pass.

---

## 10. Data and code availability

- **Code.** All forecasting source under `src/scifield/forecasting/` (the feature pipeline `data.py` and the baselines package `baselines/`). Tests under `tests/test_forecasting_*.py` (leakage assertions, label math, ARIMA→naive fallback, MLP plumbing, and a CLI smoke test). The commit SHA at OSF submission time will be pasted into the front matter above (`Project commit SHA at submission`).
- **Configuration.** The single source of truth for the split, the label thresholds (`GAMMA`, `V_min`, `H`, `delta`), the noise/denominator policy, the feature column lists, and the baseline hyperparameters is `conf/forecasting/v1.yaml`, which also carries the `preregistration` block (`osf_url`, `pr_doc`).
- **Data.** The corpus DuckDB (`data/v1/papers.duckdb`) is reproducible from the V1-S04 harvest plus V1-S05 enrichment/dedup; the topic join (`data/v1/archetypes.parquet`) is the V1-S06b deliverable. The forecasting feature matrix (`data/v1/forecasting_features.parquet`) and the validation-metrics table (`data/v1/forecasting_baselines.parquet`) are committed at submission time, each with a `.run.json` sidecar that records the git SHA, the full resolved config, and (for the baselines run) the `preregistration` block, so every artifact is recoverable and carries its OSF provenance.
- **Sealed test set.** No 2021–2022 (test-origin) rows are read, materialized, or shared in this session; they remain sealed until V1-S14.
- **License.** Repository's existing LICENSE (Apache-2.0 per the repository's `LICENSE` file).

---

## 11. OSF submission workflow

1. Claude drafts this markdown file at `docs/preregistrations/PR2_forecasting.md` with `OSF DOI / URL: PENDING_OSF_SUBMISSION` and `Project commit SHA at submission: PENDING_OSF_SUBMISSION`, and builds/runs the full V1-S12 pipeline so that all runnable tests pass and the four baselines produce emergence AUC + share MAPE on the validation set, with every baseline sidecar carrying the `preregistration` block (`osf_url: PENDING_OSF_SUBMISSION`).
2. The PI uploads this document to the Open Science Framework as a registered pre-registration.
3. OSF mints a DOI / shortlink for the registration.
4. The PI pastes the resulting DOI into the front matter above on the `OSF DOI / URL` line **and** into `conf/forecasting/v1.yaml` (`preregistration.osf_url`), and fills in the `Project commit SHA at submission` line.
5. The PI re-runs `scifield forecasting baselines` (a cheap, CPU-only run) so that the regenerated baseline sidecars carry the real OSF URL, then commits the updated document, config, and sidecars. The acceptance grep specified in the plan (a substring match against this file's `OSF DOI / URL` line for the registration's host domain) then passes.
6. **V1-S13 (GNN training on the Brev A100) must NOT begin until step 5's commit is in.**
