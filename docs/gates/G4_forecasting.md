# Gate G4 — 3-year emergence forecasting (graph vs. graph-free)

**Session:** V1-S14 (Sealed test-set evaluation + Gate G4 report)  **Date generated:** 2026-06-05  **Plan ref:** `plans/Session-Objectives-MAP.md` §V1-S14 (and PR2 §6/§9)

G4 tests whether the heterogeneous-graph model (HGT) beats the validation-selected graph-free baseline at forecasting 3-year topic emergence on the **sealed test set**, by a pre-registered margin. If HGT clears the bar, F3 is a positive "graph structure helps forecasting" result; if it does not, F3 stands as a characterized null. This is the **STOP gate before V1-S15** — under PR2 §9 / Gate G4 there is no retraining and no label/horizon re-tuning after this evaluation. Every number below is reproduced end-to-end by `notebooks/10_forecasting_eval.ipynb`.

---

## Pass criteria

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| H1 — HGT vs. graph-free lift @3yr (PR2 §6 ablation) | `HGT − no_graph > +5.0pp AUC @3yr` | **−2.34pp** (0.7809 vs. 0.8043) | **FAIL** |
| H2 — paired per-topic significance | `paired Brier-loss Wilcoxon p < 0.05` | `p = 2.63e-11` — significant, but **direction = favors_baseline (HGT worse)** | **does NOT support F3** |

> **Comparator rationale.** The comparator is `no_graph` because it is (a) the validation-selected best baseline and (b) exactly PR2 §6's GNN-minus-graph ablation — the apples-to-apples "what does adding graph structure buy you" contrast. **H2 operationalization.** H2 was operationalized (locked with Samer) as a paired Wilcoxon on per-row **Brier loss** (HGT vs. `no_graph`; unit = (topic, origin_year); lower = better), because the pre-registered *intent* is "is HGT's per-topic forecast meaningfully better," which is a calibrated-error comparison rather than a raw-score comparison; the raw-score Wilcoxon is reported as a secondary robustness check (§3). **Joint reading.** H2 is mechanically `p < 0.05`, but the significant direction **favors the baseline** (HGT carries higher Brier loss), so H2 does **not** support the HGT-superiority hypothesis. H2's significance running *against* HGT, combined with H1's outright failure, makes this an **unambiguous null** — not a near-miss. Row 2 is therefore marked "does NOT support F3" rather than a bare PASS.

## Provenance

Sidecar: `data/v1/forecasting_test_wilcoxon.json.run.json`

| Field | Value |
|---|---|
| config_hash | `6d739ccdc650bfdd742992330c7c0d04ae9abd490f1bf12afad254b5a8574a05` |
| git_sha | `7c737d402976a2f2c65722a4972c6e029ac28ef1` |
| git_dirty | `true` |
| timestamp | `2026-06-05T04:34:57.344322+00:00` |
| OSF DOI (PR2) | `https://doi.org/10.17605/OSF.IO/XP94F` |

- Artifact: `data/v1/forecasting_test_metrics.parquet` (+ sidecar `data/v1/forecasting_test_metrics.parquet.run.json`) — the 5-model test table.
- Artifact: `data/v1/forecasting_test_wilcoxon.json` (+ sidecar `data/v1/forecasting_test_wilcoxon.json.run.json`) — H1/H2 verdict, primary Brier + secondary raw-score Wilcoxon.
- Artifact: `data/v1/forecasting_test_calibration.parquet` (+ sidecar `data/v1/forecasting_test_calibration.parquet.run.json`) — per-bin calibration.
- Artifact: `data/v1/forecasting_test_sensitivity.parquet` (+ sidecar `data/v1/forecasting_test_sensitivity.parquet.run.json`) — alternate-label sensitivity.
- Artifact: `data/v1/forecasting_test_per_unit.parquet` (+ sidecar `data/v1/forecasting_test_per_unit.parquet.run.json`) — per-(topic, origin_year) scores feeding the paired tests.
- Notebook: `notebooks/10_forecasting_eval.ipynb` (executes end-to-end; reproduces every number in this report).
- Figure: `docs/figures/F3_forecasting.png`.

---

## 1. Full 5-model test table

Sealed test set: forecast-**origin** years 2021–2022. `n_train = 1298`, `n_test = 138`, `n_test_pos = 13` (≈ 9.4% positive). Source: `data/v1/forecasting_test_metrics.parquet`.

| model | emergence_auc | share_mape | fallback_frac |
|---|---:|---:|---:|
| naive | 0.4074 | 0.2605 | — |
| arima | 0.6529 | 0.2744 | 0.1449 |
| mlp | 0.7009 | 0.3954 | — |
| no_graph | **0.8043** | 0.4610 | — |
| hgt | **0.7809** | 0.5729 | — |

The graph-free `no_graph` model is the strongest on test-set emergence AUC (0.8043). HGT (0.7809) is below it. On the share-forecast error (`share_mape`), HGT is the **worst** of the five models (0.5729) — a first indication of the calibration problem characterized in §4.

## 2. Ablation — HGT vs. no_graph (PR2 §6)

The PR2 §6 GNN-minus-graph ablation isolates the marginal value of the heterogeneous graph: same supervised target and horizon, with vs. without graph message-passing. The result is a **negative** graph contribution:

- HGT emergence AUC = **0.7809**; no_graph = **0.8043**; delta = **−2.3384615384615337 pp** (≈ **−2.34pp**).

This is not merely short of the +5.0pp pre-registered bar — it is on the **wrong side of zero**. A positive result would have required HGT ≥ 0.8543 AUC (no_graph + 5.0pp); instead HGT *trails* the graph-free model. On this corpus and test split, adding the heterogeneous graph did not help and modestly hurt.

## 3. Wilcoxon detail (H2)

Source: `data/v1/forecasting_test_wilcoxon.json`. Paired over n_pairs = 138 units = (topic, origin_year).

**Primary — paired Brier-loss (HGT vs. no_graph; lower = better):**
- statistic = **1659.0**, p = **2.63e-11**, median_diff = **+0.233**, direction = **favors_baseline**.
- Interpretation: the difference is highly significant, but the **sign is against HGT** — HGT's per-row Brier loss is systematically *higher* (worse-calibrated). Mechanically `p < 0.05` is satisfied, but it certifies that HGT is significantly **worse**, not better. This does **not** support the HGT-superiority hypothesis.

**Secondary — raw-score Wilcoxon (robustness):**
- statistic = **0.0**, p = **2.16e-24**, direction = **model_higher**.
- Interpretation: HGT emits systematically **higher** emergence scores than no_graph across essentially every paired unit. This is consistent with the over-prediction mechanism in §4 and explains *why* HGT loses the Brier comparison.

Both tests point the same way: HGT systematically predicts higher, is significantly worse-calibrated, and the significance therefore reinforces the null rather than rescuing F3.

## 4. Calibration summary — the over-prediction mechanism

Source: `data/v1/forecasting_test_calibration.parquet`. The two models occupy almost disjoint regions of score space.

| model | where its predictions live | observed emergence in those bins |
|---|---|---|
| no_graph | concentrated near 0: **129 / 138** rows in [0.0, 0.1] (mean predicted **0.026** vs. observed **0.093**); 7 rows in [0.1, 0.2]; 2 rows in [0.3, 0.4] | matches the ≈9.4% base rate — slightly under-confident but well-located |
| hgt | spread across [0.1, 0.8], mass in [0.4, 0.7] (26 rows in 0.4–0.5, 37 in 0.5–0.6, 34 in 0.6–0.7) | observed = **0.0** in bins 0.1–0.5; only **0.135 / 0.206 / 0.143** in bins 0.5–0.6 / 0.6–0.7 / 0.7–0.8 |

HGT **over-predicts severely**: it assigns 0.4–0.8 emergence probability to many (topic, origin_year) units whose realized emergence frequency is 0 or near it, while the true positive rate is ≈9.4%. This single mechanism drives both adverse outcomes — the worst-in-class share-MAPE (0.5729, §1) and the losing paired Brier comparison (§3). no_graph, by contrast, keeps its mass near the base rate and is roughly calibrated.

## 5. Sensitivity — alternate emergence labels

Source: `data/v1/forecasting_test_sensitivity.parquet`. HGT emergence AUC under alternate label definitions (descriptive only — sensitivity **cannot** convert a fail into a pass):

| variant | label_column | n_test_pos | HGT emergence_auc | Δ vs. primary (pp) |
|---|---|---:|---:|---:|
| primary (leaf_only) | `emergent` | 13 | 0.7809 | 0.00 |
| additive-jump | `emergent_additive` | 30 | 0.6315 | −14.94 |
| count-surge | `emergent_count_surge` | 5 | 0.7068 | −7.42 |

The null is **robust to the label definition**: HGT is no better — in fact lower — under every variant, never approaching the +5pp bar over its own baseline.

> **Noted deviation / limitation.** The include-noise *denominator* variant was **not** recomputed in this gate run. `leaf_only` is the pre-registered primary denominator (PR2), so the primary verdict is unaffected; the missing denominator-variant sweep is logged here as a transparent deviation rather than silently omitted.

## 6. Reconciliation note — origin years vs. outcome window

The forecasting map / figure labels the test span "**2021–2025**"; the verdict and tables here use test **origin** years **2021–2022**. These are consistent: the map's "2021–2025" is the **outcome** window (origin 2022 → forward 3-year window 2023–2025), while PR2 §3.3 locks the test **origin** years to 2021–2022. The corpus ends 2026, so a 3-year-ahead label needs origin ≤ 2023; the config additionally caps the test origin at 2022. Both descriptions refer to the same sealed evaluation; only the anchoring (origin vs. outcome window) differs.

---

## Recommendation

The objective pre-registered criteria are **not** met: H1 fails outright (HGT −2.34pp *below* the graph-free baseline, far from the +5.0pp bar), and H2's paired significance runs **against** HGT (significantly worse-calibrated, `p = 2.63e-11`, direction = favors_baseline). `overall_pass = H1 ∧ H2 = False`. Mechanically this is a **NULL FINDING**. A clean, well-characterized null is publishable (plan §6): F3 stands as "**graph structure did not improve 3-yr emergence forecasting over a graph-free baseline on the sealed test set**," with a characterized mechanism (HGT over-prediction / miscalibration, §4). The val→test ranking shifted on a small, high-variance test set (n_test = 138, 13 positives; no_graph 0.701→0.804, hgt 0.737→0.781), but even granting that variance HGT does not clear — indeed sits on the wrong side of — the bar; this is a clean FAIL/NULL, not a near-miss to be re-litigated. The final F3 disposition (and any PASS/NULL/DROP architectural call) is **Samer's**.

- [ ] PASS (F3 success)
- [x] NULL FINDING (F3 reported as null — graph adds no test-set lift)
- [ ] DROP F3

**Decision: NULL FINDING.** F3 is reported as a characterized null — on the sealed test the citation-graph (HGT) adds no forecasting value beyond the temporal/structural node-features, and is significantly worse-calibrated. No retraining or re-tuning of this gate (PR2 §9). A re-tuned follow-up (V1-S14-2) is logged separately and **requires a fresh pre-registration / holdout** — it may not re-score the now-used 2021–2022 origin-year test as confirmatory evidence.

Signed: _Samer G. Salman_  Date: _2026-06-05_

Gate G4 is resolved (NULL FINDING); per the Session-Objectives map, **V1-S15 is unblocked.**
