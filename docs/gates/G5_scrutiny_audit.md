# Gate G5 — scrutiny audit (numbers-verified integrity record)

**Session:** V1-S15 scrutiny  **Date generated:** 2026-06-09  **Plan ref:** `plans/i-just-made-a-cuddly-moth.md` (Phase 0 — scrutiny record)

This document is the permanent record of the scrutiny Samer asked for: an independent re-verification of every headline number behind Gate **G5 = DOWNSCOPE** (signed 2026-06-08), a plain statement of the integrity finding, and the three substantive flags that surfaced during the audit. It was produced by independently re-deriving the numbers from the saved artifacts — not by transcribing the gate report or the results draft — and every entry below was confirmed against the file named in its row.

**This document does NOT alter signed Gate G5.** Signed `docs/gates/G5_v1_findings.md`, and the pre-registrations PR2 (`docs/preregistrations/`, OSF DOI 10.17605/OSF.IO/XP94F) and PR3 (`docs/preregistrations/PR3_epistemic_cascade.md`), stand as the honest, unedited record of what the *differenced* (F1) and *original-test* (F3) analyses found. Nothing here re-counts the ≥2-of-3 bar, re-scores any finding, or reopens the DOWNSCOPE disposition. The three flags are interpretation / power / framing issues; **none is a code bug, a leakage path, or a pre-registration violation.** Any consequence they may have is routed through the additive, self-pre-registered path described in §5 — never a retroactive edit to a signed gate.

---

## 1. Purpose & scope

Gate G5 closed V1 with a signed **DOWNSCOPE**: of the three pre-registered findings only **F2 (dual novelty)** holds with statistical support; **F1 (epistemic cascade — the pre-registered deciding vote)** and **F3 (graph forecasting)** are honest, pre-specified nulls, landing the statistical count at **1 of 3**, below the **≥2-of-3** bar. V1 ships as a methods-and-resource paper; the V2 scale-up is not triggered.

Samer asked for two deliverables: (1) **scrutinize each finding** — confirm it is correctly computed and truthfully reported, with no bugs; and (2) where flags surface, **a plan to resolve them**. This document is deliverable (1): the scrutiny record. It is preceded by a full read-only audit (3 Explore agents over the F1/F2/F3 compute + 2 Plan agents designing the resolution). The resolution plan (deliverable 2) lives in `plans/i-just-made-a-cuddly-moth.md` and is summarized in §6.

Scope of this record:

- **In scope:** re-verifying the audited numbers (§2), stating the integrity finding (§3), documenting the three flags with evidence and a TRUE-vs-APPARENT assessment (§4), and stating the integrity model that keeps any reopening clean (§5).
- **Out of scope:** changing any signed gate, re-scoring the ≥2-of-3 count, running any new statistic, or pre-judging the Phase-2 human checkpoint. This document records; it does not decide.

---

## 2. Numbers-verified table

Every number below was re-derived in this session from the named artifact. ✓ = reproduces exactly (to the stated precision); ✗ = flag (none found among the audited numbers).

| # | Quantity | Verified value | Artifact source | Status |
|---|---|---|---|---|
| **F1 — epistemic cascade (pre-registered NULL)** | | | | |
| 1 | Primary directional count | **0 / 138** (frac_directional 0.000; n_none 138; dominant_direction "none") | `data/v1/f1_cascade_verdict.json` `primary` | ✓ reproduces |
| 2 | Primary panel p-values | quality→volume **p = 0.27712**, volume→quality **p = 0.73928** | `data/v1/f1_cascade_verdict.json` `panel_primary` | ✓ reproduces |
| 3 | RCT-share robustness panel | quality→volume **p = 0.95023**, volume→quality **p = 0.95623** (agree = true) | `data/v1/f1_cascade_verdict.json` `panel_rct_share` | ✓ reproduces |
| 4 | F1 verdict | **"NULL"**; n_qualifying 138, n_non_evaluable (primary) 10, n_evaluable_primary 128, n_evaluable_rct_share 138 | `data/v1/f1_cascade_verdict.json` | ✓ reproduces |
| 5 | **BH multiple-comparison m** | **m = 252** (see derivation below) | `data/v1/f1_cascade_results.parquet` | ✓ reproduces |
| 6 | Smallest survivable p at BH | **0.05 / 252 = 1.984 × 10⁻⁴**; actual min observed **7.745 × 10⁻⁴**; **15** raw p < 0.05; **0** survive BH-FDR | `data/v1/f1_cascade_results.parquet` (recomputed) | ✓ reproduces |
| **F2 — dual novelty (HOLDS)** | | | | |
| 7 | Kruskal–Wallis omnibus | **H = 1938.5836728582**, p < 1e-300 (scipy returns 0.0), k = 4, **n = 81,738** | `data/v1/archetypes.parquet` via `src/scifield/findings/impact.py` | ✓ reproduces |
| 8 | Rank-effect size ε² | **0.0236815** = (H − k + 1)/(n − k); hand-recompute identical | same | ✓ reproduces (artifact 0.02368; results_draft rounds to 0.024) |
| 9 | Archetype medians (within-year impact pctile) | disruptive-novel **0.4329** < novel-consolidating **0.5023** < conventional-disruptive **0.5234** < incremental **0.6069** (monotone inversion) | same | ✓ reproduces |
| 10 | Circularity check (CD-index vs impact) | **NEGATIVE** — cd5 Pearson **−0.126** / Spearman **−0.061**; cd10 Pearson **−0.119** / Spearman **−0.071** (all p ≪ 1e-60) | `data/v1/archetypes.parquet` (recomputed) | ✓ reproduces (against mechanical coupling) |
| 11 | Axis independence (G3, context) | max\|ρ\| = 0.18 (Pearson 0.1828 / Spearman 0.1751, sem_nov_min × cd10) | `docs/gates/G3_dual_novelty.md` | ✓ consistent |
| **F3 — graph forecasting (registered NULL)** | | | | |
| 12 | Sealed test AUC | no_graph **0.8043077** vs hgt **0.7809231** → Δ = **−2.3385 pp** (wrong side of +5pp bar) | `data/v1/forecasting_test_metrics.parquet` + `forecasting_test_wilcoxon.json` `verdict` | ✓ reproduces |
| 13 | Paired Brier Wilcoxon | **p = 2.6290 × 10⁻¹¹**, direction = **favors_baseline**, n_pairs = 138, statistic 1659.0, median_diff 0.2330 | `data/v1/forecasting_test_wilcoxon.json` `primary_brier` | ✓ reproduces |
| 14 | Test positives / base rate | n_train 1298, n_test **138**, **13 positives** (≈ 9.4%); 7 in 2021, 6 in 2022 | `data/v1/forecasting_test_metrics.parquet` + `forecasting_test_per_unit.parquet` | ✓ reproduces |
| 15 | HGT calibration / share error | hgt share-MAPE **0.5729** (worst of five models; no_graph 0.4610) | `data/v1/forecasting_test_metrics.parquet` | ✓ reproduces |
| 16 | Winner's-curse margin (val) | best-of-**5 completed** Optuna trials (15 pruned of 20); winner val_auc **0.736680** vs runner-up **0.728981** → **+0.77 pp** | `data/v1/forecasting_sweep.parquet` (recomputed) | ✓ reproduces (plan's "+0.8pp" is the loose form; true = +0.77pp) |

### BH m = 252 — full derivation

`notebooks/11_F1_epistemic_cascade.ipynb` pools the two directional Granger p-values (`p_q_to_v`, `p_v_to_q`) across topics, then applies `bh_fdr`. The pool spans the **128 "ok" topics × 2 directions = 256 slots**, but `bh_fdr` operates only on **finite** p-values — and **2 of the 128 ok topics carry a NaN in each direction** (126 finite `p_q_to_v` + 126 finite `p_v_to_q`). The BH denominator is therefore the count of finite pooled p-values:

```
m = 126 (p_q_to_v finite) + 126 (p_v_to_q finite) = 252
0.05 / 252 = 1.984 × 10⁻⁴   ← smallest p that could survive at q = 0.05
```

This matches `docs/results_draft.md` line 19 ("m = 252 hypotheses requires the smallest p-value to be ≤ 1.98 × 10⁻⁴"). The actual minimum observed p is **7.745 × 10⁻⁴ > 1.984 × 10⁻⁴**, so **0 of 252** survive — confirming F1 is a true null, not an underpowered miss. The 10 non-evaluable topics are retained in the 138-topic denominator (they can only make the 20% fraction harder to clear).

---

## 3. Finding: no bugs, no leakage, pre-registrations honored

Stated plainly, and confirmed by independent re-derivation of every number in §2:

- **No bugs.** Every reported statistic reproduces from the saved artifacts. F1's 0/138 and the m = 252 BH math; F2's H = 1938.58, ε² = 0.0236815, and the four archetype medians; F3's −2.34pp and the paired Brier p = 2.63 × 10⁻¹¹ all re-derive exactly. The ε² is hand-checkable from H/k/n and matches.
- **No leakage.** F3's evaluation is on a sealed test set fixed in advance (forecast-origin years 2021–2022; n_train 1298 / n_test 138), with model selection done on the validation split only and the HGT frozen at test time (Gate G4 record). The F1/F2 computes are cross-sectional or within-topic over the fixed corpus with no train/test split to leak.
- **Pre-registrations honored.** F1's PR3 (`bab917d`, 2026-06-08 21:28:50) was git-timestamped **before** the F1 compute (`015ee9b`, 2026-06-08 21:44:42) — F1 was computed once, after PR3 (Gate G5 provenance table). F3's PR2 is OSF-registered (DOI XP94F). The reported verdicts use the pre-registered bars (F1: ≥20% directional at BH-FDR q < 0.05 + panel; F3: HGT − no_graph > +5pp test AUC). Both nulls were specified before their analyses were run.

**The signed DOWNSCOPE is defensible on its own terms.** The differenced F1 test and the original-test F3 analysis each answered exactly the question they pre-registered, and each found a null. Gate G5 is the honest record of those findings.

---

## 4. The three flags

These are **NOT code bugs, leakage, or pre-registration violations** — every number in §2 reproduces. They are interpretation / power / framing issues, drawn from the audit's flag table (`plans/i-just-made-a-cuddly-moth.md` lines 19–23). Each is recorded with the flag, the evidence, and a TRUE-vs-APPARENT assessment of whether the underlying deficiency is real.

### 🚩 Flag 1 — F1: differenced test answered a narrower question than the level language claims → most likely APPARENT

- **The flag.** The pre-registered F1 test (PR3) runs the Granger probe on **once-differenced** series (`docs/results_draft.md` line 15: "a Granger test (fixed lag 3, on differenced series)"). Differencing **structurally removes the long-run level / cointegration relationship** between quality and volume. But an "epistemic cascade" — higher-quality topics attracting more research volume over time — is most naturally a **level** phenomenon: it lives in whether the *levels* of quality and volume co-move and lead/lag, not in whether their *year-to-year changes* do.
- **Evidence.** Series length is fine — the qualifying bar is ≥8 years at ≥5 papers/yr with v_min = 30, and the median qualifying series is ~30 observations. So the null is **not** a sample-size / power problem. It is a **specification-scope** problem: the differenced test correctly answered a narrower question (do short-run *changes* lead/lag?) and found a clean null, but it never tested the level relationship. The framing then drifts: `docs/results_draft.md` line 19 describes the null in **level** language — "evidence quality and research volume **rise together** without either systematically leading the other" — which is a statement about co-movement of *levels* that a differenced test cannot support. The differenced null says nothing about whether levels rise together.
- **TRUE vs APPARENT.** **Most likely APPARENT** — the high-value thread. The deficiency the gate reports (no detectable cascade) may be an artifact of testing the short-run-changes question instead of the level question. This does not impugn the signed gate (which honestly reports the differenced result); it means the *level* cascade was never tested, and the results-draft language overreaches the differenced evidence. Phase 1A (a self-pre-registered Toda–Yamamoto levels diagnostic + a power simulation, §6) is designed to classify this as TRUE / APPARENT / AMBIGUOUS against a pre-stated table.

### 🚩 Flag 2 — F3: the AUC arm is underpowered; the paired Brier test is the real evidence → most likely TRUE

- **The flag.** The headline F3 arm — HGT − no_graph = **−2.34pp** against a **+5pp** bar — is **underpowered** at only **13 positives** in the sealed test (n = 138). The audit estimates the paired AUC-gap 95% CI at **≈ [−10pp, +5.5pp]** with roughly **~25% power** to detect a true +5pp effect. *(This CI is the audit's estimate; it is rigorously confirmed in the Phase 1B characterization, `notebooks/16_F3_characterization.ipynb` — a teammate is computing the stratified paired bootstrap now.)*
- **Evidence.** The real statistical teeth are the **paired per-topic Brier-loss Wilcoxon test**, which uses **all 138 units** (well-powered) and is highly significant at **p = 2.63 × 10⁻¹¹** in the direction that **favors the baseline** — HGT is systematically worse-calibrated, over-predicting (mass at 0.4–0.8 where realized emergence ≈ 0; worst-in-class share-MAPE 0.5729). This calibration test, not the underpowered AUC arm, **should be the lead** for F3. Separately, the validation→test ranking flip (no_graph 0.701→0.804, HGT 0.737→0.781) is better explained by **winner's-curse** than by a regime shift: the HGT config was the best of **5 completed** Optuna trials (15 of 20 pruned), and the winner beat the runner-up by only **+0.77 pp** on validation (0.736680 vs 0.728981) — note this is the precise margin, *not* the +0.8pp loosely stated in the plan. On the test set the **baseline improved more** than HGT (HGT did not collapse — it rose 0.737→0.781), exactly the signature of an over-fit val-selected winner regressing while a simpler model generalizes.
- **TRUE vs APPARENT.** **Most likely TRUE.** The graph genuinely adds no calibration value: the well-powered, all-138-unit Brier test is decisively against the graph, and the AUC arm — though underpowered — also lands on the wrong side of zero. The deficiency is real; the only correction is framing (lead with Brier, flag the AUC arm's CI and power, correct the winner's-curse margin to +0.77pp / best-of-5).

### 🚩 Flag 3 — F2: holds honestly but modestly; needs only proportion language → sound

- **The flag.** F2 holds, but **modestly** — ε² ≈ **0.024** (≈ 2.4% of rank variance) — on the **fuzziest** of the pre-registered bars (the "supported impact gradient" / "≥1 surprising-to-expert finding" sub-criterion).
- **Evidence.** The omnibus is overwhelmingly reliable (H = 1938.58 on n = 81,738, p < 1e-300) and the ordering is cleanly monotone (0.433 < 0.502 < 0.523 < 0.607). The two potential confounds were checked and came back clean: the impact metric is an **age-fair within-year percentile** (older papers are not credited for longer accrual), and the **circularity check is negative** — CD-index vs impact correlates **−0.13 / −0.12** (Pearson, cd5/cd10), the *opposite* sign of what mechanical coupling between the CD construction and the impact metric would produce. So the inversion is not an artifact of the metric.
- **TRUE vs APPARENT.** **Sound.** F2 is a genuine, reliable finding; it simply needs **proportion language** — the strength is the *consistency and reliability of the ordering*, not the magnitude of the gap. `docs/results_draft.md` already frames it this way ("small but highly reliable"); no statistical change is warranted.

---

## 5. Integrity model for any reopening

Retroactively flipping a pre-registered null to a "hold" is p-hacking regardless of intent. So if any flag warrants follow-up, it routes through a **clean, additive path** — never a retroactive edit (summarizing `plans/i-just-made-a-cuddly-moth.md` lines 26–31):

1. **Signed G5, PR2, and PR3 are preserved** as the honest record of what the *differenced* (F1) / *original-test* (F3) analyses found. Nothing in any follow-up edits them.
2. New analyses split into two kinds:
   - **(a) Diagnostics that re-describe existing data** — bootstrap CIs, power/MDE statements, descriptive decompositions — need **no new pre-registration** (they add no new statistic, only characterize a saved result).
   - **(b) New statistics** — e.g. a levels test that was not in PR3 — need a **self-pre-registration committed and git-timestamped BEFORE compute**, run **once**, with a **symmetric** decision rule (a null is as reportable as a positive). Same anti-drift discipline as PR3.
3. **A finding can only "flip" via a fresh, pre-registered *confirmatory* study**, recorded as an **additive new gate (G5′ / G6)** that *adds to* — never overwrites — the signed G5. Even then, statistical re-passage (e.g. F1′ + F2 = 2-of-3) **still leaves the co-author narrative-coherence sub-criterion to human judgment**; it is not auto-satisfied by a statistical pass.

---

## 6. What happens next

- **Phase 1 diagnostics** (CPU-only, $0, no new data) characterize each null:
  - **F1 levels diagnostic** — a Toda–Yamamoto level-causality test plus a power simulation, self-pre-registered as **PR3D** (`docs/preregistrations/PR3D_levels_diagnostic.md`, committed before compute) and driven by **`notebooks/15_F1_levels_diagnostic.ipynb`**. It reuses the same 138-topic denominator / FDR / 20% / panel bars as PR3 so the verdict is structurally comparable, and the power simulation tests whether the differenced test was *blind* to an injected level cascade while the levels test was powered.
  - **F3 characterization** — a stratified paired bootstrap (13-pos / 125-neg strata, fixed seed) for the AUC-gap CI, an explicit power/MDE statement for the +5pp bar, a foregrounded-Brier narrative, and a winner's-curse decomposition — all from saved artifacts, **no new pre-registration**, in **`notebooks/16_F3_characterization.ipynb`**.
- Each diagnostic carries a **pre-stated decision table** classifying the null as **TRUE / APPARENT / AMBIGUOUS**, and records `does_not_flip_g5: true`.
- **Phase 2 is a human checkpoint:** the verdicts go to Samer, who decides per finding whether to escalate to a fresh confirmatory pre-registration (Phase 3). **No G5 reopening happens without his sign-off** — matching the project's gate discipline.

---

## Provenance

| Item | Value |
|---|---|
| Git SHA at time of writing | `b5a696cd06b57deb94aba870d6655cb5533144ce` |
| Branch | `v1-s09-epistemic-validation` |
| Plan ref | `plans/i-just-made-a-cuddly-moth.md` (Phase 0) |
| Signed gate this audits (unaltered) | `docs/gates/G5_v1_findings.md` (DOWNSCOPE, Samer, 2026-06-08) |
| Pre-registrations (unaltered) | PR2 (OSF DOI 10.17605/OSF.IO/XP94F); PR3 `docs/preregistrations/PR3_epistemic_cascade.md` (`bab917d`) |
| Verified artifacts (read-only) | `data/v1/f1_cascade_verdict.json`, `data/v1/f1_cascade_results.parquet`, `data/v1/forecasting_test_metrics.parquet`, `data/v1/forecasting_test_wilcoxon.json`, `data/v1/forecasting_sweep.parquet`, `data/v1/archetypes.parquet` |
| F2 impact module | `src/scifield/findings/impact.py` (pure; statistic computed live in `notebooks/12_F2_dual_novelty_trends.ipynb`) |
| Verification method | every number in §2 independently re-derived this session from the named artifact; not transcribed |

This is a scrutiny record only. It does not alter the signed Gate G5 disposition. An optional acknowledgment by Samer is left blank below; he countersigns if and when he chooses.

Acknowledged: ____________________  Date: __________
