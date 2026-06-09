# F3 characterization — proposed Results prose (for T6 to fold into `results_draft.md`)

**Status: proposed block, not yet integrated.** This is a re-description of the ALREADY-LOCKED Gate
G4 / V1-S14 sealed-test null (signed NULL, Samer 2026-06-05; OSF PR2 DOI `10.17605/OSF.IO/XP94F`).
It introduces **no new experiment, no model retraining, and no new pre-registration** — only
diagnostics over saved artifacts (bootstrap CIs, power/MDE, decompositions), which the Phase-1
integrity model explicitly permits without a pre-reg. Every number below is computed in
`notebooks/16_F3_characterization.ipynb` and written to
`data/v1/forecasting_characterization.json` (+ `…_bootstrap.parquet`); the figure is
`docs/figures/F3_characterization.png`. **It does not edit, overwrite, or reopen Gate G4 — it
firms up the null and sharpens how it is reported.** The finding remains a **TRUE null**: the
citation graph adds no calibration value beyond temporal/feature dynamics.

T6 should treat the three paragraphs below as a drop-in replacement for the current F3 sub-section,
keeping the lead order (Brier first, AUC caveat second, winner's-curse third).

---

## Proposed prose

**Lead (well-powered evidence): the graph model is significantly worse-calibrated.** On the sealed
test set (forecast-origin years 2021–2022; *n* = 138 topic-years, 13 emergent ≈ 9.4% base rate), the
decisive, well-powered comparison is the **paired per-topic Brier-loss Wilcoxon signed-rank test**,
which uses **all 138 units** (every unit contributes, not only the 13 positives). It is highly
significant **against** the graph model: *p* = 2.63 × 10⁻¹¹ (signed-rank statistic = 1659.0,
*n* = 138 pairs, median per-unit Brier difference = +0.233, direction = *favors_baseline*), with the
heterogeneous-graph transformer (HGT) carrying the higher Brier loss on **125 of 138 units**. The
mechanism is over-prediction: HGT concentrates predicted-emergence mass in the 0.4–0.8 range — **104
of 138 units** fall there, where realized emergence is only ≈12.5% (and 0% in the 0.4–0.5 bin) —
whereas the graph-free baseline (`no_graph`) keeps **129 of 138 units** below 0.1 predicted
probability, with an observed frequency of 9.3% that tracks the 9.4% base rate almost exactly. The
same over-prediction shows up in the point forecasts: HGT's share-MAPE (0.573) is the **worst of all
five models** (vs `no_graph` 0.461). The citation graph does not merely fail to help — it actively
degrades calibration, and that conclusion rests on a test that uses every test unit and is not
sample-size–limited.

**The headline AUC arm is underpowered and cannot adjudicate the +5pp bar.** The pre-registered H1
margin (HGT must beat `no_graph` by > 5 AUC percentage points) was missed: test emergence-AUC is
0.804 for `no_graph` versus 0.781 for HGT, a gap of **−2.34 pp**. But with only 13 positives this
single point estimate is uninformative about a 5 pp effect. A label-stratified paired bootstrap
(10,000 resamples, resampling within the 13 positives and 125 negatives separately while keeping
units paired across models) puts the gap's **95% CI at [−10.1 pp, +5.6 pp]** (bootstrap SE 3.98 pp);
the analytic DeLong method for correlated AUCs gives an essentially identical gap SE of 3.99 pp
(model-score correlation ρ = 0.58, confirming the paired analysis is the correct one and that a
naïve independence assumption would over-state the SE roughly threefold). Translating that SE into a
power statement, the study had only **~24% power** (α = 0.05, two-sided) to detect the pre-registered
+5 pp gap, and the **minimum detectable effect at 80% power is ≈ 11 pp** — more than double the bar.
In other words, the AUC arm could not have reliably confirmed a true +5 pp graph benefit even if one
existed; the calibration (Brier) result above is where the statistical evidence actually lives.

**The validation→test "flip" is winner's-curse plus small-sample variance, not a regime shift.** On
validation, the selected HGT configuration (AUC 0.737) led `no_graph` (0.701) by +3.6 pp; on test the
ordering reversed (−2.34 pp). Two facts explain this without invoking any change in the data. First,
HGT was chosen as the **best of 5 *completed*** hyperparameter trials (the other 15 of 20 trials were
pruned at the first epoch; single seed 1729), and its validation AUC beat the runner-up by only
**+0.77 pp** — a margin well inside selection noise, so the selected validation AUC is an optimistic
estimate. Second, decomposing the move shows HGT did **not** collapse on test: it *gained* +4.4 pp
(0.737 → 0.781), while the **baseline gained more**, +10.3 pp (0.701 → 0.804), and overtook it. The
sign change is the baseline catching up on a 13-positive test set, the textbook signature of
winner's-curse selection plus small-sample variance — not a distributional regime shift. The
per-origin-year split is consistent with noise of this magnitude and is reported descriptively only:
2021 (67 units, 7 positives) favors HGT (0.867 vs 0.824, +4.3 pp; gap CI [−5.7, +15.2] pp) while 2022
(71 units, 6 positives) favors `no_graph` (0.687 vs 0.787, −10.0 pp; gap CI [−19.7, −0.8] pp), but
with 6–7 positives per year each per-year gap CI spans ≈30 pp, so neither year is interpretable on
its own.

> **Transparent deviation.** PR2 configured `n_trials = 40` for the HGT sweep; the committed run
> executed **20 trials, of which 5 completed** (15 pruned early by Optuna's pruner). Model selection
> therefore drew from 5 completed candidates rather than 40. This does not change the locked Gate G4
> verdict — it is disclosed here for completeness and strengthens, rather than weakens, the
> winner's-curse reading (a thinner completed pool makes the +0.77 pp selection margin even more
> clearly within noise).

**Net.** Across a well-powered calibration test (Brier, all 138 units) and an underpowered
discrimination test (AUC, 13 positives), the evidence is consistent and points one way: the
citation-graph topology adds **no** forecasting value beyond the temporal/feature node dynamics, and
where it changes anything it makes calibration **worse**. F3 is reported as a characterized **TRUE
null**.

---

## Figure caption (proposed — `docs/figures/F3_characterization.png`)

**F3 characterization: the calibration deficit is the well-powered evidence; the AUC gap is
underpowered.** *(A)* Stratified paired bootstrap (10,000 resamples) of the test AUC gap
(HGT − `no_graph`): point estimate −2.34 pp, 95% CI [−10.1, +5.6] pp; the +5 pp pre-registered bar
sits in the right tail (≈24% power, MDE@80% ≈ 11 pp). *(B)* Per-origin-year test AUC with bootstrap
CIs; at 6–7 positives per year the CIs span ≈30 pp, so the 2021↔2022 swing is small-sample noise.
*(C)* Calibration: HGT piles predicted-emergence mass into 0.4–0.8 (104/138 units) where realized
emergence is ≈12.5%, while `no_graph` stays near the 9.4% base rate (129/138 units below 0.1) — the
mechanism behind the paired Brier result (*p* = 2.63 × 10⁻¹¹, favors the graph-free baseline).

---

## HANDOFF FLAGS → T6 (exact numbers for the results-draft F3 paragraph)

All values from `notebooks/16_F3_characterization.ipynb` →
`data/v1/forecasting_characterization.json`. **No material deviation from the plan's estimates** —
all three plan anchors confirmed.

**Test set / labels (cross-checked vs audit, byte-exact):**
- *n* = 138 units; 13 positives (9.4% base rate); 125 negatives. Split: 2021 = 67 units / 7 pos;
  2022 = 71 units / 6 pos.
- Test AUC: `no_graph` = **0.8043077**, `hgt` = **0.7809231**, gap = **−2.3385 pp** (bar was +5 pp).
  Reproduces the audit exactly.

**LEAD — paired Brier Wilcoxon (well-powered, all 138 units):**
- *p* = **2.6290 × 10⁻¹¹**, statistic = **1659.0**, *n*_pairs = **138**, median_diff = **+0.23297**,
  direction = **favors_baseline**. Recomputed here from saved per-unit scores → identical to the
  frozen `forecasting_test_wilcoxon.json`.
- HGT worse-calibrated on **125 / 138** units (better on 13/138).
- Calibration mechanism: HGT puts **104 / 138** units in predicted-prob [0.4, 0.8], observed
  emergence there ≈ **12.5%**; `no_graph` keeps **129 / 138** units in [0.0, 0.1], observed freq
  **9.3%** ≈ base rate. HGT share-MAPE **0.5729** (worst of 5) vs `no_graph` **0.4610**.

**AUC arm bootstrap CIs (10,000 resamples, seed 1729, stratified 13 pos / 125 neg, paired):**
- `no_graph` AUC 95% CI = **[0.7200, 0.8794]**; `hgt` AUC 95% CI = **[0.6886, 0.8646]**.
- **Gap (hgt − no_graph) 95% CI = [−10.09 pp, +5.60 pp]**, mean −2.37 pp, bootstrap SE **3.979 pp**.
  - *Plan anchor [−10 pp, +5.5 pp]: CONFIRMED* (low end −0.09 pp, high end +0.10 pp from plan — not
    material).
- DeLong cross-check: gap SE = **3.985 pp** (agrees with bootstrap to 0.006 pp); per-AUC DeLong SE
  hgt 4.58 pp / `no_graph` 4.10 pp; model-score ρ = **0.583**.

**Power / MDE for the +5 pp bar (α = 0.05, two-sided):**
- Using the paired gap SE: **power(+5 pp) = 24.2%** (DeLong 24.1%), **MDE@80% = 11.15 pp**.
  - *Plan anchor ~25% power: CONFIRMED* (−0.8 pp from 25% — not material). MDE ≈ 11 pp = >2× the bar.
- Hanley–McNeil single-AUC SE: hgt 7.78 pp, `no_graph` 7.51 pp. Naïve independent combination
  (over-states SE; ignores ρ = 0.58) gives SE 10.82 pp → power 7.5%, MDE@80% 30.3 pp. **Use the
  PAIRED numbers (~24% / 11 pp) as the headline**; the independent figure is a conservative contrast
  only.

**Winner's-curse / val→test decomposition:**
- Sweep: **20 trials ran, 5 completed, 15 pruned**, single seed **1729**. **PR2 configured
  `n_trials = 40`** → transparent deviation (selection pool = 5 completed). DISCLOSE this.
- Winner = trial 1 (val AUC **0.7367**); runner-up = trial 4 (val AUC **0.7290**); margin =
  **+0.77 pp**.
  - *Plan anchor: use +0.77 pp (NOT the loose "+0.8 pp").* CONFIRMED to +0.7699 pp.
- Val→test gains: `no_graph` **+10.34 pp** (0.7010 → 0.8043); `hgt` **+4.42 pp** (0.7367 → 0.7809).
  Val gap +3.57 pp → test gap −2.34 pp. **HGT did NOT collapse; the baseline gained more and
  overtook.** Frame as winner's-curse + small-sample variance, NOT a regime shift.

**Per-origin-year (DESCRIPTIVE ONLY — do not over-interpret; 6–7 pos/yr → ~30 pp gap CIs):**
- 2021 (67/7 pos): `no_graph` 0.824 [0.710, 0.919], `hgt` 0.867 [0.757, 0.952], gap **+4.29 pp**
  [−5.71, +15.24] pp.
- 2022 (71/6 pos): `no_graph` 0.787 [0.662, 0.895], `hgt` 0.687 [0.551, 0.813], gap **−10.00 pp**
  [−19.74, −0.77] pp.

**Integrity reminders for T6:**
- Do **not** present this as reopening Gate G4. It re-describes the locked null; the verdict stands.
- Keep the lead order: Brier (well-powered) → AUC caveat (underpowered, CI + power/MDE) →
  winner's-curse (with the 40-configured / 5-completed disclosure).
- Numbers are exact above; the JSON `data/v1/forecasting_characterization.json` is the source of
  truth if any rounding question arises.
