# Gate G5 — V1 findings integration (≥2-of-3)

**Session:** V1-S15 (Findings integration + Results draft + Gate G5 report)  **Date generated:** 2026-06-08  **Plan ref:** `plans/Session-Objectives-MAP.md` §V1-S15 (and the V1-S15 design spec §2)

G5 decides whether the V1 narrative is strong enough to scale to V2. The pre-registered criterion is **≥2 of {F1, F2, F3} hold with statistical support AND co-authors agree the narrative is coherent.** This is the **STOP gate before V2**: if the bar is not met, V1 ships as a methods-and-resource paper rather than proceeding to a V2 scale-up. Every number below is reproduced end-to-end by `notebooks/11_F1_epistemic_cascade.ipynb`, `notebooks/12_F2_dual_novelty_trends.ipynb`, `notebooks/13_F3_forecasting_narrative.ipynb`, and `notebooks/14_bonus_cross_journal.ipynb`. This report produces a **mechanical** count + recommendation only; the narrative-coherence judgment and the final disposition are **Samer's + a co-author's**. The sign-off below records **Samer's signed disposition (2026-06-08): DOWNSCOPE.**

---

## Pass criteria

| Finding | Bar | Result (key statistic) | Status |
|---|---|---|---|
| F1 — epistemic cascade (PRE-REGISTERED, deciding vote) | ≥20% of qualifying topics show an FDR-significant directional Granger relationship | **0 / 138** directional (frac = 0.000 ≪ 0.20); panel non-significant both ways (quality→volume p = 0.277, volume→quality p = 0.739); RCT-share robustness agrees (panel p = 0.950, 0.956) | **NULL** |
| F2 — dual novelty (Gate G3 PROCEED) | two near-independent axes + a supported impact gradient | max\|ρ\| = **0.18** (Pearson 0.1828 / Spearman 0.1751, both < 0.4); archetype × within-year impact KW **H = 1938.58, p < 1e-300, ε² = 0.024** (monotone inversion) | **HOLDS** |
| F3 — graph forecasting (Gate G4 NULL) | HGT − no_graph > +5pp test AUC @3yr | HGT − no_graph = **−2.34pp** (0.7809 vs 0.8043, wrong side of zero); paired Brier Wilcoxon **p = 2.63e-11, direction = favors_baseline** | **NULL** |
| **Summary** | **threshold ≥ 2 of 3** | **Findings holding with statistical support = 1 of 3 (only F2)** | **BELOW BAR** |

> The count is over **statistical support only**. F2 holds. F1 is a pre-registered null and F3 is an OSF-registered null — each is an honest *result*, not a "hold" toward the ≥2-of-3 bar. F1 was the pre-registered deciding vote and it came back null, so the statistical bar lands at **1 of 3**. The second sub-criterion — co-author agreement that the narrative is coherent — is a **separate human judgment** (§5) and is **not** something this report can certify.

## Provenance

| Item | Value |
|---|---|
| F1 pre-registration | `docs/preregistrations/PR3_epistemic_cascade.md` (committed `bab917d`, internal — no OSF this session) |
| F1 modules commit | `6ad5dea` (F1 cascade + bonus seeding modules) |
| F1 compute commit | `015ee9b` (notebook 11 — F1 = NULL) |
| Anti-p-hacking ordering | PR3 `bab917d` (2026-06-08 21:28:50) committed **before** the F1 compute `015ee9b` (2026-06-08 21:44:42); git-timestamped. **F1 was computed once, after PR3.** |
| Fan-out commit | `fc203ce` (notebooks 12–14 — F2 trends, F3 narrative, bonus cross-journal) |
| Gate ref — F2 | `docs/gates/G3_dual_novelty.md` — signed **PROCEED** (Samer, 2026-05-31) |
| Gate ref — F3 | `docs/gates/G4_forecasting.md` — signed **NULL FINDING** (Samer, 2026-06-05); pre-reg **PR2**, OSF DOI 10.17605/OSF.IO/XP94F |
| PR3 registration | internal (git-timestamped); no OSF deposit this session |

- Notebooks: `notebooks/11_F1_epistemic_cascade.ipynb`, `notebooks/12_F2_dual_novelty_trends.ipynb`, `notebooks/13_F3_forecasting_narrative.ipynb`, `notebooks/14_bonus_cross_journal.ipynb` (each executes end-to-end; reproduces every number in this report and in `docs/results_draft.md`).
- Figures (F1): `docs/figures/F1_epistemic_cascade.png`.
- Figures (F2): `docs/figures/F2_dual_novelty.png` (G3 2×2), `docs/figures/F2_dual_novelty_trends.png` (NEW: by-year + by-topic heatmap + archetype×impact).
- Figures (F3): `docs/figures/F3_forecasting.png` (G4 3-panel diagnostic), `docs/figures/F3_forecasting_narrative.png` (NEW integrated panel).
- Figures (bonus, non-gating): `docs/figures/bonus_seeding_network.png`, `docs/figures/bonus_specialty_contrast.png`, `docs/figures/bonus_geographic.png`.
- Regenerable artifacts (gitignored): `data/v1/f1_cascade_verdict.json`, `data/v1/f1_cascade_results.parquet`; plus the G3/G4 artifacts referenced in their own gate reports.

---

## 1. Findings assessment

**F1 — epistemic cascade (pre-registered NULL).** Across the 138 qualifying topics (≥30 papers, ≥8 yrs @ ≥5/yr; 69,339 papers, 149 leaf topics, 1995–2026), **0** show an FDR-significant directional Granger relationship in either direction (frac_directional = 0.000, far below the 0.20 bar; dominant_direction = "none"), and the pooled panel test is non-significant both ways (quality→volume p = 0.277, volume→quality p = 0.739). This is a *true* null, not an underpowered miss: 15 topics reach raw p < 0.05, but BH-FDR over m = 252 hypotheses requires the smallest p ≤ 1.98e-4 to survive, while the actual minimum is 7.75e-4 — exactly the handful of nominal hits one expects by chance at this many tests — and the CCF peak-lag distribution is broad and roughly symmetric (median 0 yr) with no displaced peak. The 138-topic denominator conservatively retains 10 quality-non-evaluable topics, which can only make the fraction harder to clear. F1 was the pre-registered deciding vote; reported as a null per PR3, it shifts the count to 1 of 3.

**F2 — dual novelty (HOLDS).** Semantic and structural novelty behave as weakly-correlated, distinct axes: the largest of the four pairings is max\|ρ\| = 0.18 (Pearson 0.1828 / Spearman 0.1751, both < 0.4; N = 81,733 complete cases), and the 2×2 partitions the corpus cleanly with no degenerate quadrant. The V1-S15 enrichment links the archetypes to age-fair (within-year-percentile) citation impact: a Kruskal-Wallis test is highly reliable (H = 1938.58, p < 1e-300; n = 81,738 labeled) with a *monotone* inversion — disruptive-novel 0.433 < novel-consolidating 0.502 < conventional-disruptive 0.523 < incremental 0.607 — reconfirming and quantifying the G3 "surprising-to-expert" finding. The honest caveat, which this report deliberately does **not** oversell: the effect size is **small**, ε² = 0.024 (≈ 2.4% of rank variance). The strength of F2 is the *reliability and consistency of the ordering*, not the magnitude of the gap; it should be reported that way.

**F3 — graph forecasting (registered NULL).** On the sealed test (origin years 2021–2022; n_train = 1298, n_test = 138, 13 positives ≈ 9.4%), the graph-free `no_graph` baseline is the strongest of five models (AUC 0.8043), ahead of HGT (0.7809). The pre-registered ablation HGT − no_graph = −2.34pp is not a near-miss but on the *wrong side of zero* against the +5pp bar, and the paired Brier-loss Wilcoxon (p = 2.63e-11) is significant in the direction that *favors the baseline* — HGT is systematically worse-calibrated, over-predicting (mass at 0.4–0.8 where observed emergence ≈ 0; worst-in-class share-MAPE 0.5729). The graph adds nothing over a topic's temporal/feature dynamics; G4 already records this as a signed NULL FINDING and it carries into G5 as a registered null.

## 2. Narrative-coherence assessment (for co-authors to weigh — not certified here)

The Results draft (`docs/results_draft.md`) frames V1 as **one cross-sectional positive (F2) against two pre-specified temporal-evolution nulls (F1, F3)**. In my assessment, that thru-line reads as a coherent story: F1 and F3 are independent, pre-registered probes of how a topic *evolves over time* — whether evidence quality leads/lags volume, and whether citation-graph topology forecasts emergence — and both return honest nulls, while the finding that survives (F2) is about the *cross-sectional* structure of innovation. The two nulls are not loose ends; they sharpen the claim by bounding what the framework can and cannot detect, and each was specified before its analysis was run (F1 internally pre-registered; F3 OSF-registered), so they are credible negative results rather than failed searches. The draft also keeps F2 in proportion — explicitly flagging the small ε² and the thin pre-2000 CD coverage — which is the right posture for a single-positive narrative.

That is my reading. **Whether the narrative is coherent enough to carry the paper is a co-author judgment, and it is the second, currently UNMET, sub-criterion of G5.** I cannot certify it. A coherent-narrative sign-off from Samer + a co-author is required before this sub-criterion can be marked satisfied; this report records it as open.

## 3. Sensitivity / caveats

- **F1 robustness.** The primary metric (mean evidence tier, 65.4% coverage) and the RCT-share robustness metric **agree** — both null (RCT-share panel p = 0.950, 0.956) — so the null does not hinge on the quality operationalization.
- **F3 test set.** The sealed test is small and high-variance (n = 138, 13 positives), and the val→test ranking shifted (no_graph 0.701→0.804, HGT 0.737→0.781). Even granting that variance, HGT lands on the *wrong side* of the bar (−2.34pp), not near it, and is robust to alternate emergence-label definitions; the variance does not rescue F3.
- **F2 effect size + coverage.** ε² = 0.024 is small — the result is the reliable *ordering*, not a large gap — and the temporal drift (conventional-disruptive +0.083, novel-consolidating −0.055) should not be over-read at its left edge given thinner pre-2000 CD coverage (~0.95 → ~0.90).
- **Bonus is exploratory and never enters the count.** The cross-journal / specialty / institutional seeding observations (notebook 14) are descriptive context only; the geographic panel carries a documented coverage caveat (country_code present on 56% of institutions, 16,258/29,159; blank-country rows dropped, never imputed; 94% country coverage on the rows actually used). **The bonus does not contribute to the ≥2-of-3 count.**

---

## Recommendation

Findings holding with statistical support = **1 of 3** (only F2; F1 and F3 are honest, pre-specified nulls). The pre-registered statistical bar for proceeding to V2 is **≥ 2 of 3**, so the **statistical bar is not met**. F1 — the pre-registered deciding vote — came back null, which is what brings the count below threshold. On the statistical criterion alone, the **mechanical recommendation is to DOWNSCOPE to a methods-and-resource paper**: that paper is still a real contribution — the validated multi-axis pipeline, the F2 dual-novelty finding, and two rigorously characterized nulls — but it is **not** a proceed-to-V2 on the ≥2-of-3 bar. The narrative-coherence sub-criterion (§2) and the final disposition are **human** and are left to Samer + a co-author.

- [ ] PROCEED to V2 (≥2 of 3 findings hold + coherent narrative)
- [x] DOWNSCOPE to a methods-and-resource paper (<2 of 3 hold)  ← **(mechanical recommendation; accepted)**
- [ ] other disposition (co-authors' call)

**Decision: DOWNSCOPE to a methods-and-resource paper.** F1 — the pre-registered deciding vote — and F3 are both honest, pre-specified nulls; only F2 holds (1 of 3, below the ≥2-of-3 bar). V1 ships as a methods-and-resource paper: the validated multi-axis pipeline, the F2 dual-novelty finding (with its small-but-reliable impact inversion), and two rigorously characterized nulls (F1 internally pre-registered, F3 OSF-registered). The V2 scale-up is not triggered on the current findings. The narrative-coherence sub-criterion (§2) is affirmed on this sign-off for the downscoped framing.

Signed: _Samer G. Salman_  Date: _2026-06-08_

Gate G5 is resolved (DOWNSCOPE). Per the Session-Objectives map, V1 proceeds as a methods-and-resource paper; the V2 scale-up is **not** triggered on the current findings.
