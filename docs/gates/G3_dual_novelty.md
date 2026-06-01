# Gate G3 — Dual-novelty independence (semantic × structural)

**Session:** V1-S11 (Dual-novelty 2×2 archetype + Gate G3 report)  **Date generated:** 2026-05-31  **Plan ref:** `plans/Session-Objectives-MAP.md` §V1-S11 (and the V1-S11 plan)

G3 tests whether the two novelty axes (semantic novelty vs. structural CD index) carry **independent** signal. If they were strongly correlated, the dual-novelty idea (finding F2) collapses to a single axis and the paper rescopes to F1+F3. This is the **STOP gate before V1-S12**. Every number below is reproduced end-to-end by `notebooks/08_dual_novelty.ipynb`.

---

## Pass criteria

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| Row 1 — Max \|ρ\| across the four pairings (most conservative) | `\|ρ\| < 0.4` | **Pearson 0.1828 / Spearman 0.1751** (both from `sem_nov_min × cd10`) | **PASS** |
| Row 2 — ≥1 surprising-to-expert finding | qualitative | _affirmed on sign-off (see §4 — impact-by-archetype inversion)_ | **PASS — Samer** |

> The numeric sub-criterion uses the **MAX** (most conservative / largest) \|ρ\| of the four pairings. **Both** the max Pearson (**0.1828**) and the max Spearman (**0.1751**) are below 0.4, so the numeric sub-criterion is objectively **PASS**. Row 2 — the "surprising-to-expert" judgment — is **left for Samer**.

## Provenance

Sidecar: `data/v1/archetypes.parquet.run.json`

| Field | Value |
|---|---|
| config_hash | `458c6a94f3c314f778712d6c3d64e7b1c34018fd11f4975fc6c3d33b31f102a2` |
| git_sha | `f58cf9bec8212c9348a0e44c39b6121c55394244` |
| git_dirty | `true` |
| timestamp | `2026-06-01T02:50:06.079525+00:00` |

- Artifact: `data/v1/archetypes.parquet` (+ sidecar `data/v1/archetypes.parquet.run.json`).
- Notebook: `notebooks/08_dual_novelty.ipynb` (executes end-to-end; reproduces every number in this report).
- Figure: `docs/figures/F2_dual_novelty.png`.

---

## 1. Full 4-pairing correlation table (complete cases, N = 81,733)

| pairing | N | Pearson r | Spearman ρ |
|---|---:|---:|---:|
| sem_nov_mean × cd5 | 81,733 | 0.0779 | 0.0805 |
| sem_nov_mean × cd10 | 81,733 | 0.0822 | 0.0906 |
| sem_nov_min × cd5 | 81,733 | 0.1663 | 0.1458 |
| sem_nov_min × cd10 | 81,733 | 0.1828 | 0.1751 |

All four \|ρ\| ≪ 0.4; the two axes are **weakly correlated / largely independent**. N = 81,733 complete cases (all four metrics non-null) out of 89,230 papers; ~7,497 papers are null on ≥1 axis (earliest-in-field `n_prior==0`, or no CD citers) and are excluded from the 2×2 — **documented, not silently dropped**.

## 2. 2×2 distribution + temporal interpretation

Per-quadrant counts + shares for the primary **mean×cd5** pairing:

| archetype | n | share |
|---|---:|---:|
| disruptive-novel (HH) | 22,662 | 0.2773 |
| novel-consolidating (HL) | 18,210 | 0.2228 |
| conventional-disruptive (LH) | 19,474 | 0.2382 |
| incremental (LL) | 21,392 | 0.2617 |

The four quadrants are well-populated and balanced — **no degenerate quadrant**.

**Temporal trend.** From ≤2000 to ≥2020, the 'conventional-disruptive' share moves **+0.083** and 'novel-consolidating' moves **−0.056**.

> **Caveat.** Per-year CD coverage runs ~0.95 (≤2000) to ~0.90 (≥2020). Note this drift runs *opposite* the usual "early years sparse" intuition — coverage is slightly **higher** pre-2000 here — so this is flagged for Samer's eye rather than smoothed over. The early-year shares should **not** be over-read on a thin CD denominator.

## 3. Corpus-CD upward-bias caveat (NOT a blocker)

Stated plainly: the corpus CD index is an **upward-biased approximation** — it omits type-*k* citers (works citing the focal's references but not the focal itself). Relative ranking and the median split are fine; the **absolute** CD sign is shifted. This is precisely **why a CD=0 split was rejected in favor of the global median split**. Pre-2000 CD coverage is thinner, which is why the temporal panel annotates per-year CD coverage.

## 4. Candidate observations (for Samer to flag which — if any — is "surprising")

(Neutral framing — none labeled "surprising"; that is Samer's call.)

- **Dual-novelty separation:** largest \|Pearson r\| = 0.183 and largest \|Spearman ρ\| = 0.175 (both `sem_nov_min × cd10`); all four pairings < 0.4 — semantic and structural novelty behave as weakly-correlated, distinct axes.
- **Temporal drift (mean×cd5):** ≤2000 → ≥2020, 'conventional-disruptive' +0.083, 'novel-consolidating' −0.056; CD coverage 0.95 → 0.90 over the same span (thin early-year denominator caveat applies).
- **Journal contrasts (mean×cd5):** 'disruptive-novel' share ranges 0.195 (Arthroscopy) to 0.396 (JAMA Surgery); 'incremental' share ranges 0.155 (JAMA Surgery) to 0.365 (Arthroscopy).
- **Impact by archetype (age-fair within-year citation percentile, mean×cd5):** median percentile highest for 'incremental' (0.607) and lowest for 'disruptive-novel' (0.433); full set: disruptive-novel 0.433, novel-consolidating 0.502, conventional-disruptive 0.523, incremental 0.607. (I.e. in this corpus the most novel-and-disruptive papers sit LOWER on within-year citation impact than incremental ones.)

### Samer's adjudication (Row 2)

| # | Finding | Category | Surprise | Framing |
|---|---|---|---|---|
| 1 | Dual-novelty separation | Construct validity / method validation | Low | Shows the framework works: semantic novelty and structural disruption are related but distinct. |
| 2 | Temporal drift | Longitudinal trend / field evolution | Moderate | The literature has shifted over time toward more conventional-disruptive work and away from novel-consolidating work. |
| 3 | Journal contrasts | Cross-journal heterogeneity / editorial phenotype | Moderate | Journals differ meaningfully in the kinds of innovation they publish — interesting, but not shocking. |
| 4 | Impact by archetype | Counterintuitive impact finding | **High** | **The genuinely surprising result:** the most novel-and-disruptive papers had a LOWER age-adjusted citation percentile than incremental papers. |

**Designated surprising-to-expert finding (Row 2): #4 — impact by archetype.**

---

## Recommendation

The objective numeric criterion (max \|ρ\| < 0.4) is met **decisively** (Pearson 0.1828, Spearman 0.1751), so the data **support** treating semantic and structural novelty as independent axes and retaining F2 as a two-axis construct. But the "≥1 surprising-to-expert finding" judgment and the final PROCEED-to-V1-S12 / DROP-F2 architectural decision are **Samer's** — the candidate observations in §4 are offered for his adjudication.

- [x] PROCEED to V1-S12 (retain F2 dual-novelty)
- [ ] QUALIFIED PROCEED
- [ ] DROP F2 (rescope to F1+F3)

**Decision: PROCEED** — the two novelty axes are independent (max \|ρ\| = 0.18 ≪ 0.4), ≥1 surprising-to-expert finding is affirmed (§4, the impact-by-archetype inversion), and F2 dual-novelty is retained as a two-axis construct. **V1-S12 is unblocked.**

Signed: _Samer G. Salman_  Date: _2026-05-31_

Per the Session-Objectives map, Gate G3 is resolved (PROCEED); V1-S12 may begin.
