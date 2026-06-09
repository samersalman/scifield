# Novelty Origins — Robustness Verdict (V2-S06b)

**Status:** EXECUTED 2026-06-09 (robustness substantiation for the S06 origin findings; Gate-G6 ingredient (iii))
**Author:** Samer Salman (executed by subagent S06b)
**Builds on:** V2-S06 (`origins.py`, `build_origins.py`, `V2/data/origins/{sector,geo}_novelty.parquet`)
**Code:** `src/scifield/cartography/origins.py` (pure: `split_half_stability`, `bootstrap_delta_ci`) · `V2/scripts/build_origins_robustness.py` (I/O)
**Tables:** `V2/data/origins/{geo_robustness,sector_robustness}.parquet` (+ `.run.json`, task `V2-S06b`)

> **One-line verdict:** the **sector** finding is a *robust near-null* (bootstrap CI
> spans zero on both novelty axes) and the **geography** finding is *moderately
> stable* (split-half Spearman ρ ≈ 0.49–0.67 across modes and axes — reproducible
> ordering, not a single-paper artifact, but not razor-sharp). For Gate-G6 ingredient
> (iii) the **strongest robust finding is the sector near-null** (a clean, fully
> substantiated result); geography is a *moderately-robust* secondary candidate.

---

## 0. Why topic-granularity Method-D is N/A here (the reviewer's point)

The validation protocol's **Method-D is topic-granularity sensitivity** (re-run the
analysis at leaf vs mid topic grain; require a high per-entity rank correlation). That
is the right robustness check for the *journal cascade*, whose entities (journals) are
ranked *within topics*, so the grain matters.

The S06 novelty-origin findings are different: they are keyed on author-institution
`country_code` (geography) and `type` (sector). **Those keys do not change when the
topic grain changes** — a paper's countries and sectors are identical whether you bin
its topic at leaf or mid level. So topic-granularity Method-D is **N/A by
construction** for the per-institution origin findings; running it would just return
ρ ≈ 1.0 trivially and prove nothing. The original S06 hand-off asserted geography was
"the most robust candidate" and sector was "a NULL" *without a quantified robustness
number* — this doc supplies the appropriate, grain-independent number instead:

- **Geography → split-half stability** (`split_half_stability`): split the papers in
  two, rank each country's mean novelty within each half, correlate the two rankings
  (Spearman ρ). Run in **`random`** mode (sampling robustness, seed `20260609`) and
  **`temporal`** mode (early-vs-late at the median year — an out-of-period check).
  Only countries with ≥ 30 papers in **both** halves are ranked.
- **Sector → bootstrap delta CI** (`bootstrap_delta_ci`): bootstrap the
  company-minus-non-company mean-novelty delta (1000 reps, 95% percentile CI,
  independent within-group resampling). A CI that **spans zero** substantiates the
  near-null; one that excludes zero would mean a real sector effect.

Both are run on `sem_nov_mean` (semantic novelty) and `cd5` (disruption). The join
recipe reuses the S06 builder exactly: `archetypes.pmid` (noise topic dropped) →
`paper_institutions.pmid` → `institutions.country_code`/`type`, any-author tagging,
blanks dropped never imputed. 67,821 papers; 5,091 company vs 60,068 non-company.

---

## 1. Sector finding — robust NEAR-NULL

Bootstrap of (company − non-company) mean novelty, 1000 reps, 95% percentile CI:

| novelty | delta (company − non-company) | 95% CI | spans zero? | verdict |
|---|---|---|---|---|
| `sem_nov_mean` | **−0.0006** | [−0.0031, +0.0019] | **yes** | genuine near-null |
| `cd5` | **+0.0034** | [−0.0106, +0.0173] | **yes** | genuine near-null |

On **both** novelty axes the CI comfortably contains zero, and the point estimate is a
rounding error relative to the spread (semantic delta ≈ 6e-4; disruption delta ≈ 3e-3,
CI half-width ≈ 1.4e-2). This is **not** "we lacked power to find an effect" — the
company group has 5,091 papers and the CI is *tight* (±0.0025 on `sem_nov_mean`), so a
sector novelty premium of any practically meaningful size would have been detected.

> **Sector verdict: robust near-null.** Novelty in this 10-journal surgical/ortho
> corpus is **not** concentrated in the tech/industry (`company`) sector — and that
> null is now substantiated, not merely asserted. This is a clean, defensible result.

---

## 2. Geography finding — moderately stable spread

Split-half Spearman ρ of the per-country mean-novelty ranking (countries with ≥ 30
papers in both halves):

| novelty | mode | Spearman ρ | n_keys | reading |
|---|---|---|---|---|
| `sem_nov_mean` | random | **+0.669** | 49 | moderately stable |
| `sem_nov_mean` | temporal | **+0.508** | 42 | moderately stable (stricter) |
| `cd5` | random | **+0.490** | 48 | moderate |
| `cd5` | temporal | **+0.653** | 42 | moderately stable (stricter) |

All four ρ are **positive and moderate (≈ 0.49–0.67)**: the country novelty ordering
reproduces across an independent 50/50 split *and* across an early-vs-late temporal
split, on both novelty axes. The ranking is therefore **not a single-paper or
single-period artifact** — there is a real, reproducible between-country signal. But it
is also **not razor-sharp**: ρ in the 0.5–0.67 band means a meaningful share of the
per-country rank reshuffles between halves (consistent with the S06 read that the
spread is partly topic-mix-driven and descriptive). The temporal ρ holding up (0.51 /
0.65) is the more demanding result and is reassuring.

> **Geography verdict: moderately robust.** The cross-country novelty spread (LMIC/
> emerging-research countries ranking high; e.g. GH/HU/UG top, KR/RS/TH bottom in S06)
> is a *reproducible ordering*, but with moderate stability — surface it on the map as
> a *descriptive, panel-conditional* layer, not as a sharp causal claim.

---

## 3. Which is the strongest ingredient-(iii) candidate for Gate G6?

Reading the **actual computed numbers** (not the S06 prejudgement that geography was
"most robust"):

- The **sector near-null is the strongest, cleanest robust finding.** Its robustness
  is unambiguous: the bootstrap CI spans zero on both axes with a tight interval, so
  "novelty is not sector-concentrated in this corpus" is fully substantiated. A
  well-powered null is a legitimate, robust ingredient-(iii) finding.
- **Geography is a moderately-robust secondary candidate.** Split-half ρ ≈ 0.49–0.67
  (random and temporal, both axes) shows the country ordering is real and reproducible,
  but the stability is moderate rather than strong — so it should be presented as a
  *descriptive* spread with its ρ stated honestly, not as the headline robust finding.

**Recommendation for G6:** ingredient (iii) "≥ 1 robust novelty-origin finding" is
**satisfied** — primarily by the **sector near-null** (robust, well-powered,
CI-substantiated), with the **geography spread** as a corroborating moderately-stable
secondary. Both numbers are now quantified and reproducible (seed `20260609`), so Samer
can weigh ingredient (iii) on evidence rather than assertion.

---

## 4. Caveats (carry into the map and the G6 synthesis)

- **Panel-conditional.** 10-journal surgical/ortho corpus, not all of science; "company
  novelty" and "country novelty" mean *within this corpus*.
- **Conditioned on resolved affiliation metadata** (~35% blank `type`, ~44% blank
  `country_code` excluded; never imputed).
- **Split-half ρ is a stability check, not a significance test** — it asks whether the
  *ordering* reproduces, not whether between-country novelty differences are
  individually significant.
- **Bootstrap is a within-group resample** preserving each group's size; it quantifies
  the sampling uncertainty of the *mean delta*, which is exactly the near-null question.
- The descriptive cross-country pattern is **likely topic-mix-driven** (per S06); no
  causal interpretation is claimed.
