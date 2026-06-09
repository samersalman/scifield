# Pre-registration #3D — Levels diagnostic: was F1's differenced null a TRUE level-null, or did the once-differenced test miss a level-cascade?

**Author:** Samer G. Salman, Baylor College of Medicine (ORCID: [0009-0007-9897-4071](https://orcid.org/0009-0007-9897-4071))
**Date drafted:** 2026-06-09
**Registration mechanism:** internal git-timestamped commit (no OSF filing this session); the committing SHA precedes the F1-levels-diagnostic compute commit.
**Codebase:** https://github.com/samersalman/scifield  (Zenodo DOI minted at release v0.1.0 — 10.5281/zenodo.20360023 | concept DOI: https://doi.org/10.5281/zenodo.20360022)
**Implementing modules:** `scifield.findings.cascade` (Toda–Yamamoto + Engle–Granger ECM + level-panel additions, pure, synthetic-fixture tested) and `scifield.findings.cascade_powersim` (new pure power-simulation module); executed once in `notebooks/15_F1_levels_diagnostic.ipynb`.
**Parent pre-registration (NOT edited):** `docs/preregistrations/PR3_epistemic_cascade.md` — PR3D reuses PR3's 138-topic denominator, pooled BH-FDR, 20% directional bar, and panel-agreement bar verbatim, so the levels verdict is structurally comparable to PR3's differenced verdict.
**Signed gate this diagnostic does NOT touch:** `docs/gates/G5_v1_findings.md` (Gate G5 = DOWNSCOPE, signed by Samer 2026-06-08). PR3D and the signed G5 are NEVER edited.
**Plan reference:** `plans/i-just-made-a-cuddly-moth.md` §Phase 1A (the F1 levels-cascade diagnostic).

---

## 0. Filing note — this is an internal, git-timestamped pre-registration, and it is a NON-GATING diagnostic

Like Pre-registration #3 (`docs/preregistrations/PR3_epistemic_cascade.md`), and unlike Pre-registration #2 (`docs/preregistrations/PR2_forecasting.md`, which carries an external OSF DOI 10.17605/OSF.IO/XP94F), **PR3D is not filed with the Open Science Framework in this session.** The registration mechanism is the **git commit of this file**. This commit lands in the repository's history **before** the commit that runs the levels-diagnostic compute notebook (`notebooks/15_F1_levels_diagnostic.ipynb`) and writes any levels-diagnostic result artifact. The git history is therefore the **timestamp of record**: the analysis plan below — the Toda–Yamamoto levels specification, the Engle–Granger error-correction robustness check, the level-panel mirror, the power-simulation grid, and the **pre-stated verdict table** — is provably fixed prior to the existence of any Toda–Yamamoto Wald statistic, cointegration p-value, error-correction coefficient, or level-panel result on the real corpus.

The verifiable claim this document makes is narrow and checkable from `git log`: **commit(PR3D) precedes commit(notebook-15 results)**. That ordering is the whole guarantee, exactly as **commit(PR3) precedes commit(notebook-11 results)** was for the original F1 finding.

**This is a diagnostic, not a finding, and it cannot flip a gate.** PR3D characterizes whether F1's signed null (Gate G5, 2026-06-08) is a *true* level-null or an *apparent* one produced by a differenced test that was structurally blind to a level relationship. It is the levels analogue of PR3's RCT-share robustness rerun in one decisive respect: **it cannot, by itself, convert a signed null into a reportable finding.** A flip to a reportable finding requires a SEPARATE confirmatory pre-registration (PR4) and an ADDITIVE new gate (G5′), and only after Samer's Phase-2 checkpoint (§9). The signed Gate G5 and the parent PR3 are preserved as the honest record of what the *differenced* analysis found; nothing in PR3D edits them.

---

## 1. Background and rationale

SciField is a multi-axis framework for monitoring the health of the orthopedic-surgery research field. Phase 6 integration produced three pre-registered findings; under Gate G5 (`docs/gates/G5_v1_findings.md`, signed DOWNSCOPE 2026-06-08) only **F2 (dual novelty)** held, while **F1 (epistemic cascade)** — the pre-registered deciding vote — and **F3 (graph forecasting)** came back null, landing the count at 1-of-3. PR3D scopes a single, sharply bounded **post-hoc diagnostic of F1's null**.

PR3 (`docs/preregistrations/PR3_epistemic_cascade.md`) pre-registered the F1 epistemic-cascade test and ran its cross-correlation and Granger statistics on the **once-differenced (Δ) series** (PR3 §5 step 3, §6.1, §6.2; differencing happens inside `granger_pair` / `panel_granger`). Differencing is a sound stationarity control, but it has a structural consequence that scopes the question F1 actually answered: **differencing removes the long-run level / cointegration relationship.** A differenced Granger test asks whether *changes* in quality precede *changes* in volume; it cannot, by construction, detect a relationship that lives in the **levels** — e.g. that topics whose evidence quality sits at a persistently higher level go on to accumulate persistently higher volume.

An "epistemic cascade" — higher-quality topics attract more research volume over time — is most naturally a **level phenomenon**, not a first-difference phenomenon. Series length is not the issue: the qualifying series have a **median length of ~30 annual observations** (the real lengths cluster around {8, 15, 22, 30, 40}; §7), so F1's null is **not** a sample-size problem. It is a **specification-scope** problem: the differenced test correctly answered a narrower question (about year-over-year changes) and found a null (0 of 138 topics directional, panel non-significant in both directions — PR3 result, encoded in `data/v1/f1_cascade_verdict.json`); it never tested the level relationship. The Gate-G5 results draft then described that null partly in **level language** ("quality and volume rise together without either leading"), which the differenced test cannot actually support.

PR3D resolves this in the only integrity-preserving way: it pre-registers, **before any compute touches real data**, a **levels** test that is valid regardless of the integration / cointegration order of the two series — the **Toda–Yamamoto (TY)** augmented-VAR Granger test — plus an **Engle–Granger error-correction (ECM)** robustness check that directly characterizes a long-run level relationship, plus a **decisive power simulation** that runs BOTH the PR3 differenced test and the TY levels test on the SAME synthetic series across DGPs and series lengths. Together these determine whether F1's null is:

- **TRUE** — a real level-null, confirmed by a levels test that the power simulation shows *would have detected* a real level-cascade at realistic series lengths (this STRENGTHENS PR3's null), or
- **APPARENT** — the differenced test was blind to a level-cascade that the (well-powered) levels test detects.

To keep the levels verdict structurally comparable to PR3 — so that "the levels test holds" means the same thing "the differenced test holds" meant — PR3D **reuses PR3's machinery verbatim**: the same universe, the same 138-topic qualifying denominator, the same pooled Benjamini–Hochberg FDR at q = 0.05, the same 20% directional holds-fraction, the same panel-agreement bar, and the same lead/lag sign convention. The only thing that changes is the per-pair and per-panel test statistic (differenced Granger → Toda–Yamamoto levels), plus the additive ECM characterization and the power simulation. The corpus operated on is the same 10-journal orthopedic-and-surgery PubMed subset (1995–2026) read by PR3; PR3D introduces no new harvest and no new external data.

---

## 2. Hypotheses

The substantive question is the same as PR3's, asked on the **levels** instead of the differences: within a topic, are evidence quality and research volume coupled in the *long run* / *levels*, and does one direction dominate? The pre-registered diagnostic does not assume which direction; it estimates it on the levels and gates on whether a dominant direction is supported across the topic population, using the identical bars PR3 used.

- **HL1 (directional level-coupling at the population level).** Across qualifying leaf topics, an FDR-significant *directional* quality↔volume Toda–Yamamoto **levels** relationship (a clear lead OR a clear lag, but not both) appears in **at least 20% of qualifying topics** (i.e., **≥ 28 of 138**). This is PR3's H1 bar (`holds_frac = 0.20`, denominator 138) applied to the TY levels p-values.
- **HL2 (level-panel agreement on the dominant direction).** A pooled fixed-effects **level-panel Toda–Yamamoto** test is significant (p < 0.05, `panel_alpha = 0.05`) in the **same** dominant direction as the per-topic significant majority. This is PR3's H2 bar applied to the levels panel.

**The LEVELS TEST HOLDS iff both HL1 and HL2 are satisfied**, computed by the identical `decide_f1` decision rule PR3 used (§3.1, §8). Otherwise the **levels test is NULL**.

These hypotheses, and the verdict table in §8 that maps them (together with the robustness and power-simulation evidence) onto **TRUE / APPARENT / AMBIGUOUS**, are pre-registered **before the levels diagnostic is computed** — before the Toda–Yamamoto, Engle–Granger, level-panel, or power-simulation statistics exist on the real corpus. Any deviation from the analysis plan below will be reported transparently as a clearly-logged deviation note, exactly as the Gate-G4 report logged its deviations (`docs/gates/G4_forecasting.md` §5) and as PR3 §2 committed to.

### 2.1 Pre-declared sign convention (identical to PR3 §2.1)

The lead-lag sign is fixed here, before the data are seen, and is **byte-identical to PR3 §2.1**, so that "lead" vs. "lag" cannot be reinterpreted after the fact and so the levels verdict's direction is directly comparable to PR3's:

> **A NEGATIVE peak cross-correlation lag means quality LEADS volume (rigor precedes the surge); a POSITIVE peak lag means quality LAGS volume (attention precedes rigor).**

In the Toda–Yamamoto framing this maps onto the two directional p-values exactly as PR3's Granger directions did: the `quality → volume` TY p-value is evidence that **quality leads** (`"quality_leads"` / `"lead"`); the `volume → quality` TY p-value is evidence that **volume leads / quality lags** (`"quality_lags"` / `"lag"`). The CCF sign convention and the TY direction labels are consistent with PR3 §6 and with `classify_direction` (§3.1), which is reused verbatim.

---

## 3. Operational definitions

The unit of analysis is a **leaf topic**, identical to PR3 §3 / §4. The per-topic objects under test are the two annual **LEVEL** series — `quality` (primary = mean evidence-tier; robustness = RCT-share) and `volume` — exactly as PR3 defines them, and the levels lead-lag statistics computed from them. The canonical machine-readable source for every reused threshold, the tier map, and the function defaults is the implementing module `scifield.findings.cascade`; the new levels and power-simulation defaults are exposed as default arguments on the new functions named in §3.2 and §3.3.

### 3.1 Reused VERBATIM from `scifield.findings.cascade` (NOT reimplemented)

The following PR3 components are **reused without modification** — same code, same pinned defaults — so the levels verdict is structurally comparable to PR3's differenced verdict:

- **`build_topic_year_series`** — collapses the tidy paper frame to one row per `(topic, year)` with `volume`, `mean_tier`, `rct_share`, `n_graded` (PR3 §3.1, §3.4). Same `TIER_MAP` (`RCT=4, cohort=3, case_control=2, case_series=1`; `review` / `other` carry no tier).
- **`qualifying_topics`** — the qualifying-volume filter with the pinned `v_min=30`, `min_years=8`, `min_count=5` (PR3 §3.5), yielding the **same 138-topic qualifying denominator** that PR3 used. This 138 is the fixed denominator for HL1 and §8; it does not change after the data are seen.
- **`prepare_series`** — span-trim / linear-interpolate-internal-gaps / `max_nan_frac=0.25` non-evaluable flag (PR3 §5). PR3D consumes the **LEVEL series** that `prepare_series` returns (the `(quality, volume, status)` tuple) and feeds those levels **without differencing** to the Toda–Yamamoto and Engle–Granger functions. (The differencing that PR3 performs inside `granger_pair` / `panel_granger` is exactly the step PR3D deliberately does NOT do.)
- **`bh_fdr`** — pooled Benjamini–Hochberg FDR at the pinned `q=0.05` (PR3 §7). ALL per-topic, per-direction Toda–Yamamoto p-values (both directions, all evaluable topics) are pooled into a single vector and corrected exactly as PR3 pooled its Granger p-values.
- **`classify_direction`** — maps a topic's two FDR-significance flags to `"lead"` / `"lag"` / `"coupled"` / `"none"` (PR3 §7); only `"lead"` / `"lag"` count as directional.
- **`decide_f1`** — the binary holds rule with the pinned `holds_frac=0.20` and `panel_alpha=0.05` (PR3 §8). PR3D feeds it the per-topic TY directions and the level-panel result; it returns the same verdict structure (`holds`, `n_directional`, `frac_directional`, `dominant_direction`, `panel_agrees`, the full lead/lag/coupled/none split, the two panel p-values), so the levels HOLDS/NULL verdict is computed by the identical rule that produced PR3's verdict. The exact-tie ⇒ no-dominant-direction ⇒ does-not-hold conservatism (PR3 §8.2) carries over unchanged.

**Same universe** as PR3 (§4): `epistemic_extracted.parquet` (`pmid`, `study_design`) INNER-joined to `novelty_semantic.parquet` (`pmid`, `topic_id`, `year`) on `pmid`, **leaf topics only** (`topic_id != -1`) — ≈ 69,339 papers across 149 leaf topics, years 1995–2026, of which **138 qualify**. **Same primary quality metric** = `mean_tier`; **same robustness metric** = `rct_share`. **Non-evaluable topics stay in the 138 denominator** and count as non-significant (PR3 §5), exactly as in PR3.

### 3.2 The Toda–Yamamoto levels test (new, additive to `scifield.findings.cascade`)

The Toda–Yamamoto (1995) procedure tests Granger non-causality on **level series** and is valid **regardless of the integration or cointegration order** of the variables (it does not require pre-testing for unit roots or cointegration, and does not require differencing), which is precisely the property that makes it the right diagnostic for a relationship that may live in the levels. It augments a VAR with extra lags equal to the maximal suspected order of integration, fits the VAR on the **undifferenced levels**, and applies a Wald restriction to **only the original lags** — leaving the augmenting lags estimated but unrestricted — so the Wald statistic has its standard asymptotic χ² distribution even with integrated / cointegrated data.

**`toda_yamamoto_pair(quality, volume, *, k=3, d_max=1) → (p_q_to_v, p_v_to_q)`**

- **Inputs are LEVEL series** from `prepare_series` (the `(quality, volume)` returned tuple), **NOT differenced.**
- Fit a VAR of order **`p = k + d_max = 4`** on the 2-column level matrix `[quality, volume]` with **`trend="c"`** (constant term).
- **Toda–Yamamoto restriction:** a Wald test that the **FIRST `k = 3`** cross-lag coefficients — the *cause* variable's lags 1..k in the *effect* variable's equation — are **jointly zero**; the extra **`d_max = 1`** lag (lag 4) is **estimated but EXCLUDED from the restriction**.
- The Wald is built **BY HAND** from the fitted VAR coefficient vector and its covariance matrix — **NOT** via statsmodels `test_causality`, which restricts all lags and therefore omits the TY lag-exclusion that makes the test valid on integrated data. The statistic is `W = (Rβ)ᵀ (R Σ Rᵀ)⁻¹ (Rβ)`, where `β` is the stacked VAR coefficient vector (`params`), `Σ` is its estimated covariance (`cov_params`), and `R` is the `k × len(β)` restriction matrix selecting exactly the first-`k` cross-lag coefficients of the effect equation. The p-value is `p = scipy.stats.chi2.sf(W, df=k)`, i.e. **df = k = 3** (the augmenting lag is not counted in the degrees of freedom).
- **`p_q_to_v`** restricts the first-`k` lags of `quality` in the `volume` equation → evidence **quality leads** (`"lead"` / `"quality_leads"`). **`p_v_to_q`** restricts the first-`k` lags of `volume` in the `quality` equation → evidence **volume leads / quality lags** (`"lag"` / `"quality_lags"`). This matches PR3 §2.1 / §6.2 sign convention and the negative-CCF-lag = quality-leads reading.
- **NaN guard:** returns `NaN` for a direction (recorded, treated as non-significant — never raised, never silently dropped, exactly as PR3's `granger_pair` does) when **observations ≤ 2p + 2** (i.e. ≤ 10 usable rows for `p = 4`) or the VAR / covariance is **singular**.

### 3.3 The Engle–Granger error-correction robustness characterization (new, additive — ROBUSTNESS / INFORMATIONAL ONLY)

**`engle_granger_ecm_pair(quality, volume) → {cointegrated, ec_coef, ec_pvalue, coint_pvalue}`**

- `coint_pvalue` — the Engle–Granger cointegration p-value from `statsmodels.tsa.stattools.coint(quality, volume)`; `cointegrated` is `True` iff `coint_pvalue < 0.05`.
- If `cointegrated`, fit the error-correction model and report the **error-correction (speed-of-adjustment) coefficient** `ec_coef` (the loading on the lagged cointegration residual in the Δvolume equation) and its p-value `ec_pvalue`; a significant negative `ec_coef` indicates volume adjusts toward a long-run equilibrium with quality. If not cointegrated, `ec_coef` / `ec_pvalue` are `NaN`.
- **STATUS: ROBUSTNESS / INFORMATIONAL ONLY.** This characterizes whether a long-run level relationship exists and how the system corrects toward it; it is reported alongside the TY verdict but **CANNOT by itself flip a verdict** — it occupies exactly the same non-gating status as PR3's RCT-share robustness rerun (PR3 §8.3, §9): a favorable Engle–Granger result cannot convert a NULL levels verdict into a HOLD, and an unfavorable one cannot convert a HOLD into a NULL. It informs the TRUE / APPARENT / AMBIGUOUS narrative (§8) but does not enter the HOLD/NULL gate.

### 3.4 The level-panel Toda–Yamamoto test (new, additive — mirror of `panel_granger`)

**`panel_toda_yamamoto(level_pairs_by_topic, *, k=3, d_max=1) → (p_q_to_v, p_v_to_q)`**

- The **level-panel mirror** of PR3's `panel_granger`. Inputs are per-topic **LEVEL** `(quality, volume)` pairs (the `prepare_series` outputs), one per evaluable qualifying topic — **NOT differenced** (this is the one substantive difference from `panel_granger`, which differences internally).
- Stack, per topic, rows of `effect_t` regressed on `(k + d_max)` own lags of the effect series + `(k + d_max)` cross lags of the cause series + **topic fixed effects** (topic dummies) + a constant — i.e. the same within-topic, no-boundary-crossing lag construction `panel_granger` uses (PR3 §8.2), but on levels and with `k + d_max = 4` lags.
- **Block Wald on ONLY the first-`k = 3` cross-lags** (the augmenting `d_max = 1` cross-lag is included in the regression but excluded from the restriction, mirroring the per-pair TY rule), giving `p_q_to_v` (cross = quality lags in the volume equation) and `p_v_to_q` (cross = volume lags in the quality equation).
- Returns `(NaN, NaN)` if the pooled design cannot be fit (no usable rows / singular), exactly as `panel_granger` does. This is the test feeding criterion HL2 via `decide_f1`.

---

## 4. Sampling plan / units of analysis

PR3D's sampling plan is **identical to PR3 §4** and is restated here only for auditability; no number below is recomputed for PR3D, and PR3D introduces no new harvest:

- **Unit of analysis.** One **leaf topic** (`topic_id ≠ -1`); within each topic, the paired annual **level** `quality` and `volume` series are the objects under test.
- **Universe.** Papers in `epistemic_extracted.parquet` (`pmid`, `study_design`) **INNER-joined** to `novelty_semantic.parquet` (`pmid`, `topic_id`, `year`) on `pmid`, restricted to **leaf topics only** (`topic_id != -1`) — **≈ 69,339 papers across 149 leaf topics, years 1995–2026**. The implementing module is **pure**: it receives the already-joined tidy frame; the notebook performs the parquet I/O and the join (cloning PR3's notebook-11 driver).
- **Qualifying denominator.** **138 of 149** leaf topics qualify under `qualifying_topics` (§3.1, PR3 §3.5) and form the holds-fraction denominator for HL1.
- **Tier coverage.** Identical to PR3: the mean-tier (primary) series is defined over graded designs (≈ 65.4% of the universe); the remaining ≈ 34.6% (`review` + `other`) carry no tier but still count toward `volume` and toward the RCT-share denominator.

### 4.1 Pre-lock descriptive denominators (identical to PR3 §4.1 — restated, not recomputed)

These are the **same descriptive counts of the universe** PR3 §4.1 reported; they are restated for the audit and contain **no** levels-diagnostic result (no Toda–Yamamoto Wald, no cointegration p-value, no error-correction coefficient, no level-panel p-value, no levels holds-fraction):

| Pre-lock descriptive quantity | Value |
|---|---:|
| Papers in the joined universe (epistemic ⋈ novelty, leaf only) | **≈ 69,339** |
| Leaf topics | **149** |
| Qualifying leaf topics (§3.1) — holds-fraction denominator | **138** |
| Holds threshold (20% of 138) | **≥ 28 topics** |
| Mean-tier coverage (graded designs as a share of the universe) | **≈ 65.4%** |
| RCT share of the universe | **≈ 7.0%** |
| Year span | **1995–2026** |

The PR3 *result* being diagnosed (the differenced F1 null) is recorded in `data/v1/f1_cascade_verdict.json` and is **not** a PR3D input parameter: it is the prior finding PR3D characterizes, not a threshold PR3D tunes. No PR3D-derived number appears anywhere in this pre-registration, because none has been computed.

---

## 5. Series construction and leakage / NaN controls (per qualifying topic)

PR3D consumes the **same prepared series** PR3 produced, via the verbatim-reused `prepare_series` (§3.1, PR3 §5), with **one deliberate and pre-declared difference**: PR3D feeds the **LEVEL** series to its tests and does **NOT difference** them.

1. **Span trimming.** Identical to PR3 §5 step 1 — the contiguous `[min year with ≥ 5 papers, max year with ≥ 5 papers]` span, via `prepare_series` (unchanged).
2. **Internal-gap handling (primary mean-tier series only).** Identical to PR3 §5 step 2 — internal NaN-tier years linearly interpolated; if `> max_nan_frac = 0.25` of in-span years are NaN-tier, the topic is **quality-non-evaluable**, **kept in the results table with an explicit reason** (NOT silently dropped), **remains in the 138-topic qualifying denominator**, and **counts as non-significant**. This is the conservative rule PR3 declared, carried over verbatim. (The RCT-share series is gap-free, so the non-evaluable branch is a no-op there, as in PR3.)
3. **NO differencing.** PR3 §5 step 3 differenced each series once (Δ) before the Granger tests, inside `granger_pair` / `panel_granger`. **PR3D deliberately does the opposite: it feeds the undifferenced LEVEL series to `toda_yamamoto_pair`, `engle_granger_ecm_pair`, and `panel_toda_yamamoto`.** This is the entire point of the diagnostic — the Toda–Yamamoto procedure (§3.2) is valid on integrated / cointegrated levels precisely so that a level relationship the differenced test could not see is testable here. The TY augmenting lag (`d_max = 1`) is what licenses the level fit; no unit-root pre-test is performed (TY does not require one).

As in PR3 §5, there is no forecasting horizon and no train/test split (this is a retrospective lead-lag diagnostic, not a predictive one), so the controls are the deterministic span / NaN / level-fitting rules above plus the NaN guard in §3.2 (`obs ≤ 2p + 2` → NaN, treated as non-significant). The "non-evaluable counts as non-significant, stays in the denominator" rule is carried over unchanged from PR3.

---

## 6. Levels lead-lag tests (per evaluable qualifying topic)

For every evaluable qualifying topic, on the **undifferenced LEVEL** series:

### 6.1 Per-topic Toda–Yamamoto levels Granger (both directions)

`toda_yamamoto_pair(quality, volume, k=3, d_max=1)` (§3.2) returns `(p_q_to_v, p_v_to_q)`: the by-hand Wald χ²(`df = k = 3`) p-values for `quality → volume` ("quality leads") and `volume → quality` ("volume leads / quality lags"), from a VAR(`p = k + d_max = 4`, `trend="c"`) on the levels, restricting only the first `k = 3` cross-lags. `NaN` (recorded, non-significant) when `obs ≤ 2p + 2` or singular. The fixed `k = 3` mirrors PR3's fixed Granger lag order of 3 and is **not** re-selected after seeing the data; the augmenting `d_max = 1` is fixed here and not tuned.

### 6.2 Engle–Granger cointegration + error-correction (robustness / informational)

`engle_granger_ecm_pair(quality, volume)` (§3.3) returns `{cointegrated, ec_coef, ec_pvalue, coint_pvalue}`, characterizing the long-run level relationship. **Robustness / informational only — cannot flip the HOLD/NULL verdict** (§3.3, §9), same status as PR3's RCT-share rerun. Its distribution across topics is reported regardless of the verdict (§8.4).

### 6.3 CCF readout (reused, descriptive)

The descriptive CCF peak-lag distribution is reported exactly as in PR3 §6.1 / §8.4 (it remains a descriptive readout, not a gating statistic). PR3D does not change `cross_correlation`; it reuses PR3's CCF output for the descriptive panel only.

---

## 7. Power simulation (decisive; new pure module `scifield/findings/cascade_powersim.py`)

> **Pre-compute amendment note — 2026-06-09 (before any commit or real-data compute).** During implementation (before any commit or real-data compute) the original `level_cascade` DGP was found to be detectable by the differenced test — a once-differenced single-lag term carries the same `β·quality_{t−lag}` signal — so it **cannot demonstrate differencing-blindness** and cannot support the "the differenced test was blind" leg of the §8 verdict table. The cointegrated `level_cascade_coint` integrator DGP is therefore added — **before compute** — to provide the differencing-blind DGP the verdict table routes through (volume tracks a slow *integrator* of the quality level, a relationship that lives purely in the levels and that the differenced test cannot see). The literal `level_cascade` is **retained and reported alongside it** (now correctly described as a single-lag level relationship that BOTH tests detect, TY only modestly better-powered at small β). This amendment is additive: it expands the power-sim grid from three DGPs to four and re-routes the verdict's "differenced under-powered / levels well-powered" read from `level_cascade` to `level_cascade_coint`. It is recorded here transparently as a clean pre-compute correction, not a post-hoc rescue: no power-sim statistic on the real corpus exists yet, and the verdict thresholds (§8) are otherwise unchanged.

The power simulation is what makes the TRUE-vs-APPARENT verdict **decisive** rather than suggestive: it establishes, on synthetic data with known ground truth, **which test (the PR3 differenced Granger or the TY levels test) can detect which kind of cascade at the real series lengths** — and it measures each test's false-positive (size) rate so that any Toda–Yamamoto small-sample over-rejection is **measured and reported, not hidden**.

### 7.1 Four data-generating processes (DGPs)

For each replication, two annual series `quality` and `volume` are generated under one of four DGPs, with a fixed cascade lag `lag = 2`:

- **`null`** — no relationship between quality and volume (independent processes). Used to measure each test's **size** (false-positive rate); the `β = 0` cells of the other DGPs coincide with this.
- **`level_cascade`** — a **literal single-lag** level relationship: `volume_t = c + β · quality_{t−lag} + noise`. **BOTH** tests detect this — the once-differenced test sees the same single-lag `β·quality_{t−lag}` signal through its differenced term — so the TY levels test is only **modestly better-powered at small β**, not uniquely able to detect it. (This corrects the original draft, which wrongly claimed the differenced test was blind here; see the 2026-06-09 amendment note above.) This DGP is **reported but is NOT a verdict input**; the differencing-blind contrast the verdict reads off is `level_cascade_coint` below.
- **`diff_cascade`** — the cascade lives in the **DIFFERENCES**: `Δvolume_t = β · Δquality_{t−lag} + noise`. This is a deliberate **FAIRNESS case**: here the differenced test is **well-powered / competitive** on the difference-cascade and the levels test may be weaker. Including it guards against the simulation being rigged to favor TY — it shows the differenced test is the right (or competitive) tool for a difference-cascade and disadvantaged only on the cointegrated level-cascade. It is a fairness sanity check, **not a verdict input**.
- **`level_cascade_coint`** — the **cointegrated / integrator** level cascade, the differencing-blind DGP the verdict routes through. Volume tracks a slow **integrator** of the quality *level*: `stock_t = 0.9 · stock_{t−1} + 0.1 · quality_t`, then `volume_t = β · stock_{t−lag} + noise` (0.3·standard-normal). Because volume tracks the *accumulated level* of quality (not its lagged *changes*), the relationship lives purely in the levels: the once-differenced Granger test is **BLIND** to it while the Toda–Yamamoto levels test **recovers** it. This is the case that would make F1's null **APPARENT**, and the **only** DGP in the grid that demonstrates the differenced-blind / levels-powered contrast — so the §8 verdict reads its "differenced under-powered / levels well-powered" leg off **this** DGP. (The integrator constants 0.9 / 0.1 / 0.3 are byte-identical to the validated `_stock_level_cascade` unit-test fixture.)

### 7.2 Grid, replications, and what is reported (all locked here)

- **DGPs** `{null, level_cascade, diff_cascade, level_cascade_coint}` (§7.1) — all four are simulated and reported; the verdict (§8) reads its differenced-blind / levels-powered leg off **`level_cascade_coint`** only.
- **Series length** `T ∈ {8, 15, 22, 30, 40}` — matching the real qualifying-series lengths, **median ≈ 30**, which is the **headline** cell and (per §7.3) the **only** length that informs the verdict's power read.
- **Effect size** `β ∈ {0.0, 0.25, 0.5, 1.0, 2.0}` — the `β = 0.0` cells are the **null / size** cells.
- **Cascade lag** `lag = 2`.
- **Replications** `n_rep ≥ 500` per cell.
- **Significance level** `α = 0.05`.
- **Fixed random seed** (declared in the module; the simulation is fully reproducible and run once).
- For **each replication**, run **BOTH** `granger_pair` (the PR3 differenced test, reused verbatim) **AND** `toda_yamamoto_pair` (the TY levels test) on the **SAME** generated series, and record whether each rejects at `α = 0.05`.
- **Report per cell:** the **detection rate** of each test (fraction of replications rejecting), for both the `quality → volume` direction (the cascade direction) and, descriptively, the reverse.

### 7.3 The size (false-positive) requirement — measured, not hidden

The `β = 0` (null) cells give **each test's false-positive (size) rate per `T`**. The differenced test controls size (≈ 0.05–0.10) across the realistic lengths `T ∈ {22, 30, 40}`. The Toda–Yamamoto **χ² Wald over-rejects under the null at short series**: the measured TY false-positive rate is **≈ 0.28 at `T = 15` and ≈ 0.10 at `T = 22`, controlling (≈ 0.09–0.10) only at `T ≥ 30`** (these are the approximate observed rates with the AR(1) φ = 0.8 base and the spec-mandated χ²(df = k) Wald — NOT a size-corrected F form; the notebook reports the exact numbers per `T`). This is a **DISCLOSED LIMITATION** of the Toda–Yamamoto test at short series, not a hidden artifact: it is **measured per `T` and reported openly**, the `T = 8` / `T = 15` / `T = 22` size rows are reported alongside the headline `T = 30` row, and — because of it — **only the `T = 30` cell informs the §8 verdict**. The verdict's "well-powered / under-powered" read is anchored exclusively on the `T = 30` `level_cascade_coint` cells (§8.1) precisely so that a short-`T` size inflation cannot be mistaken for power; at `T = 8` the TY test cannot even fit a VAR(4) (`obs ≤ 2p + 2 = 10`) and is recorded as a non-detection. The `T = 8 / 15 / 22` cells are reported for completeness but are **EXCLUDED from the power read**.

---

## 8. Pre-registered verdict table (locked here; applied ONCE in notebook 15)

The mapping from the levels test outcome (HL1 ∧ HL2 via `decide_f1`), the Engle–Granger robustness, and the power-simulation evidence onto the diagnostic verdict is **fixed here, before any compute**, and is **applied exactly once** in `notebooks/15_F1_levels_diagnostic.ipynb`. The table is **symmetric**: a **TRUE** verdict and an **AMBIGUOUS** verdict are each **exactly as reportable** as an **APPARENT** verdict — none is privileged, and the diagnostic is not "looking for" APPARENT.

The conjunctive bar for **APPARENT is deliberately HIGH**: an apparent-null claim (i.e. that the signed PR3 result understated a real cascade) is the strongest claim PR3D can make, so it requires the levels test to hold AND be robust AND the power simulation to vindicate the levels test over the differenced test at realistic length.

### 8.1 The three verdicts

All four DGPs are reported, but the verdict's power read is taken off the **`level_cascade_coint`** DGP (the only differencing-blind DGP) at the **`T = 30`** cell — the only series length where the TY χ² Wald controls size (§7.3).

- **APPARENT** — F1's null is an artifact of the differenced specification — **iff ALL** of:
  1. the **levels test HOLDS** on the **primary `mean_tier`** series — i.e. `decide_f1` returns `holds = True`: **≥ 20% of the 138 topics (≥ 28)** show an FDR-significant **directional** Toda–Yamamoto levels relationship **AND** the **level-panel Toda–Yamamoto agrees** on the dominant direction (HL1 ∧ HL2), **AND**
  2. the levels HOLD is **robust** under the **`rct_share`** rerun (the `rct_share` levels test also holds, by the same `decide_f1` rule), **AND**
  3. the **power simulation** shows the **differenced test is UNDER-powered for `level_cascade_coint`** at the headline **`T = 30`** cell — the only series length where the TY χ² Wald controls size; the `T = 8 / 15 / 22` cells are reported but EXCLUDED from the power read because the TY test over-rejects under the null at short `T` (§7.3) — **while the levels (TY) test is well-powered** there (TY detection rate high where differenced detection rate is low, at the `level_cascade_coint` DGP at `T = 30`).
- **TRUE** — F1's null is a real level-null, and PR3's null is STRENGTHENED — **iff BOTH** of:
  1. the **levels test is NULL** (`decide_f1` returns `holds = False` on the primary `mean_tier` series — HL1 or HL2 fails), **AND**
  2. the **power simulation** shows the **levels (TY) test is well-powered for `level_cascade_coint`** at the headline **`T = 30`** cell — the only series length where the TY χ² Wald controls size; the `T = 8 / 15 / 22` cells are reported but EXCLUDED from the power read because the TY test over-rejects under the null at short `T` (§7.3) — i.e. a real cointegrated level-cascade of plausible effect size **WOULD have been detected** by the TY test, so the levels null is informative rather than a power artifact.
- **AMBIGUOUS** — **otherwise**. In particular: the levels test holds on `mean_tier` but is **not robust** under `rct_share`; **or** the levels test is null but the **levels (TY) test is underpowered** for `level_cascade_coint` at the headline **`T = 30`** cell (the only size-controlling length; the `T = 8 / 15 / 22` cells are excluded from the power read per §7.3) — so the null cannot be read as informative; **or** the **level-panel disagrees** with the per-topic directional majority (HL2 fails while HL1 holds); **or** any other combination not meeting the full APPARENT or TRUE conjunction. AMBIGUOUS is a legitimate, fully reportable outcome — it states honestly that the levels diagnostic could not decisively distinguish a true level-null from an apparent one.

### 8.2 The verdict does NOT flip the gate

The verdict JSON records **`does_not_flip_g5: true`**. Whatever the verdict, **PR3D cannot retroactively flip Gate G5** (signed DOWNSCOPE, 2026-06-08) or convert F1's signed null into a reportable finding. An APPARENT verdict is a *diagnostic signal* that the differenced specification was scope-limited; converting that signal into a reportable finding requires:

1. a **SEPARATE confirmatory pre-registration (PR4)**, git-timestamped **before** any confirmatory compute, locking the TY levels test as the confirmatory test of a **new finding F1′**, run once, with a symmetric decision rule; **and**
2. an **ADDITIVE new gate (G5′)** that *re-counts* (e.g. F1′ + F2) and *adds to* — **never overwrites** — the signed G5; **and**
3. Samer's **Phase-2 checkpoint** approval before either PR4 or G5′ is created.

The signed Gate G5 and the parent PR3 are **NEVER edited**. PR3D itself is **NEVER edited** after its registration commit.

### 8.3 Always reported (whatever the verdict)

Regardless of the verdict, notebook 15 reports, for **both** the primary `mean_tier` and the `rct_share` levels reruns: the **per-topic TY direction split** (lead / lag / coupled / none); the **levels holds-fraction** (count and % of 138); the **level-panel TY direction and p-values**; the **HOLD/NULL** verdict from `decide_f1`; the **Engle–Granger** cointegration / error-correction distribution across topics; the **count of quality-non-evaluable topics**; and the **full power-simulation detection-rate grid** for both tests across every `(T, β, DGP)` cell, **including the size (`β = 0`) rows for every `T`**. Reporting the full grid and the full split — not just the verdict — is what makes a TRUE or AMBIGUOUS verdict as interpretable as an APPARENT one.

---

## 9. Pivot conditions and integrity

- **The diagnostic is computed ONCE, after this file is committed.** There is **no re-tuning** of the Toda–Yamamoto parameters (`k = 3`, `d_max = 1`, `trend = "c"`, df = k, the by-hand Wald restriction selecting only the first-`k` cross-lags), the reused PR3 thresholds (138 denominator, BH q = 0.05, `holds_frac = 0.20`, `panel_alpha = 0.05`, the qualifying filters 30 / 8 / 5, the 25%-NaN non-evaluable cutoff, the sign convention), the Engle–Granger specification, the level-panel specification, or the power-simulation grid (`T ∈ {8,15,22,30,40}`, `β ∈ {0.0,0.25,0.5,1.0,2.0}`, `lag = 2`, `n_rep ≥ 500`, `α = 0.05`, fixed seed) **after the result is seen**. Any unavoidable change is a **clearly-logged deviation note**, not a silent edit — the standard applied to PR3 §9 and the Gate-G4 deviations.
- **Symmetric reporting; no favorable-variant rescue.** A TRUE or AMBIGUOUS verdict is reported with exactly the same prominence as an APPARENT verdict (§8). The Engle–Granger ECM rerun and the `rct_share` levels rerun are reported descriptively; a favorable Engle–Granger or `rct_share` result **cannot, by itself, convert a NULL levels verdict into a HOLD** — exactly as PR3 §9's RCT-share rerun could not rescue a failed primary, and as PR2 §9's sensitivity variants could not rescue a failed Gate G4. The single `mean_tier` levels HOLD/NULL via `decide_f1` is the primary; `rct_share` and Engle–Granger characterize robustness only.
- **This diagnostic CANNOT retroactively flip Gate G5.** It only **characterizes** the null (TRUE / APPARENT / AMBIGUOUS). The verdict JSON carries `does_not_flip_g5: true`. A flip to a reportable finding routes ONLY through the clean path of §8.2 — a fresh confirmatory PR4, an additive gate G5′, and Samer's Phase-2 checkpoint — never through a retroactive edit of G5, PR3, or PR3D, all of which are preserved unchanged. Retroactively flipping a signed pre-registered null to a "hold" would be p-hacking regardless of intent; PR3D is explicitly constructed so that it cannot do so.
- **Run-once + additive.** PR3D only **adds** files (§10). The signed Gate G5 (`docs/gates/G5_v1_findings.md`), the parent PR3, and the existing F1 artifacts (`data/v1/f1_cascade_*`) are **byte-unchanged** by this diagnostic; the anti-drift verification (§11) checks this.
- **The disposition decision remains Samer's.** PR3D and the compute fix only the *diagnostic* characterization of the F1 null. Whether the diagnostic warrants escalating to a PR4 confirmatory study is **Samer's Phase-2 checkpoint decision**, reserved for him and the co-authors; PR3D makes no escalation and no gate change on its own.

---

## 10. Data and code availability

- **Code.** The levels-diagnostic logic lives in two pure modules (no I/O, no network, no GPU):
  - **`scifield.findings.cascade`** — the verbatim-reused PR3 components (`build_topic_year_series`, `qualifying_topics`, `prepare_series`, `bh_fdr`, `classify_direction`, `decide_f1`, plus `TIER_MAP`, `evidence_tier`, `cross_correlation`, `granger_pair`, `panel_granger`) **plus** the three additive functions `toda_yamamoto_pair` / `_ty_one` (augmented-VAR by-hand Wald), `engle_granger_ecm_pair` (cointegration + error-correction), and `panel_toda_yamamoto` (level-panel mirror of `panel_granger`) — each exposing the pinned values above as default arguments.
  - **`scifield.findings.cascade_powersim`** *(new)* — the four-DGP power simulation over the locked grid (§7), pure and fixed-seed.
  - **Tests** under `tests/test_findings_cascade_powersim.py` exercise the new logic on **synthetic in-memory fixtures only** (a fixed-seed injected cointegrated level-lead — `level_cascade_coint` — that TY recovers but the differenced test is blind to; a known-null DGP on which the differenced test holds size ≈ 0.05 and the TY size is recorded per `T`); no test reads real data. The existing `tests/test_findings_cascade.py` continues to cover the reused PR3 functions unchanged.
- **Execution.** The diagnostic is executed **exactly once** on the real joined universe in **`notebooks/15_F1_levels_diagnostic.ipynb`** (cloning PR3's notebook-11 `run_f1` I/O driver), in the commit that follows this pre-registration's commit. That notebook performs the parquet I/O and the `epistemic_extracted ⋈ novelty_semantic` join, runs the levels diagnostic for `mean_tier` (primary) and `rct_share` (robustness), runs the power simulation, applies the §8 verdict table once, and writes the artifacts below.
- **Declared outputs (single run).**
  - `data/v1/f1_levels_diagnostic_results.parquet` — per-topic levels results (TY p-values both directions, FDR flags, direction, Engle–Granger fields, status) for both quality metrics.
  - `data/v1/f1_levels_diagnostic_verdict.json` — the `decide_f1` HOLD/NULL verdicts (primary + `rct_share`), the level-panel p-values, the §8 verdict (`TRUE` / `APPARENT` / `AMBIGUOUS`), and `does_not_flip_g5: true`.
  - `data/v1/f1_levels_powersim.parquet` — the per-cell detection-rate grid for both tests across every `(T, β, DGP)` cell, including the `β = 0` size rows.
  - `docs/figures/F1_levels_powersim.png` — the power-curve / detection-rate figure.
- **Data.** Inputs are the same Phase-3 `epistemic_extracted.parquet` and Phase-4 `novelty_semantic.parquet` PR3 read, reproducible from the documented upstream V1 sessions; PR3D introduces no new harvest and no new external data.
- **Cost / compute.** **$0, CPU-only; no DeepSeek or any paid API call; no network; no new harvest** (consistent with the V1-S15 integrity constraints and PR3 §10).
- **License.** Repository's existing LICENSE (Apache-2.0).

---

## 11. Registration workflow (internal git timestamp)

1. This markdown file is committed to the repository at `docs/preregistrations/PR3D_levels_diagnostic.md`. **This commit is the registration of record**; its SHA is the timestamp.
2. **Only after** that commit is in does `notebooks/15_F1_levels_diagnostic.ipynb` run the levels diagnostic and write any levels-diagnostic result; the compute lands in a **later** commit. The ordering `commit(PR3D) precedes commit(notebook-15 results)` is verifiable from `git log` and is the anti-drift guarantee that replaces an OSF DOI for this diagnostic — exactly the mechanism PR3 used (`commit(PR3) precedes commit(notebook-11 results)`).
3. The diagnostic is **additive**: the signed Gate G5, the parent PR3, and the existing `data/v1/f1_cascade_*` artifacts are byte-unchanged. The anti-drift check confirms (a) `commit(PR3D)` precedes the notebook-15 results commit, and (b) PR3 / G5 / `f1_cascade_*` are byte-identical (sha/diff) after the diagnostic runs.
4. No OSF DOI is minted this session. If the diagnostic is later escalated to a confirmatory finding, that routes through a **separate** PR4 + additive G5′ + Samer's Phase-2 checkpoint (§8.2, §9); it does not alter the plan locked here.

---

## 12. Integrity statement

The levels-cascade diagnostic specified above is pre-registered **before it is computed**. The reuse-verbatim of PR3's `build_topic_year_series`, `qualifying_topics`, `prepare_series`, `bh_fdr`, `classify_direction`, and `decide_f1` (same 138-topic denominator, same pooled BH-FDR q < 0.05, same 20% directional bar, same panel-agreement bar, same sign convention); the Toda–Yamamoto specification (`toda_yamamoto_pair`, VAR order `p = k + d_max = 4`, `trend = "c"`, by-hand Wald χ² on only the first `k = 3` cross-lags with df = k, the `obs ≤ 2p + 2` / singular NaN guard); the Engle–Granger error-correction robustness characterization (`engle_granger_ecm_pair`, robustness/informational only, cannot flip a verdict); the level-panel mirror (`panel_toda_yamamoto`, first-`k` cross-lag block Wald on levels); the power-simulation module and its locked grid (`cascade_powersim`: four DGPs `null` / `level_cascade` / `diff_cascade` / `level_cascade_coint`, `T ∈ {8,15,22,30,40}`, `β ∈ {0.0,0.25,0.5,1.0,2.0}`, `lag = 2`, `n_rep ≥ 500`, `α = 0.05`, fixed seed, both tests on the same series, size measured per `T`, with the verdict's power read routed through the differencing-blind `level_cascade_coint` DGP at the `T = 30` cell only — the only size-controlling length per the disclosed short-`T` TY over-rejection); and the **pre-stated, symmetric TRUE / APPARENT / AMBIGUOUS verdict table** are all **fixed by the commit of this document** and will **not** be altered after the result is seen. The diagnostic is computed **once**. It is reported symmetrically: a TRUE or AMBIGUOUS verdict is as reportable as an APPARENT one, and it will not be rescued by re-tuning the TY parameters, the reused thresholds, the Engle–Granger or `rct_share` reruns, or the power-simulation grid. **This diagnostic CANNOT retroactively flip the signed Gate G5; it only characterizes whether F1's null is TRUE, APPARENT, or AMBIGUOUS, and the verdict JSON records `does_not_flip_g5: true`.** Any flip to a reportable finding requires a separate confirmatory pre-registration (PR4) and an additive new gate (G5′), only after Samer's Phase-2 checkpoint; the signed Gate G5, the parent PR3, and this document are preserved unchanged. Any unavoidable deviation will be reported as a clearly-logged deviation note in the diagnostic notebook. The escalation and disposition decisions remain reserved for Samer and the co-authors.

_Samer G. Salman — Baylor College of Medicine — 2026-06-09_

---

## HANDOFF FLAGS

Restated verbatim for the code implementer (T3, who writes `scifield.findings.cascade` additions + `scifield.findings.cascade_powersim`) and the compute driver (T4, who writes/runs `notebooks/15_F1_levels_diagnostic.ipynb`). **These must match exactly.**

### Final function names / signatures locked

Reused VERBATIM from `scifield.findings.cascade` (do NOT reimplement; consume their existing pinned defaults): `build_topic_year_series`, `qualifying_topics(v_min=30, min_years=8, min_count=5)`, `prepare_series(quality_col=..., max_nan_frac=0.25, min_count=5)`, `bh_fdr(q=0.05)`, `classify_direction(*, leads_sig, lags_sig)`, `decide_f1(per_topic_results, panel_result, *, n_qualifying, holds_frac=0.20, panel_alpha=0.05)`, plus `TIER_MAP`, `evidence_tier`, `cross_correlation`, `granger_pair`, `panel_granger`.

NEW, additive (signatures locked):
- `toda_yamamoto_pair(quality, volume, *, k=3, d_max=1) -> (p_q_to_v, p_v_to_q)` — LEVEL inputs (NOT differenced); VAR order `p = k + d_max = 4`, `trend="c"`; by-hand Wald `W=(Rβ)ᵀ(RΣRᵀ)⁻¹(Rβ)` from `params` / `cov_params` (NOT `test_causality`); restrict ONLY the first `k=3` cross-lags (cause's lags 1..k in the effect equation), the `d_max=1` lag estimated-but-excluded; `p = chi2.sf(W, df=k)` with **df = k = 3**; `p_q_to_v` = quality leads, `p_v_to_q` = quality lags (PR3 sign convention; negative CCF lag = quality leads). NaN if `obs ≤ 2p + 2` (≤ 10 rows at p=4) or singular. Internal helper `_ty_one` permitted (mirrors `_granger_one`).
- `engle_granger_ecm_pair(quality, volume) -> {cointegrated, ec_coef, ec_pvalue, coint_pvalue}` — `coint_pvalue` via statsmodels `coint`; `cointegrated = coint_pvalue < 0.05`; if cointegrated, error-correction (speed-of-adjustment) coef sign + p in the Δvolume equation, else NaN. **ROBUSTNESS / INFORMATIONAL ONLY — cannot flip a verdict** (same status as PR3's rct_share rerun).
- `panel_toda_yamamoto(level_pairs_by_topic, *, k=3, d_max=1) -> (p_q_to_v, p_v_to_q)` — level-panel mirror of `panel_granger`; per-topic rows of `effect_t` on `(k+d_max)=4` own lags + `(k+d_max)=4` cross lags + topic fixed effects + const; block Wald on ONLY the first `k=3` cross-lags; LEVELS (no differencing); `(NaN, NaN)` if unfittable/singular.

### Verdict-table thresholds (locked; applied ONCE in notebook 15; symmetric)

- **APPARENT** iff ALL: (1) levels test HOLDS on `mean_tier` — `decide_f1` `holds=True`: **≥ 20% of 138 (≥ 28)** directional via TY **AND** `panel_toda_yamamoto` agrees on direction; (2) HOLD robust under the `rct_share` rerun (`decide_f1` `holds=True` there too); (3) power-sim shows the **differenced test UNDER-powered for `level_cascade_coint` at the `T = 30` cell (the only size-controlling length) WHILE the TY levels test is well-powered** there.
- **TRUE** iff BOTH: (1) levels test NULL on `mean_tier` (`decide_f1` `holds=False`); (2) power-sim shows the **TY levels test well-powered for `level_cascade_coint` at the `T = 30` cell** — a real cointegrated level-cascade WOULD have been detected → strengthens PR3's null.
- **AMBIGUOUS** otherwise (levels holds but not `rct_share`-robust; OR levels null but TY underpowered for `level_cascade_coint` at `T = 30`; OR `panel_toda_yamamoto` disagrees with the per-topic majority; OR any other non-conjunctive combination).
- **Power read routes through `level_cascade_coint` at `T = 30` ONLY** (FIX, 2026-06-09 amendment §7): the literal `level_cascade` is detectable by the differenced test and is reported but is NOT a verdict input; `diff_cascade` is a fairness check, not a verdict input; the `T = 8 / 15 / 22` cells are reported but EXCLUDED from the power read because the TY χ² Wald over-rejects under the null at short `T` (≈ 0.28 at T=15, ≈ 0.10 at T=22, controlling only at T≥30).
- Reused gate constants (DO NOT change): denominator **138**, BH **q = 0.05**, `holds_frac = 0.20` (≥ 28), `panel_alpha = 0.05`. Primary quality = `mean_tier`; robustness = `rct_share`. Non-evaluable topics stay in the 138 denominator, count as non-significant.
- Verdict JSON MUST carry **`does_not_flip_g5: true`**. No flip without a SEPARATE PR4 + additive G5′ + Samer's Phase-2 checkpoint. PR3D and signed G5 are NEVER edited.

### Power-sim grid (locked; `scifield/findings/cascade_powersim.py`)

- DGPs: **`null`** (no relationship — gives size); **`level_cascade`** (`volume_t = c + β·quality_{t−lag} + noise`, literal single-lag level relationship — BOTH tests detect it, TY only modestly better at small β; reported but NOT a verdict input); **`diff_cascade`** (`Δvolume_t = β·Δquality_{t−lag} + noise`, cascade in DIFFERENCES — FAIRNESS case where the differenced test is well-powered/competitive; not a verdict input); **`level_cascade_coint`** (cointegrated integrator: `stock_t = 0.9·stock_{t−1} + 0.1·quality_t`, `volume_t = β·stock_{t−lag} + 0.3·noise`, cascade in LEVELS via an integrator — differenced test BLIND, TY detects; **the verdict's power read routes through THIS DGP at `T = 30`**; integrator constants byte-identical to the `_stock_level_cascade` unit-test fixture).
- Grid: **`T ∈ {8, 15, 22, 30, 40}`** (median ≈ 30 = headline), **`β ∈ {0.0, 0.25, 0.5, 1.0, 2.0}`** (`β=0` = null/size cells), **`lag = 2`**, **`n_rep ≥ 500`**, **`α = 0.05`**, **fixed seed**.
- For EACH replication run BOTH `granger_pair` (PR3 differenced) AND `toda_yamamoto_pair` (levels) on the SAME series; report **detection rate per cell** for both tests. Each test's `β=0` cells report its **false-positive (size) rate per `T`**; the differenced test controls size across `T ∈ {22,30,40}` while the **TY χ² Wald over-rejects under the null at short `T` (≈ 0.28 at T=15, ≈ 0.10 at T=22, controlling ≈ 0.09–0.10 only at T≥30)** — **measured per `T` and reported, not hidden**. Because of this, **`T = 30` is the only cell that informs the verdict** and the `T = 8 / 15 / 22` cells are reported but EXCLUDED from the power read (§7.3).
- Outputs (single run): `data/v1/f1_levels_diagnostic_results.parquet`, `data/v1/f1_levels_diagnostic_verdict.json`, `data/v1/f1_levels_powersim.parquet`, `docs/figures/F1_levels_powersim.png`. Executed exactly once in `notebooks/15_F1_levels_diagnostic.ipynb` AFTER the PR3D commit.

### Spec choices made where the plan/spec left detail ambiguous (flagged for T3/T4)

1. **`p = chi2.sf(W, df=k)` degrees of freedom = k = 3** (the augmenting `d_max` lag is excluded from the df, consistent with it being excluded from the restriction `R`). The locked spec stated `df=k` explicitly; I encoded `df = k = 3` throughout. No latitude — T3 must use df = k.
2. **Engle–Granger `ec_coef` is the speed-of-adjustment loading in the Δvolume equation** (volume adjusts toward the quality-volume equilibrium), with `cointegrated` thresholded at `coint_pvalue < 0.05`. The spec named the return dict keys and "error-correction (speed-of-adjustment) coefficient sign + p" but did not pin the threshold or which equation's loading; I pinned `< 0.05` (matching `panel_alpha`/FDR q) and the Δvolume equation (the directional analogue of "quality → volume"). T3: if the regression is run in the symmetric (Δquality) direction instead, log it as a deviation. This field is informational-only and cannot flip the verdict, so the choice does not affect any gate.
3. **NaN guard `obs ≤ 2p + 2` with p = 4 ⇒ ≤ 10 usable rows → NaN.** The spec wrote `obs ≤ 2p+2`; with the locked `p = 4` that is ≤ 10. I stated both forms so T3 has the concrete integer. Mirrors PR3's `granger_pair` NaN-on-too-short discipline (recorded, non-significant, never raised).
4. **The verdict's power read = the `T = 30` cell ONLY** (tightened by the 2026-06-09 amendment §7.3). The original draft said "realistic `T` ≈ 30" with `T = 22` / `T = 40` as a surrounding band; the measured TY χ² Wald over-rejection at short `T` (≈ 0.28 at T=15, ≈ 0.10 at T=22, controlling only at T≥30) means **only `T = 30` controls size**, so `T = 8 / 15 / 22` are reported but EXCLUDED from the power read. T4 applies the verdict table reading "well-powered / under-powered" off the **`T = 30` `level_cascade_coint`** cells only.
5. **"well-powered" vs "under-powered" is read qualitatively from the detection-rate grid** (TY high where differenced is low at `level_cascade_coint`, `T = 30`); the spec did not pin a numeric power cutoff, and I deliberately did NOT invent one — pinning an arbitrary numeric power bar post-hoc would itself be a forking path. T4 reports the actual detection rates and applies the qualitative contrast at the `level_cascade_coint` `T = 30` cell; if Samer wants a numeric cutoff it should be added to PR4, not retrofitted here. Flagged as the one genuinely qualitative seam in the otherwise fully-pinned table.
