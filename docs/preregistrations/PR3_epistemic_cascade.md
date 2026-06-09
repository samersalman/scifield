# Pre-registration #3 — Epistemic cascade: does evidence quality lead or lag research volume in the orthopedic-surgery literature?

**Author:** Samer G. Salman, Baylor College of Medicine (ORCID: [0009-0007-9897-4071](https://orcid.org/0009-0007-9897-4071))
**Date drafted:** 2026-06-08
**Registration mechanism:** internal git-timestamped commit (no OSF filing this session); the committing SHA precedes the F1 compute commit.
**Codebase:** https://github.com/samersalman/scifield  (Zenodo DOI minted at release v0.1.0 — 10.5281/zenodo.20360023 | concept DOI: https://doi.org/10.5281/zenodo.20360022)
**Implementing module:** `scifield.findings.cascade` (pure, synthetic-fixture tested); executed once in `notebooks/11_F1_epistemic_cascade.ipynb`.
**F1 spec contract (source of truth):** `/tmp/v1s15_f1_contract.md` (locked by Samer 2026-06-08); this document encodes that contract in prose.
**Plan reference:** `plans/Session-Objectives-MAP.md` §V1-S15 + Gate G5.
**Design-spec reference:** `docs/superpowers/specs/2026-06-07-v1-s15-integration-design.md` (§5 F1; §2 Gate G5 logic).

---

## 0. Filing note — this is an internal, git-timestamped pre-registration

Unlike Pre-registration #2 (`docs/preregistrations/PR2_forecasting.md`), which carries an external OSF DOI (10.17605/OSF.IO/XP94F), **PR3 is not filed with the Open Science Framework in this session.** The registration mechanism is the **git commit of this file**. This commit lands in the repository's history **before** the commit that runs the F1 compute notebook (`notebooks/11_F1_epistemic_cascade.ipynb`) and writes any F1 result artifact. The git history is therefore the **timestamp of record**: the analysis plan below is provably fixed prior to the existence of any lead-lag, cross-correlation, Granger, or panel result. Where PR2 cited an OSF DOI as its anti-drift anchor, PR3 cites its own committing SHA, which precedes the F1 compute commit. No external OSF DOI is minted this session; if this analysis is later promoted to an external filing, that is a separate, additively-logged step and does not alter the plan locked here.

The verifiable claim this document makes is narrow and checkable from `git log`: **commit(PR3) precedes commit(notebook-11 results)**. That ordering is the whole guarantee.

---

## 1. Background and rationale

SciField is a multi-axis framework for monitoring the health of the orthopedic-surgery research field. Across phases, it layers (a) a deduplicated 10-journal PubMed corpus (V1-S03/S04), (b) a topic landscape over abstract embeddings (V1-S05/S06), (c) an LLM-assisted epistemic-quality layer over abstracts (Phase 3), (d) a novelty signal (Phase 4), and (e) a forecasting layer (Phase 5), with downstream integration analyses in Phase 6. This pre-registration scopes **Finding F1 — the epistemic cascade**, the central Phase-6 question and the deciding vote for Gate G5.

F1 asks a single, sharply scoped question about the temporal ordering of rigor and attention **within a research topic**: does evidence *quality/maturity* **lead** research-volume growth (rigor precedes the surge — high-quality studies appear, then volume swells), or does it **lag** (attention precedes rigor — volume swells first, and methodological quality catches up later)? The two competing readings have opposite implications for how a surgical subfield matures, and the literature offers anecdotes for both, so the direction is genuinely uncertain a priori and worth pre-registering rather than asserting.

This pre-registration locks the F1 analysis protocol — the evidence-tier map, the two annual time series, the qualifying-topic filter, the series construction (trimming, interpolation, differencing), the cross-correlation and fixed-lag Granger specifications, the multiple-comparison correction, the panel-Granger confirmation, and the binary "F1 holds" decision rule — **before the analysis is computed**. Per the project's anti-drift discipline (`plans/Session-Objectives-MAP.md`; mirrored from PR2 §1 and from the signed Gate-G4 null in `docs/gates/G4_forecasting.md`), the protocol must be fixed prior to computation so that the tier map, lag orders, FDR threshold, qualifying filters, and the holds rule cannot be tuned post hoc to a favorable result. F1 is the open finding in the V1 narrative: F2 (dual novelty) already holds under the signed Gate G3 (PROCEED, 2026-05-31), and F3 (graph forecasting) is a signed, characterized **null** under Gate G4 (NULL FINDING, 2026-06-05; HGT −2.34pp below the graph-free baseline). Under the Gate-G5 criterion (≥2 of {F1, F2, F3} hold with statistical support **and** co-authors agree the narrative is coherent), **F1 is the deciding vote**: if F1 holds, F1+F2 = 2 findings and G5 can pass on the statistical bar; if F1 comes back null, only F2 holds and the mechanical recommendation shifts toward downscoping to a methods-and-resource paper (still publishable). Pre-registering F1 — exactly as F3 was pre-registered — is what lets a null F1 read as an honest result rather than as a failure to be re-litigated.

The corpus operated on is the 10-journal orthopedic-and-surgery subset of PubMed spanning 1995–2026. F1 reads the intersection of the Phase-3 epistemic layer and the Phase-4 novelty layer; the universe and denominators are stated in §4.

---

## 2. Hypotheses

The substantive hypothesis is **directional but two-sided**: within a topic, evidence quality and research volume are temporally coupled, and one direction dominates. The pre-registered test does not assume which direction; it estimates it and gates on whether a dominant direction is supported across the topic population.

- **H1 (directional coupling at the population level).** Across qualifying leaf topics, an FDR-significant *directional* quality↔volume Granger relationship (a clear lead OR a clear lag, but not both) appears in **at least 20% of qualifying topics** (i.e., **≥ 28 of 138**).
- **H2 (panel agreement on the dominant direction).** A pooled fixed-effects **panel-Granger** test is significant (p < 0.05) in the **same** dominant direction as the per-topic significant majority.

**F1 HOLDS iff both H1 and H2 are satisfied** (§8). Otherwise **F1 is NULL** and is reported as a null.

These hypotheses are pre-registered **before the F1 analysis is computed** — before `scifield.findings.cascade` is run on real data and before any cross-correlation, Granger, or panel statistic exists. Any deviation from the analysis plan below will be reported transparently as a clearly-logged deviation note, exactly as the Gate-G4 report logged its deviations (`docs/gates/G4_forecasting.md` §5).

### 2.1 Pre-declared sign convention

The cross-correlation lead-lag sign is fixed here, before the data are seen, so that "lead" vs. "lag" cannot be reinterpreted after the fact:

> **A NEGATIVE peak cross-correlation lag means quality LEADS volume (rigor precedes the surge); a POSITIVE peak lag means quality LAGS volume (attention precedes rigor).**

The directional verdict reported as the dominant direction is `"quality_leads"` (quality→volume) or `"volume_leads"` / equivalently "quality lags" (volume→quality), consistent with this sign convention and with the Granger direction labels in §6.

---

## 3. Operational definitions

The unit of analysis is a **leaf topic** (§4); the per-topic objects under test are two annual time series and the lead-lag statistics computed from them. The canonical machine-readable source for every threshold, the tier map, and the function defaults is the implementing module `scifield.findings.cascade`; the operational rules below are exactly what that module applies, with the pinned values exposed as default arguments.

### 3.1 The two per-topic annual series

For each topic and each calendar year `y` in the topic's span:

- **`volume(topic, y)`** = the topic's **paper count** in year `y`, over **all study designs** (no design is excluded from volume).
- **`quality(topic, y)` — PRIMARY = mean evidence-tier.** The mean of the ordinal evidence-tier values over the topic's papers that year **whose design is graded**, NaN-skipping. A year with **zero graded papers** has `quality = NaN` for that year (handled in §5).

### 3.2 The evidence-tier map and its rationale

Each paper's extracted `study_design` is mapped to an ordinal evidence-pyramid tier (higher = stronger evidence):

| `study_design` | Tier |
|---|---:|
| `RCT` | **4** |
| `cohort` | **3** |
| `case_control` | **2** |
| `case_series` | **1** |
| `review` | **excluded — no tier (NaN)** |
| `other` | **excluded — no tier (NaN)** |

**Rationale for the two exclusions (pre-declared).** The Phase-3 extraction's `review` category does **not** separate systematic-review / meta-analysis (top of the evidence pyramid) from narrative review (which sits below primary studies); because these two occupy opposite ends of the pyramid and the extraction cannot tell them apart, `review` is **not safely rankable** and is given no tier. The `other` category is an **undetermined-design catch-all** (≈25% of papers) whose members cannot be placed on the pyramid, so it too is given no tier. Consequently the **mean-tier series covers the graded-design subset only — ≈65.4% of the universe** (see §4). Both excluded categories still count toward **`volume`** (which uses all designs) and toward the RCT-share denominator (§3.3); they are excluded **only** from the mean-tier numerator/denominator.

Because the excluded `review` tier would otherwise have carried the highest-rigor signal (systematic reviews / meta-analyses), the **RCT-share robustness series (§3.3)** is **pre-declared to carry the high-rigor signal** that the mean-tier series, by excluding `review`, omits. The robustness rerun is therefore not a cosmetic check: it is the pre-registered guard that the mean-tier exclusions did not discard the very signal F1 is about.

### 3.3 The quality robustness series — RCT-share

The entire F1 test (§§5–8) is **re-run** with the following series substituted for mean-tier:

- **`RCT_share(topic, y)`** = `#RCT(topic, y) / #all-papers(topic, y)`, where the **denominator is ALL designs in that topic-year**, including `review` and `other`. Every paper carries an extracted design, so the denominator equals the full topic-year paper count and `RCT_share` is defined (and gap-free) for every year in which the topic has any papers.

The robustness verdict is reported alongside the primary verdict; agreement or disagreement of the two is reported descriptively (§8.3). A favorable robustness rerun cannot, by itself, convert a null primary verdict into a hold (§9).

### 3.4 Per-topic-year intermediate columns

The module materializes, per (topic, year): `volume`, `mean_tier`, `rct_share`, and `n_graded` (the count of graded-design papers that year, i.e., the support behind `mean_tier`). These are the inputs to qualification (§3.5) and series construction (§5).

### 3.5 Qualifying topic

A leaf topic is **qualifying** iff **both**:

1. it has **≥ 30 papers total** (all designs), **and**
2. it has **≥ 8 distinct years each with ≥ 5 papers** (total count, all designs),

so that the tier ratio is not estimated from 0/1-paper noise. **138 of the 149 leaf topics qualify.** This 138 is the **fixed denominator** for the holds-fraction in H1 and §8; it does not change after the data are seen.

---

## 4. Sampling plan / units of analysis

- **Unit of analysis.** One **leaf topic** (`topic_id ≠ -1`); within each topic, the paired annual `quality` and `volume` series are the objects under test.
- **Universe.** Papers in `epistemic_extracted.parquet` (`pmid`, `study_design`) **INNER-joined** to `novelty_semantic.parquet` (`pmid`, `topic_id`, `year`) on `pmid`, restricted to **leaf topics only** (`topic_id != -1`). This yields **≈ 69,339 papers across 149 leaf topics, years 1995–2026.** The implementing module is **pure**: it receives this already-joined tidy frame as input; the notebook performs the parquet I/O and the join.
- **Qualifying denominator.** **138 of 149** leaf topics qualify under §3.5 and form the holds-fraction denominator.
- **Tier coverage.** The mean-tier (primary quality) series is defined only over graded designs (`RCT`/`cohort`/`case_control`/`case_series`), which cover **≈ 65.4% of the universe**; the remaining ≈34.6% (`review` + `other`) carry no tier (§3.2) but still count toward `volume` and toward the RCT-share denominator.

### 4.1 Pre-lock descriptive denominators (class balance, computed before locking)

The following are **descriptive counts of the universe**, computed as pre-lock class-balance / coverage checks **before** any lead-lag statistic — exactly as PR2 §4 reported train/validation class balance before freezing its thresholds. They contain **no** correlation, Granger, lead-lag, or panel result; they are stated here only to make the qualifying denominators and coverage auditable:

| Pre-lock descriptive quantity | Value |
|---|---:|
| Papers in the joined universe (epistemic ⋈ novelty, leaf only) | **≈ 69,339** |
| Leaf topics | **149** |
| Qualifying leaf topics (§3.5) — holds-fraction denominator | **138** |
| Holds threshold (20% of 138) | **≥ 28 topics** |
| Mean-tier coverage (graded designs as a share of the universe) | **≈ 65.4%** |
| RCT share of the universe (class balance of the top tier) | **≈ 7.0%** |
| Year span | **1995–2026** |

No other data-derived number appears anywhere in this pre-registration. In particular, no peak-lag, no per-topic or panel Granger p-value, no holds-fraction, and no directional split is reported here, because none has been computed.

---

## 5. Series construction and leakage / NaN controls (per qualifying topic)

For each qualifying topic, the two series are prepared identically before any lead-lag test:

1. **Span trimming.** Restrict both series to the **contiguous span `[min year with ≥ 5 papers, max year with ≥ 5 papers]`**, trimming thin leading and trailing years so the test runs on the topic's well-populated era.
2. **Internal-gap handling (primary mean-tier series only).** The mean-tier series may carry **internal NaN years** (years inside the span with zero graded papers). **Internal gaps are linearly interpolated.** If **more than 25%** of the in-span years are NaN-tier, the topic is recorded as **quality-non-evaluable**: it is **kept in the results table with an explicit reason** (NOT silently dropped), it **remains in the 138-topic qualifying denominator**, and it **counts as non-significant** (no directional relationship). This is deliberately **conservative** — non-evaluable topics can only make F1 harder to hold, never easier. (The RCT-share series is gap-free within any year having papers, §3.3, so it does not require interpolation.)
3. **Differencing for stationarity.** Each series is **differenced once (Δ)** before the Granger tests, to remove trend/non-stationarity. (The CCF in §6 is computed on the **Δ-differenced (stationarised) series — the same series fed to the Granger tests — so the lead-lag estimators are consistent and a shared trend cannot inflate the cross-correlation.** The differencing for the CCF, Granger, and panel-Granger all happens inside those test functions.)

There is no forecasting horizon and no train/test split in F1 (it is a retrospective lead-lag analysis, not a predictive one), so the leakage controls are the deterministic span/NaN/differencing rules above rather than a temporal seal. The "non-evaluable counts as non-significant, stays in the denominator" rule is the F1 analogue of PR2's "recorded, not silently dropped" volume-guard discipline.

---

## 6. Lead-lag tests (per evaluable qualifying topic)

Two complementary tests are computed for every evaluable qualifying topic:

### 6.1 Cross-correlation function (CCF)

The cross-correlation of `(quality, volume)` is computed on the **Δ-differenced (stationarised) series — the same series fed to the Granger tests** — over **integer lags ∈ [−5, +5] years** (lag window **L = ±5**), so the lead-lag estimators are consistent and a shared trend cannot inflate the cross-correlation. For each topic the module reports the **peak-|correlation| lag and its sign**, under the §2.1 sign convention (negative = quality leads; positive = quality lags). The **peak-lag distribution across all topics is reported regardless of the gate outcome** (§8.4) — it is a descriptive readout, not itself a gating statistic.

### 6.2 Granger causality (both directions, fixed lag order 3)

**Granger causality is tested in BOTH directions on the Δ-differenced series at a FIXED lag order of 3** (no lag selection, no search over lag orders). The implementation is statsmodels `grangercausalitytests`, statistic = **`ssr_ftest`**, taking the **single p-value at lag 3**:

- **`quality → volume`** p-value = evidence that **quality leads** (rigor precedes volume).
- **`volume → quality`** p-value = evidence that **volume leads / quality lags** (attention precedes rigor).

Each topic's Granger computation is wrapped in `try`/`except`: a **degenerate or non-fittable** topic yields **p = NaN in both directions**, which is **recorded and treated as non-significant** (never raised, never silently dropped). The fixed lag order of 3 is locked here and is **not** re-selected after seeing the data.

---

## 7. Multiple-comparison correction and the per-topic directional rule

- **Pooling and FDR.** ALL per-topic, per-direction Granger p-values (both directions, all evaluable topics) are pooled into a single vector and corrected with the **Benjamini–Hochberg false-discovery-rate procedure at q < 0.05**.
- **Per-topic directional rule.** A topic shows a **directional quality↔volume Granger relationship iff EXACTLY ONE direction is FDR-significant** — a clear lead OR a clear lag. If **both** directions are FDR-significant, the topic is **"coupled, no clear direction"** and **does NOT count as directional**. If **neither** is significant, the topic shows **no relationship**. Only topics with exactly one significant direction contribute to the H1 holds-fraction.

The q-threshold (0.05) and the "exactly one direction" rule are locked here and are not adjusted after the data are seen.

---

## 8. Pre-registered pass/fail criteria

### 8.1 The "F1 holds" decision rule

**F1 HOLDS iff BOTH of the following are true; otherwise F1 = NULL.**

> **(a)** **≥ 20% of qualifying topics (≥ 28 of 138)** show an FDR-significant **directional** quality↔volume Granger relationship (per §7 — exactly one significant direction), **AND**
> **(b)** a pooled **panel-Granger** test **agrees on the dominant direction**.

### 8.2 The panel-Granger confirmation (criterion b)

The panel test is a **pooled fixed-effects OLS on the within-topic Δ-differenced series**, with **topic fixed effects** (within-transformation or topic dummies):

- Regress **`Δvolume_t`** on **lags 1–3 of `Δvolume`** and **lags 1–3 of `Δquality`**, with topic fixed effects; **F-test the block of `Δquality` lags** → **p_{quality→volume}**.
- The **symmetric** regression (regress `Δquality_t` on lags 1–3 of `Δquality` and lags 1–3 of `Δvolume`, F-test the `Δvolume` block) → **p_{volume→quality}**.

**"Panel agrees"** = the panel test is **significant (p < 0.05) in the SAME dominant direction as the per-topic significant majority.** That is: take the dominant direction implied by the per-topic directional topics (whichever of quality-leads vs. volume-leads is the majority among the topics that passed §7), and require the panel's corresponding directional p-value to be < 0.05. If the panel is non-significant, or significant only in the *other* direction, criterion (b) fails and **F1 = NULL**.

**Exact tie ⇒ NULL (conservative reading).** If the count of per-topic lead topics exactly equals the count of per-topic lag topics, there is **no dominant direction** (there is no panel tie-break), so criterion (b) is **undefined and F1 = NULL** — regardless of how strongly the panel is significant in either direction.

### 8.3 Robustness rerun (RCT-share)

Criteria (a) and (b) are **re-computed in full with `RCT_share` (§3.3) substituted for mean-tier**, and the robustness verdict is reported next to the primary verdict. Whether the two verdicts **agree** is reported descriptively. The **primary verdict is the mean-tier verdict**; the RCT-share rerun characterizes robustness and **cannot convert a null primary into a hold** (§9).

### 8.4 Always reported (whether F1 holds or is null)

Regardless of the verdict, the F1 notebook reports: the **CCF peak-lag distribution** across topics; the **per-topic direction split** (lead / lag / coupled / none); the **holds-fraction** (count and %); the **panel direction and p-values**; **both** the primary (mean-tier) and the RCT-share verdicts; and the **count of quality-non-evaluable topics** (§5). Reporting the full distribution and split, not just the gate outcome, is what makes a null F1 interpretable rather than a bare "fail."

---

## 9. Pivot conditions and integrity

- **A null F1 is reported as a null.** If either criterion (a) or (b) fails, **F1 = NULL**: there is no FDR-significant directional cascade in ≥20% of topics confirmed by an agreeing panel test. Per the Gate-G5 logic (`docs/superpowers/specs/2026-06-07-v1-s15-integration-design.md` §2; `plans/Session-Objectives-MAP.md` §V1-S15), a null F1 **shifts the Gate-G5 mechanical recommendation toward downscope** — a methods-and-resource paper (F2 holding alone), still publishable — rather than proceed-to-V2. This mirrors how the Gate-G4 forecasting null was honored (`docs/gates/G4_forecasting.md`): a pre-registered null is a *result*, not a miss to be re-litigated.
- **F1 is computed ONCE, after this file is committed.** There is **no re-tuning** of the tier map, the lag window (L = ±5), the Granger lag order (3), the FDR q-threshold (0.05), the qualifying filters (≥30 papers / ≥8 years / ≥5 per year), the 25%-NaN non-evaluable cutoff, the panel-Granger specification, or the "holds" rule (≥20% directional AND agreeing panel) **after the result is seen**. Any necessary change is a **clearly-logged deviation note**, not a silent edit — the same standard applied to the Gate-G4 deviations.
- **No favorable-variant rescue.** The RCT-share robustness rerun (§8.3) is reported descriptively; a favorable RCT-share verdict **cannot** convert a failed primary (mean-tier) F1 into a hold, just as PR2 §9's sensitivity variants could not rescue a failed primary Gate G4.
- **The Gate-G5 decision is Samer's and the co-authors'.** This document and the F1 compute fix the *statistical* input to G5; the G5 report (`docs/gates/G5_v1_findings.md`) states the ≥2-of-3 count and makes a *mechanical* recommendation, but is left **unsigned** for Samer and a co-author to adjudicate narrative coherence.

---

## 10. Data and code availability

- **Code.** The F1 logic lives entirely in the pure module **`scifield.findings.cascade`** (no I/O, no network, no GPU): the tier map and `evidence_tier`, the per-(topic, year) series builder, the qualifying-topic filter, the series-preparation (trim / interpolate / non-evaluable flag) function, the cross-correlation function, the per-pair Granger wrapper (differences internally, `try`/`except` → NaN), the BH-FDR procedure, the panel-Granger function, and the `decide_f1` decision rule — each exposing the pinned values above as **default arguments**. Tests under `tests/test_findings_cascade.py` exercise it on **synthetic in-memory fixtures only** (a fixed-seed injected-lead case, a known-null random-walk case, a BH-FDR truth test, a decision-rule truth test at the 20% boundary, and a non-evaluable-topic case); no test reads real data.
- **Execution.** The module is executed **exactly once** on the real joined universe in **`notebooks/11_F1_epistemic_cascade.ipynb`**, in the commit that follows this pre-registration's commit. That notebook performs the parquet I/O and the `epistemic_extracted ⋈ novelty_semantic` join, calls the module, and writes the F1 figure and result artifacts.
- **Data.** The inputs are the Phase-3 `epistemic_extracted.parquet` (per-paper `pmid`, `study_design`) and the Phase-4 `novelty_semantic.parquet` (per-paper `pmid`, `topic_id`, `year`), both reproducible from the documented upstream V1 sessions; F1 introduces no new harvest and no new external data.
- **Cost / compute.** $0, CPU-only; no DeepSeek or any paid API call; no new harvest (consistent with the V1-S15 integrity constraints).
- **License.** Repository's existing LICENSE (Apache-2.0 per the repository's `LICENSE` file).

---

## 11. Registration workflow (internal git timestamp)

1. This markdown file is committed to the repository at `docs/preregistrations/PR3_epistemic_cascade.md`. **This commit is the registration of record**; its SHA is the timestamp.
2. **Only after** that commit is in does `notebooks/11_F1_epistemic_cascade.ipynb` run the F1 analysis and write any F1 result; the F1 compute lands in a **later** commit. The ordering `commit(PR3) precedes commit(F1 results)` is verifiable from `git log` and is the anti-drift guarantee that replaces PR2's OSF DOI for this finding.
3. No OSF DOI is minted this session. If F1 is later promoted to an external OSF filing, that is a separate, additively-logged step that does not alter the analysis plan locked here.

---

## 12. Integrity statement

The F1 epistemic-cascade analysis specified above is pre-registered **before it is computed**. The evidence-tier map, the `review`/`other` exclusions and their rationale, the two annual series, the RCT-share robustness series, the qualifying filter (≥30 papers / ≥8 years / ≥5 per year → 138 of 149 topics), the span-trim / linear-interpolation / 25%-NaN-non-evaluable rules, the CCF window (L = ±5), the fixed Granger lag order (3) with `ssr_ftest`, the Benjamini–Hochberg q < 0.05 correction, the "exactly one direction" per-topic directional rule, the fixed-effects panel-Granger specification, the sign convention (negative lag = quality leads), and the binary holds rule (≥20% directional AND agreeing panel) are all **fixed by the commit of this document** and will **not** be altered after the result is seen. F1 is computed **once**. A null F1 will be reported as a null and will shift the Gate-G5 recommendation toward downscope; it will not be rescued by re-tuning the map, lags, thresholds, filters, or the holds rule, nor by substituting the RCT-share robustness verdict for the primary. Any unavoidable deviation will be reported as a clearly-logged deviation note in the F1 notebook and the Gate-G5 report. The Gate-G5 decision itself is reserved for Samer and the co-authors.

_Samer G. Salman — Baylor College of Medicine — 2026-06-08_
