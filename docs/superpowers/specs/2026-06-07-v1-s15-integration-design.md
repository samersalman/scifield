# V1-S15 — Integration design spec (F1 + F2 + F3, bonus, results draft, Gate G5)

**Session:** V1-S15 (Integration — Phase 6)  **Date:** 2026-06-07  **Status:** design (pre-plan)
**Plan ref:** `plans/Session-Objectives-MAP.md` §V1-S15 + Gate G5; master plan §5 Phase 6.
**Depends on:** V1-S14 + Gate G4 resolved (NULL, signed 2026-06-05). **Next step:** `writing-plans` → `/awesome-execute`.

---

## 1. Goal

Synthesize the four pipeline axes into the three target findings, add a bonus cross-journal/institutional layer, and produce a draft Results section, the publication figure set, and the **Gate G5** report that decides whether the V1 narrative is strong enough to scale to V2. The deliverable of *this planning phase* is a V1-S15 implementation plan; this spec is its design input.

## 2. The three findings & the Gate G5 logic

| Finding | What it claims | Gate | Status entering V1-S15 |
|---|---|---|---|
| **F1** — epistemic cascade | Within a topic, evidence *quality/maturity* **leads or lags** research-**volume** growth | G2 validated the *extraction*; the *finding* is computed **here** | **Open — the pivot** |
| **F2** — dual novelty | Semantic × structural novelty are near-independent and form an interpretable 2×2 | G3 = PROCEED | **Holds** |
| **F3** — graph forecasting | An HGT beats a graph-free baseline at 3-yr emergence forecasting | G4 = NULL (signed 2026-06-05) | **Characterized null** |

**Gate G5 criterion (from the map):** ≥2 of {F1, F2, F3} hold with statistical support **and** co-authors agree the narrative is coherent. F2 holds; F3 is an honest null (a *result*, not a "hold"); **F1 is the deciding vote.** If F1 holds → F1+F2 = 2 → G5 can pass on the stat bar (coherence still requires co-author sign-off). If F1 is also weak → only F2 holds → recommend downscoping to a methods + resource paper (still publishable). The G5 report states this count plainly and makes a *mechanical* recommendation; the decision is Samer's + co-authors'.

## 3. Decisions locked in brainstorming (2026-06-07)

1. **F1 is pre-registered and computed honestly** — F1 can come back null without it reading as p-hacking, mirroring the F3 model.
2. **PR3 form = internal, git-committed** `docs/preregistrations/PR3_epistemic_cascade.md`, committed **before** the notebook-11 compute commit (timestamped via git history; no external OSF filing this session).
3. **F1 primary quality metric = evidence-tier index** (ordinal study-design pyramid); **RCT-share = pre-registered robustness check.**
4. **Sequencing = "F1-first, then parallel fan-out"** (Approach 1): lock PR3 → compute F1 → early G5-viability read → parallel {F2 trends, F3 re-frame, full bonus} → converge on figures + results draft + G5.
5. **Bonus scope = full** — cross-journal seeding + per-specialty contrast + institutional/geographic, with an up-front metadata-availability guard (degrade gracefully, don't trigger a new harvest).
6. **Cost = $0, CPU-only, no DeepSeek / no new harvest** — everything runs on already-extracted data.
7. **Packaging rule (project cross-cutting):** new analysis logic lives in tested `scifield` modules; notebooks import, never reimplement.

## 4. Architecture & sequencing

```
PR3 (lock F1 spec, commit)
  └─> notebook 11: F1 compute  (scifield.findings.cascade)  ── early G5-viability read ──┐
                                                                                          v
        ┌──────────────── parallel fan-out (independent files) ───────────────┐   (decide narrative spine
        │ notebook 12: F2 trends   (reuse scifield.novelty.archetypes)         │    from F1's actual result)
        │ notebook 13: F3 re-frame (reuse V1-S14 artifacts; no new compute)    │
        │ notebook 14: full bonus  (scifield.findings.seeding)                 │
        └──────────────────────────────────────────────────────────────────────┘
                                          │ converge
                                          v
   figure set (F1, F2, F3, ≤3 bonus) → docs/results_draft.md → docs/gates/G5_v1_findings.md (UNSIGNED — STOP)
```

**Packaging:** new modules `src/scifield/findings/cascade.py` (F1 lead-lag) and `src/scifield/findings/seeding.py` (bonus networks) — a new `findings` subpackage, unless an existing package (e.g. `scifield.epistemic`) is the more natural home; the implementation plan places them per existing conventions. F2-trends reuses `scifield.novelty.archetypes`; F3-reframe reuses the V1-S14 forecasting artifacts.

## 5. F1 — epistemic cascade (the pre-registered pivot)

**Hypothesis.** Within a topic, does evidence *quality/maturity* **lead** research-volume growth (rigor precedes the surge) or **lag** it (attention precedes rigor)?

**Two per-topic annual time series.**
- **Volume**: paper count (and corpus-share) per year for the topic.
- **Evidence quality (primary)**: the **evidence-tier index** — map each paper's extracted `study_design` onto an ordinal evidence-pyramid tier (e.g. systematic-review/meta-analysis > RCT > prospective cohort > case-control > cross-sectional/case series > case report), aggregated per topic-year as the **mean tier** (with **share ≥ RCT-tier** as a secondary readout). **Robustness:** re-run the whole test with **RCT-share** as the quality series.

**Method (locked in PR3).**
- **Cross-correlation function (CCF)** between the two series over lags ±L → the peak-correlation lag and its sign (quality leads vs lags).
- **Granger causality** in both directions (quality→volume and volume→quality) at a pre-specified max lag, on **differenced** (stationary) series.

**Proposed pre-registered parameters** (concrete defaults — finalized in PR3; adjust at spec review):
- Lag window **L = ±5 years**; Granger **max lag = 3 years**.
- Multiple comparisons across topics: **Benjamini-Hochberg FDR, q < 0.05.**
- **Qualifying topic** = ≥ `v_min` (**30**) papers **and** ≥ **8** years with an annual paper count ≥ **5** (so the tier ratio isn't 0/1 noise).
- **F1 "holds"** iff (a) an FDR-significant *directional* quality↔volume Granger relationship appears in **≥ 20%** of qualifying topics, **and** (b) a **pooled panel-Granger** test agrees on the dominant direction. (Both the fraction and the panel test are pre-registered; reporting also includes the CCF peak-lag distribution.)

**Deliverables:** `scifield.findings.cascade` (pure, synthetic-fixture tested) → `notebooks/11_F1_epistemic_cascade.ipynb` → **figure F1** (exemplar quality-vs-volume trajectories + a lead-lag summary panel: peak-lag distribution and the per-topic direction split).

**Integrity:** F1 is computed **once** after PR3 is committed; no re-tuning the spec, metric, lag window, or threshold after seeing the result. A null F1 is reported as a null (and shifts the G5 recommendation toward downscope).

## 6. F2 — dual-novelty trends (`notebooks/12_F2_dual_novelty_trends.ipynb`)

Reuses `data/v1/archetypes.parquet` + `scifield.novelty.archetypes`. New analysis: **archetype distribution by year and by topic**, and **archetype × citation impact** (do certain archetypes attract more citations? — Kruskal-Wallis across the four archetypes, with a rank-based effect size; pre-specify the impact metric, e.g. `cited_by_pctile`). F2 already holds (G3), so this enriches the narrative rather than re-gating it. Output extends the existing `F2_dual_novelty.png` with a temporal/impact panel. Any new statistic lives in a tested helper.

## 7. F3 — forecasting narrative (`notebooks/13_F3_forecasting_narrative.ipynb`)

**No new compute, no model.** Reads the V1-S14 artifacts (`forecasting_test_*`, the verdict JSON) + `F3_forecasting.png` and re-frames the **null** in the integrated story: *temporal dynamics alone (`no_graph` AUC 0.80) carry the emergence signal; the citation-graph topology adds nothing on the sealed test.* Threads F3 to F1 — both concern a topic's temporal evolution. May add one summary panel; primarily narrative.

## 8. Bonus — cross-journal / specialty / institutional (`notebooks/14_bonus_cross_journal.ipynb` + `scifield.findings.seeding`)

- **Cross-journal seeding:** per topic, identify which journals publish *first* (seed) vs *follow* → a per-journal seeding/lead-follow score and a directed seeding network.
- **Per-specialty contrast:** group the 10 journals (surgical vs medical); compare the F1 cascade, F2 archetype mix, and seeding patterns across specialties (feeds the V2 specialty-dependence question).
- **Institutional/geographic:** seeding by institution / country — **gated by an up-front metadata-availability check** (per-paper affiliation + country from the OpenAlex enrichment). If that metadata is thin/absent, this panel **degrades to a documented limitation** rather than triggering a new harvest.
- **≤ 3 figures, clearly labeled exploratory / non-gating.** Logic in `scifield.findings.seeding` (tested). This layer never affects the G5 count.

## 9. Integration — figure set + results draft + G5 report

- **Figure set:** F1 (cascade), F2 (2×2 + trends), F3 (forecasting null), + ≤ 3 bonus — consistent style, **< 1 MB each**, each with a `record_run` sidecar.
- **`docs/results_draft.md`:** a **Results-section narrative only** (intro/methods/discussion are out of scope → V3-S05). Must be readable by a co-author who has not seen the pipeline; every claim ties to a figure + its statistic; F3 is framed honestly as the null. Walks F1 → F2 → F3 → bonus.
- **`docs/gates/G5_v1_findings.md`:** mirrors the G3/G4 gate-report structure — a pass-criteria table (which of F1/F2/F3 held + the ≥2-of-3 count), a provenance block, narrative-coherence assessment, sensitivity/caveats, and a **mechanical recommendation** (proceed-to-V2 vs downscope-to-methods/resource) with a **blank sign-off** for Samer + a co-author-review note. **STOP** — resolve G5 before V2.

## 10. New modules & packaging

| Module | Purpose | Tested with |
|---|---|---|
| `scifield.findings.cascade` | F1 series construction, CCF, Granger wrapper (statsmodels), BH-FDR, the pre-registered decision rule | synthetic fixtures with a known injected lead-lag and a known null |
| `scifield.findings.seeding` | per-journal/institution seeding & lead-follow scores, directed seeding network, specialty grouping | synthetic fixtures with a known seeding order |

Reused (no modification): `scifield.novelty.archetypes`, the V1-S14 forecasting artifacts, `scifield.repro.record_run`. Module placement follows the existing package layout.

## 11. Testing & rigor

- **Unit tests** for both new modules (pure, synthetic, no I/O / no network / no GPU): cascade recovers an injected lead-lag and correctly returns "no relationship" on independent series; FDR + decision rule truth-tested; seeding recovers a known first-publisher order.
- **PR3 committed before the F1 compute commit** (git-timestamped); F1 computed once, honestly; a null is reported as a null.
- **Acceptance tests (definition of done):**
  1. `docs/preregistrations/PR3_epistemic_cascade.md` committed *before* notebook 11's results commit.
  2. Notebooks 11–14 each render end-to-end via `nbconvert`.
  3. Figure set committed: F1, F2 (+trends), F3, ≤3 bonus — each < 1 MB with a `record_run` sidecar.
  4. `docs/results_draft.md` committed and comprehensible to a pipeline-naïve co-author.
  5. `docs/gates/G5_v1_findings.md` committed with the ≥2-of-3 count + mechanical recommendation; **sign-off blank**.
  6. `uv run pytest` green (incl. new `tests/test_findings_cascade.py`, `tests/test_findings_seeding.py`); `pre-commit` green.
  7. Integrity: $0, CPU-only, no DeepSeek / no new harvest; F1 not re-tuned post-hoc; the institutional/geographic panel either runs on present metadata or is a documented limitation.
  8. **STOP** — present the G5 count + recommendation for Samer's + co-authors' decision (do not self-sign).

## 12. Out of scope (per the map)

v2 corpus expansion (→ V2-S01); OSS release / docs polish (→ V3); manuscript intro/methods/discussion (→ V3-S05); re-running anything from earlier phases; re-tuning or re-scoring F3 (the V1-S14-2 retry is a separate, freshly-pre-registered effort).

## 13. Open items to finalize at spec review / in PR3

- The exact `study_design` → evidence-tier mapping (depends on the extracted category vocabulary; the plan will read the actual categories and pin the ordinal map in PR3).
- The F1 pre-registered numbers in §5 (L, max lag, q, the 20%/panel decision rule, the ≥30/≥8/≥5 filters) — proposed defaults; confirm or adjust before PR3 is committed.
- The F2 impact metric (`cited_by_pctile` vs raw citations) and the specialty grouping of the 10 journals.
