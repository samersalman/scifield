# Per-Topic Trajectories ("Future" Layer) — SciField V2 Cartography

**Status:** Exploratory result (2026-06-21)
**Author:** Samer Salman
**Program:** V2 — Literature Cartography (Phase G, V2-S10)
**Companion documents:** [`charter.md`](charter.md), [`validation_protocol.md`](validation_protocol.md),
[`roles_velocity.md`](roles_velocity.md) (sibling cartography layer this one mirrors),
[`../../CONTEXT.md`](../../CONTEXT.md).
**Code:** `scifield.cartography.trajectory` (pure logic) · `V2/scripts/build_trajectory.py` (builder)
· `V2/notebooks/v2_05_trajectories.ipynb` (figures).
**Data:** `V2/data_v2/trajectory/trajectory_series.parquet` (per topic×year, observed + projected, bands)
· `V2/data_v2/trajectory/trajectory_summary.parquet` (per topic: slope, direction, projection + bands).

This document records the per-topic **trajectory layer** of the V2 cartography: for each of the 149 leaf
topics, where its annual **share** (and raw **volume**) of corpus output has been trending across 1995–2025
and where a state-space local-linear-trend model extrapolates it to 2030, with a stated uncertainty band.

> **This is a descriptive trend projection, clearly labeled exploratory — it is NOT a predictive claim,
> and it is explicitly NOT the dead F3 emergence classifier.** F3 was a *predictive* heterogeneous-graph-
> transformer that tried to *classify* which topics would emerge; it was **signed NULL at Gate G4** (sealed
> test AUC 0.781 < the 0.804 no-graph baseline, H1 FAIL). V2-S10 revives none of that machinery: there is
> **no held-out classification and no predictive-performance claim** — only "where each topic's annual
> share/volume has been trending, extrapolated forward with a stated uncertainty range." This framing is
> restated in §5 (Limits).

---

## 1. Method (what the trajectory of a topic means)

Every leaf topic gets a single annual time series of its **share** of corpus output, the trend of that
series is estimated, and the fitted state is rolled forward five years. Two quantities are carried:

| Quantity | Definition | Role |
|---|---|---|
| **share** (headline) | `n_papers(topic, year) / total_assigned_papers(year)` — the topic's fraction of that year's *assigned* (non-noise) output | primary; removes the corpus-growth confound |
| **volume** (secondary) | `n_papers(topic, year)` — the raw annual count | reported as context; a topic can fall in share while rising in volume |

**Data.** `data/v2/archetypes.parquet` — **802,139** papers carrying a `topic_id` and a `year`, the
**78-journal V2 corpus**, years 1995–2026. There are **149 real leaf topics** (BERTopic `topic_id` 0–148).
BERTopic **noise** (`topic_id == -1`) is removed from **both** the per-topic numerator **and** the per-year
share denominator, so the share is a fraction of *assigned* output only.

**Partial-year exclusion.** The 2026 slice is incomplete at build time (~3,872 papers vs a ~28k/yr
full-corpus average), so the model is **fit on 1995–2025** only (`max_complete_year = 2025`). 2026 is then
emitted as the **first projected year**, not an observed one — it is never used to fit the trend.

**Model (state-space first, deterministic fallback).** Per topic, a **local-linear-trend** state-space model
— `statsmodels UnobservedComponents(level="local linear trend")`, fit by maximum likelihood (L-BFGS, **no
RNG**) — then `.get_forecast(5)` for the predicted mean and `.conf_int(alpha = 1 − ci_level)` for the band.
**Share is fit in logit space** (`log(p / (1−p))` after epsilon-clipping into `(ε, 1−ε)`, back-transformed with
`expit`) so its band stays inside `[0, 1]`; **volume is fit in log space** (`log1p`, back-transformed with
`expm1`, clipped at 0) so its band stays `≥ 0`. When a series is too short (`< 8` observed years) or the MLE
does not converge, a deterministic **log/logit-linear fallback** (OLS slope + analytic horizon-widening
prediction interval) is used instead; the chosen estimator is recorded in the `model` column. The pipeline is
fully deterministic — no RNG, analytic bands, byte-stable output.

**Settings.** `ci_level = 0.80` (80% bands), horizon = **+5 years to 2030**, grain = **leaf** (149 topics).

**Direction label.** `direction ∈ {rising, flat, falling}` is the sign of the projected share slope with a
dead-band: `|slope| < 1e-4` share/yr → **flat**. At corpus scale (~28k papers/yr) that threshold is roughly
three papers per year of share-equivalent drift — below the year-to-year noise of a single leaf topic — so
"flat" means "no trend distinguishable from a topic's own jitter," not literally zero.

---

## 2. Why this construction

The cartography needs a *forward-looking* layer to sit beside the static role/velocity layers, but the F3
post-mortem (Gate G4 NULL) is explicit that a *predictive emergence classifier* did not beat a no-graph
baseline on sealed data. So this layer deliberately makes the **weakest possible forward claim**: it does not
predict a label, it *describes the observed trend and draws its straight-line continuation with an honest
band*. That is a reproducible summary statistic of the past, annotated forward — not a model whose accuracy
we are asserting.

**Share, not volume, as the headline.** The 78-journal corpus grows over the panel, so a topic's raw count
rises mechanically even when its *relative* prominence is flat or shrinking. Fitting the **share** removes
that confound: a rising share means the topic is gaining ground *relative to everything else being published*.
Volume is kept alongside so the two can disagree out loud (Caveat 3).

**Logit/log space, not raw.** Shares are bounded in `[0, 1]` and volumes in `[0, ∞)`; a naïve linear fit can
forecast a negative share or volume, or a share above 1. Fitting in logit (share) / log (volume) space and
back-transforming guarantees every projected mean *and band edge* is in range, and lets the band compress
naturally as a saturating topic approaches its ceiling.

**State-space local-linear-trend, not a point regression.** A local-linear-trend model lets both the *level*
and the *slope* drift, so a topic that bent upward late in the panel (e.g. covid, digital health) is fit by
its recent trajectory rather than a single 1995–2025 average line; the forecast interval then widens with
horizon in a principled way. The OLS fallback exists only to keep the 3 hardest series (short or
non-converging) from failing the run, and it is flagged in the `model` column so a reader can discount those.

---

## 3. Caveats

1. **1995 left-censoring.** The corpus begins in 1995, so early-year shares are **panel-conditional** — they
   describe the topic's fraction of *this* corpus from its start, not an absolute historical origin. A topic
   that was already mature in 1995 has no visible ramp-up here.
2. **2026 partial-year dropped from the fit.** 2026 is incomplete (~3,872 papers vs a ~28k/yr full-corpus
   average) and is **excluded from estimation**; it appears only as the first *projected* year, so nothing in
   the trend is driven by a half-counted final year.
3. **Share vs volume can disagree.** Share removes the corpus-growth confound, so a topic can be **falling in
   share while still rising in absolute volume** (the whole corpus is growing underneath it). Both columns are
   in the tables; "falling" below always means *relative re-weighting*, not necessarily fewer papers.
4. **Extrapolation widens — and can compress at the boundary.** Bands widen monotonically with horizon in the
   **transformed** (logit/log) space. After back-transformation a share band can *narrow* near the 0 or 1
   boundary for a saturating topic — so a tight late-horizon share band reflects **saturation geometry, not
   extra confidence**. Read the band width in context with the projected level.
5. **Descriptive, not predictive — and not F3.** Restated in §5: this layer makes no held-out classification
   and no predictive-accuracy claim, and revives none of the Gate-G4-NULL F3 emergence classifier.

---

## 4. Results

**Coverage.** All **149/149** topics fit successfully (`fit_ok = True`). Estimator split:
**state_space_llt = 146**, **loglinear_fallback = 3**. Each topic has 25–31 observed years on the 1995–2025
fit window; the horizon year is 2030 for all.

**Direction breakdown.** Most topics are trend-stable; the moving set splits slightly toward falling.

| direction | n topics | reading |
|---|---:|---|
| **flat** | **81** | no share trend distinguishable from a single topic's year-to-year jitter |
| **falling** | **38** | losing *relative* share (not necessarily volume — Caveat 3) |
| **rising** | **30** | gaining relative share |
| total | 149 | |

**Top rising topics** (share slope/yr; last observed share → projected 2030 share, with 80% band; volume
context). Numbers are read directly from `trajectory_summary.parquet`.

| topic_id | label (top terms) | slope (share/yr) | last share → 2030 [80% band] | 2030 volume [80% band] | model |
|---:|---|---:|---|---|---|
| 2 | health, information, data, care, digital, learning | **+0.00331** | 0.1389 → 0.1553 [0.110 – 0.214] | ~5,068 [2,852 – 9,003] | fallback |
| 13 | covid19, influenza, sarscov2, coronavirus, vaccine | +0.00133 | 0.0196 → 0.0267 [0.006 – 0.114] | ~528 [111 – 2,494] | llt |
| 0 | knee, cartilage, acl, ligament, cruciate, tibial | +0.00132 | 0.0575 → 0.0532 [0.042 – 0.068] | ~935 [735 – 1,190] | llt |
| 3 | shoulder, cuff, rotator, rotator cuff, elbow | +0.00050 | 0.0268 → 0.0297 [0.024 – 0.037] | ~544 [424 – 696] | llt |
| 19 | pancreatic, pancreatectomy, resection, pancreas | +0.00044 | 0.0190 → 0.0225 [0.016 – 0.031] | ~439 [330 – 585] | llt |
| 1 | colorectal, resection, liver, cancer, rectal | +0.00042 | 0.0355 → 0.0325 [0.028 – 0.038] | ~575 [477 – 693] | llt |
| 35 | infection, periprosthetic, ssi, arthroplasty, wound | +0.00037 | 0.0115 → 0.0136 [0.010 – 0.019] | ~229 [161 – 325] | llt |
| 72 | opioid, pain, opioid use, opioids, postoperative | +0.00033 | 0.0073 → 0.0095 [0.005 – 0.019] | ~184 [90 – 377] | llt |
| 55 | training, skills, surgical, residents, laparoscopic | +0.00027 | 0.0070 → 0.0061 [0.004 – 0.009] | ~107 [68 – 168] | llt |
| 59 | quantum, topological, spin, magnetic, optical | +0.00026 | 0.0095 → 0.0127 [0.006 – 0.028] | ~234 [100 – 542] | llt |

> Note the share-slope sign can lead the level: topics 0 and 1 carry a positive *recent* slope yet their
> point projection dips slightly below the last observed share, because the local-linear-trend level and the
> drifting slope are estimated jointly on the full window — the band, not the point, is the honest object.

**Top falling topics** (most negative share slope/yr).

| topic_id | label (top terms) | slope (share/yr) | last share → 2030 [80% band] | 2030 volume [80% band] | model |
|---:|---|---:|---|---|---|
| 10 | coronary, heart, myocardial, cardiac, heart failure | **−0.00080** | 0.0103 → 0.0086 [0.006 – 0.011] | ~169 [126 – 225] | llt |
| 32 | sepsis, shock, septic, septic shock | −0.00070 | 0.0019 → 0.0012 [0.0008 – 0.0017] | ~20 [13 – 29] | llt |
| 9 | renal, kidney, glomerular, dialysis | −0.00063 | 0.0114 → 0.0095 [0.007 – 0.014] | ~185 [119 – 288] | llt |
| 7 | dna, rna, transcription, protein, chromatin | −0.00054 | 0.0134 → 0.0118 [0.009 – 0.015] | ~217 [165 – 286] | llt |
| 29 | pylori, colitis, gastric, intestinal, crohns | −0.00049 | 0.0058 → 0.0047 [0.004 – 0.006] | ~86 [71 – 105] | llt |
| 45 | liver, bile, hepatic, hepatocytes, mice | −0.00045 | 0.0032 → 0.0020 [0.0016 – 0.0025] | ~38 [30 – 47] | llt |
| 20 | hiv, hiv1, immunodeficiency, antiretroviral | −0.00044 | 0.0068 → 0.0060 [0.005 – 0.007] | ~111 [92 – 133] | llt |
| 6 | valve, aortic, ventricular, mitral, aortic valve | −0.00043 | 0.0103 → 0.0064 [0.005 – 0.009] | ~129 [90 – 184] | llt |
| 11 | ventilation, icu, pressure, respiratory, intensive care | −0.00042 | 0.0110 → 0.0098 [0.008 – 0.012] | ~179 [139 – 230] | llt |
| 27 | coronary, artery, cabg, coronary artery, bypass | −0.00042 | 0.0050 → 0.0039 [0.003 – 0.005] | ~71 [49 – 104] | llt |

**Interpretation (honest, not over-claimed).** The **rising** set mixes genuinely surging clinical/digital
topics — digital-health/ML (topic 2, the steepest riser and the largest non-procedural topic), covid (13),
opioids (72), periprosthetic infection (35) — with steady high-volume **surgical staples** (knee 0, shoulder
3, pancreatic 19, colorectal 1) whose recent slope is mildly positive. The **falling** set is led by
**basic-science, cardiology, and critical-care** topics (dna/rna 7, sepsis 32, renal 9, hiv 20, aortic valve
6, ICU/ventilation 11, CABG 27). Several of the falling basic-science topics — and `quantum` (59), which
appears on the *rising* side — are **generalist-journal** topics that only entered the model when the
78-journal expansion added Nature / Science / PNAS / NEJM / Lancet; their share movement is partly the corpus
composition settling after that expansion. All share shifts should therefore be read as **relative
re-weighting within a growing corpus**, not absolute decline of any field (Caveat 3).

---

## 5. Limits / Gate framing

**What this layer is.** A reproducible, deterministic, in-range **descriptive extrapolation** of each topic's
observed 1995–2025 share (and volume) trend, carried five years to 2030 with an 80% band. It is a summary
statistic of the past with an honestly-banded straight-line continuation attached.

**What this layer is NOT — stated plainly.** This is a **descriptive trend projection, clearly labeled
exploratory; it is NOT a predictive claim, and it is explicitly NOT the dead F3 GNN.** F3 was a *predictive*
heterogeneous-graph-transformer **emergence classifier** that was **signed NULL at Gate G4** (sealed test AUC
0.781 < the 0.804 no-graph baseline; H1 FAIL). V2-S10 revives **none** of that: there is **no held-out
classification, no predictive-performance metric, and no claim that these projections will come true** — only
a statement of *where each topic's annual share/volume has been trending, extrapolated forward with a stated
uncertainty range.* A wide band is the model telling the truth about how little the past constrains the
future for that topic.

**Honest limits carried forward.**
1. **Panel-conditional** — 1995 left-censoring means early shares describe this corpus from its start, not
   absolute field origins (Caveat 1); topics mature before 1995 show no ramp.
2. **Composition shift** — the 78-journal expansion injected generalist basic-science topics mid-history, so
   some share trends (especially the falling basic-science set) reflect the corpus settling, not the field.
3. **Band geometry** — late-horizon share bands can *compress* at the 0/1 boundary for saturating topics;
   narrowness there is saturation, not confidence (Caveat 4).
4. **Fallback rows** — 3 topics use the OLS log/logit-linear fallback (incl. the headline riser, topic 2);
   their bands are analytic, not state-space, and should be read as the coarser of the two estimators.

**Gate framing.** This layer is **exploratory cartography**, contributing a forward-looking *view* to the V2
map alongside the cascade, roles/velocity, and novelty layers. It carries **no Gate hypothesis test** of its
own — there is deliberately no accuracy bar to pass, because making a held-out predictive claim is exactly
what the Gate-G4 F3 NULL ruled out of scope. Any future move from *description* to *prediction* would require
a fresh, pre-registered held-out evaluation and its own gate; nothing here should be read as having cleared
one.
