# Gate G6 — Readiness Packet (for Samer's decision)

**Date prepared:** 2026-06-09 · **Prepared by:** automated V2 prove-phase execution (Phases A–E)
**Status:** ✅ **SIGNED — PASS by Samer Salman, 2026-06-09.** → Fund Phases F–G. See sign-off block at end.
Gate G6 is a human call by Samer, like G1–G5; it was signed by the human, not self-signed.

---

## What G6 decides

G6 is the **spend gate**. It replaces V1's "≥2-of-3 findings" bar with a cartography-appropriate one.

- **PASS** → the cartographic methods produce non-artifactual, interpretable structure → **fund
  Phases F–G** (hybrid corpus expansion + public map release; DeepSeek/GPU spend begins, under the
  `feedback-deepseek-spend-gating` rule).
- **FAIL** → the structure is a panel artifact or too unstable to trust → **do not spend**; reconsider
  corpus design or method before scaling. The V1 paper (Track 0) ships regardless.

The pass bar (locked in `validation_protocol.md` §5, before any compute): cascade structure survives
the null/permutation + jackknife; journal roles are stable and interpretable; ≥1 novelty-origin
finding is robust; the map v0 is coherent to a domain reader.

---

## The four ingredients — verdicts

### (i) Does the cascade survive null + jackknife? → **PASS (with qualification)**
Source: `cascade_validation.md`, `V2/data/cascade/{null,jackknife,granularity}_results.parquet`
(n_perm=1000, seed=20260609).

- **Primary, gate-critical (S1 seeding-spread vs within-topic null, Null-1):** significant at all four
  cells — leaf z=+3.27 p=0.0020, mid z=+4.04 p=0.0010.
- **The decisive 1995 test:** on the *resolvable* subset (topics with ≥2 distinct first-appearance
  years, i.e. removing the ~80% all-tied-at-1995 topics), the primary z is essentially unchanged
  (leaf 3.27→3.22, mid 4.04→3.67). **The structure lives in genuinely-ordered topics, not the
  left-censoring artifact.** This was the single biggest threat and it does not hold.
- **Granularity (Method D):** leaf-vs-mid journal-ranking Spearman ρ=0.855 — robust across grain.
- **Origin jackknife (Method C, S04 half):** drop-one-journal flip-rate=0.0.

**Honest qualifications (do not skip these):**
- The secondary statistic **S2 (net directed lead-lag asymmetry) FAILS under Null-1** (z=0.41) because
  a within-topic label shuffle conserves total asymmetry *magnitude* while destroying *who-leads-whom*;
  S2 passes under Null-2 (year-shuffle, z=3.35). The protocol's designated primary is S1, which passes
  cleanly — but the Method-B *block* is therefore PARTIAL, not a clean sweep. On record, not hidden.
- The jackknife flip-rate of 0.0 is **partly structural** (origin = `argmin(first_year)` with a
  deterministic alphabetical tie-break, so dropping a non-origin journal cannot change the earliest).
  It confirms non-fragility rather than being a strong independent test.
- **Held-out years (Method A) is weak** (leaf ρ=0.30, mid ρ=0.09) — but it was pre-designated
  SUPPORTING-only because the 2019–2025 window is just 7 years and inherits the 1995 censoring. It does
  not sink the gate; B/C/D are the gate-critical checks.

### (ii) Are journal roles stable and interpretable? → **PASS**
Source: `roles_velocity.md`, `V2/data/roles/{role_scores,velocity}.parquet`.

- **Stability:** drop-one-journal `role_rank_correlation` = 0.945 (leaf) / 0.962 (mid), both clearing
  the ≥0.7 bar. Robust across grain too.
- **Interpretable, matches within-corpus priors:** general-surgery generalists (Br J Surg, Surgery,
  J Am Coll Surg) score **source**; subspecialty journals (Arthroscopy, Spine) score **bridge**;
  prestige authorities (Ann Surg, JBJS, CORR, JAMA Surg) score **terminal**.
- **Velocity:** median time-to-citation 4–7y; J Arthroplasty fastest (4y), prestige terminals slowest
  (6–7y) but accrue the most citations with the longest tails.
- **Caveat (panel-conditional):** "terminal" here = a *within-panel* citation sink (net importer), NOT
  a global dead-end. The true source generalists (Nature/NEJM/Lancet) are not in the corpus yet — this
  is exactly what the Phase-F source layer is designed to correct.

### (iii) Is ≥1 novelty-origin finding robust? → **PASS (sector near-null is the cleanest; geography moderate)**
Source: `novelty_origins.md`, `V2/data/origins/{sector,geo}_robustness.parquet` (seed 20260609).
Note: topic-granularity Method-D is **N/A by construction** for per-institution country/sector findings
(they don't depend on topic grain); robustness is established by split-half stability + bootstrap CI.

- **Sector = robust, well-powered NEAR-NULL.** Company-minus-non-company novelty delta CIs span zero on
  both axes: `sem_nov_mean` −0.0006 [−0.0031, +0.0019]; `cd5` +0.0034 [−0.0106, +0.0173]. With 5,091
  company-affiliated papers, this is a *real* null, not underpowered. **Honest finding: in this corpus,
  novelty is NOT concentrated in the tech/industry sector.**
- **Geography = moderately robust** (secondary): split-half ρ ≈ 0.49–0.67 (random + temporal) — a
  reproducible country ordering, but not razor-sharp.
- **Correction on record:** S06 initially asserted geography was "the most robust" origin finding; the
  computed numbers say the **sector well-powered near-null is the cleaner robust finding**.
- **Deferred (coverage notes, not failures):** funding-origin (OpenAlex `grants` parser added but not
  wired through the store / re-harvested — $0 this session) and citation-intent origin (`citation_intents.parquet`
  empty; needs `SEMANTIC_SCHOLAR_API_KEY` + a re-run). Both are Phase-F-adjacent re-harvests.

### (iv) Is the map v0 coherent to a domain reader? → **COHERENT (PASS-ready)**
Source: `map_v0_README.md`, `V2/map_v0/index.html` (self-contained static HTML, 126.8 KB, 9 Plotly
panels across 4 tabs; opens by double-click, no server).

Renders all three required layers — cascades, roles, origins — on the current corpus, each with its
caveat box and upstream verdict surfaced. The narrative lines up: generalist-surgery journals seed →
subspecialty journals bridge → prestige journals are within-panel citation sinks. The panel-conditional
+ 1995-censoring + deferred-layer limitations are foregrounded, not buried.

---

## Bottom line for the decision

| Ingredient | Bar | Verdict |
|---|---|---|
| (i) Cascade survives null + jackknife | primary null PASS + jackknife + granularity | **PASS (qualified)** |
| (ii) Roles stable + interpretable | role_rank_correlation ≥ 0.7 | **PASS** (0.945 / 0.962) |
| (iii) ≥1 robust novelty-origin finding | one finding robust | **PASS** (sector near-null; geography moderate) |
| (iv) Map v0 coherent | coherent to a domain reader | **PASS-ready** |

**All four ingredients clear their bars.** The honest reading is: the cartographic method produces
**real, non-artifactual, interpretable structure on the 10-journal corpus** — the cascade is not a
1995/panel artifact (it survives the resolvable-subset control), roles are stable and match priors, and
there is at least one robust novelty-origin finding (a clean, well-powered sector near-null). On the
locked bar this is a **PASS → fund Phases F–G**.

**What Samer should weigh before signing:**
1. The cascade PASS rests on the **S1/Null-1 primary**; the Method-B block is PARTIAL because S2 fails
   under Null-1 (mechanistically explained). If you want the block-level sweep, that's a reason to call
   it PASS-with-qualification rather than clean PASS — but it is not a fail.
2. Ingredient (iii)'s strongest robust finding is a **null** (sector). A robust null is a legitimate
   "interpretable structure," but if G6's intent was a robust *positive* origin signal, geography is
   only moderate (ρ≈0.5–0.67) and the positive signals (funding/intent) are deferred.
3. Everything is **panel-conditional**: origins mean "first in our 10-journal set," and "terminal"
   journals are within-panel sinks. Phase F's source layer (Nature/NEJM/Lancet) is precisely the
   correction — so a PASS here funds the work that tests whether these attributions survive the true
   source generalists entering the panel.
4. The held-out (Method A) result is weak; any forward-looking/trajectory claim (Phase G / V2-S10) must
   stay descriptive and cautious until the expanded corpus's longer history is available.

**If PASS:** proceed to V2-S08 (expansion harvest) — and the `feedback-deepseek-spend-gating` rule
triggers (dry-run cost estimate + explicit OK before any extraction).
**If FAIL / DOWNSCOPE:** the V1 methods-and-resource paper (Track 0) ships regardless; reconsider corpus
design or method before spending.

---

## Sign-off (human)

- **Decision (PASS / PARTIAL / FAIL):** **PASS** → fund Phases F–G (corpus expansion + public map).
- **Signed:** Samer Salman  **Date:** 2026-06-09
- **Rationale / conditions:** All four G6 ingredients cleared their pre-locked bars — cascade is REAL
  (with qualification) and survives the 1995-left-censoring resolvable-subset control; roles are stable
  (role_rank_correlation 0.945/0.962) and interpretable; ≥1 novelty-origin finding is robust (the
  well-powered sector near-null); map v0 is coherent. The cartographic method is proven on the current
  10-journal corpus, which is exactly what G6 gates. **Condition:** Phase F spend is now authorized in
  principle, but the `feedback-deepseek-spend-gating` rule applies — a dry-run cost estimate + explicit
  per-run OK from Samer is required before ANY DeepSeek extraction call in V2-S08.
