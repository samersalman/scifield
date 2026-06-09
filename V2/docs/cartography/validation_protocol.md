# Exploratory-Rigor Validation Protocol — SciField V2 Cartography

**Status:** LOCKED before Phase B compute (2026-06-09)
**Author:** Samer Salman
**Program:** V2 — Literature Cartography (Phase A, V2-S01)
**Companion:** [`charter.md`](charter.md) (the map's questions and stance), [`../../CONTEXT.md`](../../CONTEXT.md).

This protocol **replaces the V1 pre-registered hypothesis gates (G1–G5)**. It specifies the
four rigor methods that make an *exploratory* map trustworthy, with each method given a concrete
name, definition, locked parameters, and PASS/PARTIAL/FAIL rubric that the cascade-validation
task (**V2-S04**) and the roles task (**V2-S05**) must implement *exactly as written*. It is
locked **before any Phase B (cascade) compute** so results cannot be quietly reverse-engineered.

Downstream tasks (V2-S03 cascade engine, V2-S04 validation, V2-S05 roles) implement these
definitions. The flow data model and the null/jackknife primitives (T1b, `flow.py`) provide the
shared (`pmid → journal_slug`) resolver and the permutation/jackknife machinery; this protocol
defines *what* those primitives compute and *how the verdict is read*.

---

## 0. Locked global constants (every method uses these)

| Constant | Locked value | Rationale |
|---|---|---|
| Corpus | current 10-journal panel, 1995–2025 | prove-phase, $0 (charter §7) |
| **Held-out split** | learn on **1995–2018**, validate on **2019–2025** | §1 |
| **Permutation count** | **n_perm = 1000** | §2 |
| **RNG seed** | **seed = 20260609** | §2 — fixed, reproducible; = the lock date |
| **Granularities** | **leaf** (149 `topic_id`) and **mid** (96 `mid_level_id`) | §4 |
| Min papers per (topic, journal) to count an appearance | **min_papers = 1** | matches `seeding.first_publication_year` default; sensitivity at min_papers ∈ {1, 3} reported as a secondary robustness pass |
| First-appearance definition | earliest `year` in which a (topic, journal) reaches `min_papers` | reuses `scifield.findings.seeding.first_publication_year` |

These constants are the contract. V2-S04 and V2-S05 must read these exact values; if a task
needs to deviate, it logs the deviation and reason in `V2/EXECUTION_LOG.md` before computing.

---

## 1. Method A — Held-out years (temporal split)

**What it checks.** That the cascade structure is not an over-fit description of the full
history but a regularity that *holds out of sample in time*: structure learned on early years
should still describe later years.

**Definition.**
- **Train window:** first-appearance and lead-lag structure are fit on papers with
  `year ≤ 2018` (1995–2018, 24 years).
- **Holdout window:** `year ≥ 2019` (2019–2025, 7 years).
- **Procedure.** (i) On the train window, compute per-journal **seeding scores** and the
  directed lead-lag (seeding) network (`scifield.findings.seeding.seeding_score` /
  `directed_seeding_network`). (ii) On the holdout window, recompute the same per-journal
  seeding scores *independently*. (iii) Compare: the test statistic is the **Spearman
  rank-correlation `ρ_holdout` between the train-window journal seeding-score ranking and the
  holdout-window journal seeding-score ranking** (10 journals → 10 paired ranks). A journal that
  systematically leads should keep leading.

**Why this split.** The corpus spans 1995–2025 (31 years). 1995–2018 gives a long, deep learning
window with enough years per topic to estimate first-appearance order; 2019–2025 leaves a 7-year
holdout that is long enough for a topic to appear in several journals (so lead-lag is estimable)
yet recent enough that any structural drift would show up. A topic-cascade is a slow process, so
the holdout must be multi-year; 7 years is the shortest window that still admits a within-window
diffusion ordering. The split is *temporal*, never random, to respect the arrow of time and
avoid leakage of late information into the "learn" set.

**Rubric.**
- **PASS** — `ρ_holdout ≥ 0.5` (journal lead/lag ranking is stably reproduced out of sample).
- **PARTIAL** — `0.2 ≤ ρ_holdout < 0.5` (ordering partially reproduces; report and qualify).
- **FAIL** — `ρ_holdout < 0.2` (the structure does not generalize across the time split).

**Caveat to report.** Topics that only appear after 2019 are unscorable in the train window and
are excluded from the held-out comparison; report the count excluded.

---

## 2. Method B — Null / permutation models

**What it checks.** That the observed inter-journal lead-lag structure exceeds what arises by
chance when the journal-to-topic assignment carries *no* genuine ordering — the core defense
against "this cascade is a panel/sampling artifact."

**Two locked null models.** Both run at **n_perm = 1000** permutations with the global
**seed = 20260609** (use a `numpy.random.default_rng(20260609)` stream; each permutation draws
the next state, so the full sequence is reproducible).

1. **Null-1: journal-label shuffle (primary).** *Within each topic*, randomly permute the
   journal labels across that topic's papers, holding the set of (topic, year) appearances
   fixed. This destroys any systematic journal-leads-journal ordering while preserving each
   topic's temporal footprint and each journal's overall volume per year. This is the primary
   null for the cascade and for journal roles.
2. **Null-2: first-appearance-year shuffle (secondary).** For each topic, randomly permute the
   *first-appearance years* across the journals that ever publish that topic, holding the journal
   set fixed. This destroys the lead-lag ordering while preserving which journals are involved.
   Reported as a secondary cross-check on Null-1.

**Test statistics (two, reported for each null).**
- **S1 — seeding-score spread.** The standard deviation across the 10 per-journal seeding scores
  (`seeding_score`). A real cascade makes some journals consistent leaders and others consistent
  followers → high spread; a no-structure null → low spread.
- **S2 — net directed lead-lag asymmetry.** From the directed seeding network
  (`directed_seeding_network`, edge weight = `n_precedes / n_shared`), the sum over journal pairs
  of `|weight(A→B) − weight(B→A)|`. A real cascade has directional edges (A leads B more than B
  leads A) → high asymmetry; a no-structure null → near-symmetric edges → low asymmetry.

**Reported outputs (for each null × each statistic).**
- **Permutation p-value:** `p = (1 + #{perm stat ≥ observed stat}) / (1 + n_perm)` (the +1
  add-one form, so p is never exactly 0 and is valid at n_perm = 1000; one-sided, upper tail).
- **Effect size:** a standardized **z-score** `z = (observed − mean(null)) / sd(null)` *and* the
  observed value's **percentile** within the null distribution. Report both; do not report a
  p-value without the effect size.

**Rubric (applied per statistic; the cascade must pass on S1, the primary statistic, under
Null-1).**
- **PASS** — `p < 0.01` **and** `z ≥ 2` under Null-1 for S1 (seeding-score spread), with the
  same direction corroborated by S2 and by Null-2.
- **PARTIAL** — `p < 0.05` but `z < 2`, **or** S1 passes under Null-1 but S2 or Null-2 does not
  corroborate (structure present but weak/inconsistent across statistics).
- **FAIL** — `p ≥ 0.05` on S1 under Null-1 (observed structure indistinguishable from the
  no-ordering null → the cascade is a panel artifact).

---

## 3. Method C — Drop-one-journal jackknife

**What it checks.** Stability of origin attributions and journal roles against the choice of
panel — the operational proxy for the panel-conditional caveat (charter §5). If removing any
single journal flips many origin attributions or reshuffles the role ranking, the structure is
fragile to which journals we happened to harvest.

**Definition.** Run the full pipeline **10 times**, each time **removing exactly one of the 10
journals** and recomputing on the remaining 9: (i) per-topic **origin-journal attribution**
(which surviving journal published each topic first), (ii) per-journal **seeding scores**, and
(iii) per-journal **role scores** (source/bridge/terminal, V2-S05). Each leave-one-out run is
otherwise identical to the full run (same constants, §0).

**Stability metric — name: `jackknife_stability`.** A two-part reported metric:
- **`attribution_flip_rate`** — the fraction of topics whose origin-journal attribution *changes*
  across the leave-one-out runs, computed on the topics whose original origin journal is **not**
  the removed journal (a topic whose true origin journal was removed *must* be reattributed and is
  excluded from the flip-rate denominator for that run; report it separately as
  `forced_reattribution_count`). Averaged over the 10 runs.
- **`role_rank_correlation`** — the mean Spearman rank-correlation between the full-run journal
  role-score ranking and each leave-one-out run's role-score ranking (averaged over the 9
  surviving journals per run, then over the 10 runs).

**Rubric.**
- **PASS** — `attribution_flip_rate ≤ 0.20` **and** mean `role_rank_correlation ≥ 0.7` (origins
  and roles are stable to dropping any single journal).
- **PARTIAL** — `0.20 < attribution_flip_rate ≤ 0.40` **or** `0.5 ≤ role_rank_correlation < 0.7`
  (moderate sensitivity; report which journal's removal is most destabilizing and qualify claims).
- **FAIL** — `attribution_flip_rate > 0.40` **or** `role_rank_correlation < 0.5` (the structure
  is an artifact of the specific panel composition; do not trust origin/role claims).

---

## 4. Method D — Topic-granularity sensitivity

**What it checks.** That a conclusion is not an artifact of the topic resolution. The topic model
has 149 **leaf** topics (`topic_id` in `topic_hierarchy.parquet`) and 96 **mid-level** topics
(`mid_level_id`). A conclusion is called **robust** only if it holds at **both grains**.

**Definition.** Every headline cascade/role/origin conclusion is computed twice:
- **leaf** grain — keyed on `topic_id` (149 topics).
- **mid** grain — keyed on `mid_level_id` (96 topics), obtained by joining
  `archetypes.topic_id → topic_hierarchy.topic_id → mid_level_id` and re-aggregating
  appearances to the mid level (a (mid_topic, journal) appears in the earliest year any of its
  leaf children reaches `min_papers` in that journal).

The same statistic (e.g. journal seeding-score ranking, S1 spread, role ranking) is computed at
both grains; the comparison statistic is the **Spearman rank-correlation between the
leaf-grain and mid-grain journal rankings**.

**Rubric.**
- **PASS** — the qualitative conclusion (e.g. "journal X is a source; Y is a terminal") holds at
  *both* grains **and** the leaf-vs-mid journal ranking Spearman ρ ≥ 0.5.
- **PARTIAL** — holds at one grain but is attenuated/borderline at the other (0.2 ≤ ρ < 0.5).
- **FAIL** — the conclusion reverses or vanishes at one grain (ρ < 0.2) → grain-dependent
  artifact; report it as not robust.

---

## 5. How the methods feed Gate G6

Gate G6 is the **spend gate** (a human decision by Samer, like G1–G5). It replaces V1's
"≥2-of-3 findings" bar with a cartography-appropriate bar. The four methods feed it as follows.

**G6 PASS requires ALL of:**
1. **Cascade survives null + jackknife** — the cascade earns **PASS** on Method B (null/
   permutation, S1 under Null-1) **and** **PASS** on Method C (drop-one-journal jackknife). A
   PARTIAL on either is a PARTIAL for the gate, not a pass.
2. **Roles stable** — journal source/bridge/terminal role scores (V2-S05) earn **PASS** on the
   Method C `role_rank_correlation` rubric (roles survive the jackknife) and are interpretable to
   a domain reader.
3. **≥1 robust novelty-origin finding** — at least one novelty-by-sector/geography/funding finding
   (V2-S06) holds at **both granularities** (Method D PASS) with its coverage limits stated.
4. **Map coherent** — the proof-of-concept map v0 (V2-S07) renders cascades, roles, and origins
   coherently to a domain reader on the current corpus.

**G6 FAIL =** the cascade is a panel artifact (Method B or C FAIL) or too unstable to trust, **or**
no novelty-origin finding is robust, **or** the map is incoherent. → **Do not spend.** Reconsider
corpus design or method before scaling. (The V1 paper still ships regardless — charter §6.1.)

**Method A (held-out years)** is a *supporting* line of evidence reported alongside the gate
inputs: a PASS strengthens the case that the cascade is a genuine temporal regularity, but a
PARTIAL/FAIL on A alone does not sink the gate if B, C, and D pass — it is reported as a
qualification, because the held-out window is short (7 years) and not every topic is scorable in
both windows. The gate-critical, make-or-break checks are **B (null) and C (jackknife)**.

**Reporting commitment.** Every headline cascade/role/origin claim in V2-S03–S07 is published
with its Method B p-value + effect size, its Method C stability metric, and its Method D
both-grains result attached. A claim without these three is not a claim — it is a hypothesis for
the expanded corpus. The panel-conditional caveat (charter §5) is attached to every origin claim.

---

## 6. Reproducibility

All four methods run on the current corpus at $0. Every output artifact (validation tables,
permutation-null distributions, jackknife runs) gets a `scifield.repro.record_run` sidecar JSON
(git SHA, config hash, input hashes, software versions). The permutation null is reproducible
from `seed = 20260609` and `n_perm = 1000`; the jackknife is deterministic (10 fixed leave-one-out
runs); the held-out split (`≤2018` / `≥2019`) and the two granularities (leaf 149 / mid 96) are
fixed constants (§0). Anyone re-running V2-S04/S05 with these locked values must reproduce the
verdicts exactly.
