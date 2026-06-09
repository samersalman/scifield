# Cascade Validation — Verdict (V2-S04)

**Status:** EXECUTED 2026-06-09 (the make-or-break step; primary input to Gate G6)
**Author:** Samer Salman (executed by subagent S04)
**Protocol:** [`validation_protocol.md`](validation_protocol.md) (LOCKED before this compute)
**Code:** `src/scifield/cartography/cascade_validation.py` (pure) · `V2/scripts/run_cascade_validation.py` (I/O)
**Tables:** `V2/data/cascade/{null_results,jackknife_results,granularity_results}.parquet` (+ `.run.json`)
**Notebook (figures):** `V2/notebooks/v2_01b_cascade_validation.ipynb`

> **One-line verdict:** the inter-journal cascade is **REAL** on the gate-critical
> checks (seeding-spread null + origin jackknife + granularity), with a documented
> **PARTIAL** qualification on the secondary lead-lag-asymmetry statistic and on the
> short held-out window. Crucially, it **survives the 1995-censoring sensitivity** —
> the structure is not an artifact of the year-one ties.

---

## 0. What was tested, and the locked constants

The four rigor methods of the locked protocol were run against the V2-S03 cascade
outputs (the shipped flow tables `flow_leaf.parquet` 149 leaf topics / `flow_mid.parquet`
96 mid topics, 10 canonical journals). All constants are the protocol lock, used exactly:

| Constant | Value |
|---|---|
| Permutations | **1000** (`n_perm`, never capped — full run is ~80 s) |
| RNG seed | **20260609** (replicate `i` uses `seed + i`) |
| p-value | add-one upper-tail `(1 + #{null ≥ obs}) / (1 + n_perm)` |
| Null-1 (primary) | journal-label shuffle **within topic** |
| Null-2 (secondary) | first-appearance-year shuffle within topic |
| S1 (primary stat) | seeding-score spread (population std across 10 journals) |
| S2 (secondary stat) | net directed lead-lag asymmetry (Σ pairs `|w(A→B) − w(B→A)|`) |
| Granularity | leaf (149 `topic_id`) vs mid (96 `mid_level_id`), Spearman ρ bar 0.5 |
| Held-out (supporting) | train 1995–2018 / validate 2019–2025 (Method A) |

### The 1995 left-censoring threat (handled, not ignored)

The corpus starts 1995, so **139/149 leaf topics "originate" in 1995** and the origin
tie-rate is **79% (leaf) / 82% (mid)** (S03 caveat #1). A 1995 origin is read as *"present
at panel start"*, never *"born in 1995."* Two defenses are built in:

1. **Null-1 preserves each topic's year footprint** (it shuffles only *which* journal
   gets which year, within the topic), so the all-tied-at-1995 topics contribute the
   *same* (zero) ordering information to both the observed statistic and its null — they
   cannot inflate significance.
2. **Every verdict is recomputed on the resolvable subset** — the topics with ≥ 2
   distinct first-appearance years (138/149 leaf, 86/96 mid), i.e. those that actually
   carry an ordering. If the structure lived only in the 1995 ties, it would vanish on
   this subset. **It does not.**

---

## 1. Method B — Null / permutation models (gate-critical)

Per-cell verdict rubric (protocol §2): **PASS** = `p < 0.01` and `z ≥ 2`; **PARTIAL** =
`p < 0.05` but `z < 2`; **FAIL** = `p ≥ 0.05`. The gate-critical statistic is **S1 under
Null-1**; S2 and Null-2 corroborate.

### S1 — seeding-score spread (PRIMARY) — PASS everywhere, including resolvable

| grain | subset | observed | null mean | z | p | percentile | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| leaf | all | 0.0907 | 0.0600 | **+3.27** | **0.0020** | 0.999 | **PASS** |
| leaf | resolvable | 0.0907 | 0.0589 | **+3.22** | **0.0020** | 0.999 | **PASS** |
| mid | all | 0.1097 | 0.0653 | **+4.04** | **0.0010** | 1.000 | **PASS** |
| mid | resolvable | 0.1047 | 0.0642 | **+3.67** | **0.0010** | 1.000 | **PASS** |

The observed per-journal seeding scores spread **~50% wider** than the within-topic
shuffle null produces (0.091 vs 0.060 leaf), z > 3 at both grains. **The resolvable-subset
z (3.2 leaf / 3.7 mid) is essentially unchanged from the all-topics z** — the lead/follow
structure is carried by the genuinely-ordered topics, *not* by the 1995 ties. This is the
single most important result in the prove phase: **the cascade is not a left-censoring
artifact.**

Null-2 (first-year shuffle) corroborates even more strongly for S1 (z 4.7–5.2, p ≤ 0.001).

### S2 — lead-lag asymmetry (SECONDARY) — does NOT corroborate under Null-1

| grain | subset | observed | null mean | z | p | verdict |
|---|---|---:|---:|---:|---:|---|
| leaf | all | 6.798 | 6.498 | +0.41 | 0.357 | **FAIL** |
| leaf | resolvable | 6.972 | 6.535 | +0.60 | 0.277 | **FAIL** |
| mid | all | 7.993 | 6.289 | +2.16 | 0.025 | PARTIAL |
| mid | resolvable | 8.219 | 6.479 | +2.11 | 0.023 | PARTIAL |
| leaf/mid | all/resolvable | — | — | +3.3–4.0 | ≤0.004 | **PASS (Null-2)** |

**Why S2 fails under Null-1 (this is real, not a bug).** S2 sums the *magnitude* of
pairwise asymmetry. The within-topic journal-label shuffle (Null-1) preserves the *set*
of first-appearance years inside each topic and merely reassigns which journal holds each
year. That largely conserves the total magnitude of pairwise lead/lag gaps — it destroys
*who* leads *whom* (which S1 detects) but not *how much* total asymmetry exists (which S2
measures). So S2/Null-1 sits just above its null (z 0.4 leaf). Under Null-2 (which permutes
the *years themselves*), S2 passes strongly (z 3.3–4.0). At the coarser mid grain S2/Null-1
is borderline-PARTIAL (z ~2.1). **Read:** S2 is the wrong instrument for the Null-1 contrast;
S1 is the protocol's designated primary, and it is unambiguous.

### Method-B verdict per cell-block

Combining the four cells per the protocol rubric ("S1 passes under Null-1 but S2 does not
corroborate" → **PARTIAL**): every grain × subset block reads **Method-B = PARTIAL**,
driven entirely by S2/Null-1. **The gate-critical primary (S1/Null-1) is PASS at all four.**

---

## 2. Method C — Drop-one-journal jackknife of ORIGIN attributions

Rubric (protocol §3): **PASS** = `attribution_flip_rate ≤ 0.20`; PARTIAL ≤ 0.40; FAIL > 0.40.
Forced reattributions (a topic whose origin journal *is* the dropped journal must
reattribute) are excluded from the flip numerator and counted separately.

| grain | subset | flip_rate | forced_reattribution_count | genuine flips | verdict |
|---|---|---:|---:|---:|---|
| leaf | all | **0.0000** | 149 | 0 / 1341 eligible | **PASS** |
| leaf | resolvable | **0.0000** | 138 | 0 / 1242 | **PASS** |
| mid | all | **0.0000** | 96 | 0 / 864 | **PASS** |
| mid | resolvable | **0.0000** | 86 | 0 / 774 | **PASS** |

Origin attributions are **perfectly stable** to dropping any single journal — even on the
resolvable subset. The `forced_reattribution_count` equals the topic count at each grain
(exactly one forced reattribution per topic, in the single run that drops that topic's
origin journal). The genuine (non-forced) flip rate is **0.0000** everywhere.

**Honest caveat (do not over-claim):** a 0.0 flip rate is partly *structural* — origin =
`argmin(first_year)` with a deterministic alphabetical tie-break, so dropping a *non-origin*
journal can never make a later journal become the earliest. The jackknife therefore confirms
origins are not fragile to panel composition, but it is a weaker test than the null. The
**resolvable-subset** flip rate (also 0.0) shows the stability is not merely the 1995 ties.
**Scope split:** this task (S04) owns the jackknife of **origin attributions**; **V2-S05
owns the jackknife of role scores (`role_rank_correlation`)** — the G6 synthesis must
combine *both* halves of Method C.

---

## 3. Method D — Topic-granularity sensitivity

Rubric (protocol §4): **PASS** = leaf-vs-mid journal seeding-rank Spearman ρ ≥ 0.5.

| comparison | Spearman ρ | bar | verdict |
|---|---:|---:|---|
| leaf vs mid per-journal seeding ranking | **0.8545** | 0.5 | **PASS** |

The per-journal lead/follow ranking is robust across topic resolution (ρ = 0.85, well above
the 0.5 bar; corroborates S03's net-lead ρ = 0.96 from the cascade notebook). The conclusion
"some journals systematically seed, others follow" is **not** a topic-granularity artifact.

---

## 4. Method A — Held-out years (SUPPORTING ONLY)

Rubric (protocol §1): PASS ρ ≥ 0.5; PARTIAL 0.2 ≤ ρ < 0.5; FAIL < 0.2. Method A is
explicitly *supporting* — a PARTIAL/FAIL here does not sink the gate if B/C/D hold (the
holdout window is only 7 years).

| grain | train→holdout seeding-rank ρ | topics excluded (origin > 2018) | verdict |
|---|---:|---:|---|
| leaf | 0.297 | 1 | **PARTIAL** |
| mid | 0.091 | 0 | **FAIL** |

The journal seeding ranking only **partially** reproduces out of sample in time, and at the
mid grain it does not (ρ 0.09). **This is expected and acceptable:** (i) the window is short
(7 years) and the panel is left-censored, so the 1995–2018 ranking is dominated by the
year-one structure while 2019–2025 re-ranks on recent first-appearances only; (ii) Method A
is supporting evidence, not gate-critical. It is reported as a **qualification**, not a
failure of the cascade — but it does mean any *forward-looking* (trajectory) claim must be
made cautiously. This weakness should be revisited with the expanded corpus (longer history,
less censoring), where Method A becomes more informative.

---

## 5. Overall verdict: REAL (with documented PARTIAL qualification)

**Synthesis of the gate-critical checks:**

- **Method B (null), S1/Null-1 — the make-or-break statistic — PASS at both grains AND on
  the resolvable subset** (z 3.2–4.0, p ≤ 0.002). Corroborated by S1/Null-2 (z 4.7–5.2).
- **Method C (origin jackknife) — PASS everywhere** (flip rate 0.0, both grains, both subsets).
- **Method D (granularity) — PASS** (ρ 0.85).

→ **The inter-journal cascade structure is REAL, not a panel/temporal artifact.** Some
journals systematically lead and others follow, the effect exceeds the within-topic shuffle
null, origin attributions are stable to panel composition, the ranking is grain-robust, and —
the decisive 1995 test — **the structure survives on the resolvable subset.**

**The PARTIAL qualifications (report alongside every claim):**

1. The **secondary** S2 (lead-lag-asymmetry) statistic does **not** corroborate under Null-1
   (it is conserved by the within-topic label shuffle by construction); it passes under
   Null-2. So the formal Method-B *block* verdict is PARTIAL even though the primary S1/Null-1
   is a clean PASS.
2. **Method A (held-out) is PARTIAL/FAIL** — the ranking reproduces out of sample only
   partially (leaf ρ 0.30, mid ρ 0.09); supporting-only, attributed to the 7-year window +
   left-censoring.
3. The jackknife's 0.0 flip rate is partly structural (deterministic `argmin` origin), so it
   is a confirmation of non-fragility rather than a strong independent test.

**Honest framing:** this is an exploratory map, and the gate-critical evidence (B-primary +
C + D) is genuinely positive while the secondary lines (S2/Null-1, Method A) are weaker. The
correct call is **REAL on the cascade, PARTIAL on its strength/forward-stability** — not an
artifact, not an unqualified strong finding.

---

## 6. Input to Gate G6

The V2-S04 cascade-validation half of the Gate-G6 spend decision is **PASS-with-qualification**:
the cascade earns a clean PASS on the protocol's gate-critical primary (S1 seeding-spread under
Null-1, p ≤ 0.002 / z ≥ 3.2 at both grains, **and surviving the resolvable-subset 1995-censoring
sensitivity**) and a PASS on the drop-one-journal origin jackknife (flip rate 0.0) and on
topic-granularity robustness (ρ 0.85); the structure is therefore **not a panel or left-censoring
artifact**. The qualification G6 must weigh is that the *secondary* lead-lag-asymmetry statistic
(S2) does not corroborate under the primary null (it is conserved by the within-topic shuffle),
the held-out-years supporting check is only PARTIAL (7-year window, left-censored), and the
jackknife's perfect stability is partly structural. **G6 synthesis must combine this with V2-S05's
role-jackknife (`role_rank_correlation`) — the other half of Method C — and the ≥1-robust-novelty
finding (V2-S06) before the spend gate is read.** On the cascade alone, the recommendation is
**proceed**: the core cartographic claim (systematic inter-journal seeding) is real on the current
corpus, with its known left-censoring and short-history limits explicitly attached and best
resolved by the very corpus expansion G6 gates.
