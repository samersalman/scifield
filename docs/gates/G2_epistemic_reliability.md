# Gate G2 — Epistemic extraction reliability (label-free v2)

**Session:** V1-S09 (Epistemic validation v2 — label-free Gate G2)  **Date generated:** 2026-05-29  **Plan ref:** `plans/Session-Objectives-MAP.md` §V1-S09 (epistemic validation v2 + Gate G2 report)

This is the second of the human-decision gates in the SciField execution plan
and the **STOP gate before V1-S10 (novelty)**. The downstream dual-novelty (F2)
and forecasting (F3) framework rests on the F1 epistemic-cascade extraction
being *reliable* — i.e. that the per-paper study-design / control / sample-size /
statistical-claim / COI fields the LLM emits can be trusted at corpus scale.

**Gate G2 was redefined on 2026-05-29 (Samer).** The original G2 design hinged on
hand-labeled inter-rater κ against a human gold set. That criterion is **dropped**.
G2 v2 instead measures reliability through **three label-free lenses**:

- **C1 — cross-tool agreement:** DeepSeek `study_design` vs. an *independent*
  reference signal already in the corpus (the PubMed `Randomized Controlled Trial`
  structured publication-type flag).
- **C2 — model-vs-model agreement:** DeepSeek vs. Claude-Code on the same PMIDs.
- **C3 — internal-validity priors:** do the extracted field distributions match
  pre-registered domain expectations for a surgical-literature corpus?

Every number below is reproduced end-to-end by `notebooks/06_epistemic_validation.ipynb`.

---

## Pass criteria

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| C1 simple_agreement (DeepSeek vs PubMed RCT flag) | ≥ 0.85 | **0.9832** | **PASS** |
| C1 Cohen's κ | ≥ 0.70 | **0.8629** | **PASS** |
| C2 study_design exact_match (DeepSeek vs Claude-Code) | ≥ 0.80 | **0.8920** (κ 0.8560) | **PASS** |
| C2 has_control exact_match | ≥ 0.80 | **0.9568** (κ 0.9087, n=1,575) | **PASS** |
| C2 sample_size Spearman ρ | ≥ 0.75 | **0.9867** (n=1,522) | **PASS** |
| C3 (a) RCT fraction | in [2%, 15%] (exp ≈6.73%) | **0.0674** | **PASS** |
| C3 (b) statistical_claim_present fraction | in [45%, 85%] (exp ≈66.5%) | **0.6610** | **PASS** |
| C3 (c) coi_disclosed_in_abstract fraction | in (0%, 5%] (exp ≈0.05%) | **0.0005** | **PASS** |
| C3 (d) RCT ⇒ has_control | in [90%, 100%] (exp ≈99.05%) | **0.9904** | **PASS** |
| C3 (e) sample_size median | median in [10, 1000] (exp ≈105) | **103** | **PASS** |
| C3 (f) effect_direction 'na' fraction | in [20%, 55%] (exp ≈34.9%) | **0.3512** | **PASS** |

> C2 gates on all three paired-field thresholds from the maintained pass criteria
> (`study_design` exact_match ≥ 0.80, `has_control` exact_match ≥ 0.80,
> `sample_size` Spearman ρ ≥ 0.75) — all three clear comfortably. C3 gates on the
> count of failing priors: the gate **fails only if ≥ 2 of the 6 priors fail**.

## Provenance

Sidecar: `data/v1/epistemic_extracted.parquet.run.json`

| Field | Value |
|---|---|
| config_hash | `bfd393f24169b199015a74ccdd72bac92b737e8d7de78337f512ede2f7fa3f52` |
| git_sha | `e7ea1aeea58ee2887a2c38a281ce32f810dbc5ca` |
| git_dirty | `true` |
| timestamp | `2026-05-29T20:34:00Z` |

Notebook: `notebooks/06_epistemic_validation.ipynb` (executes end-to-end; reproduces every number in this report).

---

## C1 — Cross-tool RCT agreement

DeepSeek `study_design == 'RCT'` vs. the PubMed `Randomized Controlled Trial`
structured publication-type flag as an independent reference. This is the
strongest label-free lens because the reference is a human-curated NLM
publication type that exists entirely outside the LLM pipeline.

**N = 89,230** — full DeepSeek coverage, deduped to one row per PMID.

| Metric | Threshold | Result | Status |
|---|---|---|---|
| simple_agreement | ≥ 0.85 | **0.9832** | **PASS** |
| Cohen's κ | ≥ 0.70 | **0.8629** | **PASS** |
| sensitivity | report | 0.8991 | — |
| precision | report | 0.8462 | — |

Confusion matrix (DeepSeek RCT call vs PubMed RCT flag):

| | PubMed RCT = 1 | PubMed RCT = 0 |
|---|---:|---:|
| **DeepSeek RCT = 1** | TP = 5,089 | FP = 925 |
| **DeepSeek RCT = 0** | FN = 571 | TN = 82,645 |

Figure: `docs/figures/G2_C1_rct_confusion.png`.

> **Consistency with the pre-rerun trajectory.** An earlier pass reported
> N=87,268, simple_agreement 98.3%, κ≈0.865. The current larger N=89,230
> reflects full DeepSeek coverage after the C2 rerun; agreement and κ are
> materially unchanged, confirming the C2 rerun did not perturb C1.

**C1 VERDICT: PASS.**

---

## C2 — Model vs. model

DeepSeek vs. Claude-Code on the **1,981 paired PMIDs**. All three paired-field
criteria gate (`study_design` and `has_control` exact_match ≥ 0.80, `sample_size`
Spearman ρ ≥ 0.75); all three pass.

**N = 1,981 paired PMIDs.**

| Field | n | Metric | Threshold | Result | Status |
|---|---:|---|---|---|---|
| study_design | 1,981 | exact_match | ≥ 0.80 | **0.8920** (κ 0.8560) | **PASS** |
| has_control | 1,575 | exact_match | ≥ 0.80 | **0.9568** (κ 0.9087) | **PASS** |
| sample_size | 1,522 | Spearman ρ | ≥ 0.75 | **0.9867** | **PASS** |

Figure: `docs/figures/G2_C2_model_agreement.png`.

> **What the paired set actually is.** This is the *full DeepSeek rerun* of all
> 1,981 Claude-Code PMIDs. The two V1-S08 model sets were **disjoint** — true
> cross-model overlap was 0. The frequently-cited "19 overlap" were
> DeepSeek-*internal* duplicate rows, not cross-model pairs. The V1-S09 C2 rerun
> deliberately re-extracted every Claude-Code PMID under DeepSeek so each of the
> 1,981 PMIDs now carries both `model_id`s. The comparison is therefore
> well-powered at N=1,981 rather than resting on an accidental handful.

**C2 VERDICT: PASS.**

---

## C3 — Internal-validity priors

Full DeepSeek set, deduped. Each prior is a pre-registered domain expectation for
a surgical-literature corpus; the gate fails only if **≥ 2 of 6** priors fall
outside their band.

| key | label | value | threshold | passed |
|---|---|---:|---|:--:|
| a_rct_fraction | RCT fraction (study_design=='RCT') | 0.0674 | in [2%, 15%] (exp ≈6.73%) | PASS |
| b_statistical_claim_fraction | statistical_claim_present fraction | 0.6610 | in [45%, 85%] (exp ≈66.5%) | PASS |
| c_coi_fraction | coi_disclosed_in_abstract fraction | 0.0005 | in (0%, 5%] (exp ≈0.05%) | PASS |
| d_rct_implies_control | RCT ⇒ has_control | 0.9904 | in [90%, 100%] (exp ≈99.05%) | PASS |
| e_sample_size_median | sample_size median (max adjudicated) | 103 | median in [10, 1000] (exp ≈105) | PASS |
| f_effect_direction_na_fraction | effect_direction 'na' fraction | 0.3512 | in [20%, 55%] (exp ≈34.9%) | PASS |

Figure: `docs/figures/G2_C3_priors.png`.

### C3(e) outlier adjudication

The (e) prior keys on the **median** sample_size (103, which is sane) — not the
max. The max sample_size in the corpus is **68,205,695**, and there are **23
PMIDs with sample_size > 10M**. Manual inspection confirms these are **legitimate
national-database / registry / population-level studies — not extraction
errors**. Examples:

- PMID 31115918 — "Age of patients undergoing surgery" (NHS England) — 68,205,695
- PMID 25876008 — "Seasonal Variation in Emergency General Surgery" (National Inpatient Sample) — 63,911,033
- PMID 40801367 — "U.S. Surgical Practice: 23-Year Trends in Medicare Procedures" (Medicare) — 53,514,927
- PMID 21576609 — "Mortality rate after nonelective hospital admission" (Nationwide Inpatient Sample) — 29,991,621

Full >10M PMID set (23): 25876008, 27234633, 28267693, 32611513, 34955287,
36017938, 21576609, 26466334, 28657950, 30048311, 33404647, 34757424, 38811327,
40592060, 17015592, 23813242, 24867450, 33263743, 36727966, 27192350, 31115918,
40801367, 30993676.

**Adjudicated verdict: (e) PASS.** These are real cohort/database sizes, so a
max-bound failure would be spurious; the median (103) is the correct statistic
for this prior and sits comfortably inside [10, 1000].

**C3 VERDICT: PASS (0 of 6 priors failing; gate fails only if ≥ 2 fail).**

---

## Recommendation

- [ ] PASS
- [ ] QUALIFIED PASS
- [ ] FAIL

The notebook computes an **overall G2 = PASS**: C1 ✓ (simple_agreement 0.9832,
κ 0.8629 against an independent PubMed reference at N=89,230), C2 ✓ (study_design
exact_match 0.8920 / κ 0.8560 on the full N=1,981 paired set, with has_control
and sample_size ρ both very high), and C3 ✓ (0 of 6 internal-validity priors
failing, including the adjudicated (e) outlier case). All three label-free lenses
clear their thresholds, so the F1 epistemic extraction is reliable enough to
carry the downstream F2/F3 framework. **This is the STOP gate before V1-S10
(novelty)** — V1-S10 is unblocked only on a PROCEED decision here. The
proceed / qualify / fail decision is **Samer's**: the checkboxes above are left
unchecked for his sign-off.

Signed: _Samer Salman_  Date: ____________
