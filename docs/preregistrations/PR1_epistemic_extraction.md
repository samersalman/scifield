# Pre-registration #1: Epistemic-quality extraction from PubMed abstracts

**Authors:** Samer G. Salman, Baylor College of Medicine (ORCID: [0009-0007-9897-4071](https://orcid.org/0009-0007-9897-4071))
**Date drafted:** 2026-05-22
**OSF DOI / URL:** _(to be filled after submission — see §11 below)_
**Project commit SHA at submission:** _(to be filled at submission time)_
**Codebase:** https://github.com/samersalman/scifield  (Zenodo DOI minted at release v0.1.0 — paste here once available)
**Plan reference:** `plans/2026-05-22-v1-s07-epistemic-prep.md`
**Master plan reference:** `plan/scifield_plan.md` §5 Phase 3

---

## 1. Background and rationale

SciField is a multi-axis framework for monitoring the health of the orthopedic-surgery research field. Across phases, it layers (a) a deduplicated 10-journal PubMed corpus (V1-S03/S04), (b) a topic landscape over abstract embeddings (V1-S05/S06), (c) an LLM-assisted epistemic-quality layer over abstracts (Phase 3), (d) a novelty signal, and (e) a forecasting layer, with downstream analyses in Phase 6. Phase 3 introduces a per-abstract structured judgement covering six epistemic features — `study_design`, `sample_size`, `has_control`, `effect_direction`, `statistical_claim_present`, and `coi_disclosed_in_abstract` — that subsequent analyses will combine with the topic landscape to track methodological practice over time and across subfields.

This pre-registration scopes the hand-labeling protocol and the LLM-vs-human extraction agreement study that gates Phase 3 before any downstream use. The corpus operated on is the 10-journal orthopedic-and-surgery subset of PubMed spanning 1995–2026, comprising the abstract-bearing subset of `papers_distinct` (~99,938 papers after V1-S05 carryover dedup).[^1] The topic landscape referenced for stratification spot-checks is the V1-S06 deliverable (149 leaf topics, G1 PROCEED override clearance documented in `docs/gates/G1_topic_interpretability.md`).

[^1]: The plan document `plans/2026-05-22-v1-s07-epistemic-prep.md` cites 89,244 abstract-bearing papers; that figure was an earlier estimate from before V1-S05 carryover dedup completed. The operative number used for sampling and proportional allocation in this pre-registration is **99,938**.

---

## 2. Hypotheses

- **H1 (extractability).** Six epistemic-quality features (`study_design`, `sample_size`, `has_control`, `effect_direction`, `statistical_claim_present`, `coi_disclosed_in_abstract`) are extractable from PubMed abstracts at inter-rater Cohen's κ at or above the pre-specified per-field thresholds enumerated in §8.
- **H2 (LLM ≈ human).** Claude (via Claude Code, prompt v0.1) achieves field-by-field κ_LLM-vs-arbitrated within 10 percentage points of the field-by-field κ_inter-rater on every one of the six fields.

These hypotheses are pre-registered prior to running the 500-abstract sprint. Any deviation from the pre-registered analysis plan will be reported transparently in the registered report.

---

## 3. Operational definitions

The six fields below are the canonical extraction targets. The Pydantic schema source enumerating the allowed values and nullability is reproduced verbatim in Appendix B. The operational rules below are what raters (human or LLM) apply when populating each field.

### 3.1 study_design

Closed enum: `RCT`, `cohort`, `case_control`, `case_series`, `review`, `other`.

- **RCT** — Abstract explicitly states "randomized" / "randomised" AND describes at least two parallel (or cross-over) arms.
- **cohort** — Prospective or retrospective longitudinal observational follow-up of a defined group, NOT randomized.
- **case_control** — Explicit cases-vs-controls retrospective design, controls selected on outcome status.
- **case_series** — Fewer than 20 patients with no comparator, OR an explicit "case series" / "case report" label.
- **review** — Narrative review, scoping review, systematic review, or meta-analysis.
- **other** — Technique reports, biomechanical / bench studies, surveys, editorials, modeling studies, qualitative research, and anything not covered above.

### 3.2 sample_size

Integer ≥1 or JSON `null`. If the abstract states a top-line N, use it. For meta-analyses, prefer **N studies** (not pooled patients). If the abstract reports separate arms (e.g., 50 vs 50), report the **total** (100). `null` means *the abstract did not state a sample size*, not *we don't know*. Zero and negative values are never valid and are rejected at schema validation time.

### 3.3 has_control

Boolean or `null`. `True` if the design has an explicit comparator arm — an RCT control arm, a cohort comparator group, a case-control control group, an alternative-treatment arm, etc. `False` for single-arm case series. `null` for reviews, editorials, methods papers, modeling studies, and any other design where the question is ill-posed.

### 3.4 effect_direction

Closed enum: `positive`, `null`, `negative`, `mixed`, `na`.

- **positive** — Intervention/exposure beat the comparator on the primary outcome as reported in the abstract.
- **null** — Abstract states no significant difference / findings did not reach significance / no effect.
- **negative** — Intervention/exposure was worse than the comparator on the primary outcome.
- **mixed** — Multiple primary endpoints disagree, or significant effects in conflicting directions.
- **na** — Concept does not apply (review, technique report, case series with no effect claim, descriptive epidemiology with no hypothesis).

Note: the literal string `"na"` is used for non-applicability; JSON `null` is reserved as a soft fallback for malformed model output and is *not* the labeler's intended value.

### 3.5 statistical_claim_present

Boolean. `True` if the abstract contains any of: a p-value, a confidence interval, a hazard ratio, an odds ratio, a risk ratio, a mean ± SD reported against a comparator, a regression coefficient, an effect size, a power calculation, or the words "significant" / "not significant" in a statistical sense. `False` if the abstract is purely descriptive with no statistical language.

### 3.6 coi_disclosed_in_abstract

Boolean. `True` only if a conflict-of-interest, funding, or industry-sponsorship disclosure appears in the **abstract text itself**. `False` otherwise. **Conflict-of-interest information present in the full-text body but absent from the abstract is labeled `False` per this operational definition.** This is an intentionally conservative measure of *abstract-level disclosure*; most journals print COI in the body, not the abstract, and the design choice here is to measure what the abstract reader sees.

---

## 4. Sampling plan

- **Population.** The `papers_distinct` view (V1-S05 carryover-deduped) filtered to `abstract IS NOT NULL AND length(abstract) > 50`. This is approximately 99,938 abstract-bearing papers across 10 orthopedic-and-surgery journals over the 1995–2026 span. The 10 journal slugs are: `ann_surg`, `arthroscopy`, `br_j_surg`, `clin_orthop_relat_res`, `j_am_coll_surg`, `j_arthroplasty`, `j_bone_joint_surg_am`, `jama_surg`, `spine`, `surgery`.
- **Stratification.** Journal × era. **40 cells** = 10 journals × 4 era bins (`pre2000`, `2000-2009`, `2010-2019`, `2020+`). The floor cell size (smallest non-empty cell) is **624 abstracts** (in `j_am_coll_surg` × `pre2000`), well above the per-cell allocation that any cell will receive at n=500.
- **Allocation rule.** Proportional allocation across the 40 cells, with **largest-remainder (Hare) rounding** to hit exactly n=500 paired-rater rows.
- **Random seed.** `20260522` (UTC date of pre-registration drafting). Within-cell selection is uniform random under this seed.
- **Topic-coverage check.** At least **80 of the 149 leaf topics** from the V1-S06 topic landscape must appear in the n=500 sample. This is enforced as an assertion inside `scifield.epistemic.sampling.stratified_sample` and the run aborts if violated, prompting a seed / stratification investigation rather than a silent skewed sample.
- **Output artifact.** `data/v1/handlabel_sample.parquet`, written by the CLI command `scifield epistemic sample`. A sidecar `.run.json` captures git SHA + the full sampling config hash, enabling exact replay.
- **Footnote on cell count.** An earlier scoping document (plan §B as drafted) gave 41 cells and a floor of 689 abstracts. Both numbers were re-derived after V1-S05 carryover dedup completed; the operative numbers used here are **40 cells, floor 624**, which match what the implementation produces on the current corpus snapshot.

---

## 5. Hand-labeling protocol

- **Raters.** Two — Samer Salman (PI / corresponding author) and one co-rater. The two raters label the same 500-abstract sample independently. Neither rater sees the other's labels until both have finished and submitted their workbook.
- **Tool.** Excel workbook (`data/v1/labels_<rater>.xlsx`) with `openpyxl`-applied `DataValidation` enum dropdowns on each rater-fill column, generated by `scifield epistemic export-labels --rater <name>`. Excel was chosen over a custom Streamlit tool so a non-coder co-rater can use familiar tooling; the dropdowns enforce the closed enums at entry time, and Pydantic re-validates everything at import time.
- **Schema enforcement.** On import (`scifield epistemic import-labels`), each row is validated through `EpistemicLabel` (see Appendix B). Bad enum values, out-of-range `sample_size`, or extra columns trigger a per-row error report; the import is otherwise idempotent (re-importing the same `(pmid, rater)` overwrites the prior row).
- **Arbitration.** After both raters finish, every disagreement is reviewed in a single arbitration meeting between the two raters. Arbitrated decisions are written to `data/v1/epistemic_handlabel_final.parquet` (V1-S08 scope). The arbitration log captures which rater was overridden on which field for every disagreement.
- **Schema version stamp.** `LABEL_SCHEMA_VERSION = "v0.1"` (see Appendix B) is recorded on every labeled row so downstream consumers can detect any future schema migration.

---

## 6. LLM extraction protocol

- **Model.** Claude, accessed via the Claude Code CLI (`claude --print`). The specific model id resolved by the local Claude Code install is captured per-call via `subprocess.run` stdout and stored verbatim in the `raw_response` field of every output row (and in the `model_id` field via the surrounding `EpistemicExtraction` wrapper). Claude Code was chosen for the V1-S07 pilot to avoid Anthropic API spend during pre-registration scoping; the `extract_one()` interface is model-agnostic, so V1-S08 can flip transports (e.g., to the Anthropic Batch API) by swapping `cfg.claude_cmd` for an HTTP transport without changing any other code.
- **Prompt version.** `v0.1` (full text in Appendix A). The prompt may iterate during the 50-abstract pilot to `v0.1.1` and/or `v0.1.2` based on observed parse failures or systematic miscategorization; the iteration log lives in `docs/phases/epistemic.md`. **The final prompt version that ships into the 500-abstract batch is stamped on every row of `data/v1/epistemic_pilot.parquet` (and on every V1-S08 batch row) so post-hoc analyses can stratify by prompt version.**
- **Output schema.** Pydantic-validated JSON object with exactly the six fields enumerated in Appendix B. On JSON parse failure or Pydantic validation failure, `extract_one()` retries exactly once with a stricter "respond with valid JSON only — no prose" suffix appended. Persistent failures after the single retry are recorded as separate rows in `data/v1/epistemic_pilot_failed.parquet` with the raw response retained for forensic review.
- **Determinism.** The pilot sample is sorted by `pmid` before iteration to make the n=50 pilot subset deterministic given the n=500 sample. Each pilot run writes a sidecar `.run.json` with git SHA + resolved model id + prompt version, enabling exact replay.

---

## 7. Primary analysis

- **Inter-rater agreement (human-vs-human).** Cohen's κ per field across the 500 paired labels.
- **LLM-vs-arbitrated agreement.** Cohen's κ per field between the Claude extraction and the arbitrated 2-rater human label set.
- **Secondary statistic.** Krippendorff's α at the nominal level for the four categorical fields (`study_design`, `effect_direction`, `statistical_claim_present`, `coi_disclosed_in_abstract`, with the latter two treated as 2-level categorical). Ordinal-level α is **not** invoked because `effect_direction` is conceptually unordered (`mixed` does not lie between `positive` and `negative`).
- **Confusion matrices.** One per field, rendered in the registered report.
- **Sample-size justification.** With n=500 paired observations, the 95% Wilson CI half-width on a κ of 0.7 is approximately ±0.04 for the binary fields and approximately ±0.05 for 5-way categorical fields — sufficient to distinguish κ=0.7 from κ=0.6 at α=0.05, power=0.8.

---

## 8. Pre-registered pass/fail criteria

Per-field κ thresholds for inter-rater agreement:

| Field | Threshold |
|---|---|
| `study_design` | κ ≥ 0.7 |
| `has_control` | κ ≥ 0.8 |
| `effect_direction` | κ ≥ 0.6 |
| `statistical_claim_present` | κ ≥ 0.7 |
| `coi_disclosed_in_abstract` | κ ≥ 0.7 |

For `sample_size`, Cohen's κ is undefined (continuous integer-valued); we instead report **Lin's concordance correlation coefficient ≥ 0.85** as the analogue threshold.

**LLM-vs-human delta.** For every one of the six fields, `|κ_LLM-vs-arbitrated − κ_inter-rater| ≤ 0.10` (i.e., within 10 percentage points). For `sample_size`, the analogue delta is on Lin's CCC and is held to the same 10pp tolerance.

---

## 9. Pivot conditions

**If H1 fails** — any per-field threshold in §8 missed by ≥0.10:

- **Option A.** Fine-tune a small BERT-class classifier on the 500 human-labeled abstracts (insert new session V1-S09b) and re-evaluate against held-out test rows.
- **Option B.** Drop feature F1 (epistemic-quality extraction) from the manuscript and proceed with F2 (novelty) and F3 (forecasting) only.
- **Decision rule.** Choose Option A if at least one field has κ ≥ 0.5 in the failing setup (signal-bearing enough for a fine-tune to plausibly recover). Choose Option B if every failing field has κ < 0.5.

**If H2 fails** — LLM-vs-human delta exceeds 10pp on any field — the LLM extraction is reported as an **exploratory / secondary** feature in the manuscript rather than a primary feature, and downstream Phase 6 analyses weight epistemic features accordingly (or drop them and use only human-labeled subsets, where applicable).

---

## 10. Data and code availability

- **Code.** All source under `src/scifield/epistemic/`, tests under `tests/test_epistemic_*.py`. The commit SHA at OSF submission time will be pasted into the front matter of this document above (`Project commit SHA at submission`).
- **Data.** The corpus DuckDB (`data/v1/papers.duckdb`) is reproducible from the V1-S04 harvest pipeline plus the V1-S05 enrichment + dedup steps; the handlabel sample (`data/v1/handlabel_sample.parquet`) and its `.run.json` sidecar are committed at submission time so the exact 500-row sample is recoverable.
- **Raw rater workbooks.** The `.xlsx` files filled in by individual raters are **not** redistributed (they are PHI-free but contain unpublished judgment calls and arbitration drafts that pre-date the agreed final label set). The import-time long-form parquet (one row per `(pmid, rater, field, value)`) **is** shared after the arbitration meeting concludes.
- **License.** Repository's existing LICENSE (Apache-2.0 per the repository's `LICENSE` file).

---

## 11. OSF submission workflow

1. Claude drafts this markdown file at `docs/preregistrations/PR1_epistemic_extraction.md`.
2. The PI uploads it to the Open Science Framework as a registered preprint pre-registration.
3. OSF mints a DOI / shortlink for the registration.
4. The PI pastes the resulting DOI into the front matter above on the `OSF DOI / URL` line.
5. The PI commits the updated file with the DOI populated. The acceptance grep specified in the plan (a substring match against this file looking for the registration's host domain) then passes.
6. **V1-S08 (the 500-abstract hand-labeling sprint and the 200k-abstract batch run) must NOT begin until step 5's commit is in.**

---

## Appendix A — Full system prompt and few-shot examples (prompt version v0.1)

The following is the verbatim source of the system prompt and few-shot exemplars used by `scifield.epistemic.extract.extract_one`. The single source of truth lives in `src/scifield/epistemic/prompt.py` — this appendix is a snapshot for OSF registration purposes. The constant `PROMPT_VERSION = "v0.1"` is what is stamped onto every pilot and batch row.

### A.1 System prompt (`SYSTEM_PROMPT_V0_1`)

```text
You are an epistemic-quality extractor over PubMed abstracts. Your
job is to read a single biomedical abstract and emit a structured
judgement about six features of the study it describes. You are not
summarizing the abstract, not assessing whether the findings are
correct, and not making causal claims of your own — you are reporting
what the abstract itself says (or fails to say) about its own design
and statistical conduct.

For each abstract, extract exactly these six fields:

1. study_design — the kind of study reported. Closed enum, one of:
   RCT, cohort, case_control, case_series, review, other.
   Use "RCT" for randomized controlled trials (any randomization
   described). Use "cohort" for prospective or retrospective
   observational follow-up of a defined group. Use "case_control"
   for studies that compare cases to controls selected on outcome.
   Use "case_series" for descriptive reports of a series of
   patients without a comparator. Use "review" for narrative
   reviews, systematic reviews, and meta-analyses. Use "other" for
   anything that does not fit (methods papers, editorials,
   commentaries, basic-science / in vitro, modeling studies,
   qualitative research).

2. sample_size — the integer N of the primary analytic sample as
   reported in the abstract. If the abstract states a number,
   return it as a JSON integer >= 1. If the abstract does not state
   a sample size, return JSON null. Do NOT return 0; do NOT guess.

3. has_control — boolean. True if the study has a comparator group
   (placebo arm, untreated controls, matched controls, an
   alternative therapy arm, etc.). False if there is no comparator
   (e.g., single-arm case series). Return JSON null when the
   question is ill-posed for the design — most reviews, editorials,
   methods papers, modeling studies.

4. effect_direction — direction of the primary reported effect.
   Closed enum, one of: positive, null, negative, mixed, na.
   Use "positive" when the abstract states the intervention/exposure
   was beneficial or the hypothesized effect was found. Use "null"
   when the abstract states no significant effect / no difference /
   findings did not reach significance. Use "negative" when the
   abstract states the intervention was harmful or moved the
   outcome the wrong way. Use "mixed" when multiple primary
   endpoints disagree. Use the literal string "na" (NOT JSON null)
   when the concept does not apply — reviews, methods papers, case
   series with no effect claim, descriptive epidemiology with no
   hypothesis.

5. statistical_claim_present — boolean. True if the abstract makes
   any statistical claim — a p-value, a confidence interval, a
   hazard / odds / risk ratio, the words "significant" or "not
   significant", a regression coefficient, an effect size, a power
   calculation. False if the abstract is purely descriptive with no
   statistical language at all.

6. coi_disclosed_in_abstract — boolean. True only if a
   conflict-of-interest, funding, or sponsorship disclosure appears
   in the abstract text itself. False if no such disclosure is in
   the abstract (the default — most journals put COI in the
   full-text footer, not the abstract).

Output format — read this carefully:

Respond with VALID JSON ONLY. Exactly one JSON object. No markdown
code fence. No leading or trailing prose. No commentary. The object
must have exactly these six keys: study_design, sample_size,
has_control, effect_direction, statistical_claim_present,
coi_disclosed_in_abstract. Use JSON null (not the string "null",
except for the effect_direction enum value) for unreported
sample_size and not-applicable has_control. Use the string "na"
(NOT JSON null) for effect_direction when the concept does not
apply.
```

### A.2 Few-shot examples (`FEW_SHOT_EXAMPLES`)

The following is the verbatim Python literal value of `FEW_SHOT_EXAMPLES` as exported from `src/scifield/epistemic/prompt.py`. Six worked exemplars cover RCT (positive outcome), observational cohort, case series, systematic review, RCT (null outcome), and case-control.

```python
FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "abstract": (
            "Background: We conducted a randomized, double-blind, placebo-"
            "controlled trial of drug X in adults with moderate hypertension. "
            "Methods: 480 patients were randomized 1:1 to drug X or placebo "
            "for 12 weeks. The primary endpoint was change in systolic blood "
            "pressure. Results: Drug X reduced systolic blood pressure by 8.4 "
            "mmHg more than placebo (95% CI 6.1-10.7; p<0.001). "
            "Conclusions: Drug X significantly lowers blood pressure."
        ),
        "label": {
            "study_design": "RCT",
            "sample_size": 480,
            "has_control": True,
            "effect_direction": "positive",
            "statistical_claim_present": True,
            "coi_disclosed_in_abstract": False,
        },
    },
    {
        "abstract": (
            "Objective: To evaluate whether long-term coffee consumption is "
            "associated with incident type 2 diabetes. Design: Prospective "
            "cohort study with 14 years of follow-up. Setting: A community-"
            "based cohort of 12,750 adults aged 40-70 at baseline. Results: "
            "After adjustment for age, sex, BMI, smoking, and physical "
            "activity, each additional cup of coffee per day was associated "
            "with a hazard ratio of 0.93 (95% CI 0.89-0.97) for incident "
            "diabetes. Conclusion: Higher coffee consumption was associated "
            "with lower diabetes risk. Funded by NIH; the authors report no "
            "conflicts of interest."
        ),
        "label": {
            "study_design": "cohort",
            "sample_size": 12750,
            "has_control": False,
            "effect_direction": "positive",
            "statistical_claim_present": True,
            "coi_disclosed_in_abstract": True,
        },
    },
    {
        "abstract": (
            "We describe a series of 14 consecutive patients presenting to "
            "our tertiary referral center with a rare hepatic manifestation "
            "of systemic amyloidosis between 2015 and 2022. Imaging findings, "
            "biopsy results, and clinical courses are summarized. Median "
            "survival from diagnosis was 9 months. No formal statistical "
            "comparisons were performed given the small sample."
        ),
        "label": {
            "study_design": "case_series",
            "sample_size": 14,
            "has_control": False,
            "effect_direction": "na",
            "statistical_claim_present": False,
            "coi_disclosed_in_abstract": False,
        },
    },
    {
        "abstract": (
            "This systematic review and meta-analysis synthesizes evidence "
            "on the efficacy of cognitive behavioral therapy (CBT) for adult "
            "insomnia. We searched MEDLINE, Embase, and PsycINFO through "
            "March 2024, identifying 38 randomized trials. The methodology, "
            "study heterogeneity, and risk-of-bias profile of the literature "
            "are discussed. No pooled effect estimate is reported in this "
            "abstract; full quantitative synthesis appears in the main text."
        ),
        "label": {
            "study_design": "review",
            "sample_size": None,
            "has_control": None,
            "effect_direction": "na",
            "statistical_claim_present": False,
            "coi_disclosed_in_abstract": False,
        },
    },
    {
        "abstract": (
            "Background: A multicenter Phase III randomized trial evaluated "
            "compound Y versus standard of care in metastatic pancreatic "
            "cancer. Methods: 612 patients were randomized 1:1 and followed "
            "for overall survival. Results: Median overall survival was 8.1 "
            "months with compound Y vs 8.4 months with standard of care "
            "(hazard ratio 1.02, 95% CI 0.86-1.21, p=0.78). "
            "Conclusions: Compound Y did not improve overall survival "
            "compared to standard of care."
        ),
        "label": {
            "study_design": "RCT",
            "sample_size": 612,
            "has_control": True,
            "effect_direction": "null",
            "statistical_claim_present": True,
            "coi_disclosed_in_abstract": False,
        },
    },
    {
        "abstract": (
            "Objective: To compare prior antibiotic exposure between patients "
            "with Clostridioides difficile infection and matched hospital "
            "controls. Design: Case-control study. Setting: Two academic "
            "medical centers, 2018-2021. Participants: 312 cases and 624 "
            "controls matched on age, sex, and admission unit. Results: Any "
            "antibiotic use in the prior 30 days was strongly associated "
            "with C. difficile infection (adjusted OR 4.7, 95% CI 3.2-6.9). "
            "Conclusion: Recent antibiotic exposure remains the dominant "
            "modifiable risk factor."
        ),
        "label": {
            "study_design": "case_control",
            "sample_size": 312,
            "has_control": True,
            "effect_direction": "positive",
            "statistical_claim_present": True,
            "coi_disclosed_in_abstract": False,
        },
    },
]
```

---

## Appendix B — Pydantic schema source (v0.1)

The following is the verbatim source of `src/scifield/epistemic/schema.py`. This is the canonical schema referenced throughout this pre-registration (`EpistemicLabel`, `EpistemicExtraction`, `LABEL_SCHEMA_VERSION`). Validation against this schema is performed both at Excel-import time (human labels) and at LLM-extraction time (machine labels).

```python
"""Pydantic v2 schema for V1-S07 epistemic-quality extraction.

Defines the six fields the labeling sprint (and the LLM pilot) will
populate per abstract, plus a thin :class:`EpistemicExtraction` wrapper
that bundles a single :class:`EpistemicLabel` with provenance — which
``pmid`` it came from, which model id produced it, which prompt version
was in flight, and the raw model response (kept verbatim for forensic
review).

Field set, enums, and nullability mirror master plan §5 Phase 3 exactly;
see :file:`plans/2026-05-22-v1-s07-epistemic-prep.md` §A for the table.

Module-level constant :data:`LABEL_SCHEMA_VERSION` is the source of
truth that gets stamped onto every persisted row. Bump it (and add a
migration note) before changing any field semantics — downstream
parquets keyed on the older version must remain readable.

The models are frozen + ``extra="forbid"`` so accidental field
additions raise at validation time rather than silently entering a
parquet column nobody planned for.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "LABEL_SCHEMA_VERSION",
    "StudyDesign",
    "EffectDirection",
    "EpistemicLabel",
    "EpistemicExtraction",
]


LABEL_SCHEMA_VERSION: str = "v0.1"


StudyDesign = Literal["RCT", "cohort", "case_control", "case_series", "review", "other"]
"""Closed enum of study designs the labelers will see in the Excel dropdown."""


EffectDirection = Literal["positive", "null", "negative", "mixed", "na"]
"""Closed enum of effect-direction labels; ``na`` is reserved for
abstracts where no effect direction is reportable (e.g. reviews,
methods papers)."""


class EpistemicLabel(BaseModel):
    """One labeler's (or one LLM's) judgement about a single abstract.

    Field set is fixed by master plan §5 Phase 3. Nullability is meaningful:

    * :attr:`sample_size` — ``None`` means *the abstract did not state
      one*, NOT *we don't know*. The validator rejects values <1 so
      callers can't smuggle in sentinel zeros.
    * :attr:`has_control` — ``None`` for designs where the question is
      ill-posed (most reviews).
    * :attr:`effect_direction` — ``None`` allowed at the schema level,
      but raters are instructed to use the ``"na"`` enum value
      explicitly; ``None`` is a soft fallback for malformed model
      output.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    study_design: StudyDesign = Field(
        description="Closed-enum study design.",
    )
    sample_size: int | None = Field(
        default=None,
        description="Reported N (≥1) or None if not stated in abstract.",
    )
    has_control: bool | None = Field(
        default=None,
        description="True/False or None if not applicable (e.g., reviews).",
    )
    effect_direction: EffectDirection | None = Field(
        default=None,
        description="Direction of the primary reported effect.",
    )
    statistical_claim_present: bool = Field(
        description="True if the abstract makes any statistical claim.",
    )
    coi_disclosed_in_abstract: bool = Field(
        description="True if any conflict-of-interest disclosure appears in the abstract.",
    )

    @field_validator("sample_size")
    @classmethod
    def _sample_size_positive(cls, v: int | None) -> int | None:
        """Reject sample sizes <1 unless explicitly ``None``.

        ``None`` is the legitimate sentinel for *unreported*; zero or
        negative is never meaningful and almost always indicates a
        parsing bug we want to surface loudly.
        """
        if v is None:
            return v
        if v < 1:
            raise ValueError(f"sample_size must be >= 1 or None; got {v}")
        return v


class EpistemicExtraction(BaseModel):
    """One labeled abstract + the provenance needed to audit it later.

    Wraps :class:`EpistemicLabel` with the four pieces of metadata that
    let us replay the extraction:

    * :attr:`pmid` — the abstract this label is for.
    * :attr:`model_id` — opaque string identifying the producer (e.g.
      ``"claude-via-claude-code"`` for the V1-S07 pilot, or a specific
      API model id once V1-S08 promotes the pipeline).
    * :attr:`prompt_version` — matches :data:`LABEL_SCHEMA_VERSION`-ish
      semantics but versions the prompt independently; raw string so
      future revisions (``v0.1.1``, ``v0.2``) can co-exist in the
      parquet.
    * :attr:`raw_response` — verbatim model output, kept so we can
      diff parse failures and rebuild the JSON parser without re-paying
      for inference.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    pmid: int = Field(description="PubMed ID of the abstract this extraction labels.")
    label: EpistemicLabel = Field(description="The structured judgement itself.")
    model_id: str = Field(description="Identifier for the producing model/agent.")
    prompt_version: str = Field(description="Prompt version that produced this extraction.")
    raw_response: str | None = Field(
        default=None,
        description="Verbatim model response, retained for parse-failure forensics.",
    )
```
