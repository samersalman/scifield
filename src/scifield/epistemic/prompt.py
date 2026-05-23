"""V1-S07 epistemic-quality extraction prompt module (v0.1 baseline).

Owns the *single source of truth* for the system prompt, the few-shot
exemplars, and the final stdin-ready string that ``extract.extract_one``
feeds to ``claude --print``. The Claude Code CLI surface is single-stream
stdin -> single-stream stdout: there is no role separation, so the
"system prompt" and "user prompt" are concatenated into one coherent
text block by :func:`build_prompt`.

Versioning:

* :data:`PROMPT_VERSION` is the single string stamped onto every
  :class:`scifield.epistemic.schema.EpistemicExtraction` row in the
  pilot parquet. It MUST match ``prompt.version`` in
  ``conf/epistemic/v1.yaml``.
* Any change to :data:`SYSTEM_PROMPT_V0_1` or :data:`FEW_SHOT_EXAMPLES`
  must bump :data:`PROMPT_VERSION` (``v0.1`` -> ``v0.1.1`` for
  in-pilot iteration, ``v0.2`` for breaking changes) and add a new
  H3 entry under "Prompt iteration log" in
  ``docs/phases/epistemic.md``.

Output contract enforced on the model side via prompt text (and on the
Python side via :class:`EpistemicLabel`):

* Response is a single JSON object, no markdown fences, no prose.
* Keys: exactly the 6 schema fields.
* ``study_design`` and ``effect_direction`` use closed enums.
* ``"na"`` (string) is the effect_direction sentinel for non-applicable
  cases; JSON ``null`` is reserved for *malformed* output and only
  tolerated by the schema as a soft fallback.
* ``sample_size`` and ``has_control`` use JSON ``null`` when not
  reported / not applicable.
"""

from __future__ import annotations

import textwrap

__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT_V0_1",
    "FEW_SHOT_EXAMPLES",
    "build_prompt",
]


PROMPT_VERSION: str = "v0.1"
"""Version stamp for the prompt + few-shot bundle. Must equal
``prompt.version`` in ``conf/epistemic/v1.yaml`` and match the minor
of :data:`scifield.epistemic.schema.LABEL_SCHEMA_VERSION`."""


SYSTEM_PROMPT_V0_1: str = textwrap.dedent(
    """\
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
    """
).strip()


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
"""Worked exemplars covering RCT (positive + null), observational cohort,
case-control, case series, and review. Each ``label`` dict is constructed
to validate cleanly through :class:`EpistemicLabel` — see
``tests/test_epistemic_prompt.py`` for the schema-validation guard."""


def build_prompt(abstract: str) -> str:
    """Compose the full stdin payload for ``claude --print``.

    The Claude Code CLI does not accept role-separated messages — it
    reads a single string from stdin and emits a single string on
    stdout. So we concatenate (a) the system prompt, (b) the few-shot
    block, and (c) the target abstract followed by ``JSON:`` as the
    cue for the model's next output.

    Each few-shot exemplar is rendered as ``Abstract: <text>\\nJSON:
    <one-line JSON>\\n\\n``, mirroring the final query's shape so the
    model treats the target as just another step in the same pattern.

    Args:
        abstract: The PubMed abstract text to extract from.

    Returns:
        A single string ready to be passed via stdin to
        ``subprocess.run(["claude", "--print"], input=...)``.
    """
    import json

    parts: list[str] = [SYSTEM_PROMPT_V0_1, ""]
    for ex in FEW_SHOT_EXAMPLES:
        parts.append(f"Abstract: {ex['abstract']}")
        parts.append(f"JSON: {json.dumps(ex['label'])}")
        parts.append("")
    parts.append(f"Abstract: {abstract}")
    parts.append("JSON:")
    return "\n".join(parts)
