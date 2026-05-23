"""Tests for V1-S07 epistemic extraction prompt module (v0.1 baseline).

Covers the contract that ties together the schema, the prompt, the
pilot extractor, and the prompt-version field stamped onto every row
of ``data/v1/epistemic_pilot.parquet``:

* :data:`PROMPT_VERSION` is the single source of truth and must agree
  with :data:`LABEL_SCHEMA_VERSION` at the minor-version level.
* :data:`SYSTEM_PROMPT_V0_1` actually names the 6 schema fields, the
  6 ``study_design`` enum values, and the JSON-only output rule.
* :data:`FEW_SHOT_EXAMPLES` covers RCT / cohort / case_series /
  review (the four design buckets the pilot is most likely to hit
  with parse errors) and every exemplar's label dict validates
  through :class:`EpistemicLabel`.
* :func:`build_prompt` produces a single coherent stdin payload —
  not role-separated messages — that ends in the literal ``JSON:``
  cue so the model knows what to emit.
"""

from __future__ import annotations

from scifield.epistemic.prompt import (
    FEW_SHOT_EXAMPLES,
    PROMPT_VERSION,
    SYSTEM_PROMPT_V0_1,
    build_prompt,
)
from scifield.epistemic.schema import LABEL_SCHEMA_VERSION, EpistemicLabel


def test_prompt_version_constant() -> None:
    assert PROMPT_VERSION == "v0.1"


def test_prompt_version_matches_schema_minor() -> None:
    # Both ship the same v0.1 baseline; any future skew must be a
    # conscious bump on one side, not a silent divergence.
    assert PROMPT_VERSION == LABEL_SCHEMA_VERSION


def test_system_prompt_names_all_six_fields() -> None:
    for field_name in (
        "study_design",
        "sample_size",
        "has_control",
        "effect_direction",
        "statistical_claim_present",
        "coi_disclosed_in_abstract",
    ):
        assert field_name in SYSTEM_PROMPT_V0_1, f"missing field: {field_name}"


def test_system_prompt_names_all_study_design_enum_values() -> None:
    for enum_value in ("RCT", "cohort", "case_control", "case_series", "review", "other"):
        assert enum_value in SYSTEM_PROMPT_V0_1, f"missing enum value: {enum_value}"


def test_system_prompt_mentions_json_output_rule() -> None:
    assert "JSON" in SYSTEM_PROMPT_V0_1


def test_system_prompt_explains_na_vs_null_convention() -> None:
    # The "na" vs JSON null distinction is the single most common
    # source of prompt-iteration churn; lock it in explicitly.
    assert '"na"' in SYSTEM_PROMPT_V0_1
    assert "null" in SYSTEM_PROMPT_V0_1


def test_few_shot_examples_minimum_count() -> None:
    assert len(FEW_SHOT_EXAMPLES) >= 5


def test_few_shot_examples_cover_required_designs() -> None:
    designs = {ex["label"]["study_design"] for ex in FEW_SHOT_EXAMPLES}
    for required in ("RCT", "cohort", "case_series", "review"):
        assert required in designs, f"few-shot mix missing required design: {required}"


def test_few_shot_examples_include_a_negative_result_trial() -> None:
    # Plan §A explicitly calls for one negative-result trial in the
    # exemplar mix so the model sees the null-effect path.
    rct_directions = {
        ex["label"]["effect_direction"]
        for ex in FEW_SHOT_EXAMPLES
        if ex["label"]["study_design"] == "RCT"
    }
    assert "null" in rct_directions


def test_every_few_shot_label_validates_through_schema() -> None:
    for i, ex in enumerate(FEW_SHOT_EXAMPLES):
        # Schema raises on enum / nullability / extra-field violations.
        label = EpistemicLabel(**ex["label"])
        # Round-trip identity sanity-check.
        assert label.study_design == ex["label"]["study_design"], f"exemplar {i} drifted"


def test_few_shot_examples_have_required_keys() -> None:
    for ex in FEW_SHOT_EXAMPLES:
        assert set(ex.keys()) == {"abstract", "label"}
        assert isinstance(ex["abstract"], str)
        assert isinstance(ex["label"], dict)


def test_build_prompt_contains_system_prompt() -> None:
    out = build_prompt("dummy abstract text")
    assert SYSTEM_PROMPT_V0_1 in out


def test_build_prompt_contains_target_abstract() -> None:
    out = build_prompt("dummy abstract text")
    assert "dummy abstract text" in out


def test_build_prompt_ends_with_json_cue() -> None:
    out = build_prompt("dummy abstract text")
    assert out.endswith("JSON:")


def test_build_prompt_renders_each_exemplar() -> None:
    out = build_prompt("dummy abstract text")
    # Every exemplar abstract should appear verbatim in the rendered
    # prompt (truncation here would silently drop few-shots).
    for ex in FEW_SHOT_EXAMPLES:
        assert ex["abstract"] in out


def test_build_prompt_is_single_string_no_role_separators() -> None:
    # Claude Code CLI reads one stdin stream; assert no accidental
    # "system:" / "user:" / "assistant:" role markers crept in.
    out = build_prompt("dummy abstract text").lower()
    for role_marker in ("\nsystem:", "\nuser:", "\nassistant:"):
        assert role_marker not in out
