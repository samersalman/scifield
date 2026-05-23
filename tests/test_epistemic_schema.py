"""Tests for V1-S07 epistemic schema foundation.

Covers the contract the labeling sprint and the pilot extractor both
depend on: enum-restricted study_design / effect_direction, the
sample_size >=1 validator, JSON round-trip identity, and the version
constant stamped onto every downstream parquet.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scifield.epistemic.schema import (
    LABEL_SCHEMA_VERSION,
    EpistemicExtraction,
    EpistemicLabel,
)


def _rct_exemplar() -> EpistemicLabel:
    """Canonical valid RCT label used across multiple tests."""
    return EpistemicLabel(
        study_design="RCT",
        sample_size=240,
        has_control=True,
        effect_direction="positive",
        statistical_claim_present=True,
        coi_disclosed_in_abstract=False,
    )


def test_label_schema_version_constant() -> None:
    assert LABEL_SCHEMA_VERSION == "v0.1"


def test_valid_rct_construction() -> None:
    label = _rct_exemplar()
    assert label.study_design == "RCT"
    assert label.sample_size == 240
    assert label.has_control is True
    assert label.effect_direction == "positive"
    assert label.statistical_claim_present is True
    assert label.coi_disclosed_in_abstract is False


def test_label_json_round_trip_identity() -> None:
    original = _rct_exemplar()
    payload = original.model_dump_json()
    restored = EpistemicLabel.model_validate_json(payload)
    assert restored == original


def test_label_rejects_bad_study_design_enum() -> None:
    with pytest.raises(ValidationError):
        EpistemicLabel(
            study_design="experimental",
            sample_size=100,
            has_control=True,
            effect_direction="positive",
            statistical_claim_present=True,
            coi_disclosed_in_abstract=False,
        )


def test_label_rejects_bad_effect_direction_enum() -> None:
    with pytest.raises(ValidationError):
        EpistemicLabel(
            study_design="RCT",
            sample_size=100,
            has_control=True,
            effect_direction="up",
            statistical_claim_present=True,
            coi_disclosed_in_abstract=False,
        )


def test_label_rejects_zero_sample_size() -> None:
    with pytest.raises(ValidationError):
        EpistemicLabel(
            study_design="cohort",
            sample_size=0,
            has_control=False,
            effect_direction="null",
            statistical_claim_present=True,
            coi_disclosed_in_abstract=False,
        )


def test_label_rejects_negative_sample_size() -> None:
    with pytest.raises(ValidationError):
        EpistemicLabel(
            study_design="cohort",
            sample_size=-5,
            has_control=False,
            effect_direction="null",
            statistical_claim_present=True,
            coi_disclosed_in_abstract=False,
        )


def test_label_accepts_none_sample_size() -> None:
    label = EpistemicLabel(
        study_design="review",
        sample_size=None,
        has_control=None,
        effect_direction="na",
        statistical_claim_present=False,
        coi_disclosed_in_abstract=False,
    )
    assert label.sample_size is None
    assert label.has_control is None


def test_label_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EpistemicLabel(
            study_design="RCT",
            sample_size=10,
            has_control=True,
            effect_direction="positive",
            statistical_claim_present=True,
            coi_disclosed_in_abstract=False,
            unexpected_field="boom",
        )


def test_extraction_round_trip_preserves_provenance() -> None:
    extraction = EpistemicExtraction(
        pmid=12345678,
        label=_rct_exemplar(),
        model_id="claude-via-claude-code",
        prompt_version="v0.1",
        raw_response='{"study_design":"RCT","sample_size":240,"has_control":true,'
        '"effect_direction":"positive","statistical_claim_present":true,'
        '"coi_disclosed_in_abstract":false}',
    )
    payload = extraction.model_dump_json()
    restored = EpistemicExtraction.model_validate_json(payload)
    assert restored == extraction
    assert restored.pmid == 12345678
    assert restored.model_id == "claude-via-claude-code"
    assert restored.prompt_version == "v0.1"
    assert restored.raw_response is not None
    assert restored.label == _rct_exemplar()


def test_extraction_raw_response_defaults_to_none() -> None:
    extraction = EpistemicExtraction(
        pmid=1,
        label=_rct_exemplar(),
        model_id="claude-via-claude-code",
        prompt_version="v0.1",
    )
    assert extraction.raw_response is None
