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
