"""Phase 3 — Epistemic quality extraction.

V1-S07 ships the schema, prompt engineering scaffolding, sampling, and
50-abstract pilot infrastructure. This module exposes the full V1-S07
public surface (pilot helpers loaded lazily by the CLI).
"""

from scifield.epistemic.extract import (
    ExtractConfig,
    ExtractionError,
    extract_one,
)
from scifield.epistemic.kappa import (
    cohens_kappa,
    krippendorffs_alpha,
    per_field_summary,
)
from scifield.epistemic.labeling import (
    export_to_xlsx,
    import_from_xlsx,
)
from scifield.epistemic.prompt import (
    FEW_SHOT_EXAMPLES,
    PROMPT_VERSION,
    SYSTEM_PROMPT_V0_1,
    build_prompt,
)
from scifield.epistemic.sampling import (
    SamplingConfig,
    stratified_sample,
)
from scifield.epistemic.schema import (
    LABEL_SCHEMA_VERSION,
    EpistemicExtraction,
    EpistemicLabel,
)

__all__ = [
    "FEW_SHOT_EXAMPLES",
    "LABEL_SCHEMA_VERSION",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT_V0_1",
    "EpistemicExtraction",
    "EpistemicLabel",
    "ExtractConfig",
    "ExtractionError",
    "SamplingConfig",
    "build_prompt",
    "cohens_kappa",
    "export_to_xlsx",
    "extract_one",
    "import_from_xlsx",
    "krippendorffs_alpha",
    "per_field_summary",
    "stratified_sample",
]
