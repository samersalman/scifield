"""V1-S07 §D — 50-abstract Claude-Code pilot driver.

This module owns the *batch* edge of the V1-S07 pilot. Where
:mod:`scifield.epistemic.extract` handles one abstract at a time, this
module reads the stratified sample parquet (written by
:mod:`scifield.epistemic.sampling`), iterates :func:`extract_one` across
the first ``n_pilot`` rows sorted deterministically by ``pmid``, and
persists two parquets:

* ``cfg.pilot_path`` — successful extractions, one row per accepted
  :class:`EpistemicExtraction`, flattened.
* ``cfg.pilot_failed_path`` — rows that hit :class:`ExtractionError`,
  with the verbatim raw response and the human-readable reason.

Both parquets are written with explicit :mod:`pyarrow` schemas so the
column types persist even when the success or failure set is empty —
this lets V1-S08 (and the notebook smoke test) consume the parquet
without special-casing first-run-of-the-day.

Each parquet gets a ``.run.json`` sidecar via
:func:`scifield.repro.record_run` capturing inputs, run config, and
success/failure counts.

.. note::

   Per plan §D and the V1-S07 risk row: this driver MUST NOT be invoked
   on real corpus data outside the 50-abstract pilot. The full 200k
   batch run is V1-S08 territory and is gated on the OSF pre-registration
   link landing in ``docs/preregistrations/PR1_epistemic_extraction.md``.
   Sorting by ``pmid`` ascending before ``.head(n_pilot)`` keeps the
   pilot row selection reproducible across re-runs of the same sample
   parquet, even though the sample itself is randomized upstream.

Status reporting is push-only via the optional ``progress`` callback.
There are no :func:`print` calls in this module — the CLI (Batch 4B) is
responsible for any user-facing terminal output and wires its own
callback into :func:`run_pilot`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scifield.epistemic.extract import ExtractConfig, ExtractionError, extract_one
from scifield.epistemic.schema import EpistemicExtraction
from scifield.repro import record_run

__all__ = [
    "PilotConfig",
    "PILOT_SUCCESS_SCHEMA",
    "PILOT_FAILED_SCHEMA",
    "run_pilot",
]


PILOT_SUCCESS_SCHEMA: pa.Schema = pa.schema(
    [
        ("pmid", pa.int64()),
        ("study_design", pa.string()),
        ("sample_size", pa.int64()),
        ("has_control", pa.bool_()),
        ("effect_direction", pa.string()),
        ("statistical_claim_present", pa.bool_()),
        ("coi_disclosed_in_abstract", pa.bool_()),
        ("model_id", pa.string()),
        ("prompt_version", pa.string()),
        ("raw_response", pa.string()),
        ("extracted_at", pa.string()),
    ]
)
"""Explicit Arrow schema for the success parquet.

Pinned so that even when the pilot returns zero successful rows (an
edge case at ``n_pilot=0``, or a catastrophic prompt regression), the
parquet on disk still carries the 11-column type signature downstream
consumers expect."""


PILOT_FAILED_SCHEMA: pa.Schema = pa.schema(
    [
        ("pmid", pa.int64()),
        ("reason", pa.string()),
        ("raw_response", pa.string()),
        ("model_id", pa.string()),
        ("prompt_version", pa.string()),
        ("failed_at", pa.string()),
    ]
)
"""Explicit Arrow schema for the failures parquet.

Same rationale as :data:`PILOT_SUCCESS_SCHEMA` — the schema must survive
the empty-failure happy path so the failures parquet is always
readable."""


_PROGRESS_OK: str = "ok"
"""Outcome string passed to the ``progress`` callback for a row that
:func:`extract_one` returned cleanly for (possibly via the retry path —
the current :func:`extract_one` API does not distinguish first-try-ok
from retry-recovered-ok, and per plan §D that finer-grained tri-state
is explicitly a nice-to-have, not blocking for V1-S07)."""

_PROGRESS_FAILED: str = "failed"
"""Outcome string passed to the ``progress`` callback for a row that
:func:`extract_one` raised :class:`ExtractionError` on."""


@dataclass(frozen=True)
class PilotConfig:
    """Configuration for one :func:`run_pilot` invocation.

    Field order is load-bearing for the CLI wiring (Batch 4B) — the
    three required paths come first, then the small integer knob, then
    the nested :class:`ExtractConfig`. All fields except
    :attr:`extract_cfg` are intended to be sourced from
    ``conf/epistemic/v1.yaml``.

    Attributes:
        sample_path: Parquet written by
            :func:`scifield.epistemic.sampling.stratified_sample`.
            Must carry the columns ``pmid, journal, year, era,
            topic_id, title, abstract``.
        pilot_path: Output parquet for successful extractions.
            Schema matches :data:`PILOT_SUCCESS_SCHEMA`.
        pilot_failed_path: Output parquet for rows that hit
            :class:`ExtractionError`. Schema matches
            :data:`PILOT_FAILED_SCHEMA`.
        n_pilot: Number of rows from the (pmid-sorted) sample to run
            through the extractor. V1-S07 plan §D defaults to 50.
        extract_cfg: Per-row :class:`ExtractConfig`. Defaults to
            :class:`ExtractConfig` with all its V1-S07 defaults
            (``("claude", "--print")``, ``v0.1`` prompt, 120s timeout,
            retry enabled).
    """

    sample_path: Path
    pilot_path: Path
    pilot_failed_path: Path
    n_pilot: int = 50
    extract_cfg: ExtractConfig = field(default_factory=ExtractConfig)


def _flatten_success(
    extraction: EpistemicExtraction,
    extracted_at: str,
) -> dict[str, Any]:
    """Flatten one :class:`EpistemicExtraction` to a parquet row dict.

    The label sub-model is hoisted to top-level columns so the parquet
    is queryable without nested-struct gymnastics. ``extracted_at`` is
    threaded in from a single batch-level timestamp captured at the top
    of :func:`run_pilot` so every row in one pilot run shares the same
    extraction stamp (matches the sidecar's ``timestamp``).
    """
    label = extraction.label
    return {
        "pmid": extraction.pmid,
        "study_design": label.study_design,
        "sample_size": label.sample_size,
        "has_control": label.has_control,
        "effect_direction": label.effect_direction,
        "statistical_claim_present": label.statistical_claim_present,
        "coi_disclosed_in_abstract": label.coi_disclosed_in_abstract,
        "model_id": extraction.model_id,
        "prompt_version": extraction.prompt_version,
        "raw_response": extraction.raw_response,
        "extracted_at": extracted_at,
    }


def _flatten_failure(
    err: ExtractionError,
    cfg: ExtractConfig,
    failed_at: str,
) -> dict[str, Any]:
    """Flatten one :class:`ExtractionError` to a failures-parquet row dict.

    ``model_id`` / ``prompt_version`` come off the :class:`ExtractConfig`
    in flight rather than the (absent) extraction, so a failures parquet
    can still be diffed against the success parquet on those provenance
    axes.
    """
    return {
        "pmid": err.pmid,
        "reason": err.reason,
        "raw_response": err.raw_response,
        "model_id": cfg.model_id,
        "prompt_version": cfg.prompt_version,
        "failed_at": failed_at,
    }


def _write_parquet(
    rows: list[dict[str, Any]],
    out_path: Path,
    schema: pa.Schema,
) -> None:
    """Write ``rows`` to ``out_path`` under ``schema``.

    Handles the empty-rows case explicitly by constructing an Arrow
    table from typed empty arrays — :func:`pa.Table.from_pylist` would
    yield ``schema=null`` for an empty list, which loses the column
    type signature we need downstream.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
    else:
        empty_arrays = [pa.array([], type=field_.type) for field_ in schema]
        table = pa.Table.from_arrays(empty_arrays, schema=schema)
    pq.write_table(table, out_path)


def _extract_cfg_to_dict(cfg: ExtractConfig) -> dict[str, Any]:
    """JSON-serializable view of :class:`ExtractConfig` for the sidecar.

    ``asdict`` already gives us a plain dict; we only have to coerce the
    tuple-typed ``claude_cmd`` to a list so :func:`json.dumps` in
    :func:`record_run` doesn't choke on tuples (it would actually be
    fine — tuples serialize as JSON arrays — but a list keeps the
    sidecar round-trippable via :func:`json.loads`).
    """
    raw = asdict(cfg)
    if "claude_cmd" in raw and isinstance(raw["claude_cmd"], tuple):
        raw["claude_cmd"] = list(raw["claude_cmd"])
    return raw


def run_pilot(
    cfg: PilotConfig,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run the V1-S07 pilot extraction over ``cfg.n_pilot`` abstracts.

    The full lifecycle is:

    1. Read ``cfg.sample_path`` into a DataFrame; sort ascending by
       ``pmid`` for reproducibility; take the first ``cfg.n_pilot`` rows.
    2. Iterate each row, calling
       :func:`scifield.epistemic.extract.extract_one` with the row's
       ``abstract`` and ``pmid``. Successes accumulate into one list;
       :class:`ExtractionError` rows accumulate into another.
    3. After each row, invoke ``progress(i, n, outcome)`` where ``i`` is
       the 1-indexed row counter, ``n`` is ``cfg.n_pilot`` (or the
       effective n if the sample was shorter), and ``outcome`` is
       ``"ok"`` or ``"failed"``. ``progress`` may be ``None``, in which
       case status is silent.
    4. Write the success parquet to ``cfg.pilot_path`` and the failure
       parquet to ``cfg.pilot_failed_path``, using explicit Arrow
       schemas (so empty parquets are still type-pinned).
    5. Write a ``.run.json`` sidecar next to each parquet via
       :func:`scifield.repro.record_run`, with the sample parquet as the
       sole input.
    6. Return a summary dict with counts and paths.

    Args:
        cfg: :class:`PilotConfig` carrying input/output paths, ``n_pilot``,
            and the nested :class:`ExtractConfig` to pass into
            :func:`extract_one`.
        progress: Optional callback invoked once per row. Signature
            ``(i: int, n: int, outcome: str) -> None``. ``outcome`` is
            one of ``"ok"`` or ``"failed"``.

    Returns:
        ``{"n_attempted": int, "n_ok": int, "n_failed": int,
        "pilot_path": str, "pilot_failed_path": str,
        "wall_seconds": float}``.
    """
    t0 = time.monotonic()
    # Capture one timestamp at the top of the run so every row in this
    # batch shares an `extracted_at` / `failed_at` matching the sidecar's
    # `timestamp` field.
    run_stamp = datetime.now(UTC).isoformat()

    df = pd.read_parquet(cfg.sample_path)
    df = df.sort_values("pmid", ascending=True).reset_index(drop=True)
    df = df.head(cfg.n_pilot)
    n_attempted = len(df)

    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        try:
            extraction = extract_one(
                abstract=row.abstract,
                pmid=int(row.pmid),
                cfg=cfg.extract_cfg,
            )
        except ExtractionError as err:
            failure_rows.append(_flatten_failure(err, cfg.extract_cfg, run_stamp))
            if progress is not None:
                progress(i, n_attempted, _PROGRESS_FAILED)
            continue
        success_rows.append(_flatten_success(extraction, run_stamp))
        if progress is not None:
            progress(i, n_attempted, _PROGRESS_OK)

    _write_parquet(success_rows, cfg.pilot_path, PILOT_SUCCESS_SCHEMA)
    _write_parquet(failure_rows, cfg.pilot_failed_path, PILOT_FAILED_SCHEMA)

    sidecar_config = {
        "n_pilot": cfg.n_pilot,
        "extract_cfg": _extract_cfg_to_dict(cfg.extract_cfg),
        "n_attempted": n_attempted,
        "n_ok": len(success_rows),
        "n_failed": len(failure_rows),
    }
    record_run(
        artifact_path=cfg.pilot_path,
        inputs={"sample": cfg.sample_path},
        config=sidecar_config,
    )
    record_run(
        artifact_path=cfg.pilot_failed_path,
        inputs={"sample": cfg.sample_path},
        config=sidecar_config,
    )

    return {
        "n_attempted": n_attempted,
        "n_ok": len(success_rows),
        "n_failed": len(failure_rows),
        "pilot_path": str(cfg.pilot_path),
        "pilot_failed_path": str(cfg.pilot_failed_path),
        "wall_seconds": time.monotonic() - t0,
    }
