"""Tests for V1-S07 §D pilot driver :mod:`scifield.epistemic.pilot`.

Coverage:

* End-to-end happy path with mocked :func:`extract_one`: 4 successes +
  1 failure across a 5-row synthetic sample parquet.
* Both output parquets exist with the right column signatures (11
  columns success, 6 columns failures), plus ``.run.json`` sidecars.
* Deterministic ``head(n_pilot)`` selection — with ``n_pilot=3`` on a
  shuffled 5-row sample, the chosen rows are the 3 lowest ``pmid`` s.
* ``progress`` callback fires once per row with the right outcome
  strings in the right order.
* Empty-pilot edge case (``n_pilot=0``): both parquets are still
  written, are readable, and carry the pinned schema.

No test ever shells out to the real ``claude`` binary —
:func:`extract_one` is patched at the symbol the pilot module imports.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

from scifield.epistemic.extract import ExtractConfig, ExtractionError
from scifield.epistemic.pilot import (
    PILOT_FAILED_SCHEMA,
    PILOT_SUCCESS_SCHEMA,
    PilotConfig,
    run_pilot,
)
from scifield.epistemic.schema import EpistemicExtraction, EpistemicLabel

_EXPECTED_SUCCESS_COLUMNS: list[str] = [
    "pmid",
    "study_design",
    "sample_size",
    "has_control",
    "effect_direction",
    "statistical_claim_present",
    "coi_disclosed_in_abstract",
    "model_id",
    "prompt_version",
    "raw_response",
    "extracted_at",
]

_EXPECTED_FAILURE_COLUMNS: list[str] = [
    "pmid",
    "reason",
    "raw_response",
    "model_id",
    "prompt_version",
    "failed_at",
]


def _write_sample_parquet(path: Path, pmids: list[int]) -> None:
    """Build a 5-row (or N-row) synthetic ``handlabel_sample.parquet``."""
    df = pd.DataFrame(
        {
            "pmid": pmids,
            "journal": [f"J{p % 3}" for p in pmids],
            "year": [2000 + (p % 25) for p in pmids],
            "era": ["2000-2009"] * len(pmids),
            "topic_id": [p % 7 for p in pmids],
            "title": [f"Title for {p}" for p in pmids],
            "abstract": [f"Abstract body for pmid {p}." for p in pmids],
        }
    )
    df.to_parquet(path, index=False)


def _canned_extraction(pmid: int) -> EpistemicExtraction:
    """Construct a canned, schema-valid :class:`EpistemicExtraction`.

    Used as the return value of the patched :func:`extract_one` so we
    never go near the real subprocess / ``claude`` binary.
    """
    return EpistemicExtraction(
        pmid=pmid,
        label=EpistemicLabel(
            study_design="RCT",
            sample_size=100 + pmid,
            has_control=True,
            effect_direction="positive",
            statistical_claim_present=True,
            coi_disclosed_in_abstract=False,
        ),
        model_id="claude-via-claude-code",
        prompt_version="v0.1",
        raw_response=f'{{"pmid":{pmid}}}',
    )


def _make_extract_one_mock(fail_pmid: int):
    """Return a function that mocks :func:`extract_one`.

    Behavior: returns a canned extraction for any pmid except ``fail_pmid``,
    for which it raises :class:`ExtractionError`. Signature mirrors the
    real :func:`extract_one` so the call shape is realistic.
    """

    def _fake_extract_one(
        abstract: str,
        pmid: int,
        cfg: ExtractConfig | None = None,
    ) -> EpistemicExtraction:
        if pmid == fail_pmid:
            raise ExtractionError(
                pmid=pmid,
                reason="canned failure for test",
                raw_response="garbage from model",
            )
        return _canned_extraction(pmid)

    return _fake_extract_one


def test_run_pilot_happy_path_4_ok_1_failed(tmp_path: Path) -> None:
    """4 successes + 1 failure across a 5-row sample."""
    sample_path = tmp_path / "sample.parquet"
    pilot_path = tmp_path / "pilot.parquet"
    failed_path = tmp_path / "pilot_failed.parquet"

    pmids = [1001, 1002, 1003, 1004, 1005]
    _write_sample_parquet(sample_path, pmids)

    cfg = PilotConfig(
        sample_path=sample_path,
        pilot_path=pilot_path,
        pilot_failed_path=failed_path,
        n_pilot=5,
    )

    with patch(
        "scifield.epistemic.pilot.extract_one",
        side_effect=_make_extract_one_mock(fail_pmid=1003),
    ):
        summary = run_pilot(cfg)

    assert summary["n_attempted"] == 5
    assert summary["n_ok"] == 4
    assert summary["n_failed"] == 1
    assert summary["pilot_path"] == str(pilot_path)
    assert summary["pilot_failed_path"] == str(failed_path)
    assert summary["wall_seconds"] >= 0.0

    # Both parquets exist on disk.
    assert pilot_path.exists()
    assert failed_path.exists()

    # Both sidecars exist next to the parquets.
    pilot_sidecar = Path(str(pilot_path) + ".run.json")
    failed_sidecar = Path(str(failed_path) + ".run.json")
    assert pilot_sidecar.exists()
    assert failed_sidecar.exists()

    # Sidecars carry the counts we tucked into config.
    pilot_payload = json.loads(pilot_sidecar.read_text())
    assert pilot_payload["config"]["n_ok"] == 4
    assert pilot_payload["config"]["n_failed"] == 1
    assert pilot_payload["config"]["n_pilot"] == 5

    # Success parquet column signature.
    success_table = pq.read_table(pilot_path)
    assert success_table.column_names == _EXPECTED_SUCCESS_COLUMNS
    assert success_table.num_rows == 4
    # And the failed pmid did NOT land in the success parquet.
    success_pmids = set(success_table.column("pmid").to_pylist())
    assert success_pmids == {1001, 1002, 1004, 1005}

    # Failure parquet column signature.
    failure_table = pq.read_table(failed_path)
    assert failure_table.column_names == _EXPECTED_FAILURE_COLUMNS
    assert failure_table.num_rows == 1
    assert failure_table.column("pmid").to_pylist() == [1003]
    assert failure_table.column("reason").to_pylist() == ["canned failure for test"]


def test_run_pilot_head_is_pmid_sorted(tmp_path: Path) -> None:
    """``n_pilot=3`` on a shuffled 5-row sample selects the 3 smallest pmids."""
    sample_path = tmp_path / "sample.parquet"
    pilot_path = tmp_path / "pilot.parquet"
    failed_path = tmp_path / "pilot_failed.parquet"

    # Deliberately out-of-order pmids — sampling.stratified_sample shuffles.
    pmids = [9000, 1000, 5000, 3000, 7000]
    _write_sample_parquet(sample_path, pmids)

    cfg = PilotConfig(
        sample_path=sample_path,
        pilot_path=pilot_path,
        pilot_failed_path=failed_path,
        n_pilot=3,
    )

    with patch(
        "scifield.epistemic.pilot.extract_one",
        # Never fails — we only care about which pmids get picked.
        side_effect=_make_extract_one_mock(fail_pmid=-1),
    ):
        summary = run_pilot(cfg)

    assert summary["n_attempted"] == 3
    assert summary["n_ok"] == 3
    assert summary["n_failed"] == 0

    success_table = pq.read_table(pilot_path)
    selected = success_table.column("pmid").to_pylist()
    # Sorted ascending, lowest 3.
    assert selected == [1000, 3000, 5000]


def test_run_pilot_progress_callback_fires_once_per_row(tmp_path: Path) -> None:
    """``progress`` is called once per attempted row with the right outcome."""
    sample_path = tmp_path / "sample.parquet"
    pilot_path = tmp_path / "pilot.parquet"
    failed_path = tmp_path / "pilot_failed.parquet"

    pmids = [1, 2, 3, 4, 5]
    _write_sample_parquet(sample_path, pmids)

    cfg = PilotConfig(
        sample_path=sample_path,
        pilot_path=pilot_path,
        pilot_failed_path=failed_path,
        n_pilot=5,
    )

    calls: list[tuple[int, int, str]] = []

    def _progress(i: int, n: int, outcome: str) -> None:
        calls.append((i, n, outcome))

    with patch(
        "scifield.epistemic.pilot.extract_one",
        # Fail pmid==3 so we get a mixed outcome sequence.
        side_effect=_make_extract_one_mock(fail_pmid=3),
    ):
        run_pilot(cfg, progress=_progress)

    assert len(calls) == 5
    # i indices are 1-based and monotonic.
    assert [c[0] for c in calls] == [1, 2, 3, 4, 5]
    # n is the effective attempted count throughout.
    assert all(c[1] == 5 for c in calls)
    # Outcomes: row 3 (pmid==3) fails; rest are ok. Sample is sorted by
    # pmid ascending so row order matches pmid order.
    assert [c[2] for c in calls] == ["ok", "ok", "failed", "ok", "ok"]


def test_run_pilot_empty_n_pilot_writes_typed_empty_parquets(tmp_path: Path) -> None:
    """``n_pilot=0`` still writes both parquets with the pinned schema."""
    sample_path = tmp_path / "sample.parquet"
    pilot_path = tmp_path / "pilot.parquet"
    failed_path = tmp_path / "pilot_failed.parquet"

    _write_sample_parquet(sample_path, [10, 20, 30])

    cfg = PilotConfig(
        sample_path=sample_path,
        pilot_path=pilot_path,
        pilot_failed_path=failed_path,
        n_pilot=0,
    )

    progress_calls: list[tuple[int, int, str]] = []

    with patch(
        "scifield.epistemic.pilot.extract_one",
        side_effect=_make_extract_one_mock(fail_pmid=-1),
    ) as mock_extract:
        summary = run_pilot(
            cfg,
            progress=lambda i, n, outcome: progress_calls.append((i, n, outcome)),
        )

    # extract_one should never have been invoked.
    assert mock_extract.call_count == 0
    assert progress_calls == []
    assert summary["n_attempted"] == 0
    assert summary["n_ok"] == 0
    assert summary["n_failed"] == 0

    # Both parquets exist and are empty but still typed.
    success_table = pq.read_table(pilot_path)
    failure_table = pq.read_table(failed_path)
    assert success_table.num_rows == 0
    assert failure_table.num_rows == 0
    assert success_table.schema.equals(PILOT_SUCCESS_SCHEMA)
    assert failure_table.schema.equals(PILOT_FAILED_SCHEMA)

    # Sidecars still drop next to empty parquets.
    assert Path(str(pilot_path) + ".run.json").exists()
    assert Path(str(failed_path) + ".run.json").exists()


def test_run_pilot_passes_extract_cfg_through(tmp_path: Path) -> None:
    """Custom :class:`ExtractConfig` is threaded to every :func:`extract_one` call."""
    sample_path = tmp_path / "sample.parquet"
    pilot_path = tmp_path / "pilot.parquet"
    failed_path = tmp_path / "pilot_failed.parquet"

    _write_sample_parquet(sample_path, [101, 202])

    custom_extract_cfg = ExtractConfig(
        claude_cmd=("/bin/true",),
        model_id="test-model",
        prompt_version="v0.1.test",
        timeout_s=5.0,
        retry_on_parse_failure=False,
    )

    cfg = PilotConfig(
        sample_path=sample_path,
        pilot_path=pilot_path,
        pilot_failed_path=failed_path,
        n_pilot=2,
        extract_cfg=custom_extract_cfg,
    )

    with patch(
        "scifield.epistemic.pilot.extract_one",
        side_effect=_make_extract_one_mock(fail_pmid=-1),
    ) as mock_extract:
        run_pilot(cfg)

    assert mock_extract.call_count == 2
    # Each call received the same custom ExtractConfig instance.
    for call in mock_extract.call_args_list:
        _, kwargs = call
        assert kwargs["cfg"] is custom_extract_cfg
