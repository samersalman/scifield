"""Tests for V1-S08 batch extraction driver (:mod:`scifield.epistemic.batch`).

The four behaviors locked in the V1-S08 plan §"Files to create or modify"
row for ``tests/test_epistemic_batch.py``:

(a) ``list_remaining_pmids`` skips PMIDs already in the success parquet.
(b) Bad ``subprocess.run`` payloads route the failing PMIDs into the
    failures parquet without poisoning the success parquet.
(c) Each flush writes both ``.run.json`` sidecars with
    ``config.preregistration_url`` set to the V1-S08 OSF DOI.
(d) ``BatchConfig.concurrency`` is honored — i.e. the
    :class:`ProcessPoolExecutor` is constructed with
    ``max_workers=cfg.concurrency``.

ProcessPoolExecutor caveats
---------------------------
Real child processes do not see :mod:`unittest.mock` patches applied
in the parent. For behaviors (b) and (c) we therefore short-circuit
the pool entirely by patching
:attr:`scifield.epistemic.batch.concurrent.futures.ProcessPoolExecutor`
with a :class:`_InlineExecutor` shim that runs every ``submit()``
synchronously in-process, against the parent's patched
:func:`subprocess.run`. For (d) we replace the executor class with a
``MagicMock`` and verify the ``max_workers`` keyword on
construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import duckdb
import pyarrow.parquet as pq
import pytest

from scifield.epistemic.batch import (
    BatchConfig,
    list_remaining_pmids,
    run_batch,
)
from scifield.epistemic.extract import ExtractConfig
from scifield.epistemic.pilot import PILOT_SUCCESS_SCHEMA

_VALID_LABEL_DICT: dict[str, Any] = {
    "study_design": "RCT",
    "sample_size": 480,
    "has_control": True,
    "effect_direction": "positive",
    "statistical_claim_present": True,
    "coi_disclosed_in_abstract": False,
}
_VALID_LABEL_JSON: str = json.dumps(_VALID_LABEL_DICT)
"""Mirror of the helper in ``test_epistemic_extract.py``. Kept here as a
local copy so this test file does not import private fixtures from a
sibling test module."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synth_duckdb(path: Path, n_rows: int = 5) -> list[int]:
    """Plant a tiny ``papers`` table sufficient for ``ensure_papers_distinct_view``.

    Mirrors the pattern in ``tests/test_cli_epistemic.py::_make_synth_duckdb``
    but parameterized on ``n_rows`` so tests can dial up to 6 abstracts
    when they want to split success vs. failure halves.
    """
    pmids: list[int] = []
    con = duckdb.connect(str(path))
    try:
        con.execute(
            """
            CREATE TABLE papers (
                pmid BIGINT,
                journal_slug VARCHAR,
                title VARCHAR,
                abstract VARCHAR,
                year INTEGER,
                fetched_at VARCHAR
            )
            """
        )
        rows = []
        for i in range(n_rows):
            pmid = 1_000_000 + i
            pmids.append(pmid)
            rows.append(
                (
                    pmid,
                    "ann_surg",
                    f"title-{i}",
                    "A" * 80 + f" synthetic abstract {i}",
                    2015,
                    "2026-01-01T00:00:00Z",
                )
            )
        con.executemany(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    finally:
        con.close()
    return pmids


def _make_run_result(stdout: str, returncode: int = 0) -> MagicMock:
    """Build a :class:`subprocess.CompletedProcess`-shaped Mock."""
    import subprocess

    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.stdout = stdout
    proc.stderr = ""
    proc.returncode = returncode
    return proc


def _make_subprocess_router(claude_results: list[MagicMock]):
    """Return a ``side_effect`` callable that routes subprocess.run calls.

    Calls whose first argument looks like the Claude CLI argv consume the
    next item from ``claude_results``. Calls for git / platform probes
    (issued by :func:`scifield.repro.record_run`) fall through to the
    real :func:`subprocess.run` so the sidecar can still be written.
    """
    import subprocess as _subprocess

    real_run = _subprocess.run
    claude_iter = iter(claude_results)

    def _router(*args: Any, **kwargs: Any) -> Any:
        argv = args[0] if args else kwargs.get("args")
        # Crude but reliable: the extract path uses ("claude", "--print").
        if isinstance(argv, list | tuple) and len(argv) > 0 and argv[0] == "claude":
            return next(claude_iter)
        # Everything else (git, platform probes) goes to the real call.
        return real_run(*args, **kwargs)

    return _router


def _seed_success_parquet(path: Path, pmid: int) -> None:
    """Plant a one-row success parquet (used to test resumability)."""
    row = {
        "pmid": pmid,
        "study_design": "RCT",
        "sample_size": 100,
        "has_control": True,
        "effect_direction": "positive",
        "statistical_claim_present": True,
        "coi_disclosed_in_abstract": False,
        "model_id": "claude-via-claude-code",
        "prompt_version": "v0.1",
        "raw_response": _VALID_LABEL_JSON,
        "extracted_at": "2026-01-01T00:00:00+00:00",
    }
    import pyarrow as pa

    table = pa.Table.from_pylist([row], schema=PILOT_SUCCESS_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Inline executor — runs submit() synchronously in-process.
# ---------------------------------------------------------------------------


class _InlineFuture:
    """A trivial future-shaped object whose ``.result()`` is precomputed."""

    def __init__(self, value: Any = None, exc: BaseException | None = None) -> None:
        self._value = value
        self._exc = exc

    def result(self) -> Any:  # noqa: D401 - shim
        if self._exc is not None:
            raise self._exc
        return self._value


class _InlineExecutor:
    """A :class:`ProcessPoolExecutor` stand-in that runs everything inline.

    Lets tests in this module patch
    :func:`scifield.epistemic.extract.subprocess.run` and have the patch
    actually take effect at extraction time — a real
    :class:`ProcessPoolExecutor` would fork worker processes that
    re-import the module and never see the parent's mock.
    """

    def __init__(self, max_workers: int = 1, **_: Any) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> _InlineFuture:
        try:
            value = fn(*args, **kwargs)
        except BaseException as e:  # pragma: no cover - exposed via .result()
            return _InlineFuture(exc=e)
        return _InlineFuture(value=value)


def _inline_as_completed(future_to_pmid: dict[_InlineFuture, int]) -> list[_InlineFuture]:
    """Iteration helper: yields the inline futures in insertion order."""
    return list(future_to_pmid.keys())


# ---------------------------------------------------------------------------
# (a) Resumability — skip PMIDs already in the success parquet.
# ---------------------------------------------------------------------------


def test_skips_already_extracted_pmids(tmp_path: Path) -> None:
    """``list_remaining_pmids`` excludes PMIDs in the existing success parquet."""
    duckdb_path = tmp_path / "papers.duckdb"
    pmids = _make_synth_duckdb(duckdb_path, n_rows=5)
    out_path = tmp_path / "epistemic_extracted.parquet"
    failed_path = tmp_path / "epistemic_failed.parquet"

    # Pre-populate the success parquet with PMID = pmids[0].
    _seed_success_parquet(out_path, pmid=pmids[0])

    cfg = BatchConfig(
        duckdb_path=duckdb_path,
        out_path=out_path,
        failed_path=failed_path,
    )
    remaining = list_remaining_pmids(cfg)
    remaining_pmids = [p for p, _ in remaining]
    assert pmids[0] not in remaining_pmids
    assert set(remaining_pmids) == set(pmids[1:])
    # Each entry carries an abstract long enough to clear the > 50 char
    # filter built into list_remaining_pmids.
    for _, abstract in remaining:
        assert len(abstract) > 50


# ---------------------------------------------------------------------------
# (b) Failed rows go to the failures parquet, not the success parquet.
# ---------------------------------------------------------------------------


def test_failed_rows_go_to_failure_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Half-success, half-garbage subprocess output -> clean split across parquets."""
    duckdb_path = tmp_path / "papers.duckdb"
    pmids = _make_synth_duckdb(duckdb_path, n_rows=4)
    out_path = tmp_path / "epistemic_extracted.parquet"
    failed_path = tmp_path / "epistemic_failed.parquet"

    # Alternate good / bad stdout. Each abstract triggers ``extract_one``
    # which (with retry enabled) issues up to TWO subprocess calls per
    # row. We disable retry so each abstract is exactly one subprocess
    # call, which keeps the side_effect bookkeeping straightforward.
    cfg = BatchConfig(
        duckdb_path=duckdb_path,
        out_path=out_path,
        failed_path=failed_path,
        concurrency=1,
        chunk_size=500,
        extract_cfg=ExtractConfig(retry_on_parse_failure=False),
    )

    # Deterministic order: list_remaining_pmids sorts ascending by pmid,
    # so pmids[0] -> ok, pmids[1] -> bad, pmids[2] -> ok, pmids[3] -> bad.
    claude_results = [
        _make_run_result(_VALID_LABEL_JSON),
        _make_run_result("not json at all"),
        _make_run_result(_VALID_LABEL_JSON),
        _make_run_result("still not json"),
    ]

    with (
        patch(
            "scifield.epistemic.batch.concurrent.futures.ProcessPoolExecutor",
            new=_InlineExecutor,
        ),
        patch(
            "scifield.epistemic.batch.concurrent.futures.as_completed",
            new=_inline_as_completed,
        ),
        patch(
            "scifield.epistemic.extract.subprocess.run",
            side_effect=_make_subprocess_router(claude_results),
        ),
    ):
        summary = run_batch(cfg)

    assert summary["n_attempted"] == 4
    assert summary["n_ok"] == 2
    assert summary["n_failed"] == 2

    ok_table = pq.read_table(out_path)
    failed_table = pq.read_table(failed_path)
    ok_pmids = set(int(p) for p in ok_table.column("pmid").to_pylist())
    failed_pmids = set(int(p) for p in failed_table.column("pmid").to_pylist())

    assert ok_pmids == {pmids[0], pmids[2]}
    assert failed_pmids == {pmids[1], pmids[3]}
    # Union covers every input PMID; no overlap between the two parquets.
    assert ok_pmids | failed_pmids == set(pmids)
    assert ok_pmids.isdisjoint(failed_pmids)


# ---------------------------------------------------------------------------
# (c) Sidecar JSON contains preregistration_url on both parquets.
# ---------------------------------------------------------------------------


def test_sidecar_contains_preregistration_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both sidecars carry ``config.preregistration_url`` = the OSF DOI."""
    duckdb_path = tmp_path / "papers.duckdb"
    _make_synth_duckdb(duckdb_path, n_rows=2)
    out_path = tmp_path / "epistemic_extracted.parquet"
    failed_path = tmp_path / "epistemic_failed.parquet"

    cfg = BatchConfig(
        duckdb_path=duckdb_path,
        out_path=out_path,
        failed_path=failed_path,
        concurrency=1,
        chunk_size=500,
    )

    with (
        patch(
            "scifield.epistemic.batch.concurrent.futures.ProcessPoolExecutor",
            new=_InlineExecutor,
        ),
        patch(
            "scifield.epistemic.batch.concurrent.futures.as_completed",
            new=_inline_as_completed,
        ),
        patch(
            "scifield.epistemic.extract.subprocess.run",
            side_effect=_make_subprocess_router([_make_run_result(_VALID_LABEL_JSON)] * 2),
        ),
    ):
        run_batch(cfg)

    success_sidecar = Path(str(out_path) + ".run.json")
    failed_sidecar = Path(str(failed_path) + ".run.json")
    assert success_sidecar.exists()
    assert failed_sidecar.exists()

    success_payload = json.loads(success_sidecar.read_text())
    failed_payload = json.loads(failed_sidecar.read_text())
    assert (
        success_payload["config"]["preregistration_url"] == "https://doi.org/10.17605/OSF.IO/8ZJHD"
    )
    assert (
        failed_payload["config"]["preregistration_url"] == "https://doi.org/10.17605/OSF.IO/8ZJHD"
    )


# ---------------------------------------------------------------------------
# (d) Concurrency parameter is honored at executor construction time.
# ---------------------------------------------------------------------------


def test_concurrency_param_honored(tmp_path: Path) -> None:
    """``ProcessPoolExecutor`` is constructed with ``max_workers=cfg.concurrency``."""
    duckdb_path = tmp_path / "papers.duckdb"
    _make_synth_duckdb(duckdb_path, n_rows=2)
    out_path = tmp_path / "epistemic_extracted.parquet"
    failed_path = tmp_path / "epistemic_failed.parquet"

    cfg = BatchConfig(
        duckdb_path=duckdb_path,
        out_path=out_path,
        failed_path=failed_path,
        concurrency=2,
        chunk_size=500,
    )

    # Build a MagicMock executor class. It must support both the
    # constructor + context-manager protocol AND `submit` returning a
    # future-shaped object so the run loop completes without exploding.
    fake_executor_instance = MagicMock()
    fake_executor_instance.__enter__ = MagicMock(return_value=fake_executor_instance)
    fake_executor_instance.__exit__ = MagicMock(return_value=None)

    def _fake_submit(fn: Any, *args: Any, **kwargs: Any) -> _InlineFuture:
        return _InlineFuture(value=fn(*args, **kwargs))

    fake_executor_instance.submit = MagicMock(side_effect=_fake_submit)

    fake_executor_cls = MagicMock(return_value=fake_executor_instance)

    with (
        patch(
            "scifield.epistemic.batch.concurrent.futures.ProcessPoolExecutor",
            new=fake_executor_cls,
        ),
        patch(
            "scifield.epistemic.batch.concurrent.futures.as_completed",
            new=_inline_as_completed,
        ),
        patch(
            "scifield.epistemic.extract.subprocess.run",
            side_effect=_make_subprocess_router([_make_run_result(_VALID_LABEL_JSON)] * 2),
        ),
    ):
        run_batch(cfg)

    # Assert ProcessPoolExecutor was constructed with max_workers=2.
    assert fake_executor_cls.called, "ProcessPoolExecutor was never constructed"
    _, kwargs = fake_executor_cls.call_args
    assert (
        kwargs.get("max_workers") == 2
    ), f"expected max_workers=2, got call_args={fake_executor_cls.call_args!r}"
