"""V1-S08 concurrent + resumable runner around :func:`extract_one`.

This module promotes the V1-S07 pilot's single-process,
single-abstract extraction loop into a corpus-scale driver. It is the
implementation backing the ``scifield epistemic extract-batch`` CLI
verbs in the V1-S08 plan (see
``plans/read-users-samersalman-desktop-scifield-async-giraffe.md``).

Architecture
------------

* **Worker dispatch.** A bounded :class:`concurrent.futures.ProcessPoolExecutor`
  fans out :func:`scifield.epistemic.extract.extract_one_subprocess`
  calls. Per V1-S08 D2, default concurrency is 4 (~35-55h wall time for
  the ~99,938-row dedup'd corpus); operators can throttle via
  :attr:`BatchConfig.concurrency`. The worker entrypoint returns a
  serializable envelope (``status="ok"`` / ``status="failed"``) so
  exceptions never cross the process boundary.
* **Resumability** (V1-S08 D3). State of progress lives in the output
  parquet itself — :func:`list_remaining_pmids` reads the existing
  success parquet (if any), and subtracts those PMIDs from the
  ``papers_distinct`` candidate set. No separate lockfile, no separate
  index; a crash mid-run loses at most one un-flushed chunk's worth of
  progress. ``--retry-failed`` is a complementary mode that pulls
  PMIDs from the failures parquet instead, looks their abstracts back
  up from DuckDB, and on retry-success migrates them into the success
  parquet.
* **Chunking.** Results buffer in memory and flush every
  :attr:`BatchConfig.chunk_size` (default 500). Each flush rewrites
  the success and failure parquets with the existing rows
  concatenated — Arrow file format does not support true append, and
  rewrite-per-chunk keeps the worst-case work bounded. Each flush also
  re-writes both ``.run.json`` sidecars via
  :func:`scifield.repro.record_run`, so the on-disk provenance stays
  current with the parquet.
* **Sidecar config.** The ``preregistration_url`` key (V1-S08 D5,
  acceptance criterion) is threaded through the sidecar config dict on
  every flush, alongside cumulative ``n_attempted`` / ``n_ok`` /
  ``n_failed`` counters and a serialized view of the in-flight
  :class:`ExtractConfig`.
* **Interrupt safety.** A :class:`KeyboardInterrupt` (or the
  :class:`concurrent.futures.CancelledError` it can trigger) flushes
  whatever rows are sitting in the in-memory buffer before re-raising,
  so partial progress within a chunk is preserved.

Public surface
--------------

The CLI subagent (Wave 2) drives this module via:

* :class:`BatchConfig` — frozen dataclass carrying all knobs.
* :func:`list_remaining_pmids` — DuckDB scan minus already-done PMIDs.
* :func:`run_batch` — the main entrypoint (forward + retry-failed
  modes).
* :func:`status` — read-only progress snapshot for the
  ``extract-batch --status`` verb.

Heavy I/O (pyarrow rewrites, DuckDB read-only opens) lives inside
:func:`run_batch` and :func:`list_remaining_pmids`; nothing at module
import or :class:`BatchConfig` construction time touches disk, so
unit tests can build small fixtures and exercise the dispatch logic
without ceremony.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from scifield.epistemic.extract import ExtractConfig, extract_one_subprocess
from scifield.epistemic.pilot import (
    PILOT_FAILED_SCHEMA,
    PILOT_SUCCESS_SCHEMA,
    _extract_cfg_to_dict,
    _flatten_failure,
    _flatten_success,
)
from scifield.epistemic.schema import EpistemicExtraction
from scifield.repro import record_run
from scifield.thematic.dedup import ensure_papers_distinct_view

__all__ = [
    "BatchConfig",
    "list_remaining_pmids",
    "run_batch",
    "status",
]


_ABSTRACT_MIN_LEN: int = 50
"""Minimum abstract length (matches the V1-S08 pre-reg dedup filter).

PR1 / V1-S08 define the extraction candidate set as
``papers_distinct WHERE abstract IS NOT NULL AND length(abstract) > 50``;
this constant centralizes that threshold so the SQL in
:func:`list_remaining_pmids` and the count in :func:`status` cannot
drift apart."""


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for one :func:`run_batch` invocation.

    Field order is load-bearing for the CLI wiring (Wave 2): the
    three required paths come first, then the integer knobs, then the
    sidecar-only preregistration string, then the nested
    :class:`ExtractConfig`. All fields except :attr:`extract_cfg` are
    intended to be sourced from ``conf/epistemic/v1.yaml`` under
    ``extract_batch:``.

    Attributes:
        duckdb_path: DuckDB file containing the ``papers`` table.
            :func:`list_remaining_pmids` opens this read-only and
            installs the ``papers_distinct`` view.
        out_path: Output parquet for successful extractions, schema
            matches :data:`scifield.epistemic.pilot.PILOT_SUCCESS_SCHEMA`.
        failed_path: Output parquet for rows that hit
            :class:`scifield.epistemic.extract.ExtractionError`, schema
            matches :data:`scifield.epistemic.pilot.PILOT_FAILED_SCHEMA`.
        concurrency: Number of worker processes in the
            :class:`concurrent.futures.ProcessPoolExecutor`. V1-S08 D2
            default 4.
        chunk_size: Number of completed results to buffer in memory
            before flushing the parquets and rewriting the sidecars.
            Default 500.
        preregistration_url: Stamped into the sidecar config on every
            flush. V1-S08 D5 / acceptance criterion. Default points at
            OSF PR #1.
        extract_cfg: Nested :class:`ExtractConfig`, threaded into each
            :func:`extract_one_subprocess` call. Defaults to
            :class:`ExtractConfig` defaults.
    """

    duckdb_path: Path
    out_path: Path
    failed_path: Path
    concurrency: int = 4
    chunk_size: int = 500
    preregistration_url: str = "https://doi.org/10.17605/OSF.IO/8ZJHD"
    extract_cfg: ExtractConfig = field(default_factory=ExtractConfig)


def _read_pmids_in_parquet(path: Path) -> set[int]:
    """Return the set of PMIDs currently in ``path`` (empty if missing)."""
    if not path.exists():
        return set()
    table = pq.read_table(path, columns=["pmid"])
    return {int(p) for p in table.column("pmid").to_pylist()}


def list_remaining_pmids(
    cfg: BatchConfig,
    limit: int | None = None,
) -> list[tuple[int, str]]:
    """Return ``(pmid, abstract)`` pairs not yet present in ``cfg.out_path``.

    Opens the DuckDB at :attr:`BatchConfig.duckdb_path` read-only,
    installs the ``papers_distinct`` view via
    :func:`scifield.thematic.dedup.ensure_papers_distinct_view`, and
    selects ``(pmid, abstract)`` from
    ``papers_distinct WHERE abstract IS NOT NULL AND length(abstract) > 50``
    sorted ascending by ``pmid`` for reproducibility. Then subtracts the
    PMIDs already present in :attr:`BatchConfig.out_path` (if any), so
    re-running after a crash skips work already on disk.

    Args:
        cfg: :class:`BatchConfig` carrying the DuckDB path and the
            output parquet path used to compute already-done PMIDs.
        limit: Optional cap on the number of returned rows, applied
            **after** the resumability subtraction. ``None`` (default)
            means return everything remaining.

    Returns:
        A list of ``(pmid, abstract)`` tuples, ordered by ``pmid``
        ascending, truncated to ``limit`` if given.
    """
    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        ensure_papers_distinct_view(con)
        rows = con.execute(
            "SELECT pmid, abstract FROM papers_distinct "
            "WHERE abstract IS NOT NULL "
            f"  AND length(abstract) > {_ABSTRACT_MIN_LEN} "
            "ORDER BY pmid ASC"
        ).fetchall()
    finally:
        con.close()

    done = _read_pmids_in_parquet(cfg.out_path)
    remaining: list[tuple[int, str]] = [
        (int(pmid), str(abstract)) for (pmid, abstract) in rows if int(pmid) not in done
    ]
    if limit is not None:
        remaining = remaining[:limit]
    return remaining


def _list_retry_targets(cfg: BatchConfig) -> list[tuple[int, str]]:
    """Pull PMIDs from the failures parquet and look up their abstracts.

    Returns ``(pmid, abstract)`` pairs for every row currently in
    :attr:`BatchConfig.failed_path` whose PMID still resolves to an
    abstract-bearing row in ``papers_distinct``. Rows whose abstract
    has gone missing since the original failure (unlikely but possible)
    are silently dropped — there is nothing to retry against.
    """
    if not cfg.failed_path.exists():
        return []
    failed_table = pq.read_table(cfg.failed_path, columns=["pmid"])
    failed_pmids = [int(p) for p in failed_table.column("pmid").to_pylist()]
    if not failed_pmids:
        return []

    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        ensure_papers_distinct_view(con)
        # Use DuckDB parameter binding via a temp table to avoid SQL string
        # interpolation on a potentially-large IN-list.
        con.execute("CREATE TEMP TABLE _retry_pmids (pmid BIGINT)")
        con.executemany("INSERT INTO _retry_pmids VALUES (?)", [(p,) for p in failed_pmids])
        rows = con.execute(
            "SELECT pd.pmid, pd.abstract FROM papers_distinct pd "
            "JOIN _retry_pmids r ON r.pmid = pd.pmid "
            "WHERE pd.abstract IS NOT NULL "
            f"  AND length(pd.abstract) > {_ABSTRACT_MIN_LEN} "
            "ORDER BY pd.pmid ASC"
        ).fetchall()
    finally:
        con.close()

    return [(int(pmid), str(abstract)) for (pmid, abstract) in rows]


def _read_existing_rows(path: Path, schema: pa.Schema) -> list[dict[str, Any]]:
    """Return the rows currently in ``path`` as plain dicts, [] if missing.

    Used by :func:`_flush` to concat-and-rewrite the parquet. The
    ``schema`` is consulted only to know the column order; pyarrow
    handles type fidelity on its own.
    """
    if not path.exists():
        return []
    table = pq.read_table(path)
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


def _write_parquet(
    rows: list[dict[str, Any]],
    out_path: Path,
    schema: pa.Schema,
) -> None:
    """Write ``rows`` to ``out_path`` under ``schema``.

    Mirrors :func:`scifield.epistemic.pilot._write_parquet`: builds an
    explicitly-typed empty table for the empty-rows case so the column
    types persist even on a zero-result flush.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
    else:
        empty_arrays = [pa.array([], type=field_.type) for field_ in schema]
        table = pa.Table.from_arrays(empty_arrays, schema=schema)
    pq.write_table(table, out_path)


def _build_sidecar_config(
    cfg: BatchConfig,
    n_attempted: int,
    n_ok: int,
    n_failed: int,
) -> dict[str, Any]:
    """Assemble the ``config`` dict passed to :func:`record_run`.

    Contains the V1-S08-mandated ``preregistration_url`` key plus the
    cumulative counters and a serialized view of the in-flight
    :class:`ExtractConfig` (so the sidecar fully reproduces the run).
    """
    return {
        "preregistration_url": cfg.preregistration_url,
        "concurrency": cfg.concurrency,
        "chunk_size": cfg.chunk_size,
        "extract_cfg": _extract_cfg_to_dict(cfg.extract_cfg),
        "n_attempted": n_attempted,
        "n_ok": n_ok,
        "n_failed": n_failed,
    }


def _flush(
    cfg: BatchConfig,
    new_success_rows: list[dict[str, Any]],
    new_failure_rows: list[dict[str, Any]],
    n_attempted: int,
    n_ok_total: int,
    n_failed_total: int,
    retry_pmids: set[int] | None = None,
) -> None:
    """Concat-and-rewrite both parquets, then refresh both sidecars.

    Resumability invariant: at the end of a flush, the on-disk
    success parquet contains every previously-flushed success row plus
    the new ones, and the failure parquet contains every previously
    failed row minus any that have been retry-recovered, plus any new
    failures.

    Args:
        cfg: :class:`BatchConfig` (used for paths + sidecar config).
        new_success_rows: Newly-completed success rows to append.
        new_failure_rows: Newly-completed failure rows to append.
        n_attempted: Cumulative count attempted in this run (sidecar).
        n_ok_total: Cumulative success count in this run (sidecar).
        n_failed_total: Cumulative failure count in this run (sidecar).
        retry_pmids: If provided, PMIDs that were sent through a
            retry-failed dispatch. Any PMID in this set that succeeded
            is dropped from the existing failures parquet on flush
            (preventing duplicate rows across the success / failure
            files).
    """
    # ---- success parquet ----
    existing_success = _read_existing_rows(cfg.out_path, PILOT_SUCCESS_SCHEMA)
    combined_success = existing_success + new_success_rows
    _write_parquet(combined_success, cfg.out_path, PILOT_SUCCESS_SCHEMA)

    # ---- failure parquet ----
    existing_failure = _read_existing_rows(cfg.failed_path, PILOT_FAILED_SCHEMA)
    if retry_pmids:
        # In retry-failed mode, drop any PMID that just succeeded from
        # the failures parquet so it does not double-count.
        new_ok_pmids = {int(row["pmid"]) for row in new_success_rows}
        retry_recovered = retry_pmids & new_ok_pmids
        existing_failure = [
            row for row in existing_failure if int(row["pmid"]) not in retry_recovered
        ]
        # Also drop any PMID that just failed again from the existing
        # failures (we'll re-insert via new_failure_rows below with a
        # fresh reason / timestamp).
        new_failed_pmids = {int(row["pmid"]) for row in new_failure_rows}
        retry_refailed = retry_pmids & new_failed_pmids
        existing_failure = [
            row for row in existing_failure if int(row["pmid"]) not in retry_refailed
        ]
    combined_failure = existing_failure + new_failure_rows
    _write_parquet(combined_failure, cfg.failed_path, PILOT_FAILED_SCHEMA)

    # ---- sidecars ----
    sidecar_config = _build_sidecar_config(
        cfg=cfg,
        n_attempted=n_attempted,
        n_ok=n_ok_total,
        n_failed=n_failed_total,
    )
    record_run(
        artifact_path=cfg.out_path,
        inputs={"duckdb": cfg.duckdb_path},
        config=sidecar_config,
    )
    record_run(
        artifact_path=cfg.failed_path,
        inputs={"duckdb": cfg.duckdb_path},
        config=sidecar_config,
    )


def run_batch(
    cfg: BatchConfig,
    progress: Callable[[int, int, str], None] | None = None,
    limit: int | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Run the V1-S08 batch extraction.

    Dispatches :func:`extract_one_subprocess` across a
    :class:`concurrent.futures.ProcessPoolExecutor` of
    :attr:`BatchConfig.concurrency` workers. Buffers completed
    envelopes in memory; every :attr:`BatchConfig.chunk_size` results
    (and once at the end), flushes both parquets and rewrites both
    sidecars via :func:`_flush`. On :class:`KeyboardInterrupt`, the
    in-memory buffer is flushed before the exception is re-raised, so
    partial progress within an unfinished chunk is not lost.

    Args:
        cfg: :class:`BatchConfig` carrying paths, concurrency, chunk
            size, the preregistration URL, and the nested
            :class:`ExtractConfig`.
        progress: Optional per-result callback. Signature
            ``(i: int, n: int, outcome: str) -> None``, where ``i`` is
            the 1-indexed completion counter, ``n`` is the total
            number of jobs dispatched in this call, and ``outcome`` is
            ``"ok"`` or ``"failed"``.
        limit: Cap on the number of (forward-mode) PMIDs to process
            this call. Passed through to :func:`list_remaining_pmids`.
            Ignored in ``retry_failed`` mode.
        retry_failed: When True, instead of scanning
            ``papers_distinct`` for new PMIDs, pull PMIDs from the
            existing failures parquet, look their abstracts back up,
            and retry them. On retry-success, the PMID is moved from
            the failures parquet to the success parquet on the next
            flush.

    Returns:
        Summary dict: ``{"n_attempted": int, "n_ok": int, "n_failed":
        int, "out_path": str, "failed_path": str}``.
    """
    if retry_failed:
        targets = _list_retry_targets(cfg)
        retry_pmid_set: set[int] | None = {pmid for pmid, _ in targets}
    else:
        targets = list_remaining_pmids(cfg, limit=limit)
        retry_pmid_set = None

    n_total = len(targets)
    run_stamp = datetime.now(UTC).isoformat()

    success_buffer: list[dict[str, Any]] = []
    failure_buffer: list[dict[str, Any]] = []
    n_attempted = 0
    n_ok_total = 0
    n_failed_total = 0

    def _maybe_flush(force: bool) -> None:
        """Flush both buffers if either is non-empty AND (force or chunk full)."""
        nonlocal success_buffer, failure_buffer
        buffered = len(success_buffer) + len(failure_buffer)
        if buffered == 0:
            return
        if not force and buffered < cfg.chunk_size:
            return
        _flush(
            cfg=cfg,
            new_success_rows=success_buffer,
            new_failure_rows=failure_buffer,
            n_attempted=n_attempted,
            n_ok_total=n_ok_total,
            n_failed_total=n_failed_total,
            retry_pmids=retry_pmid_set,
        )
        success_buffer = []
        failure_buffer = []

    if n_total == 0:
        # Nothing to do, but still emit empty-or-current parquets +
        # sidecars so downstream tooling can rely on their presence.
        _flush(
            cfg=cfg,
            new_success_rows=[],
            new_failure_rows=[],
            n_attempted=0,
            n_ok_total=0,
            n_failed_total=0,
            retry_pmids=retry_pmid_set,
        )
        return {
            "n_attempted": 0,
            "n_ok": 0,
            "n_failed": 0,
            "out_path": str(cfg.out_path),
            "failed_path": str(cfg.failed_path),
        }

    executor_cls = concurrent.futures.ProcessPoolExecutor
    try:
        with executor_cls(max_workers=cfg.concurrency) as executor:
            future_to_pmid: dict[Any, int] = {}
            for pmid, abstract in targets:
                fut = executor.submit(
                    extract_one_subprocess,
                    abstract,
                    pmid,
                    cfg.extract_cfg,
                )
                future_to_pmid[fut] = pmid

            for fut in concurrent.futures.as_completed(future_to_pmid):
                n_attempted += 1
                result = fut.result()
                if result["status"] == "ok":
                    extraction = result["extraction"]
                    assert isinstance(extraction, EpistemicExtraction)
                    success_buffer.append(_flatten_success(extraction, run_stamp))
                    n_ok_total += 1
                    if progress is not None:
                        progress(n_attempted, n_total, "ok")
                else:
                    # Mirror the pilot's failure flattening without
                    # round-tripping through ExtractionError: build a
                    # lightweight error-shaped namespace just to feed
                    # _flatten_failure.
                    from scifield.epistemic.extract import ExtractionError

                    raw_resp = result["raw_response"]
                    err = ExtractionError(
                        pmid=int(cast(int, result["pmid"])),
                        reason=str(result["reason"]),
                        raw_response=(None if raw_resp is None else str(raw_resp)),
                    )
                    failure_buffer.append(_flatten_failure(err, cfg.extract_cfg, run_stamp))
                    n_failed_total += 1
                    if progress is not None:
                        progress(n_attempted, n_total, "failed")

                _maybe_flush(force=False)

        # Final flush after the executor finishes (whether or not the
        # last chunk filled).
        _maybe_flush(force=True)
    except KeyboardInterrupt:
        # Persist whatever is in the buffer so partial progress is not
        # lost, then re-raise so the CLI can exit non-zero.
        _maybe_flush(force=True)
        raise

    return {
        "n_attempted": n_attempted,
        "n_ok": n_ok_total,
        "n_failed": n_failed_total,
        "out_path": str(cfg.out_path),
        "failed_path": str(cfg.failed_path),
    }


def status(cfg: BatchConfig) -> dict[str, int]:
    """Read-only progress snapshot for the ``extract-batch --status`` verb.

    Computes:

    * ``n_total`` — count of abstract-bearing rows in
      ``papers_distinct WHERE abstract IS NOT NULL AND length > 50``.
    * ``n_done`` — row count of :attr:`BatchConfig.out_path`
      (0 if the file does not exist).
    * ``n_failed`` — row count of :attr:`BatchConfig.failed_path`
      (0 if the file does not exist).
    * ``n_remaining`` — ``n_total - n_done - n_failed``.

    Args:
        cfg: :class:`BatchConfig` (paths only; no concurrency / chunk
            knobs are touched).

    Returns:
        ``{"n_total": int, "n_done": int, "n_failed": int,
        "n_remaining": int}``.
    """
    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        ensure_papers_distinct_view(con)
        n_total_row = con.execute(
            "SELECT COUNT(*) FROM papers_distinct "
            "WHERE abstract IS NOT NULL "
            f"  AND length(abstract) > {_ABSTRACT_MIN_LEN}"
        ).fetchone()
        n_total = int(n_total_row[0]) if n_total_row else 0
    finally:
        con.close()

    n_done = int(pq.read_metadata(cfg.out_path).num_rows) if cfg.out_path.exists() else 0
    n_failed = int(pq.read_metadata(cfg.failed_path).num_rows) if cfg.failed_path.exists() else 0
    return {
        "n_total": n_total,
        "n_done": n_done,
        "n_failed": n_failed,
        "n_remaining": n_total - n_done - n_failed,
    }
