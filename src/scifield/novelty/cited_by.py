"""Async OpenAlex incoming-citation harvester (V1-S10 novelty / CD-index).

For every focal corpus work (an OpenAlex id present in ``openalex_works``), this
module pages through the *forward* citations — the works that cite the focal —
via ``GET /works?filter=cites:<id>``. Each citer record carries its own inline
``referenced_works``; we classify locally whether the citer also references any
of the focal's own references (``cites_focal_ref``). That inline classification
is exactly the n_i / n_j split the Funk-Owen-Smith CD-index needs, so we avoid a
second harvest of the reference side.

Engineering mirrors :mod:`scifield.corpus.openalex`: an async httpx client on the
polite pool (``mailto``), tenacity exponential backoff, a global token-bucket
rate limiter, cursor pagination (``cursor=*``), an aggressive idempotent on-disk
gzip cache keyed by focal id, and a manifest that makes the harvest resumable.

This module only fetches + classifies + writes ``cited_by.parquet``. The CLI is
wired by the integration agent and is responsible for running the entrypoint
under the DeepSeek/OpenAlex spend gate. NOTE: callers must obtain explicit
go-ahead before any live full harvest.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles  # type: ignore[import-untyped]
import duckdb
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scifield.corpus.pubmed import RateLimiter

__all__ = ["CitedByConfig", "harvest_cited_by", "run_cited_by"]

logger = logging.getLogger(__name__)

_OPENALEX_WORK_PREFIX = "https://openalex.org/"
_PER_PAGE = 200  # OpenAlex cursor-paging max.
_AVG_BYTES_PER_ROW = 80  # Conservative on-disk parquet estimate per citer row.

# Output parquet schema — the cd_index corpus path depends on this contract.
CITED_BY_SCHEMA = pa.schema(
    [
        ("focal_oa_id", pa.string()),
        ("citing_oa_id", pa.string()),
        ("citing_year", pa.int32()),
        ("cites_focal_ref", pa.bool_()),
    ]
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitedByConfig:
    """Configuration for the incoming-citation harvester.

    Parameters
    ----------
    base_url
        OpenAlex API root, e.g. ``https://api.openalex.org``.
    cache_dir
        Directory for the per-focal gzipped JSON cache (idempotent / resumable).
    manifest_dir
        Directory holding ``cited_by_manifest.parquet`` (resume bookkeeping).
    mailto
        Polite-pool email; injected from ``$OPENALEX_EMAIL`` by the CLI.
    concurrency
        Max focal works harvested concurrently (socket / memory ceiling). The
        rate limiter is the real throttle on OpenAlex load.
    batch_size
        Reserved for symmetry with the corpus harvester; cursor paging uses
        ``_PER_PAGE`` (200) regardless. Bounds the in-memory row flush size.
    rate_limit
        Sustained requests/second across all workers (token bucket).
    request_timeout_s
        Per-request httpx timeout in seconds.
    max_retries
        Tenacity ``stop_after_attempt`` ceiling for transient HTTP errors.
    dry_run_sample_size
        Number of focal works sampled to project the reference-side cost.
    """

    base_url: str = "https://api.openalex.org"
    cache_dir: Path | None = None
    manifest_dir: Path | None = None
    mailto: str = ""
    # Optional OpenAlex API key (authenticated pool). Injected from
    # $OPENALEX_API_KEY by the CLI; sent as the ``api_key`` query param. The
    # key VALUE is never written to YAML or sidecars.
    api_key: str | None = None
    concurrency: int = 4
    batch_size: int = 200
    rate_limit: float = 8.0
    request_timeout_s: float = 60.0
    max_retries: int = 5
    dry_run_sample_size: int = 200
    # Only fetch citers within the CD window (focal_year, focal_year+N] via a
    # server-side publication-date filter. N must be >= the largest CD window
    # (CD_10 -> 10). Out-of-window citers are discarded by CD anyway, so this is
    # lossless for the metric and cuts volume sharply for old, heavily-cited
    # papers. Focals with unknown year (0) are not windowed.
    citer_window_years: int = 10
    # OR-batch up to this many same-year focals per `cites:W1|W2|...` query, then
    # attribute each citer back to the focal(s) it references. Collapses the
    # ~1-request-per-focal cost of the low-cited tail ~focal_batch_size x.
    # OpenAlex allows up to 50 OR'd values per filter.
    focal_batch_size: int = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_prefix(value: str | None, prefix: str = _OPENALEX_WORK_PREFIX) -> str:
    if not value:
        return ""
    value = value.strip()
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _cache_path(cache_dir: Path, focal_id: str) -> Path:
    # Shard on the first two chars after the leading "W" so any one directory
    # stays small across the ~118k-focal corpus.
    body = focal_id[1:] if focal_id.startswith("W") else focal_id
    shard = body[:2] if len(body) >= 2 else body.zfill(2)
    return cache_dir / shard / f"{focal_id}.json.gz"


async def _write_cache(cache_dir: Path, focal_id: str, rows: list[dict[str, Any]]) -> None:
    """Atomically persist the classified citer rows for one focal work."""
    path = _cache_path(cache_dir, focal_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(json.dumps(rows, ensure_ascii=False).encode("utf-8"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    async with aiofiles.open(tmp, "wb") as fh:
        await fh.write(payload)
    tmp.replace(path)


def _read_cache(cache_dir: Path, focal_id: str) -> list[dict[str, Any]] | None:
    """Synchronous read of cached citer rows; ``None`` if missing/corrupt."""
    path = _cache_path(cache_dir, focal_id)
    if not path.exists():
        return None
    try:
        raw = gzip.decompress(path.read_bytes())
        loaded: list[dict[str, Any]] = json.loads(raw)
        return loaded
    except (OSError, gzip.BadGzipFile, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# DuckDB loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FocalWork:
    """One focal corpus work: its OpenAlex id, references, year, and volume."""

    focal_oa_id: str
    ref_ids: frozenset[str]
    cited_by_count: int
    pub_year: int  # 0 if unknown -> citer query is not date-windowed


def _load_focal_works(duckdb_path: Path, limit: int | None = None) -> list[_FocalWork]:
    """Load focal works + their reference sets + reported citer volume.

    Reads ``openalex_works`` (``openalex_id``, ``pmid``, ``cited_by_count``) and
    joins ``references_out`` (``citing_pmid`` -> focal ``pmid``,
    ``ref_openalex_id``) to assemble each focal's own reference set. Works with
    an empty / missing ``openalex_id`` are dropped.

    Parameters
    ----------
    duckdb_path
        Path to ``papers.duckdb`` (V1-S03 + V1-S04 views attached).
    limit
        Optional cap on the number of focal works (for smoke tests).

    Returns
    -------
    list[_FocalWork]
        Deduplicated focal works ordered by descending ``cited_by_count`` so the
        heaviest (most informative for sizing) come first.
    """
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        works = conn.execute(
            """
            SELECT pmid, openalex_id, COALESCE(cited_by_count, 0) AS cited_by_count,
                   COALESCE(publication_year, 0) AS publication_year
            FROM openalex_works
            WHERE openalex_id IS NOT NULL AND openalex_id <> ''
            ORDER BY cited_by_count DESC
            """
        ).fetchall()
        refs = conn.execute(
            """
            SELECT r.citing_pmid, r.ref_openalex_id
            FROM references_out r
            WHERE r.ref_openalex_id IS NOT NULL AND r.ref_openalex_id <> ''
            """
        ).fetchall()
    finally:
        conn.close()

    refs_by_pmid: dict[str, set[str]] = {}
    for citing_pmid, ref_id in refs:
        if not citing_pmid:
            continue
        refs_by_pmid.setdefault(citing_pmid, set()).add(_strip_prefix(ref_id))

    out: list[_FocalWork] = []
    seen: set[str] = set()
    for pmid, oa_id, cited_by_count, publication_year in works:
        focal_oa_id = _strip_prefix(oa_id)
        if not focal_oa_id or focal_oa_id in seen:
            continue
        seen.add(focal_oa_id)
        out.append(
            _FocalWork(
                focal_oa_id=focal_oa_id,
                ref_ids=frozenset(refs_by_pmid.get(pmid, set())),
                cited_by_count=int(cited_by_count or 0),
                pub_year=int(publication_year or 0),
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


_MANIFEST_NAME = "cited_by_manifest.parquet"
_MANIFEST_SCHEMA = pa.schema(
    [
        ("focal_oa_id", pa.string()),
        ("fetched_at", pa.string()),
        ("status", pa.string()),
        ("n_citers", pa.int64()),
    ]
)


def _manifest_path(manifest_dir: Path) -> Path:
    return manifest_dir / _MANIFEST_NAME


def _read_manifest(manifest_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Return ``{focal_oa_id: {fetched_at, status, n_citers}}``."""
    if manifest_dir is None:
        return {}
    path = _manifest_path(manifest_dir)
    if not path.exists():
        return {}
    table = pq.read_table(path)
    ids = table.column("focal_oa_id").to_pylist()
    fetched = table.column("fetched_at").to_pylist()
    statuses = table.column("status").to_pylist()
    counts = table.column("n_citers").to_pylist()
    out: dict[str, dict[str, Any]] = {}
    for fid, fa, st, nc in zip(ids, fetched, statuses, counts, strict=True):
        if not fid:
            continue
        out[fid] = {"fetched_at": fa or "", "status": st or "", "n_citers": int(nc or 0)}
    return out


def _write_manifest(manifest_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    existing = _read_manifest(manifest_dir)
    for row in rows:
        existing[row["focal_oa_id"]] = {
            "fetched_at": row["fetched_at"],
            "status": row["status"],
            "n_citers": int(row["n_citers"]),
        }
    merged = [{"focal_oa_id": fid, **payload} for fid, payload in sorted(existing.items())]
    table = pa.Table.from_pylist(merged, schema=_MANIFEST_SCHEMA)
    path = _manifest_path(manifest_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def _write_cited_by(out_path: Path, rows: list[dict[str, Any]]) -> Path:
    """Atomically (re)write the full ``cited_by.parquet`` from ``rows``.

    Empty ``rows`` still materializes a schema-only Parquet so the cd_index
    agent always has a file to read.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=CITED_BY_SCHEMA)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class CitedByClient:
    """Async OpenAlex client — cursor-paged forward-citation harvest + cache."""

    def __init__(
        self,
        cfg: CitedByConfig,
        rate_limiter: RateLimiter,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not cfg.mailto:
            raise ValueError("mailto is required for the OpenAlex polite pool")
        self._cfg = cfg
        self._rate_limiter = rate_limiter
        self._client = client
        self._owns_client = client is None

    @property
    def config(self) -> CitedByConfig:
        return self._cfg

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"User-Agent": f"scifield (mailto:{self._cfg.mailto})"}
            self._client = httpx.AsyncClient(
                timeout=self._cfg.request_timeout_s,
                headers=headers,
            )
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _batch_filter(self, batch: list[_FocalWork]) -> str:
        """Build an OR'd ``cites:`` filter for a batch of focals sharing a window.

        Issues ``cites:W1|W2|...`` so one request returns the union of citers of
        every focal in the batch. All focals MUST share a publication year (or all
        be unknown-year) so a single ``(y, y+N]`` publication-date window applies;
        the caller guarantees this by grouping pending focals by year. Unknown-year
        batches (``pub_year == 0``) are fetched unwindowed for completeness.
        """
        ids = "|".join(f.focal_oa_id for f in batch)
        parts = [f"cites:{ids}"]
        n = self._cfg.citer_window_years
        y = batch[0].pub_year
        if y > 0 and n > 0:
            parts.append(f"from_publication_date:{y}-01-01")
            parts.append(f"to_publication_date:{y + n}-12-31")
        return ",".join(parts)

    async def _fetch_page(self, cites_filter: str, cursor: str) -> dict[str, Any]:
        """Fetch one cursor page of works matching ``cites_filter``."""

        @retry(
            stop=stop_after_attempt(self._cfg.max_retries),
            wait=wait_exponential(multiplier=2, max=20),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )
        async def _do() -> httpx.Response:
            # Acquire a rate-limiter token on EVERY attempt (including retries).
            # The token MUST be taken inside the retry body: taking it only once
            # before the retry let a 429 episode fire retried requests without
            # throttling, amplifying the send rate into a 429 storm.
            await self._rate_limiter.acquire()
            params = {
                "filter": cites_filter,
                "select": "id,publication_year,referenced_works",
                "per-page": str(_PER_PAGE),
                "cursor": cursor,
                "mailto": self._cfg.mailto,
            }
            if self._cfg.api_key:
                params["api_key"] = self._cfg.api_key
            client = self._ensure_client()
            resp = await client.get(f"{self._cfg.base_url}/works", params=params)
            if resp.status_code == 429:
                # Honour Retry-After (seconds) when present; else a polite pause.
                # Bounded so a malformed header can't stall a worker indefinitely.
                ra = resp.headers.get("Retry-After")
                try:
                    delay = float(ra) if ra is not None else 10.0
                except ValueError:
                    delay = 10.0
                await asyncio.sleep(min(max(delay, 1.0), 60.0))
            resp.raise_for_status()
            return resp

        resp = await _do()
        envelope: dict[str, Any] = resp.json()
        return envelope

    async def harvest_batch(self, batch: list[_FocalWork]) -> dict[str, list[dict[str, Any]]]:
        """Harvest forward citations for a batch of focals in one OR'd query.

        Pages through the union of the batch focals' citers and attributes each
        citer to the focals it references — read off the citer's own inline
        ``referenced_works`` (the citer was returned precisely because it cites at
        least one batch focal, so ``referenced_works`` ∩ batch-ids is exactly the
        focals it cites). For each (citer, focal) pair, ``cites_focal_ref`` is
        ``True`` iff the citer also references one of that focal's own references.

        Returns
        -------
        dict[str, list[dict]]
            ``{focal_oa_id: rows}`` for EVERY focal in the batch (empty list when a
            focal has no in-window citers), so callers can mark all of them done.
        """
        ref_by_focal = {f.focal_oa_id: f.ref_ids for f in batch}
        batch_ids = set(ref_by_focal)
        result: dict[str, list[dict[str, Any]]] = {fid: [] for fid in batch_ids}

        cites_filter = self._batch_filter(batch)
        cursor: str | None = "*"
        while cursor:
            envelope = await self._fetch_page(cites_filter, cursor)
            for work in envelope.get("results", []) or []:
                citing_oa_id = _strip_prefix(work.get("id"))
                if not citing_oa_id:
                    continue
                year_raw = work.get("publication_year")
                try:
                    citing_year = int(year_raw) if year_raw is not None else 0
                except (TypeError, ValueError):
                    citing_year = 0
                citer_refs = {_strip_prefix(r) for r in (work.get("referenced_works") or []) if r}
                for foc in citer_refs & batch_ids:
                    result[foc].append(
                        {
                            "focal_oa_id": foc,
                            "citing_oa_id": citing_oa_id,
                            "citing_year": citing_year,
                            "cites_focal_ref": bool(citer_refs & ref_by_focal[foc]),
                        }
                    )
            cursor = (envelope.get("meta") or {}).get("next_cursor")
        return result

    async def harvest_focal(self, focal: _FocalWork) -> list[dict[str, Any]]:
        """Harvest one focal's citers — thin wrapper over :meth:`harvest_batch`."""
        batched = await self.harvest_batch([focal])
        return batched[focal.focal_oa_id]


# ---------------------------------------------------------------------------
# Dry-run sizing
# ---------------------------------------------------------------------------


def _project_pages(citer_total: int) -> int:
    """Cursor pages needed for ``citer_total`` citers at ``_PER_PAGE`` each.

    One trailing page is always issued to observe an exhausted ``next_cursor``,
    so even a zero-citer focal costs one request.
    """
    return max(1, math.ceil(citer_total / _PER_PAGE))


def _dry_run_report(
    focals: list[_FocalWork],
    cfg: CitedByConfig,
    pending: list[_FocalWork],
) -> dict[str, Any]:
    """Project cost WITHOUT harvesting everything.

    The forward-citation volume is the sum of ``cited_by_count`` over pending
    focal works. The reference-side (n_R) cost is *estimated* by sampling
    ``dry_run_sample_size`` focal works and projecting their reference sets'
    contribution — here proxied by the mean references-per-focal scaled across
    the corpus, since the inline classification means we do NOT actually harvest
    the reference side (it is read off each citer's inline ``referenced_works``).
    """
    n_focal = len(focals)
    n_pending = len(pending)
    citer_total = sum(f.cited_by_count for f in pending)

    # One request per cursor page, summed over pending focals.
    total_pages = sum(_project_pages(f.cited_by_count) for f in pending)

    # Sampled reference-side projection: mean refs/focal over a sample, scaled.
    sample = pending[:: max(1, n_pending // cfg.dry_run_sample_size)] if n_pending else []
    sample = sample[: cfg.dry_run_sample_size]
    mean_refs = sum(len(f.ref_ids) for f in sample) / len(sample) if sample else 0.0
    projected_ref_side_volume = int(round(mean_refs * n_pending))

    # Wall-time is throttled by the global token bucket: pages / rate_limit.
    projected_wall_time_s = total_pages / cfg.rate_limit if cfg.rate_limit > 0 else 0.0

    # Disk: rows ~= citer_total, conservative bytes/row.
    projected_disk_bytes = citer_total * _AVG_BYTES_PER_ROW

    return {
        "mode": "dry_run",
        "n_focal_works": n_focal,
        "n_focal_pending": n_pending,
        "n_focal_already_done": n_focal - n_pending,
        "focal_citer_volume": citer_total,
        "dry_run_sample_size": len(sample),
        "mean_refs_per_focal_sampled": round(mean_refs, 2),
        "projected_ref_side_volume": projected_ref_side_volume,
        "total_pages": total_pages,
        "projected_wall_time_s": round(projected_wall_time_s, 1),
        "projected_wall_time_h": round(projected_wall_time_s / 3600.0, 2),
        "projected_disk_bytes": projected_disk_bytes,
        "projected_disk_mb": round(projected_disk_bytes / (1024 * 1024), 1),
        "rate_limit": cfg.rate_limit,
        "per_page": _PER_PAGE,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def harvest_cited_by(
    *,
    duckdb_path: Path,
    out_path: Path,
    cfg: CitedByConfig,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Harvest incoming (forward) citations for every focal corpus work.

    Single async entrypoint the CLI runs via ``asyncio.run(...)`` for BOTH the
    dry-run sizing pass and the full harvest. Resumable: focal works recorded
    ``status='ok'`` in the manifest (and present in the cache) are skipped and
    rehydrated from disk.

    Parameters
    ----------
    duckdb_path
        Path to ``papers.duckdb`` exposing ``openalex_works`` + ``references_out``.
    out_path
        Destination Parquet, e.g. ``data/v1/enrichment/cited_by.parquet``.
    cfg
        Harvester configuration.
    dry_run
        When ``True``, return a sizing report and make NO full harvest. (The
        orchestrator executes the dry-run under the spend gate; this function
        performs no live calls when ``dry_run`` is set.)
    limit
        Optional cap on focal works (smoke tests).

    Returns
    -------
    dict
        For ``dry_run=True`` the sizing report (see :func:`_dry_run_report`).
        For a full harvest a completion report with ``mode='harvest'`` plus row
        / focal counts and the output path.

    Notes
    -----
    The output Parquet schema is fixed by contract::

        focal_oa_id (str), citing_oa_id (str), citing_year (int), cites_focal_ref (bool)
    """
    focals = await asyncio.to_thread(_load_focal_works, duckdb_path, limit)
    manifest = _read_manifest(cfg.manifest_dir)

    done_ids = {fid for fid, m in manifest.items() if m.get("status") == "ok"}
    pending = [f for f in focals if f.focal_oa_id not in done_ids]

    if dry_run:
        report = _dry_run_report(focals, cfg, pending)
        logger.info(
            "cited_by dry-run: %d focal (%d pending), %d citers, %d pages, ~%.1fh",
            report["n_focal_works"],
            report["n_focal_pending"],
            report["focal_citer_volume"],
            report["total_pages"],
            report["projected_wall_time_h"],
        )
        return report

    started = time.monotonic()
    rate_limiter = RateLimiter(cfg.rate_limit)
    client = CitedByClient(cfg, rate_limiter)
    semaphore = asyncio.Semaphore(max(1, cfg.concurrency))

    all_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    # Rehydrate already-harvested focals from cache so the rewritten parquet
    # stays complete across resumes.
    if cfg.cache_dir is not None:
        for f in focals:
            if f.focal_oa_id not in done_ids:
                continue
            cached = _read_cache(cfg.cache_dir, f.focal_oa_id)
            if cached is None:
                pending.append(f)  # cache lost; re-harvest below.
                continue
            all_rows.extend(cached)

    # Circuit breaker + incremental checkpointing for a long, resumable run.
    # A sustained run of 429-exhausted focals means the OpenAlex polite-pool
    # daily cap (100k req/day) is reached; rather than thrash thousands of 429s
    # under the user's email we stop launching new work and end cleanly. The
    # manifest + parquet are flushed periodically so a manual kill (or the cap)
    # never loses more than CHECKPOINT_EVERY focals of progress.
    stop_event = asyncio.Event()
    flush_lock = asyncio.Lock()
    state = {"consecutive_429": 0, "since_checkpoint": 0}
    consecutive_429_limit = 6
    checkpoint_every = 500

    async def _checkpoint() -> None:
        async with flush_lock:
            if cfg.manifest_dir is not None and manifest_rows:
                await asyncio.to_thread(_write_manifest, cfg.manifest_dir, list(manifest_rows))
            await asyncio.to_thread(_write_cited_by, out_path, list(all_rows))

    def _make_batches(works: list[_FocalWork]) -> list[list[_FocalWork]]:
        # Group by publication year (so one date window applies per batch), then
        # chunk each year-group into OR-queries of <= focal_batch_size ids. This
        # collapses the ~1-request-per-focal cost of the low-cited tail by up to
        # focal_batch_size x.
        by_year: dict[int, list[_FocalWork]] = {}
        for w in works:
            by_year.setdefault(w.pub_year, []).append(w)
        size = max(1, cfg.focal_batch_size)
        batches: list[list[_FocalWork]] = []
        for year_works in by_year.values():
            for i in range(0, len(year_works), size):
                batches.append(year_works[i : i + size])
        return batches

    def _record(focal_id: str, status: str, n_citers: int) -> None:
        manifest_rows.append(
            {
                "focal_oa_id": focal_id,
                "fetched_at": _now_iso(),
                "status": status,
                "n_citers": n_citers,
            }
        )

    async def _batch_worker(batch: list[_FocalWork]) -> None:
        if stop_event.is_set():
            return
        async with semaphore:
            if stop_event.is_set():
                return
            try:
                by_focal = await client.harvest_batch(batch)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                logger.warning(
                    "cited_by batch failed (%d focals, e.g. %s): %s",
                    len(batch),
                    batch[0].focal_oa_id,
                    exc,
                )
                for f in batch:
                    _record(f.focal_oa_id, "error", 0)
                if status == 429:
                    state["consecutive_429"] += 1
                    if (
                        state["consecutive_429"] >= consecutive_429_limit
                        and not stop_event.is_set()
                    ):
                        stop_event.set()
                        logger.error(
                            "cited_by: %d consecutive 429 batches — OpenAlex quota likely "
                            "reached; stopping cleanly. Re-run to resume.",
                            state["consecutive_429"],
                        )
                return
            except httpx.HTTPError as exc:
                logger.warning(
                    "cited_by batch failed (%d focals, e.g. %s): %s",
                    len(batch),
                    batch[0].focal_oa_id,
                    exc,
                )
                for f in batch:
                    _record(f.focal_oa_id, "error", 0)
                return
            state["consecutive_429"] = 0
            for f in batch:
                rows = by_focal.get(f.focal_oa_id, [])
                if cfg.cache_dir is not None:
                    await _write_cache(cfg.cache_dir, f.focal_oa_id, rows)
                all_rows.extend(rows)
                _record(f.focal_oa_id, "ok", len(rows))
            state["since_checkpoint"] += len(batch)
            if state["since_checkpoint"] >= checkpoint_every:
                state["since_checkpoint"] = 0
                await _checkpoint()

    batches = _make_batches(pending)
    try:
        await asyncio.gather(*(_batch_worker(b) for b in batches))
    finally:
        await client.aclose()

    if cfg.manifest_dir is not None and manifest_rows:
        await asyncio.to_thread(_write_manifest, cfg.manifest_dir, manifest_rows)

    written = await asyncio.to_thread(_write_cited_by, out_path, all_rows)
    elapsed = time.monotonic() - started

    n_ok = sum(1 for r in manifest_rows if r["status"] == "ok")
    n_err = sum(1 for r in manifest_rows if r["status"] == "error")
    report = {
        "mode": "harvest",
        "out_path": str(written),
        "n_focal_works": len(focals),
        "n_focal_harvested": n_ok,
        "n_focal_errors": n_err,
        "n_focal_skipped": len(focals) - len(pending),
        "n_citer_rows": len(all_rows),
        "n_cites_focal_ref_true": sum(1 for r in all_rows if r["cites_focal_ref"]),
        "stopped_early": stop_event.is_set(),
        "elapsed_s": round(elapsed, 1),
    }
    logger.info(
        "cited_by harvest: %d rows over %d focal works in %.1fs -> %s",
        report["n_citer_rows"],
        report["n_focal_harvested"],
        report["elapsed_s"],
        report["out_path"],
    )
    return report


def run_cited_by(
    *,
    duckdb_path: Path,
    out_path: Path,
    cfg: CitedByConfig,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Synchronous convenience wrapper around :func:`harvest_cited_by`.

    Provided so the CLI may call either ``run_cited_by(...)`` directly or
    ``asyncio.run(harvest_cited_by(...))`` — both are equivalent.
    """
    return asyncio.run(
        harvest_cited_by(
            duckdb_path=duckdb_path,
            out_path=out_path,
            cfg=cfg,
            dry_run=dry_run,
            limit=limit,
        )
    )
