"""Async OpenAlex enrichment client (V1-S04).

Looks up PubMed PMIDs against OpenAlex via the batched filter endpoint, caches
raw JSON to disk (gzipped), and projects results into the row dicts consumed
by :mod:`scifield.corpus.enrich_store`. The orchestrator (V1-S04 step 8) is
responsible for downstream author disambiguation, ROR fill-in, and Parquet
writes; this module only fetches + parses.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles  # type: ignore[import-untyped]
import httpx
import pyarrow.parquet as pq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scifield.corpus.pubmed import RateLimiter

_OPENALEX_WORK_PREFIX = "https://openalex.org/"
_OPENALEX_PMID_PREFIX = "https://pubmed.ncbi.nlm.nih.gov/"
_DOI_PREFIX = "https://doi.org/"
_ORCID_PREFIX = "https://orcid.org/"
_TOP_N_CONCEPTS = 5


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OpenAlexConfig:
    email: str
    api_key: str | None = None
    base_url: str = "https://api.openalex.org"
    batch_size: int = 50
    rate_limit: float = 8.0
    request_timeout_s: float = 60.0
    max_retries: int = 5
    cache_dir: Path | None = None
    manifest_path: Path | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_prefix(value: str | None, prefix: str) -> str:
    if not value:
        return ""
    value = value.strip()
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


def _extract_pmid(work: dict[str, Any]) -> str:
    raw = (work.get("ids") or {}).get("pmid") or ""
    pmid = _strip_prefix(raw, _OPENALEX_PMID_PREFIX)
    return pmid.strip("/")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _cache_path(cache_dir: Path, pmid: str) -> Path:
    # Two-char shard keeps any one directory < ~few thousand files for 134k corpus.
    shard = pmid[:2] if len(pmid) >= 2 else pmid.zfill(2)
    return cache_dir / shard / f"{pmid}.json.gz"


async def _write_cache(cache_dir: Path, pmid: str, work: dict[str, Any]) -> None:
    path = _cache_path(cache_dir, pmid)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(json.dumps(work, ensure_ascii=False).encode("utf-8"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    async with aiofiles.open(tmp, "wb") as fh:
        await fh.write(payload)
    tmp.replace(path)


def _read_cache(cache_dir: Path, pmid: str) -> dict[str, Any] | None:
    """Synchronous read of a cached work JSON; returns None if missing/corrupt."""
    path = _cache_path(cache_dir, pmid)
    if not path.exists():
        return None
    try:
        raw = gzip.decompress(path.read_bytes())
        loaded: dict[str, Any] = json.loads(raw)
        return loaded
    except (OSError, gzip.BadGzipFile, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpenAlexClient:
    """Async OpenAlex client — batched PMID lookups + gzip cache."""

    def __init__(
        self,
        cfg: OpenAlexConfig,
        rate_limiter: RateLimiter,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not cfg.email:
            raise ValueError("OPENALEX_EMAIL is required for the polite pool")
        self._cfg = cfg
        self._rate_limiter = rate_limiter
        self._client = client
        self._owns_client = client is None

    @property
    def config(self) -> OpenAlexConfig:
        return self._cfg

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"User-Agent": f"scifield (mailto:{self._cfg.email})"}
            self._client = httpx.AsyncClient(
                timeout=self._cfg.request_timeout_s,
                headers=headers,
            )
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_batch(self, pmids: list[str]) -> dict[str, dict[str, Any] | None]:
        """Look up up to ``batch_size`` PMIDs in one OpenAlex filter call.

        Returns ``{pmid: work_dict_or_None}`` for every requested PMID. Caches
        each hit to ``cfg.cache_dir`` as gzipped JSON.
        """
        if not pmids:
            return {}
        if len(pmids) > self._cfg.batch_size:
            raise ValueError(
                f"fetch_batch received {len(pmids)} pmids; max is {self._cfg.batch_size}"
            )

        out: dict[str, dict[str, Any] | None] = {pmid: None for pmid in pmids}
        await self._rate_limiter.acquire()

        @retry(
            stop=stop_after_attempt(self._cfg.max_retries),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )
        async def _do() -> httpx.Response:
            params = {
                "filter": f"ids.pmid:{'|'.join(pmids)}",
                "per-page": str(self._cfg.batch_size),
                "mailto": self._cfg.email,
            }
            if self._cfg.api_key:
                params["api_key"] = self._cfg.api_key
            client = self._ensure_client()
            resp = await client.get(f"{self._cfg.base_url}/works", params=params)
            resp.raise_for_status()
            return resp

        resp = await _do()
        envelope = resp.json()
        for work in envelope.get("results", []) or []:
            pmid = _extract_pmid(work)
            if not pmid or pmid not in out:
                continue
            out[pmid] = work
            if self._cfg.cache_dir is not None:
                await _write_cache(self._cfg.cache_dir, pmid, work)
        return out

    async def fetch_pmid(self, pmid: str) -> dict[str, Any] | None:
        """Single-PMID fallback via ``/works/pmid:{pmid}``.

        Used for batch misses where we want to confirm a true 404 vs. a quirk
        of the batch filter (the OR-filter occasionally drops works whose
        pmid id is structured oddly).
        """
        await self._rate_limiter.acquire()

        @retry(
            stop=stop_after_attempt(self._cfg.max_retries),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )
        async def _do() -> httpx.Response | None:
            params = {"mailto": self._cfg.email}
            if self._cfg.api_key:
                params["api_key"] = self._cfg.api_key
            client = self._ensure_client()
            resp = await client.get(
                f"{self._cfg.base_url}/works/pmid:{pmid}",
                params=params,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp

        resp = await _do()
        if resp is None:
            return None
        work: dict[str, Any] = resp.json()
        if self._cfg.cache_dir is not None:
            await _write_cache(self._cfg.cache_dir, pmid, work)
        return work


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_openalex_work(pmid: str, work: dict[str, Any]) -> dict[str, Any]:
    """Project one OpenAlex work into an ``OPENALEX_WORKS`` row dict."""
    open_access = work.get("open_access") or {}
    concepts_raw = work.get("concepts") or []
    concepts_sorted = sorted(
        concepts_raw,
        key=lambda c: float(c.get("score") or 0.0),
        reverse=True,
    )[:_TOP_N_CONCEPTS]
    concepts = [
        {
            "id": _strip_prefix(c.get("id"), _OPENALEX_WORK_PREFIX),
            "display_name": (c.get("display_name") or "").strip(),
            "score": float(c.get("score") or 0.0),
        }
        for c in concepts_sorted
    ]

    pub_year_raw = work.get("publication_year")
    try:
        publication_year = int(pub_year_raw) if pub_year_raw is not None else 0
    except (TypeError, ValueError):
        publication_year = 0

    return {
        "pmid": pmid,
        "openalex_id": _strip_prefix(work.get("id"), _OPENALEX_WORK_PREFIX),
        "oa_doi": _strip_prefix(work.get("doi"), _DOI_PREFIX),
        "type": (work.get("type") or "").strip(),
        "language": (work.get("language") or "").strip(),
        "is_retracted": bool(work.get("is_retracted") or False),
        "is_oa": bool(open_access.get("is_oa") or False),
        "oa_status": (open_access.get("oa_status") or "").strip(),
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "concepts": concepts,
        "publication_year": publication_year,
        "publication_date": (work.get("publication_date") or "").strip(),
        "fetched_at": _now_iso(),
    }


def parse_references_out(citing_pmid: str, work: dict[str, Any]) -> list[dict[str, Any]]:
    """Emit one row per referenced_work. ``ref_position`` is 0-indexed."""
    rows: list[dict[str, Any]] = []
    refs = work.get("referenced_works") or []
    for i, ref in enumerate(refs):
        ref_id = _strip_prefix(ref, _OPENALEX_WORK_PREFIX)
        if not ref_id:
            continue
        rows.append(
            {
                "citing_pmid": citing_pmid,
                "ref_openalex_id": ref_id,
                "ref_pmid_if_known": "",
                "ref_year": 0,
                "ref_position": i,
            }
        )
    return rows


def parse_authorships_staging(pmid: str, work: dict[str, Any]) -> list[dict[str, Any]]:
    """Emit raw per-author rows for the disambiguation step in ``authors.py``.

    Positional index is the row's ordering in ``work['authorships']`` (0-indexed).
    OpenAlex also returns a ``author_position`` enum ("first"/"middle"/"last")
    that we surface as ``author_position_label`` so the downstream is_first /
    is_last booleans are unambiguous even when only one author exists.
    """
    rows: list[dict[str, Any]] = []
    for i, authorship in enumerate(work.get("authorships") or []):
        author = authorship.get("author") or {}
        institutions_raw = authorship.get("institutions") or []
        institutions = [
            {
                "oa_id": _strip_prefix(inst.get("id"), _OPENALEX_WORK_PREFIX),
                "ror_id": _strip_prefix(inst.get("ror"), "https://ror.org/"),
                "display_name": (inst.get("display_name") or "").strip(),
                "country_code": (inst.get("country_code") or "").strip(),
                "type": (inst.get("type") or "").strip(),
            }
            for inst in institutions_raw
        ]
        rows.append(
            {
                "pmid": pmid,
                "author_position": i,
                "author_position_label": (authorship.get("author_position") or "").strip(),
                "author_oa_id": _strip_prefix(author.get("id"), _OPENALEX_WORK_PREFIX),
                "author_orcid": _strip_prefix(author.get("orcid"), _ORCID_PREFIX),
                "author_display_name": (author.get("display_name") or "").strip(),
                "institutions": institutions,
                "raw_affiliation_strings": list(authorship.get("raw_affiliation_strings") or []),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Return ``{pmid: {fetched_at, status, openalex_id}}`` for the manifest."""
    if not path.exists():
        return {}
    table = pq.read_table(path)
    pmids = table.column("pmid").to_pylist()
    fetched = table.column("fetched_at").to_pylist()
    statuses = table.column("status").to_pylist()
    oa_ids = table.column("openalex_id").to_pylist()
    out: dict[str, dict[str, Any]] = {}
    for pmid, fa, st, oa_id in zip(pmids, fetched, statuses, oa_ids, strict=True):
        if not pmid:
            continue
        out[pmid] = {
            "fetched_at": fa or "",
            "status": st or "",
            "openalex_id": oa_id or "",
        }
    return out


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import pyarrow as pa

    existing = _read_manifest(path)
    for row in rows:
        existing[row["pmid"]] = {
            "fetched_at": row["fetched_at"],
            "status": row["status"],
            "openalex_id": row["openalex_id"],
        }
    merged = [{"pmid": pmid, **payload} for pmid, payload in sorted(existing.items())]
    schema = pa.schema(
        [
            ("pmid", pa.string()),
            ("fetched_at", pa.string()),
            ("status", pa.string()),
            ("openalex_id", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(merged, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _chunk(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


async def enrich_openalex(
    *,
    pmids: list[str],
    cfg: OpenAlexConfig,
    rate_limiter: RateLimiter,
) -> dict[str, Any]:
    """Drive batched OpenAlex fetches over ``pmids`` (manifest-aware).

    Skips PMIDs that already have ``status='ok'`` in the manifest. Returns
    the row dicts the orchestrator will hand to ``enrich_store`` and to
    ``authors.py`` / ``ror.py``.
    """
    manifest = _read_manifest(cfg.manifest_path) if cfg.manifest_path else {}
    todo: list[str] = []
    rehydrate: list[str] = []
    seen: set[str] = set()
    for pmid in pmids:
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        prior = manifest.get(pmid)
        if prior is not None and prior.get("status") == "ok":
            rehydrate.append(pmid)
            continue
        todo.append(pmid)

    client = OpenAlexClient(cfg, rate_limiter)
    works_rows: list[dict[str, Any]] = []
    refs_rows: list[dict[str, Any]] = []
    authors_staging: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    if rehydrate and cfg.cache_dir is not None:
        for pmid in rehydrate:
            cached = _read_cache(cfg.cache_dir, pmid)
            if cached is None:
                todo.append(pmid)
                continue
            work_row = parse_openalex_work(pmid, cached)
            works_rows.append(work_row)
            refs_rows.extend(parse_references_out(pmid, cached))
            authors_staging.extend(parse_authorships_staging(pmid, cached))

    try:
        for batch in _chunk(todo, cfg.batch_size):
            attempted_at = _now_iso()
            try:
                result = await client.fetch_batch(batch)
            except httpx.HTTPError as exc:
                for pmid in batch:
                    failed_rows.append(
                        {
                            "pmid": pmid,
                            "source": "openalex",
                            "reason": f"batch_http_error: {exc}",
                            "attempted_at": attempted_at,
                        }
                    )
                    manifest_rows.append(
                        {
                            "pmid": pmid,
                            "fetched_at": attempted_at,
                            "status": "error",
                            "openalex_id": "",
                        }
                    )
                continue

            misses = [pmid for pmid, work in result.items() if work is None]
            for pmid in misses:
                try:
                    work = await client.fetch_pmid(pmid)
                except httpx.HTTPError as exc:
                    failed_rows.append(
                        {
                            "pmid": pmid,
                            "source": "openalex",
                            "reason": f"single_http_error: {exc}",
                            "attempted_at": attempted_at,
                        }
                    )
                    manifest_rows.append(
                        {
                            "pmid": pmid,
                            "fetched_at": attempted_at,
                            "status": "error",
                            "openalex_id": "",
                        }
                    )
                    result.pop(pmid, None)
                    continue
                result[pmid] = work

            for pmid, work in result.items():
                if work is None:
                    failed_rows.append(
                        {
                            "pmid": pmid,
                            "source": "openalex",
                            "reason": "not_found",
                            "attempted_at": attempted_at,
                        }
                    )
                    manifest_rows.append(
                        {
                            "pmid": pmid,
                            "fetched_at": attempted_at,
                            "status": "not_found",
                            "openalex_id": "",
                        }
                    )
                    continue

                work_row = parse_openalex_work(pmid, work)
                works_rows.append(work_row)
                refs_rows.extend(parse_references_out(pmid, work))
                authors_staging.extend(parse_authorships_staging(pmid, work))
                manifest_rows.append(
                    {
                        "pmid": pmid,
                        "fetched_at": work_row["fetched_at"],
                        "status": "ok",
                        "openalex_id": work_row["openalex_id"],
                    }
                )
    finally:
        await client.aclose()

    if cfg.manifest_path is not None and manifest_rows:
        await asyncio.to_thread(_write_manifest, cfg.manifest_path, manifest_rows)

    return {
        "works": works_rows,
        "references": refs_rows,
        "authorships_staging": authors_staging,
        "failed": failed_rows,
        "manifest_rows": manifest_rows,
    }
