"""ROR institution matcher (V1-S04 §6).

Only invoked for affiliation strings that OpenAlex could not resolve to a ROR
ID. Maintains a persistent on-disk cache (parquet) keyed by raw affiliation
string so that hits — and misses — are not re-queried across sessions.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scifield.corpus.pubmed import RateLimiter

_CACHE_SCHEMA = pa.schema(
    [
        ("raw_string", pa.string()),
        ("ror_id", pa.string()),
        ("ror_display_name", pa.string()),
        ("country_code", pa.string()),
        ("type", pa.string()),
        ("match_score", pa.float64()),
        ("fetched_at", pa.string()),
    ]
)


_WS_RE = re.compile(r"\s+")


def _normalize_raw(raw: str) -> str:
    folded = unicodedata.normalize("NFKC", raw).strip().lower()
    return _WS_RE.sub(" ", folded)


def _sha1_short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class RORConfig:
    base_url: str = "https://api.ror.org"
    rate_limit: float = 5.0
    request_timeout_s: float = 30.0
    max_retries: int = 4
    min_match_score: float = 0.85
    cache_path: Path | None = None


class RORMatcher:
    """Async affiliation → ROR matcher with persistent caching."""

    def __init__(
        self,
        cfg: RORConfig,
        rate_limiter: RateLimiter,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._cfg = cfg
        self._rate_limiter = rate_limiter
        self._client = client
        self._owns_client = client is None
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = asyncio.Lock()
        self._load_cache()

    @property
    def config(self) -> RORConfig:
        return self._cfg

    def _load_cache(self) -> None:
        path = self._cfg.cache_path
        if path is None or not Path(path).exists():
            return
        table = pq.read_table(path)
        for row in table.to_pylist():
            key = row.get("raw_string")
            if not key:
                continue
            self._cache[key] = {
                "ror_id": row.get("ror_id") or "",
                "ror_display_name": row.get("ror_display_name") or "",
                "country_code": row.get("country_code") or "",
                "type": row.get("type") or "",
                "match_score": float(row.get("match_score") or 0.0),
                "fetched_at": row.get("fetched_at") or "",
            }

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._cfg.request_timeout_s,
                headers={"User-Agent": "scifield"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def match(self, raw_string: str) -> dict[str, Any] | None:
        if not raw_string or not raw_string.strip():
            return None
        cached = self._cache.get(raw_string)
        if cached is not None:
            if cached["ror_id"] and cached["match_score"] >= self._cfg.min_match_score:
                return {
                    "ror_id": cached["ror_id"],
                    "display_name": cached["ror_display_name"],
                    "country_code": cached["country_code"],
                    "type": cached["type"],
                    "match_score": cached["match_score"],
                }
            return None

        result = await self._query(raw_string)
        async with self._cache_lock:
            self._cache[raw_string] = {
                "ror_id": result["ror_id"] if result else "",
                "ror_display_name": result["display_name"] if result else "",
                "country_code": result["country_code"] if result else "",
                "type": result["type"] if result else "",
                "match_score": result["match_score"] if result else 0.0,
                "fetched_at": datetime.now(UTC).isoformat(),
            }
        return result

    async def _query(self, raw_string: str) -> dict[str, Any] | None:
        await self._rate_limiter.acquire()

        @retry(
            stop=stop_after_attempt(self._cfg.max_retries),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )
        async def _do() -> httpx.Response:
            client = self._ensure_client()
            resp = await client.get(
                f"{self._cfg.base_url}/organizations",
                params={"affiliation": raw_string},
            )
            resp.raise_for_status()
            return resp

        try:
            resp = await _do()
        except httpx.HTTPError:
            return None

        try:
            payload = resp.json()
        except ValueError:
            return None

        items = payload.get("items") or []
        if not items:
            return None
        top = items[0]
        score = float(top.get("score") or 0.0)
        if score < self._cfg.min_match_score:
            return None
        org = top.get("organization") or {}
        ror_id = org.get("id") or ""
        if not ror_id:
            return None
        country = org.get("country") or {}
        types = org.get("types") or []
        return {
            "ror_id": ror_id,
            "display_name": org.get("name") or "",
            "country_code": country.get("country_code") or "",
            "type": types[0] if types else "",
            "match_score": score,
        }

    async def flush_cache(self) -> None:
        if self._cfg.cache_path is None:
            return
        async with self._cache_lock:
            snapshot = list(self._cache.items())
        rows: list[dict[str, Any]] = []
        for raw_string, entry in snapshot:
            rows.append(
                {
                    "raw_string": raw_string,
                    "ror_id": entry.get("ror_id", ""),
                    "ror_display_name": entry.get("ror_display_name", ""),
                    "country_code": entry.get("country_code", ""),
                    "type": entry.get("type", ""),
                    "match_score": float(entry.get("match_score", 0.0)),
                    "fetched_at": entry.get("fetched_at", ""),
                }
            )
        out_path = Path(self._cfg.cache_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=_CACHE_SCHEMA)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        pq.write_table(table, tmp_path)
        tmp_path.replace(out_path)


async def build_institution_tables(
    *,
    staging_rows: list[dict[str, Any]],
    matcher: RORMatcher,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk authorship staging rows → (institutions, paper_institutions)."""
    institutions: dict[str, dict[str, Any]] = {}
    paper_institutions: list[dict[str, Any]] = []
    min_score = matcher.config.min_match_score

    for row in staging_rows:
        pmid = row.get("pmid", "")
        author_position = int(row.get("author_position", 0))
        oa_institutions = row.get("institutions") or []
        raw_strings = row.get("raw_affiliation_strings") or []

        if oa_institutions:
            for inst in oa_institutions:
                oa_id = (inst.get("oa_id") or "").strip()
                ror_id = (inst.get("ror_id") or "").strip()
                display_name = (inst.get("display_name") or "").strip()
                country_code = (inst.get("country_code") or "").strip()
                inst_type = (inst.get("type") or "").strip()

                if oa_id:
                    canonical_id = f"OA:{oa_id}"
                    matched_by = "openalex"
                elif ror_id:
                    canonical_id = f"ROR:{ror_id}"
                    matched_by = "openalex"
                elif display_name:
                    canonical_id = f"RAW:{_sha1_short(_normalize_raw(display_name))}"
                    matched_by = "unmatched"
                else:
                    continue

                if canonical_id not in institutions:
                    institutions[canonical_id] = {
                        "institution_canonical_id": canonical_id,
                        "institution_oa_id": oa_id,
                        "ror_id": ror_id,
                        "display_name": display_name,
                        "country_code": country_code,
                        "type": inst_type,
                    }
                paper_institutions.append(
                    {
                        "pmid": pmid,
                        "author_position": author_position,
                        "institution_canonical_id": canonical_id,
                        "raw_affiliation_string": display_name,
                        "ror_matched_by": matched_by,
                    }
                )
            continue

        if not raw_strings:
            continue

        raw = (raw_strings[0] or "").strip()
        if not raw:
            continue

        match = await matcher.match(raw)
        if match is not None and match.get("match_score", 0.0) >= min_score and match.get("ror_id"):
            canonical_id = f"ROR:{match['ror_id']}"
            matched_by = "ror_api"
            if canonical_id not in institutions:
                institutions[canonical_id] = {
                    "institution_canonical_id": canonical_id,
                    "institution_oa_id": "",
                    "ror_id": match["ror_id"],
                    "display_name": match.get("display_name", ""),
                    "country_code": match.get("country_code", ""),
                    "type": match.get("type", ""),
                }
        else:
            canonical_id = f"RAW:{_sha1_short(_normalize_raw(raw))}"
            matched_by = "unmatched"
            if canonical_id not in institutions:
                institutions[canonical_id] = {
                    "institution_canonical_id": canonical_id,
                    "institution_oa_id": "",
                    "ror_id": "",
                    "display_name": raw,
                    "country_code": "",
                    "type": "",
                }

        paper_institutions.append(
            {
                "pmid": pmid,
                "author_position": author_position,
                "institution_canonical_id": canonical_id,
                "raw_affiliation_string": raw,
                "ror_matched_by": matched_by,
            }
        )

    return list(institutions.values()), paper_institutions
