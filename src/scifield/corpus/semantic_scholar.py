"""Semantic Scholar enrichment client (V1-S04).

Default-skip behavior when no ``SEMANTIC_SCHOLAR_API_KEY`` is configured —
the orchestrator continues without SS and downstream notebooks render SS=0%
until the key arrives and a backfill ``--only semantic_scholar`` run is fired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scifield.corpus.pubmed import RateLimiter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SemanticScholarConfig:
    base_url: str = "https://api.semanticscholar.org/graph/v1"
    api_key: str | None = None
    batch_size: int = 100
    rate_limit: float = 8.0
    request_timeout_s: float = 60.0
    max_retries: int = 5


_SS_FIELDS = (
    "externalIds,citationCount,references.citedPaper.externalIds,"
    "references.intents,references.isInfluential"
)


class SemanticScholarClient:
    """Thin async wrapper around the Semantic Scholar graph API batch endpoint."""

    def __init__(
        self,
        cfg: SemanticScholarConfig,
        rate_limiter: RateLimiter,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._cfg = cfg
        self._rate_limiter = rate_limiter
        self._client = client
        self._owns_client = client is None

    @property
    def config(self) -> SemanticScholarConfig:
        return self._cfg

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._cfg.request_timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._cfg.api_key:
            headers["X-API-KEY"] = self._cfg.api_key
        return headers

    async def fetch_batch(self, pmids: list[str]) -> list[dict[str, Any] | None]:
        if not pmids:
            return []
        await self._rate_limiter.acquire()

        @retry(
            stop=stop_after_attempt(self._cfg.max_retries),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )
        async def _do() -> httpx.Response:
            client = self._ensure_client()
            resp = await client.post(
                f"{self._cfg.base_url}/paper/batch",
                params={"fields": _SS_FIELDS},
                json={"ids": [f"PMID:{p}" for p in pmids]},
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp

        resp = await _do()
        payload = resp.json()
        if not isinstance(payload, list):
            return [None] * len(pmids)
        return payload


def parse_ss_paper(
    pmid: str,
    paper: dict[str, Any] | None,
    fetched_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if paper is None:
        ss_row: dict[str, Any] = {
            "pmid": pmid,
            "ss_id": "",
            "ss_doi": "",
            "citation_count": 0,
            "references_with_intent_count": 0,
            "fetched_at": fetched_at,
        }
        return ss_row, []

    external_ids = paper.get("externalIds") or {}
    ss_id = paper.get("paperId") or ""
    ss_doi = (external_ids.get("DOI") or "") if isinstance(external_ids, dict) else ""
    citation_count = int(paper.get("citationCount") or 0)

    references = paper.get("references") or []
    intent_rows: list[dict[str, Any]] = []
    intent_count = 0
    for ref in references:
        if not isinstance(ref, dict):
            continue
        cited = ref.get("citedPaper") or {}
        cited_ext = cited.get("externalIds") or {} if isinstance(cited, dict) else {}
        cited_pmid = ""
        if isinstance(cited_ext, dict):
            raw_pmid = cited_ext.get("PubMed")
            if raw_pmid:
                cited_pmid = str(raw_pmid).strip()
        cited_ss_id = ""
        if isinstance(cited, dict):
            cited_ss_id = (cited.get("paperId") or "").strip()

        if cited_pmid:
            cited_id = f"PMID:{cited_pmid}"
        elif cited_ss_id:
            cited_id = f"SS:{cited_ss_id}"
        else:
            continue

        intents_raw = ref.get("intents") or []
        intents = [str(x) for x in intents_raw if x]
        if intents:
            intent_count += 1
        is_influential = bool(ref.get("isInfluential"))

        intent_rows.append(
            {
                "citing_pmid": pmid,
                "cited_id": cited_id,
                "intents": intents,
                "is_influential": is_influential,
            }
        )

    ss_row = {
        "pmid": pmid,
        "ss_id": ss_id,
        "ss_doi": ss_doi,
        "citation_count": citation_count,
        "references_with_intent_count": intent_count,
        "fetched_at": fetched_at,
    }
    return ss_row, intent_rows


async def enrich_semantic_scholar(
    *,
    pmids: list[str],
    cfg: SemanticScholarConfig,
    rate_limiter: RateLimiter,
) -> dict[str, Any]:
    if not cfg.api_key:
        logger.warning("SS skipped — set SEMANTIC_SCHOLAR_API_KEY to enable")
        return {"papers": [], "intents": [], "skipped": True, "failed": []}

    client = SemanticScholarClient(cfg, rate_limiter)
    papers: list[dict[str, Any]] = []
    intents: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    try:
        for i in range(0, len(pmids), cfg.batch_size):
            chunk = pmids[i : i + cfg.batch_size]
            attempted_at = datetime.now(UTC).isoformat()
            try:
                response = await client.fetch_batch(chunk)
            except httpx.HTTPError as exc:
                reason = f"{type(exc).__name__}: {exc}"
                for pmid in chunk:
                    failed.append(
                        {
                            "pmid": pmid,
                            "source": "semantic_scholar",
                            "reason": reason,
                            "attempted_at": attempted_at,
                        }
                    )
                continue

            fetched_at = datetime.now(UTC).isoformat()
            # Pad/truncate response defensively so we always align indices to chunk.
            if len(response) < len(chunk):
                response = list(response) + [None] * (len(chunk) - len(response))
            for pmid, paper in zip(chunk, response[: len(chunk)], strict=True):
                ss_row, intent_rows = parse_ss_paper(pmid, paper, fetched_at)
                papers.append(ss_row)
                intents.extend(intent_rows)
    finally:
        await client.aclose()

    return {"papers": papers, "intents": intents, "skipped": False, "failed": failed}
