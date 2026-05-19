"""Async PubMed harvester (V1-S03)."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scifield.corpus.store import (
    read_manifest,
    write_bucket_parquet,
    write_manifest,
)
from scifield.repro import record_run

# Bounded concurrency across (journal, year) buckets. The rate limiter is the
# *real* throttle on Entrez load; the semaphore just keeps memory + open
# sockets sane when N_journals × N_years is large.
MAX_CONCURRENT_BUCKETS = 4

# esearch hard cap from Entrez.
ESEARCH_MAX_HARD_CAP = 9999

_MONTH_NAMES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_YEAR_RE = re.compile(r"\b(\d{4})\b")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class JournalSpec:
    slug: str
    display: str
    ta_terms: list[str]


@dataclass(slots=True)
class EntrezConfig:
    email: str
    base_url: str
    api_key: str | None = None
    request_timeout_s: float = 60.0
    max_retries: int = 5


@dataclass(slots=True)
class HarvestConfig:
    batch_size: int = 200
    rate_limit: float = 3.0
    max_papers_per_bucket: int | None = None


@dataclass(slots=True)
class OutputConfig:
    parquet_dir: Path
    duckdb_path: Path
    manifest_dir: Path
    log_dir: Path


@dataclass(slots=True)
class BucketReport:
    slug: str
    year: int
    pmid_count: int
    parsed_count: int
    skipped: bool
    error: str | None = None


@dataclass(slots=True)
class HarvestReport:
    buckets: list[BucketReport] = field(default_factory=list)
    total_papers: int = 0
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Async token-bucket rate limiter — global across all workers.

    Refills tokens lazily based on wall-clock elapsed. If no token is available,
    sleeps until one would refill. An `asyncio.Lock` serializes bookkeeping so
    bursts of concurrent `acquire()` calls don't all see the same token.
    """

    def __init__(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(rate)
        self._capacity = max(1.0, float(rate))
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            # Sleep just long enough to refill one token.
            deficit = 1.0 - self._tokens
            wait_s = deficit / self._rate
            await asyncio.sleep(wait_s)
            # After sleeping, consume the freshly-refilled token.
            self._tokens = 0.0
            self._last_refill = time.monotonic()


# ---------------------------------------------------------------------------
# Entrez client
# ---------------------------------------------------------------------------


class EntrezClient:
    """Thin async wrapper around the NCBI E-utilities endpoints."""

    def __init__(
        self,
        cfg: EntrezConfig,
        rate_limiter: RateLimiter,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._cfg = cfg
        self._rate_limiter = rate_limiter
        self._client = client
        self._owns_client = client is None

    @property
    def config(self) -> EntrezConfig:
        return self._cfg

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._cfg.request_timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _common_params(self) -> dict[str, str]:
        params: dict[str, str] = {
            "db": "pubmed",
            "email": self._cfg.email,
            "tool": "scifield",
        }
        if self._cfg.api_key:
            params["api_key"] = self._cfg.api_key
        return params

    async def esearch(
        self,
        term: str,
        retstart: int = 0,
        retmax: int = 1000,
    ) -> tuple[list[str], int]:
        """Run an esearch query; return (pmids, total_count_reported_by_entrez)."""
        await self._rate_limiter.acquire()

        @retry(
            stop=stop_after_attempt(self._cfg.max_retries),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )
        async def _do() -> httpx.Response:
            params = self._common_params()
            params.update(
                {
                    "term": term,
                    "retstart": str(retstart),
                    "retmax": str(retmax),
                    "retmode": "xml",
                }
            )
            client = self._ensure_client()
            resp = await client.get(f"{self._cfg.base_url}/esearch.fcgi", params=params)
            resp.raise_for_status()
            return resp

        resp = await _do()
        root = ET.fromstring(resp.content)
        count_text = root.findtext("Count") or "0"
        try:
            total = int(count_text)
        except ValueError:
            total = 0
        pmids = [elem.text or "" for elem in root.findall("./IdList/Id")]
        pmids = [p for p in pmids if p]
        return pmids, total

    async def efetch(self, pmids: list[str]) -> bytes:
        """POST an id-list efetch; return raw XML bytes."""
        if not pmids:
            return b"<PubmedArticleSet></PubmedArticleSet>"
        await self._rate_limiter.acquire()

        @retry(
            stop=stop_after_attempt(self._cfg.max_retries),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )
        async def _do() -> httpx.Response:
            data = self._common_params()
            data.update(
                {
                    "id": ",".join(pmids),
                    "rettype": "xml",
                    "retmode": "xml",
                }
            )
            client = self._ensure_client()
            resp = await client.post(f"{self._cfg.base_url}/efetch.fcgi", data=data)
            resp.raise_for_status()
            return resp

        resp = await _do()
        return bytes(resp.content)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _text_or_empty(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def _parse_pub_date(article: ET.Element) -> tuple[int | None, str]:
    """Return (year, iso_pub_date) extracted from PubDate / MedlineDate."""
    pubdate = article.find("./Journal/JournalIssue/PubDate")
    if pubdate is None:
        return None, ""

    year_elem = pubdate.find("Year")
    if year_elem is not None and year_elem.text:
        try:
            year = int(year_elem.text.strip())
        except ValueError:
            year = None
        if year is None:
            return None, ""
        month_text = (pubdate.findtext("Month") or "").strip()
        day_text = (pubdate.findtext("Day") or "").strip()
        month: int | None = None
        if month_text:
            if month_text.isdigit():
                try:
                    month = int(month_text)
                except ValueError:
                    month = None
            else:
                month = _MONTH_NAMES.get(month_text[:3].lower())
        day: int | None = None
        if day_text and day_text.isdigit():
            try:
                day = int(day_text)
            except ValueError:
                day = None
        if month is not None and day is not None:
            iso = f"{year:04d}-{month:02d}-{day:02d}"
        elif month is not None:
            iso = f"{year:04d}-{month:02d}-01"
        else:
            iso = ""
        return year, iso

    medline_date = pubdate.findtext("MedlineDate")
    if medline_date:
        m = _YEAR_RE.search(medline_date)
        if m:
            try:
                return int(m.group(1)), ""
            except ValueError:
                return None, ""
    return None, ""


def _parse_authors(article: ET.Element) -> list[dict[str, str]]:
    authors: list[dict[str, str]] = []
    for author_elem in article.findall("./AuthorList/Author"):
        collective = author_elem.findtext("CollectiveName")
        if collective:
            authors.append(
                {
                    "last_name": collective.strip(),
                    "fore_name": "",
                    "initials": "",
                    "affiliation": "",
                }
            )
            continue
        first_aff = author_elem.find("./AffiliationInfo/Affiliation")
        authors.append(
            {
                "last_name": (author_elem.findtext("LastName") or "").strip(),
                "fore_name": (author_elem.findtext("ForeName") or "").strip(),
                "initials": (author_elem.findtext("Initials") or "").strip(),
                "affiliation": _text_or_empty(first_aff),
            }
        )
    return authors


def _parse_mesh(medline_citation: ET.Element) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for heading_elem in medline_citation.findall("./MeshHeadingList/MeshHeading"):
        descriptor_elem = heading_elem.find("DescriptorName")
        if descriptor_elem is None:
            continue
        qualifiers: list[dict[str, Any]] = []
        for qual_elem in heading_elem.findall("QualifierName"):
            qualifiers.append(
                {
                    "name": (qual_elem.text or "").strip(),
                    "ui": qual_elem.get("UI", "") or "",
                    "major_topic": qual_elem.get("MajorTopicYN", "N") == "Y",
                }
            )
        headings.append(
            {
                "descriptor": (descriptor_elem.text or "").strip(),
                "descriptor_ui": descriptor_elem.get("UI", "") or "",
                "major_topic": descriptor_elem.get("MajorTopicYN", "N") == "Y",
                "qualifiers": qualifiers,
            }
        )
    return headings


def _parse_one_article(
    pubmed_article: ET.Element,
    source_ta_match: str,
    fetched_at: str,
) -> dict[str, Any]:
    medline_citation = pubmed_article.find("MedlineCitation")
    if medline_citation is None:
        raise ValueError("missing MedlineCitation")
    article = medline_citation.find("Article")
    if article is None:
        raise ValueError("missing Article")

    pmid = (medline_citation.findtext("PMID") or "").strip()
    title = _text_or_empty(article.find("ArticleTitle"))

    segments: list[dict[str, str]] = []
    for abstract_text in article.findall("./Abstract/AbstractText"):
        segments.append(
            {
                "label": abstract_text.get("Label", "") or "",
                "nlm_category": abstract_text.get("NlmCategory", "") or "",
                "text": "".join(abstract_text.itertext()).strip(),
            }
        )
    abstract = " ".join(seg["text"] for seg in segments if seg["text"]).strip()

    journal = _text_or_empty(article.find("./Journal/Title"))
    journal_ta = (article.findtext("./Journal/ISOAbbreviation") or "").strip()
    if not journal_ta:
        journal_ta = (medline_citation.findtext("./MedlineJournalInfo/MedlineTA") or "").strip()

    year, pub_date = _parse_pub_date(article)

    doi = ""
    for eloc in article.findall("ELocationID"):
        if eloc.get("EIdType") == "doi" and eloc.text:
            doi = eloc.text.strip()
            break

    publication_types = [
        (pt.text or "").strip()
        for pt in article.findall("./PublicationTypeList/PublicationType")
        if (pt.text or "").strip()
    ]

    authors = _parse_authors(article)
    mesh_headings = _parse_mesh(medline_citation)

    return {
        "pmid": pmid,
        "journal_slug": "",
        "title": title,
        "abstract": abstract,
        "abstract_segments": segments,
        "journal": journal,
        "journal_ta": journal_ta,
        "year": year,
        "pub_date": pub_date,
        "doi": doi,
        "publication_types": publication_types,
        "authors": authors,
        "mesh_headings": mesh_headings,
        "has_abstract": bool(abstract),
        "fetched_at": fetched_at,
        "source_ta_match": source_ta_match,
    }


def parse_pubmed_articles(
    xml_bytes: bytes,
    source_ta_match: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Parse an efetch XML payload into (rows, failures).

    `failures` collects per-record parse errors so the harvester can route them
    to `<slug>/<year>.failed.jsonl` without losing the rest of the batch.
    """
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    fetched_at = datetime.now(UTC).isoformat()
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        failures.append({"raw_xml": "<root>", "error": f"ParseError: {exc}"})
        return rows, failures

    for pubmed_article in root.findall("PubmedArticle"):
        try:
            rows.append(_parse_one_article(pubmed_article, source_ta_match, fetched_at))
        except (ValueError, ET.ParseError, AttributeError) as exc:
            failures.append(
                {
                    "raw_xml": ET.tostring(pubmed_article, encoding="unicode")[:2000],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows, failures


# ---------------------------------------------------------------------------
# Harvest orchestration
# ---------------------------------------------------------------------------


async def _esearch_all(
    entrez: EntrezClient,
    term: str,
    retmax: int = 1000,
) -> list[str]:
    """Page through esearch until we've collected `total` PMIDs (capped at 9999)."""
    pmids: list[str] = []
    retstart = 0
    first_page, total = await entrez.esearch(term, retstart=retstart, retmax=retmax)
    pmids.extend(first_page)
    while len(pmids) < min(total, ESEARCH_MAX_HARD_CAP) and first_page:
        retstart += retmax
        if retstart >= ESEARCH_MAX_HARD_CAP:
            break
        page, _ = await entrez.esearch(term, retstart=retstart, retmax=retmax)
        if not page:
            break
        pmids.extend(page)
        first_page = page
    return pmids


async def _esearch_with_overflow(
    entrez: EntrezClient,
    ta_term: str,
    year: int,
) -> tuple[list[str], list[str]]:
    """Return (pmids, query_strings_used).

    If the full-year query reports >9999 results, fall back to per-month
    queries and union the result. This shouldn't fire for our 10 journals but
    is the safety branch.
    """
    base_query = f'"{ta_term}"[TA] AND {year}[PDAT]'
    queries: list[str] = []
    queries.append(base_query)
    _, total = await entrez.esearch(base_query, retstart=0, retmax=1)
    if total <= ESEARCH_MAX_HARD_CAP:
        pmids = await _esearch_all(entrez, base_query)
        return pmids, queries

    # Overflow: split by month.
    union: dict[str, None] = {}
    for month in range(1, 13):
        month_query = f'"{ta_term}"[TA] AND {year}/{month:02d}[PDAT]'
        queries.append(month_query)
        page = await _esearch_all(entrez, month_query)
        for pmid in page:
            union.setdefault(pmid, None)
    return list(union.keys()), queries


async def harvest_journal_year(
    *,
    slug: str,
    ta_terms: list[str],
    year: int,
    entrez: EntrezClient,
    batch_size: int,
    max_papers: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Harvest one (journal, year) bucket. Returns (rows, manifest_payload)."""
    pmids_by_ta: dict[str, list[str]] = {}
    source_ta_for_pmid: dict[str, str] = {}
    query_strings: list[str] = []
    for ta in ta_terms:
        ta_pmids, queries = await _esearch_with_overflow(entrez, ta, year)
        query_strings.extend(queries)
        pmids_by_ta[ta] = sorted(set(ta_pmids), key=_pmid_sort_key)
        for pmid in ta_pmids:
            source_ta_for_pmid.setdefault(pmid, ta)

    # Union PMIDs (numeric sort for stable, deterministic ordering).
    union_pmids = sorted(source_ta_for_pmid.keys(), key=_pmid_sort_key)
    esearch_count = len(union_pmids)

    if max_papers is not None:
        union_pmids = union_pmids[:max_papers]

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for i in range(0, len(union_pmids), batch_size):
        chunk = union_pmids[i : i + batch_size]
        xml_bytes = await entrez.efetch(chunk)
        chunk_rows, chunk_failures = parse_pubmed_articles(xml_bytes)
        for row in chunk_rows:
            row["journal_slug"] = slug
            row["source_ta_match"] = source_ta_for_pmid.get(row["pmid"], "")
        rows.extend(chunk_rows)
        failures.extend(chunk_failures)

    manifest_payload: dict[str, Any] = {
        "slug": slug,
        "year": year,
        "pmids": union_pmids,
        "pmids_by_ta": {ta: pmids_by_ta[ta] for ta in ta_terms},
        "query_terms": query_strings,
        "esearch_count": esearch_count,
        "parsed_count": len(rows),
        "failure_count": len(failures),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    # Stash failures on the payload so the orchestrator can route them.
    manifest_payload["_failures"] = failures
    return rows, manifest_payload


def _pmid_sort_key(pmid: str) -> tuple[int, str]:
    """Numeric sort if PMID is all-digits; lexical fallback for safety."""
    if pmid.isdigit():
        return (0, pmid.zfill(20))
    return (1, pmid)


def _write_failures(
    failures: list[dict[str, str]],
    parquet_dir: Path,
    slug: str,
    year: int,
) -> None:
    if not failures:
        return
    out_dir = parquet_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}.failed.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in failures:
            fh.write(json.dumps(entry) + "\n")


async def _run_bucket(
    *,
    slug: str,
    ta_terms: list[str],
    year: int,
    entrez: EntrezClient,
    harvest_cfg: HarvestConfig,
    output: OutputConfig,
    refresh: bool,
    semaphore: asyncio.Semaphore,
) -> BucketReport:
    async with semaphore:
        bucket_path = output.parquet_dir / slug / f"{year}.parquet"

        # Idempotency: cheap esearch first; if its PMID set equals the prior
        # manifest's PMID set AND the Parquet exists, skip the expensive efetch.
        if not refresh and bucket_path.exists():
            try:
                fresh_pmids_set: set[str] = set()
                for ta in ta_terms:
                    ta_pmids, _ = await _esearch_with_overflow(entrez, ta, year)
                    fresh_pmids_set.update(ta_pmids)
                prior = read_manifest(slug, year, output.manifest_dir)
                if prior is not None and set(prior.get("pmids", [])) == fresh_pmids_set:
                    return BucketReport(
                        slug=slug,
                        year=year,
                        pmid_count=len(fresh_pmids_set),
                        parsed_count=0,
                        skipped=True,
                    )
            except httpx.HTTPError as exc:
                return BucketReport(
                    slug=slug,
                    year=year,
                    pmid_count=0,
                    parsed_count=0,
                    skipped=False,
                    error=f"esearch-idempotency-check failed: {exc}",
                )

        try:
            rows, manifest_payload = await harvest_journal_year(
                slug=slug,
                ta_terms=ta_terms,
                year=year,
                entrez=entrez,
                batch_size=harvest_cfg.batch_size,
                max_papers=harvest_cfg.max_papers_per_bucket,
            )
        except httpx.HTTPError as exc:
            return BucketReport(
                slug=slug,
                year=year,
                pmid_count=0,
                parsed_count=0,
                skipped=False,
                error=f"harvest failed: {exc}",
            )

        failures = manifest_payload.pop("_failures", [])
        _write_failures(failures, output.parquet_dir, slug, year)

        bucket_path = write_bucket_parquet(rows, slug, year, output.parquet_dir)
        write_manifest(output.manifest_dir, slug, year, manifest_payload)
        record_run(
            artifact_path=bucket_path,
            inputs={},
            config={
                "journal_slug": slug,
                "year": year,
                "ta_terms": list(ta_terms),
            },
        )

        return BucketReport(
            slug=slug,
            year=year,
            pmid_count=manifest_payload["esearch_count"],
            parsed_count=manifest_payload["parsed_count"],
            skipped=False,
        )


async def harvest_corpus(
    *,
    journals: list[JournalSpec],
    year_range: tuple[int, int],
    entrez: EntrezClient,
    output: OutputConfig,
    harvest_cfg: HarvestConfig,
    refresh: bool = False,
    only_journal: str | None = None,
    only_year: int | None = None,
) -> HarvestReport:
    """Drive harvest across all (journal, year) buckets and return a summary."""
    start = time.monotonic()
    y0, y1 = year_range
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BUCKETS)

    tasks: list[asyncio.Task[BucketReport]] = []
    for journal in journals:
        if only_journal is not None and journal.slug != only_journal:
            continue
        for year in range(y0, y1 + 1):
            if only_year is not None and year != only_year:
                continue
            tasks.append(
                asyncio.create_task(
                    _run_bucket(
                        slug=journal.slug,
                        ta_terms=list(journal.ta_terms),
                        year=year,
                        entrez=entrez,
                        harvest_cfg=harvest_cfg,
                        output=output,
                        refresh=refresh,
                        semaphore=semaphore,
                    )
                )
            )

    buckets: list[BucketReport] = []
    if tasks:
        buckets = list(await asyncio.gather(*tasks))

    total_papers = sum(b.parsed_count for b in buckets)
    elapsed = time.monotonic() - start
    return HarvestReport(buckets=buckets, total_papers=total_papers, elapsed_s=elapsed)
