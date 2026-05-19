"""Tests for the async PubMed harvester / parser (V1-S03 Task 7)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from scifield.corpus import (
    EntrezClient,
    EntrezConfig,
    JournalSpec,
    OutputConfig,
    RateLimiter,
    harvest_corpus,
    harvest_journal_year,
    parse_pubmed_articles,
    write_bucket_parquet,
    write_manifest,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pubmed_xml"
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _minimal_esearch_xml(pmids: list[str]) -> bytes:
    """Build a minimal esearch XML response carrying the given PMIDs."""
    id_xml = "".join(f"<Id>{pmid}</Id>" for pmid in pmids)
    return (
        f'<?xml version="1.0"?>'
        f"<eSearchResult>"
        f"<Count>{len(pmids)}</Count>"
        f"<RetMax>{len(pmids)}</RetMax>"
        f"<RetStart>0</RetStart>"
        f"<IdList>{id_xml}</IdList>"
        f"</eSearchResult>"
    ).encode()


def _read_fixture(name: str) -> bytes:
    return (FIXTURE_DIR / name).read_bytes()


def _make_entrez() -> EntrezClient:
    cfg = EntrezConfig(
        email="test@example.com",
        base_url=BASE_URL,
        api_key=None,
        request_timeout_s=10.0,
        max_retries=1,
    )
    return EntrezClient(cfg, rate_limiter=RateLimiter(rate=1000.0))


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


def test_parse_record_full() -> None:
    xml_bytes = _read_fixture("efetch_arthroscopy_2024.xml")
    rows, failures = parse_pubmed_articles(xml_bytes, source_ta_match="Arthroscopy")
    assert failures == []

    by_pmid = {row["pmid"]: row for row in rows}
    row = by_pmid["40000001"]

    assert "Anterior cruciate ligament" in row["title"]
    assert row["abstract"]
    assert row["has_abstract"] is True
    assert row["journal"]
    assert row["journal_ta"] == "Arthroscopy"
    assert row["year"] == 2024
    assert row["pub_date"] == "2024-03-15"
    assert row["doi"] == "10.1016/j.arthro.2024.01.001"
    assert row["publication_types"] == [
        "Journal Article",
        "Research Support, Non-U.S. Gov't",
    ]

    authors = row["authors"]
    assert len(authors) == 2
    assert authors[0]["last_name"] == "Smith"
    assert authors[0]["fore_name"] == "Jane A"
    assert authors[0]["initials"]

    mesh = row["mesh_headings"]
    assert len(mesh) == 2
    first = mesh[0]
    assert first["descriptor"] == "Knee Joint"
    assert first["descriptor_ui"] == "D007717"
    assert first["major_topic"] is True
    qualifiers = first["qualifiers"]
    assert any(q["name"] == "surgery" and q["major_topic"] is False for q in qualifiers)

    assert row["source_ta_match"] == "Arthroscopy"
    assert row["fetched_at"]
    assert len(row["abstract_segments"]) == 1


def test_parse_record_no_abstract() -> None:
    xml_bytes = _read_fixture("efetch_arthroscopy_2024.xml")
    rows, _ = parse_pubmed_articles(xml_bytes)
    row = next(r for r in rows if r["pmid"] == "40000002")

    assert row["has_abstract"] is False
    assert row["abstract"] == ""
    assert row["mesh_headings"] == []
    assert row["publication_types"] == ["Letter"]


def test_parse_record_structured_abstract() -> None:
    xml_bytes = _read_fixture("efetch_arthroscopy_2024.xml")
    rows, _ = parse_pubmed_articles(xml_bytes)
    row = next(r for r in rows if r["pmid"] == "40000003")

    segments = row["abstract_segments"]
    assert len(segments) == 4
    labels = [s["label"] for s in segments]
    assert labels == ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"]
    for seg in segments:
        assert seg["text"] in row["abstract"]
    assert row["mesh_headings"] == []


def test_parse_record_arch_surg_legacy() -> None:
    xml_bytes = _read_fixture("efetch_arthroscopy_2024.xml")
    rows, _ = parse_pubmed_articles(xml_bytes)
    row = next(r for r in rows if r["pmid"] == "40000004")

    assert row["journal_ta"] == "Arch Surg"
    assert row["year"] == 1998


def test_parse_record_medline_date_fallback() -> None:
    """Sanity check for the MedlineDate '1995-1996' → 1995 fallback path."""
    xml_bytes = _read_fixture("efetch_arthroscopy_2024.xml")
    rows, _ = parse_pubmed_articles(xml_bytes)
    row = next(r for r in rows if r["pmid"] == "40000005")
    assert row["year"] == 1995


# ---------------------------------------------------------------------------
# Harvest tests (respx-mocked HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_harvest_journal_year_dedup_dual_ta() -> None:
    """Overlapping PMID lists across two TA terms should be deduped into a
    single row set; per-TA manifests must reflect the raw per-TA results."""
    entrez = _make_entrez()

    # Build a 5-article PubmedArticleSet by stitching tiny inline articles.
    def _article(pmid: str, ta: str = "JAMA Surg") -> str:
        return (
            f"<PubmedArticle>"
            f"<MedlineCitation>"
            f'<PMID Version="1">{pmid}</PMID>'
            f"<Article>"
            f"<Journal>"
            f"<JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>"
            f"<Title>JAMA Surgery</Title>"
            f"<ISOAbbreviation>{ta}</ISOAbbreviation>"
            f"</Journal>"
            f"<ArticleTitle>Paper {pmid}</ArticleTitle>"
            f"<Abstract><AbstractText>Body {pmid}.</AbstractText></Abstract>"
            f"<AuthorList></AuthorList>"
            f"<PublicationTypeList>"
            f"<PublicationType>Journal Article</PublicationType>"
            f"</PublicationTypeList>"
            f"</Article>"
            f"<MedlineJournalInfo><MedlineTA>{ta}</MedlineTA></MedlineJournalInfo>"
            f"</MedlineCitation>"
            f"</PubmedArticle>"
        )

    articles_xml = "".join(_article(str(p)) for p in (1, 2, 3, 4, 5))
    efetch_body = (
        f'<?xml version="1.0"?>' f"<PubmedArticleSet>{articles_xml}</PubmedArticleSet>"
    ).encode()

    with respx.mock(assert_all_called=False) as router:

        def _esearch_side_effect(request: httpx.Request) -> httpx.Response:
            term = request.url.params.get("term", "")
            if "JAMA Surg" in term:
                return httpx.Response(200, content=_minimal_esearch_xml(["3", "4", "5"]))
            if "Arch Surg" in term:
                return httpx.Response(200, content=_minimal_esearch_xml(["1", "2", "3"]))
            return httpx.Response(200, content=_minimal_esearch_xml([]))

        router.get(url__regex=r".*/esearch\.fcgi.*").mock(side_effect=_esearch_side_effect)
        router.post(url__regex=r".*/efetch\.fcgi.*").mock(
            return_value=httpx.Response(200, content=efetch_body)
        )

        try:
            rows, manifest = await harvest_journal_year(
                slug="jama_surg",
                ta_terms=["JAMA Surg", "Arch Surg"],
                year=2024,
                entrez=entrez,
                batch_size=200,
            )
        finally:
            await entrez.aclose()

    pmids = [r["pmid"] for r in rows]
    assert len(rows) == 5
    assert len(set(pmids)) == 5
    assert set(pmids) == {"1", "2", "3", "4", "5"}
    assert set(manifest["pmids_by_ta"]["JAMA Surg"]) == {"3", "4", "5"}
    assert set(manifest["pmids_by_ta"]["Arch Surg"]) == {"1", "2", "3"}


@pytest.mark.asyncio
async def test_idempotent_skip_when_manifest_matches(tmp_path: Path) -> None:
    """If the Parquet exists and the manifest's PMID set equals the live esearch
    result, the bucket must be skipped and efetch must NEVER be called."""
    parquet_dir = tmp_path / "parquet"
    manifest_dir = tmp_path / "manifests"
    duckdb_path = tmp_path / "papers.duckdb"
    log_dir = tmp_path / "logs"

    pmids = ["40000001", "40000002", "40000003", "40000004", "40000005"]

    # Seed: schema-only Parquet + manifest with the same PMIDs that esearch
    # will return below.
    write_bucket_parquet([], slug="arthroscopy", year=2024, parquet_dir=parquet_dir)
    write_manifest(
        manifest_dir,
        slug="arthroscopy",
        year=2024,
        payload={
            "slug": "arthroscopy",
            "year": 2024,
            "pmids": pmids,
            "pmids_by_ta": {"Arthroscopy": pmids},
            "query_terms": ['"Arthroscopy"[TA] AND 2024[PDAT]'],
            "esearch_count": len(pmids),
            "parsed_count": len(pmids),
            "failure_count": 0,
        },
    )

    entrez = _make_entrez()
    output = OutputConfig(
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        manifest_dir=manifest_dir,
        log_dir=log_dir,
    )

    from scifield.corpus import HarvestConfig

    harvest_cfg = HarvestConfig(batch_size=200, rate_limit=1000.0)

    with respx.mock(assert_all_called=False) as router:
        esearch_route = router.get(url__regex=r".*/esearch\.fcgi.*").mock(
            return_value=httpx.Response(200, content=_minimal_esearch_xml(pmids))
        )
        efetch_route = router.post(url__regex=r".*/efetch\.fcgi.*").mock(
            return_value=httpx.Response(200, content=b"<PubmedArticleSet/>")
        )

        try:
            report = await harvest_corpus(
                journals=[
                    JournalSpec(
                        slug="arthroscopy",
                        display="Arthroscopy",
                        ta_terms=["Arthroscopy"],
                    )
                ],
                year_range=(2024, 2024),
                entrez=entrez,
                output=output,
                harvest_cfg=harvest_cfg,
            )
        finally:
            await entrez.aclose()

        # Bucket should have been skipped.
        assert len(report.buckets) == 1
        bucket = report.buckets[0]
        assert bucket.slug == "arthroscopy"
        assert bucket.year == 2024
        assert bucket.skipped is True
        # efetch must never have been touched on the idempotent path.
        assert efetch_route.call_count == 0
        # esearch was the only thing called (used for the idempotency probe).
        assert esearch_route.called
