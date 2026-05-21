"""Tests for `scifield.corpus.openalex` parsers + respx-mocked batch fetch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import respx

from scifield.corpus.openalex import (
    OpenAlexClient,
    OpenAlexConfig,
    enrich_openalex,
    parse_authorships_staging,
    parse_openalex_work,
    parse_references_out,
)
from scifield.corpus.pubmed import RateLimiter

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "openalex" / "sample_work.json"


def _load_fixture() -> dict[Any, Any]:
    return cast(dict[Any, Any], json.loads(FIXTURE_PATH.read_text()))


def _fixture_works() -> list[dict[Any, Any]]:
    return cast(list[dict[Any, Any]], _load_fixture()["results"])


# ---------------------------------------------------------------------------
# Config / client
# ---------------------------------------------------------------------------


def test_config_rejects_empty_email() -> None:
    cfg = OpenAlexConfig(email="")
    with pytest.raises(ValueError):
        OpenAlexClient(cfg, RateLimiter(1.0))


# ---------------------------------------------------------------------------
# parse_openalex_work
# ---------------------------------------------------------------------------


def test_parse_openalex_work_full() -> None:
    work = _fixture_works()[0]
    row = parse_openalex_work("40000001", work)

    assert row["pmid"] == "40000001"
    assert row["openalex_id"] == "W4000000001"
    # DOI must be stripped of the https://doi.org/ prefix and start with "10."
    assert row["oa_doi"].startswith("10.")
    assert not row["oa_doi"].startswith("https://")
    assert row["is_oa"] is True
    assert row["oa_status"] == "green"
    assert row["publication_year"] == 2024
    assert row["type"] == "article"
    assert row["cited_by_count"] == 12

    concepts = row["concepts"]
    assert isinstance(concepts, list)
    assert len(concepts) <= 5
    assert len(concepts) == 2
    for c in concepts:
        assert c["id"].startswith("C")
        assert not c["id"].startswith("https://")

    assert row["fetched_at"]


def test_parse_openalex_work_no_refs() -> None:
    work = _fixture_works()[1]
    refs = parse_references_out("40000002", work)
    assert refs == []


# ---------------------------------------------------------------------------
# parse_references_out
# ---------------------------------------------------------------------------


def test_parse_references_out() -> None:
    work = _fixture_works()[0]
    refs = parse_references_out("40000001", work)

    assert len(refs) == 3
    positions = [r["ref_position"] for r in refs]
    assert positions == [0, 1, 2]
    for r in refs:
        assert r["citing_pmid"] == "40000001"
        assert r["ref_openalex_id"].startswith("W")
        assert not r["ref_openalex_id"].startswith("https://")


# ---------------------------------------------------------------------------
# parse_authorships_staging
# ---------------------------------------------------------------------------


def test_parse_authorships_staging_full() -> None:
    work = _fixture_works()[0]
    authors = parse_authorships_staging("40000001", work)

    assert len(authors) == 3
    positions = [a["author_position"] for a in authors]
    labels = [a["author_position_label"] for a in authors]
    assert positions == [0, 1, 2]
    assert labels == ["first", "middle", "last"]

    # ORCID stripped of URL prefix for author 0
    a0 = authors[0]
    assert a0["author_orcid"] == "0000-0001-2345-6789"
    assert not a0["author_orcid"].startswith("https://")
    assert a0["author_oa_id"] == "A1000000001"

    # institutions preserved, oa_id stripped
    assert a0["institutions"], "expected at least one institution"
    inst = a0["institutions"][0]
    assert inst["oa_id"] == "I100"
    assert not inst["oa_id"].startswith("https://")
    assert inst["display_name"] == "Stanford University"

    # Middle author has no ORCID
    assert authors[1]["author_orcid"] == ""


def test_parse_authorships_staging_missing_orcid() -> None:
    """Third fixture work — author.id is None, so author_oa_id should be ''."""
    work = _fixture_works()[2]
    authors = parse_authorships_staging("40000003", work)
    assert len(authors) == 1
    assert authors[0]["author_oa_id"] == ""
    assert authors[0]["author_orcid"] == ""
    assert authors[0]["author_display_name"] == "Anonymous Collective"


# ---------------------------------------------------------------------------
# fetch_batch (respx-mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_batch_respx_mocked(tmp_path: Path) -> None:
    fixture_body = FIXTURE_PATH.read_bytes()
    cfg = OpenAlexConfig(
        email="test@example.com",
        cache_dir=tmp_path / "cache",
    )
    client = OpenAlexClient(cfg, RateLimiter(100.0))

    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=r".*api\.openalex\.org/works.*").mock(
            return_value=httpx.Response(200, content=fixture_body)
        )

        try:
            result = await client.fetch_batch(["40000001", "40000002", "40000003"])
        finally:
            await client.aclose()

    assert set(result.keys()) == {"40000001", "40000002", "40000003"}
    for pmid, work in result.items():
        assert work is not None, f"expected work for {pmid}"
        assert isinstance(work, dict)
        assert "id" in work


# ---------------------------------------------------------------------------
# enrich_openalex smoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_openalex_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_body = FIXTURE_PATH.read_bytes()
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.parquet"
    cfg = OpenAlexConfig(
        email="test@example.com",
        cache_dir=cache_dir,
        manifest_path=manifest_path,
    )
    rate_limiter = RateLimiter(100.0)

    pmids = ["40000001", "40000002", "40000003"]

    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=r".*api\.openalex\.org/works.*").mock(
            return_value=httpx.Response(200, content=fixture_body)
        )
        out = await enrich_openalex(
            pmids=pmids,
            cfg=cfg,
            rate_limiter=rate_limiter,
        )

    assert set(out.keys()) >= {
        "works",
        "references",
        "authorships_staging",
        "failed",
        "manifest_rows",
    }
    assert out["works"], "works should be non-empty"
    assert out["manifest_rows"], "manifest_rows should be non-empty"
    # Every requested PMID should appear in manifest_rows
    manifest_pmids = {row["pmid"] for row in out["manifest_rows"]}
    assert manifest_pmids == set(pmids)

    # Cache files exist on disk under the sharded path
    for pmid in pmids:
        shard = pmid[:2]
        cached = cache_dir / shard / f"{pmid}.json.gz"
        assert cached.exists(), f"expected cache file for {pmid} at {cached}"

    # Manifest parquet was written
    assert manifest_path.exists()


@pytest.mark.asyncio
async def test_enrich_openalex_idempotent(tmp_path: Path) -> None:
    """Pre-write manifest='ok' + cached JSON; ensure no HTTP call and rows rehydrated."""
    import gzip

    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-write the gzipped raw JSON cache so rehydration can find it.
    fixture = json.loads(FIXTURE_PATH.read_text())
    work = fixture["results"][0]  # PMID 40000001
    shard_dir = cache_dir / "40"
    shard_dir.mkdir(parents=True)
    (shard_dir / "40000001.json.gz").write_bytes(gzip.compress(json.dumps(work).encode()))

    schema = pa.schema(
        [
            ("pmid", pa.string()),
            ("fetched_at", pa.string()),
            ("status", pa.string()),
            ("openalex_id", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(
        [
            {
                "pmid": "40000001",
                "fetched_at": "2024-01-01T00:00:00+00:00",
                "status": "ok",
                "openalex_id": "W4000000001",
            }
        ],
        schema=schema,
    )
    pq.write_table(table, manifest_path)

    cfg = OpenAlexConfig(
        email="test@example.com",
        cache_dir=cache_dir,
        manifest_path=manifest_path,
    )
    rate_limiter = RateLimiter(100.0)

    empty_envelope = json.dumps({"meta": {"count": 0}, "results": []}).encode()

    with respx.mock(assert_all_called=False) as router:
        route = router.get(url__regex=r".*api\.openalex\.org/works.*").mock(
            return_value=httpx.Response(200, content=empty_envelope)
        )

        out = await enrich_openalex(
            pmids=["40000001"],
            cfg=cfg,
            rate_limiter=rate_limiter,
        )

        # No HTTP call should have been issued — PMID was already 'ok' AND cached.
        assert route.call_count == 0

    assert len(out["works"]) == 1
    assert out["works"][0]["pmid"] == "40000001"
    assert out["works"][0]["openalex_id"] == "W4000000001"
    assert out["failed"] == []
