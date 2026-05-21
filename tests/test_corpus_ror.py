"""Tests for the async ROR institution matcher (V1-S04 §6)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import respx

from scifield.corpus.pubmed import RateLimiter
from scifield.corpus.ror import (
    RORConfig,
    RORMatcher,
    build_institution_tables,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ror"
ROR_BASE_URL = "https://api.ror.org"

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


def _load_sample_response() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((FIXTURE_DIR / "sample_response.json").read_text()))


def _make_matcher(
    cache_path: Path,
    *,
    min_match_score: float = 0.85,
) -> RORMatcher:
    cfg = RORConfig(
        base_url=ROR_BASE_URL,
        rate_limit=1000.0,
        request_timeout_s=5.0,
        max_retries=1,
        min_match_score=min_match_score,
        cache_path=cache_path,
    )
    return RORMatcher(cfg, rate_limiter=RateLimiter(rate=1000.0))


def _write_cache(
    cache_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_CACHE_SCHEMA)
    pq.write_table(table, cache_path)


# ---------------------------------------------------------------------------
# Matcher tests
# ---------------------------------------------------------------------------


async def test_match_cache_hit(tmp_path: Path) -> None:
    cache_path = tmp_path / "ror_cache.parquet"
    _write_cache(
        cache_path,
        [
            {
                "raw_string": "Stanford",
                "ror_id": "https://ror.org/00f54p054",
                "ror_display_name": "Stanford University",
                "country_code": "US",
                "type": "Education",
                "match_score": 0.95,
                "fetched_at": datetime.now(UTC).isoformat(),
            }
        ],
    )

    matcher = _make_matcher(cache_path)
    try:
        result = await matcher.match("Stanford")
    finally:
        await matcher.aclose()

    assert result is not None
    assert result["ror_id"] == "https://ror.org/00f54p054"
    assert result["display_name"] == "Stanford University"
    assert result["country_code"] == "US"
    assert result["type"] == "Education"
    assert result["match_score"] == pytest.approx(0.95)


@respx.mock
async def test_match_api_call(tmp_path: Path) -> None:
    cache_path = tmp_path / "ror_cache.parquet"
    payload = _load_sample_response()

    route = respx.get(f"{ROR_BASE_URL}/organizations").mock(
        return_value=httpx.Response(200, json=payload)
    )

    matcher = _make_matcher(cache_path)
    try:
        result = await matcher.match("Stanford University")
        await matcher.flush_cache()
    finally:
        await matcher.aclose()

    assert route.called
    assert result is not None
    assert result["ror_id"] == "https://ror.org/00f54p054"
    assert result["display_name"] == "Stanford University"
    assert result["country_code"] == "US"
    assert result["type"] == "Education"
    assert result["match_score"] == pytest.approx(0.97)

    assert cache_path.exists()
    table = pq.read_table(cache_path)
    rows = table.to_pylist()
    assert any(r["raw_string"] == "Stanford University" for r in rows)


@respx.mock
async def test_match_below_threshold_returns_none(tmp_path: Path) -> None:
    cache_path = tmp_path / "ror_cache.parquet"
    payload = {
        "items": [
            {
                "organization": {
                    "id": "https://ror.org/03kk7td41",
                    "name": "Stanford Health Care",
                    "country": {"country_code": "US", "country_name": "United States"},
                    "types": ["Healthcare"],
                },
                "score": 0.5,
                "matching_type": "FUZZYMATCH",
                "substring": "Stanford",
            }
        ]
    }

    route = respx.get(f"{ROR_BASE_URL}/organizations").mock(
        return_value=httpx.Response(200, json=payload)
    )

    matcher = _make_matcher(cache_path, min_match_score=0.85)
    try:
        first = await matcher.match("Some Weak Match")
        # Re-query: should short-circuit via cached miss and NOT hit the network again.
        second = await matcher.match("Some Weak Match")
    finally:
        await matcher.aclose()

    assert first is None
    assert second is None
    assert route.call_count == 1


@respx.mock
async def test_match_no_results_returns_none(tmp_path: Path) -> None:
    cache_path = tmp_path / "ror_cache.parquet"
    route = respx.get(f"{ROR_BASE_URL}/organizations").mock(
        return_value=httpx.Response(200, json={"items": []})
    )

    matcher = _make_matcher(cache_path)
    try:
        result = await matcher.match("Nowhere Institute")
    finally:
        await matcher.aclose()

    assert route.called
    assert result is None


async def test_flush_cache_roundtrip(tmp_path: Path) -> None:
    cache_path = tmp_path / "ror_cache.parquet"

    matcher = _make_matcher(cache_path)
    # Populate in-memory cache directly to avoid network I/O.
    matcher._cache["Stanford University"] = {
        "ror_id": "https://ror.org/00f54p054",
        "ror_display_name": "Stanford University",
        "country_code": "US",
        "type": "Education",
        "match_score": 0.97,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    matcher._cache["Nowhere Clinic"] = {
        "ror_id": "",
        "ror_display_name": "",
        "country_code": "",
        "type": "",
        "match_score": 0.0,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    await matcher.flush_cache()
    await matcher.aclose()

    assert cache_path.exists()

    # New matcher should load both entries on init.
    matcher2 = _make_matcher(cache_path)
    try:
        assert "Stanford University" in matcher2._cache
        assert "Nowhere Clinic" in matcher2._cache
        assert matcher2._cache["Stanford University"]["ror_id"] == "https://ror.org/00f54p054"
        # Hit-path: cached entry above threshold should be returned.
        hit = await matcher2.match("Stanford University")
        assert hit is not None
        assert hit["ror_id"] == "https://ror.org/00f54p054"
        # Miss-path: empty ror_id → None.
        miss = await matcher2.match("Nowhere Clinic")
        assert miss is None
    finally:
        await matcher2.aclose()


# ---------------------------------------------------------------------------
# build_institution_tables
# ---------------------------------------------------------------------------


class _FakeMatcher:
    """Stub matcher exposing the subset of the RORMatcher API used by builder."""

    def __init__(
        self,
        *,
        result: dict[str, Any] | None = None,
        min_match_score: float = 0.85,
    ) -> None:
        self._result = result
        self._cfg = RORConfig(min_match_score=min_match_score)
        self.calls: list[str] = []

    @property
    def config(self) -> RORConfig:
        return self._cfg

    async def match(self, raw_string: str) -> dict[str, Any] | None:
        self.calls.append(raw_string)
        return self._result


async def test_build_institution_tables_oa_path() -> None:
    staging_rows = [
        {
            "pmid": "1",
            "author_position": 1,
            "institutions": [
                {
                    "oa_id": "I1",
                    "ror_id": "https://ror.org/abc",
                    "display_name": "Stanford",
                    "country_code": "US",
                    "type": "education",
                }
            ],
            "raw_affiliation_strings": ["Some string"],
        }
    ]
    matcher = _FakeMatcher()

    institutions, paper_institutions = await build_institution_tables(
        staging_rows=staging_rows,
        matcher=matcher,  # type: ignore[arg-type]
    )

    assert matcher.calls == []
    assert len(institutions) == 1
    assert institutions[0]["institution_canonical_id"] == "OA:I1"
    assert institutions[0]["institution_oa_id"] == "I1"
    assert institutions[0]["ror_id"] == "https://ror.org/abc"
    assert len(paper_institutions) == 1
    assert paper_institutions[0]["institution_canonical_id"] == "OA:I1"
    assert paper_institutions[0]["ror_matched_by"] == "openalex"


async def test_build_institution_tables_ror_path(tmp_path: Path) -> None:
    staging_rows = [
        {
            "pmid": "10",
            "author_position": 0,
            "institutions": [],
            "raw_affiliation_strings": ["Stanford"],
        }
    ]
    matcher = _FakeMatcher(
        result={
            "ror_id": "https://ror.org/00f54p054",
            "display_name": "Stanford University",
            "country_code": "US",
            "type": "Education",
            "match_score": 0.97,
        }
    )

    institutions, paper_institutions = await build_institution_tables(
        staging_rows=staging_rows,
        matcher=matcher,  # type: ignore[arg-type]
    )

    assert matcher.calls == ["Stanford"]
    assert len(institutions) == 1
    assert institutions[0]["institution_canonical_id"] == "ROR:https://ror.org/00f54p054"
    assert institutions[0]["ror_id"] == "https://ror.org/00f54p054"
    assert institutions[0]["display_name"] == "Stanford University"
    assert len(paper_institutions) == 1
    assert paper_institutions[0]["institution_canonical_id"] == "ROR:https://ror.org/00f54p054"
    assert paper_institutions[0]["ror_matched_by"] == "ror_api"
    assert paper_institutions[0]["raw_affiliation_string"] == "Stanford"


async def test_build_institution_tables_unmatched_path() -> None:
    staging_rows = [
        {
            "pmid": "42",
            "author_position": 2,
            "institutions": [],
            "raw_affiliation_strings": ["Some Tiny Clinic of Nowhere"],
        }
    ]
    matcher = _FakeMatcher(result=None)

    institutions, paper_institutions = await build_institution_tables(
        staging_rows=staging_rows,
        matcher=matcher,  # type: ignore[arg-type]
    )

    assert matcher.calls == ["Some Tiny Clinic of Nowhere"]
    assert len(institutions) == 1
    assert institutions[0]["institution_canonical_id"].startswith("RAW:")
    assert institutions[0]["display_name"] == "Some Tiny Clinic of Nowhere"
    assert len(paper_institutions) == 1
    assert paper_institutions[0]["institution_canonical_id"].startswith("RAW:")
    assert paper_institutions[0]["ror_matched_by"] == "unmatched"


async def test_build_institution_tables_dedup() -> None:
    staging_rows = [
        {
            "pmid": "1",
            "author_position": 0,
            "institutions": [
                {
                    "oa_id": "I1",
                    "ror_id": "https://ror.org/abc",
                    "display_name": "Stanford",
                    "country_code": "US",
                    "type": "education",
                }
            ],
            "raw_affiliation_strings": [],
        },
        {
            "pmid": "2",
            "author_position": 1,
            "institutions": [
                {
                    "oa_id": "I1",
                    "ror_id": "https://ror.org/abc",
                    "display_name": "Stanford",
                    "country_code": "US",
                    "type": "education",
                }
            ],
            "raw_affiliation_strings": [],
        },
    ]
    matcher = _FakeMatcher()

    institutions, paper_institutions = await build_institution_tables(
        staging_rows=staging_rows,
        matcher=matcher,  # type: ignore[arg-type]
    )

    assert len(institutions) == 1
    assert institutions[0]["institution_canonical_id"] == "OA:I1"
    assert len(paper_institutions) == 2
    assert {row["pmid"] for row in paper_institutions} == {"1", "2"}
    assert all(row["institution_canonical_id"] == "OA:I1" for row in paper_institutions)
