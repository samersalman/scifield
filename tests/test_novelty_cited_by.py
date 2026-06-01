"""Offline tests for the OR-batched cited_by harvester attribution logic.

No network: the OpenAlex page fetch is monkeypatched with synthetic envelopes.
These cover the parts most likely to be wrong — date-windowed filter strings and
attributing each citer (from one OR'd query) back to the focal(s) it references.
"""

from __future__ import annotations

import pytest

from scifield.corpus.pubmed import RateLimiter
from scifield.novelty.cited_by import CitedByClient, CitedByConfig, _FocalWork


def _client() -> CitedByClient:
    cfg = CitedByConfig(mailto="x@y.z", citer_window_years=10, focal_batch_size=50)
    return CitedByClient(cfg, RateLimiter(100.0))


def test_batch_filter_windowed_same_year() -> None:
    client = _client()
    batch = [
        _FocalWork("W1", frozenset(), 10, 2000),
        _FocalWork("W2", frozenset(), 10, 2000),
    ]
    assert client._batch_filter(batch) == (
        "cites:W1|W2,from_publication_date:2000-01-01,to_publication_date:2010-12-31"
    )


def test_batch_filter_unknown_year_is_unwindowed() -> None:
    client = _client()
    batch = [_FocalWork("W9", frozenset(), 3, 0)]
    assert client._batch_filter(batch) == "cites:W9"


async def test_harvest_batch_attributes_citers_to_focals() -> None:
    """Each citer is attributed only to the batch focals it actually references."""
    client = _client()
    page: dict[str, object] = {
        "results": [
            # cites W1 and references R1 (W1's own ref) -> cites_focal_ref True
            {
                "id": "https://openalex.org/C1",
                "publication_year": 2003,
                "referenced_works": ["https://openalex.org/W1", "https://openalex.org/R1"],
            },
            # cites W1 and W2, neither of their refs -> two rows, both False
            {"id": "C2", "publication_year": 2004, "referenced_works": ["W1", "W2"]},
            # cites W1 and W3; references R3 (W3's ref) -> W3 row True, W1 row False
            {"id": "C3", "publication_year": 2005, "referenced_works": ["W3", "R3", "W1"]},
            # cites no batch focal -> dropped entirely
            {"id": "C4", "publication_year": 2006, "referenced_works": ["RXX"]},
        ],
        "meta": {"next_cursor": None},
    }

    async def fake_fetch(cites_filter: str, cursor: str) -> dict[str, object]:
        return page

    client._fetch_page = fake_fetch  # type: ignore[method-assign]

    batch = [
        _FocalWork("W1", frozenset({"R1"}), 10, 2000),
        _FocalWork("W2", frozenset({"R2"}), 10, 2000),
        _FocalWork("W3", frozenset({"R3"}), 10, 2000),
    ]
    res = await client.harvest_batch(batch)

    def summary(fid: str) -> list[tuple[str, bool]]:
        return sorted((r["citing_oa_id"], r["cites_focal_ref"]) for r in res[fid])

    assert summary("W1") == [("C1", True), ("C2", False), ("C3", False)]
    assert summary("W2") == [("C2", False)]
    assert summary("W3") == [("C3", True)]
    # Every focal in the batch has an entry, even if empty (none are here).
    assert set(res) == {"W1", "W2", "W3"}


async def test_harvest_focal_delegates_to_batch() -> None:
    client = _client()
    page: dict[str, object] = {
        "results": [{"id": "C1", "publication_year": 2003, "referenced_works": ["W1", "R1"]}],
        "meta": {"next_cursor": None},
    }

    async def fake_fetch(cites_filter: str, cursor: str) -> dict[str, object]:
        return page

    client._fetch_page = fake_fetch  # type: ignore[method-assign]

    rows = await client.harvest_focal(_FocalWork("W1", frozenset({"R1"}), 10, 2000))
    assert rows == [
        {"focal_oa_id": "W1", "citing_oa_id": "C1", "citing_year": 2003, "cites_focal_ref": True}
    ]


async def test_harvest_batch_empty_focal_gets_empty_list() -> None:
    client = _client()
    page: dict[str, object] = {"results": [], "meta": {"next_cursor": None}}

    async def fake_fetch(cites_filter: str, cursor: str) -> dict[str, object]:
        return page

    client._fetch_page = fake_fetch  # type: ignore[method-assign]

    res = await client.harvest_batch([_FocalWork("W1", frozenset(), 0, 2000)])
    assert res == {"W1": []}


def test_pytest_marker_unused() -> None:
    # Guard against accidental removal of pytest import (used implicitly by the
    # asyncio_mode=auto config for the async tests above).
    assert pytest is not None
