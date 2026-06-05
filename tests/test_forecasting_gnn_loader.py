"""Unit tests for the V1-S13 leakage-safe graph snapshot loader.

Hermetic and fast: the leakage assertions run on hand-built
:class:`~torch_geometric.data.HeteroData` objects, the assembly logic runs on
tiny pandas frames via :func:`build_snapshot_from_frames`, and the only disk-
touching test builds an in-memory/temp DuckDB (never ``data/v1/*``). Mirrors the
pure-fixture conventions of ``tests/test_forecasting_data.py`` /
``tests/test_forecasting_baselines.py``: short keyword leakage messages, no real
corpus, no network.

Encoded module assumptions (see HANDOFF NOTES in the loader):
- Edges are FORWARD-only; the model applies ``ToUndirected`` itself.
- ``Topic`` nodes cover ALL leaf topics in ascending ``topic_id`` order, so the
  ``topic_id -> idx`` map is identical across years (asserted in the smoke test).
- ``CITES`` with citing.year < cited.year is a recorded artifact, NOT a leak.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402

from scifield.forecasting.gnn.loader import (  # noqa: E402
    EDGE_TYPES,
    NODE_TYPES,
    assert_snapshot_no_leakage,
    build_snapshot_from_frames,
    build_year_snapshots,
    load_snapshot,
    topic_id_to_index,
)

torch.set_num_threads(1)


# --------------------------------------------------------------------------- #
# Hand-built HeteroData helpers for the leakage assertions
# --------------------------------------------------------------------------- #


def _clean_snapshot(t: int = 2005) -> HeteroData:
    """A minimal, fully-grounded snapshot that must PASS every guard.

    3 papers (years 2003/2004/2005, all <= t), one CITES edge 0->1, one author
    grounded by both AUTHORED_BY and AFFILIATED_WITH, one institution, one
    journal, two leaf topics (ids 3, 7) with one ASSIGNED_TO edge.
    """
    data = HeteroData()
    years = torch.tensor([2003, 2004, 2005], dtype=torch.long)
    data["Paper"].x = torch.zeros(3, 2, dtype=torch.float32)
    data["Paper"].year = years
    data["Author"].x = torch.ones(1, 1, dtype=torch.float32)
    data["Institution"].x = torch.ones(1, 1, dtype=torch.float32)
    data["Journal"].x = torch.ones(1, 1, dtype=torch.float32)
    data["Topic"].num_nodes = 2
    data["Topic"].x = torch.zeros(2, 16, dtype=torch.float32)
    data["Topic"].topic_id = torch.tensor([3, 7], dtype=torch.long)

    data["Paper", "CITES", "Paper"].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    data["Paper", "AUTHORED_BY", "Author"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data["Author", "AFFILIATED_WITH", "Institution"].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    data["Paper", "PUBLISHED_IN", "Journal"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data["Paper", "ASSIGNED_TO", "Topic"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data.n_future_citation_artifacts = 0
    data.origin_year = t
    return data


# --------------------------------------------------------------------------- #
# Module-level metadata contract
# --------------------------------------------------------------------------- #


def test_node_and_edge_type_contract() -> None:
    # The model builds HGTConv metadata from these — they are part of the API.
    assert NODE_TYPES == ("Paper", "Author", "Institution", "Journal", "Topic")
    assert EDGE_TYPES == (
        ("Paper", "CITES", "Paper"),
        ("Paper", "AUTHORED_BY", "Author"),
        ("Author", "AFFILIATED_WITH", "Institution"),
        ("Paper", "PUBLISHED_IN", "Journal"),
        ("Paper", "ASSIGNED_TO", "Topic"),
    )


# --------------------------------------------------------------------------- #
# (1) assert_snapshot_no_leakage — L1 / L2 keyword assertions
# --------------------------------------------------------------------------- #


def test_clean_snapshot_passes_all_guards() -> None:
    data = _clean_snapshot(t=2005)
    # Returns the L5 artifact count (0 here) and does not raise.
    assert assert_snapshot_no_leakage(data, 2005) == 0


def test_leakage_max_year_future_isolated_paper() -> None:
    # A future paper (2006 > t=2005) with NO CITES edge falls through to L1.
    data = _clean_snapshot(t=2005)
    data["Paper"].year = torch.tensor([2003, 2004, 2006], dtype=torch.long)
    # Drop the CITES edge so L2 cannot catch the future paper first.
    data["Paper", "CITES", "Paper"].edge_index = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(AssertionError, match="leakage:max_year"):
        assert_snapshot_no_leakage(data, 2005)


def test_leakage_cites_endpoint_to_future_paper() -> None:
    # A CITES edge whose cited endpoint is a future paper (2006 > t=2005). L2 runs
    # before L1, so this raises the SPECIFIC cites_endpoint message.
    data = _clean_snapshot(t=2005)
    # idx1 (the CITES target, edge 0->1) is future at 2006.
    data["Paper"].year = torch.tensor([2003, 2006, 2004], dtype=torch.long)
    with pytest.raises(AssertionError, match="leakage:cites_endpoint"):
        assert_snapshot_no_leakage(data, 2005)


# --------------------------------------------------------------------------- #
# (2) L3 node provenance / L4 topic resolution
# --------------------------------------------------------------------------- #


def test_leakage_node_provenance_orphan_author() -> None:
    data = _clean_snapshot(t=2005)
    # Add a SECOND author node with no AUTHORED_BY / AFFILIATED_WITH edge -> orphan.
    data["Author"].x = torch.ones(2, 1, dtype=torch.float32)
    with pytest.raises(AssertionError, match="leakage:node_provenance"):
        assert_snapshot_no_leakage(data, 2005)


def test_leakage_node_provenance_orphan_journal() -> None:
    data = _clean_snapshot(t=2005)
    data["Journal"].x = torch.ones(2, 1, dtype=torch.float32)  # 2nd journal unlinked
    with pytest.raises(AssertionError, match="leakage:node_provenance"):
        assert_snapshot_no_leakage(data, 2005)


def test_isolated_topic_still_resolves() -> None:
    # A leaf topic with NO ASSIGNED_TO edge still has a node and resolves (L4).
    data = _clean_snapshot(t=2005)
    # topic ids 3 and 7 present; topic 7 has no incoming ASSIGNED_TO edge.
    assert assert_snapshot_no_leakage(data, 2005, requested_topic_ids=[3, 7]) == 0
    # The inverse map covers both even though 7 is isolated.
    assert topic_id_to_index(data) == {3: 0, 7: 1}


def test_leakage_topic_resolves_missing_requested() -> None:
    data = _clean_snapshot(t=2005)
    # Topic 99 is not in the snapshot -> requested set not a subset -> trips.
    with pytest.raises(AssertionError, match="leakage:topic_resolves"):
        assert_snapshot_no_leakage(data, 2005, requested_topic_ids=[3, 99])


# --------------------------------------------------------------------------- #
# (3) L5 diagnostic — citing.year < cited.year recorded, not raised
# --------------------------------------------------------------------------- #


def test_l5_records_future_citation_artifact_without_raising() -> None:
    data = _clean_snapshot(t=2010)
    # Paper 0 (2003) CITES paper 1 (2004) is fine; add a back-citation 2 -> 1 with
    # citing year < cited year (a real OpenAlex artifact). Both still <= t.
    data["Paper"].year = torch.tensor([2003, 2008, 2005], dtype=torch.long)
    # CITES: 0->1 (2003<2008 artifact), 2->1 (2005<2008 artifact) = 2 artifacts.
    data["Paper", "CITES", "Paper"].edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    # Drop the stored count so the assertion recomputes it defensively.
    data.n_future_citation_artifacts = -1
    n = assert_snapshot_no_leakage(data, 2010)
    assert n == 2  # recorded, no raise


# --------------------------------------------------------------------------- #
# build_snapshot_from_frames — pure pandas assembly (no DuckDB)
# --------------------------------------------------------------------------- #


def _corpus_frames() -> dict[str, pd.DataFrame]:
    """Tiny full-corpus frames shaped like the canonical kuzu queries' output.

    5 papers across 2000..2004; a citation chain; 2 authors, 2 institutions, 2
    journals, 3 leaf topics (ids 5, 8, 12). One topic (12) is never assigned ->
    must still get a node in every snapshot.
    """
    papers = pd.DataFrame(
        {
            "pmid": ["p1", "p2", "p3", "p4", "p5"],
            "year": [2000, 2001, 2002, 2003, 2004],
        }
    )
    topics = pd.DataFrame({"topic_id": [5, 8, 12]})
    # p2->p1, p3->p1, p4->p2, p5->p4 (all backward-in-time = clean, no artifacts).
    cites = pd.DataFrame(
        {
            "from_pmid": ["p2", "p3", "p4", "p5"],
            "to_pmid": ["p1", "p1", "p2", "p4"],
        }
    )
    authored_by = pd.DataFrame(
        {
            "pmid": ["p1", "p2", "p3", "p4", "p5"],
            "author_canonical_id": ["a1", "a1", "a2", "a2", "a2"],
        }
    )
    affiliated_with = pd.DataFrame(
        {
            "author_canonical_id": ["a1", "a2"],
            "institution_canonical_id": ["i1", "i2"],
        }
    )
    published_in = pd.DataFrame(
        {
            "pmid": ["p1", "p2", "p3", "p4", "p5"],
            "journal_slug": ["j1", "j1", "j2", "j2", "j2"],
        }
    )
    assigned_to = pd.DataFrame(
        {
            "pmid": ["p1", "p2", "p3", "p4", "p5"],
            "topic_id": [5, 5, 8, 8, 8],
        }
    )
    return {
        "papers": papers,
        "topics": topics,
        "cites": cites,
        "authored_by": authored_by,
        "affiliated_with": affiliated_with,
        "published_in": published_in,
        "assigned_to": assigned_to,
    }


def test_build_snapshot_from_frames_induction_and_features() -> None:
    frames = _corpus_frames()
    # t=2001 -> only p1, p2 eligible.
    data, meta = build_snapshot_from_frames(2001, **frames)

    assert meta["n_nodes"]["Paper"] == 2
    # Author a1 authored p1,p2 (both eligible) -> induced; a2's papers all > t.
    assert meta["n_nodes"]["Author"] == 1
    # a1 affiliated with i1 -> induced; i2 (a2 only) dropped.
    assert meta["n_nodes"]["Institution"] == 1
    assert meta["n_nodes"]["Journal"] == 1  # j1 only
    # ALL leaf topics present regardless of eligibility.
    assert meta["n_nodes"]["Topic"] == 3
    assert topic_id_to_index(data) == {5: 0, 8: 1, 12: 2}

    # CITES kept only if both endpoints eligible: p2->p1 only (p3/p4/p5 excluded).
    assert meta["n_edges"]["CITES"] == 1
    # Paper feature: idx of p1 has in-snapshot in-degree 1 (cited by p2).
    # Paper x col 1 = log1p(in-degree). p1 is idx 0 (sorted "p1"<"p2").
    import math

    assert data["Paper"].x[0, 1].item() == pytest.approx(math.log1p(1.0))
    assert data["Paper"].x[1, 1].item() == pytest.approx(math.log1p(0.0))
    # normalized_year for p1 (2000): (2000-1995)/(2001-1995+1) = 5/7.
    assert data["Paper"].x[0, 0].item() == pytest.approx(5.0 / 7.0)

    # Builder output passes the full guard set.
    assert assert_snapshot_no_leakage(data, 2001) == 0
    # Topic.x is a zeros placeholder of shape (n_topic, 16) (model overwrites).
    assert tuple(data["Topic"].x.shape) == (3, 16)
    assert torch.count_nonzero(data["Topic"].x).item() == 0


def test_build_snapshot_node_counts_monotonic_in_t() -> None:
    frames = _corpus_frames()
    d1, m1 = build_snapshot_from_frames(2001, **frames)
    d2, m2 = build_snapshot_from_frames(2004, **frames)
    # Strictly more (or equal) papers/edges as t grows.
    assert m2["n_nodes"]["Paper"] >= m1["n_nodes"]["Paper"]
    assert m2["n_edges"]["CITES"] >= m1["n_edges"]["CITES"]
    # Topic count is INVARIANT (all leaf topics always present).
    assert m1["n_nodes"]["Topic"] == m2["n_nodes"]["Topic"] == 3
    # topic_id->idx map identical across years.
    assert topic_id_to_index(d1) == topic_id_to_index(d2)
    # At t=2004 every paper is in; CITES = all 4 edges.
    assert m2["n_nodes"]["Paper"] == 5
    assert m2["n_edges"]["CITES"] == 4


# --------------------------------------------------------------------------- #
# (4) End-to-end build_year_snapshots smoke on a temp DuckDB (no data/v1/*)
# --------------------------------------------------------------------------- #


class _NS:
    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)

    def __getitem__(self, key: str) -> object:
        return self.__dict__[key]


def _make_temp_corpus(tmp_path: Path) -> tuple[str, str]:
    """Create a minimal DuckDB (papers + edges) + topics.parquet for the smoke test.

    ``papers`` is a base table shaped so ``ensure_papers_distinct_view`` succeeds
    (one row per pmid, has an ``abstract`` column for the dedup tiebreak). Returns
    ``(duckdb_path, topics_parquet_path)``.
    """
    import duckdb

    duckdb_path = tmp_path / "mini.duckdb"
    topics_parquet = tmp_path / "topics.parquet"

    pd.DataFrame({"topic_id": [5, 8, 12], "pmid": [1, 3, 5], "is_noise": [False] * 3}).to_parquet(
        topics_parquet
    )

    con = duckdb.connect(str(duckdb_path))
    # papers base table (VARCHAR pmid, like the real corpus) with abstract for the
    # dedup tiebreak; one row per pmid so papers_distinct == papers.
    con.execute(
        """
        CREATE TABLE papers AS SELECT * FROM (VALUES
            ('p1', 'aaaa', 't1', 2000, 'j1'),
            ('p2', 'bbbb', 't2', 2001, 'j1'),
            ('p3', 'cccc', 't3', 2002, 'j2'),
            ('p4', 'dddd', 't4', 2003, 'j2'),
            ('p5', 'eeee', 't5', 2004, 'j2')
        ) AS papers(pmid, abstract, title, year, journal_slug)
        """
    )
    # journal display column referenced by the kuzu Journal node query (unused
    # here, but the SELECT lives in shared code paths); add it to be safe.
    con.execute("ALTER TABLE papers ADD COLUMN journal VARCHAR DEFAULT 'J'")

    con.execute(
        """
        CREATE TABLE references_out AS SELECT * FROM (VALUES
            ('p2', 'oa_p1'),
            ('p3', 'oa_p1'),
            ('p4', 'oa_p2'),
            ('p5', 'oa_p4')
        ) AS r(citing_pmid, ref_openalex_id)
        """
    )
    con.execute(
        """
        CREATE TABLE openalex_works AS SELECT * FROM (VALUES
            ('p1', 'oa_p1'),
            ('p2', 'oa_p2'),
            ('p4', 'oa_p4')
        ) AS ow(pmid, openalex_id)
        """
    )
    con.execute(
        """
        CREATE TABLE authorships AS SELECT * FROM (VALUES
            ('p1', 1, 'a1', 'A1'),
            ('p2', 1, 'a1', 'A1'),
            ('p3', 1, 'a2', 'A2'),
            ('p4', 1, 'a2', 'A2'),
            ('p5', 1, 'a2', 'A2')
        ) AS a(pmid, author_position, author_canonical_id, author_display_name)
        """
    )
    con.execute(
        """
        CREATE TABLE paper_institutions AS SELECT * FROM (VALUES
            ('p1', 1, 'i1'),
            ('p2', 1, 'i1'),
            ('p3', 1, 'i2'),
            ('p4', 1, 'i2'),
            ('p5', 1, 'i2')
        ) AS pi(pmid, author_position, institution_canonical_id)
        """
    )
    con.execute(
        """
        CREATE TABLE institutions AS SELECT * FROM (VALUES
            ('i1', 'Inst1', 'US'),
            ('i2', 'Inst2', 'GB')
        ) AS inst(institution_canonical_id, display_name, country_code)
        """
    )
    con.close()
    return str(duckdb_path), str(topics_parquet)


def test_build_year_snapshots_end_to_end(tmp_path: Path) -> None:
    duckdb_path, topics_parquet = _make_temp_corpus(tmp_path)
    snap_dir = tmp_path / "snaps"
    cfg = _NS(
        input=_NS(duckdb_path=duckdb_path, topics_parquet=topics_parquet),
        output=_NS(snapshots_dir=str(snap_dir)),
        split=_NS(train=[2001, 2003], val=[2004, 2004]),
    )

    meta = build_year_snapshots(cfg, years=[2001, 2004])
    assert set(meta) == {2001, 2004}

    # Node/edge counts grow (or hold) monotonically with t.
    assert meta[2004]["n_nodes"]["Paper"] >= meta[2001]["n_nodes"]["Paper"]
    assert meta[2004]["n_edges"]["CITES"] >= meta[2001]["n_edges"]["CITES"]
    # All leaf topics present in both, count invariant.
    assert meta[2001]["n_nodes"]["Topic"] == meta[2004]["n_nodes"]["Topic"] == 3
    # At t=2004 all 5 papers in; the 4 resolvable CITES (oa_p1/p2/p4) survive.
    assert meta[2004]["n_nodes"]["Paper"] == 5
    assert meta[2004]["n_edges"]["CITES"] == 4

    # Cached files exist and reload cleanly (re-running L1/L2).
    for t in (2001, 2004):
        path = snap_dir / f"snapshot_{t}.pt"
        assert path.exists()
        assert meta[t]["path"] == str(path)
    s1 = load_snapshot(snap_dir / "snapshot_2001.pt")
    s2 = load_snapshot(snap_dir / "snapshot_2004.pt")
    # topic_id->idx map identical across the two reloaded years.
    assert topic_id_to_index(s1) == topic_id_to_index(s2) == {5: 0, 8: 1, 12: 2}
    # Forward-only: no rev_* relation in the snapshot (model adds those).
    assert ("Topic", "rev_ASSIGNED_TO", "Paper") not in s2.edge_types


def test_build_year_snapshots_default_years_from_split(tmp_path: Path) -> None:
    duckdb_path, topics_parquet = _make_temp_corpus(tmp_path)
    cfg = _NS(
        input=_NS(duckdb_path=duckdb_path, topics_parquet=topics_parquet),
        output=_NS(snapshots_dir=str(tmp_path / "snaps2")),
        split=_NS(train=[2001, 2002], val=[2003, 2004]),
    )
    # Default range = train.lo .. val.hi = 2001..2004.
    meta = build_year_snapshots(cfg)
    assert sorted(meta) == [2001, 2002, 2003, 2004]
