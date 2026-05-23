"""Tests for V1-S07 stratified hand-labeling sampler.

Uses synthetic in-memory DuckDB + a tiny topics parquet so the tests
run in <1s and don't depend on the 200k-paper V1 corpus.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from scifield.epistemic.sampling import SamplingConfig, stratified_sample

# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

JOURNALS = ("ann_surg", "arthroscopy", "br_j_surg")
ERAS = ("pre2000", "2000-2009", "2010-2019", "2020+")
# Pick a year in the middle of each era so the CASE expression buckets
# the synthetic rows the way we expect.
ERA_YEAR = {
    "pre2000": 1998,
    "2000-2009": 2005,
    "2010-2019": 2015,
    "2020+": 2022,
}


def _build_papers(
    rows_per_cell: int = 50,
    n_topics: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plant ``rows_per_cell`` papers in each (journal, era) cell.

    Returns ``(papers_df, topics_df)``. Topic IDs are spread across the
    PMIDs in round-robin order so the synthetic distinct-topic count is
    exactly ``n_topics`` (assuming total PMIDs >> n_topics).
    """
    rows = []
    pmid = 1_000_000
    for j in JOURNALS:
        for e in ERAS:
            for k in range(rows_per_cell):
                rows.append(
                    {
                        "pmid": pmid,
                        "journal_slug": j,
                        "title": f"title-{pmid}",
                        "abstract": ("A" * 60 + f" synthetic abstract {pmid} cell={j}/{e} k={k}"),
                        "year": ERA_YEAR[e],
                        "fetched_at": "2026-01-01T00:00:00Z",
                    }
                )
                pmid += 1
    papers = pd.DataFrame(rows)

    # Topics: assign topic_id = (pmid % n_topics) - 1 so most rows have
    # a non-noise topic; reserve a slice with topic_id = -1 (noise) and a
    # slice missing from the table entirely.
    topic_rows = []
    for i, p in enumerate(papers["pmid"].tolist()):
        if i % 17 == 0:
            # Skip ~6% of papers from the topics table -> NULL after join.
            continue
        if i % 23 == 0:
            tid = -1
            noise = True
        else:
            tid = i % n_topics
            noise = False
        topic_rows.append({"pmid": p, "topic_id": tid, "is_noise": noise})
    topics = pd.DataFrame(topic_rows)
    return papers, topics


def _make_db(tmp_path: Path, papers: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB with a ``papers`` table from ``papers``."""
    con = duckdb.connect(":memory:")
    con.register("papers_src", papers)
    con.execute("CREATE TABLE papers AS SELECT * FROM papers_src")
    con.unregister("papers_src")
    return con


def _write_topics(tmp_path: Path, topics: pd.DataFrame) -> Path:
    p = tmp_path / "topics.parquet"
    topics.to_parquet(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sample_returns_exact_n(tmp_path: Path) -> None:
    papers, topics = _build_papers(rows_per_cell=50, n_topics=100)
    con = _make_db(tmp_path, papers)
    topics_path = _write_topics(tmp_path, topics)

    cfg = SamplingConfig(
        duckdb_path=tmp_path / "papers.duckdb",  # unused; con is passed directly
        topics_parquet=topics_path,
        n_sample=200,
        seed=20260522,
        topic_coverage_min=20,
    )
    out = stratified_sample(con, cfg)
    assert len(out) == cfg.n_sample
    assert list(out.columns) == [
        "pmid",
        "journal",
        "year",
        "era",
        "topic_id",
        "title",
        "abstract",
    ]


def test_sample_is_deterministic(tmp_path: Path) -> None:
    papers, topics = _build_papers(rows_per_cell=50, n_topics=100)
    topics_path = _write_topics(tmp_path, topics)

    cfg = SamplingConfig(
        duckdb_path=tmp_path / "papers.duckdb",
        topics_parquet=topics_path,
        n_sample=150,
        seed=20260522,
        topic_coverage_min=10,
    )
    con1 = _make_db(tmp_path, papers)
    out1 = stratified_sample(con1, cfg)
    con2 = _make_db(tmp_path, papers)
    out2 = stratified_sample(con2, cfg)

    assert out1["pmid"].tolist() == out2["pmid"].tolist()


def test_sample_changes_with_different_seed(tmp_path: Path) -> None:
    papers, topics = _build_papers(rows_per_cell=50, n_topics=100)
    topics_path = _write_topics(tmp_path, topics)

    cfg_a = SamplingConfig(
        duckdb_path=tmp_path / "papers.duckdb",
        topics_parquet=topics_path,
        n_sample=150,
        seed=1,
        topic_coverage_min=10,
    )
    cfg_b = SamplingConfig(
        duckdb_path=tmp_path / "papers.duckdb",
        topics_parquet=topics_path,
        n_sample=150,
        seed=2,
        topic_coverage_min=10,
    )
    out_a = stratified_sample(_make_db(tmp_path, papers), cfg_a)
    out_b = stratified_sample(_make_db(tmp_path, papers), cfg_b)
    assert out_a["pmid"].tolist() != out_b["pmid"].tolist()


def test_all_cells_represented(tmp_path: Path) -> None:
    papers, topics = _build_papers(rows_per_cell=50, n_topics=100)
    con = _make_db(tmp_path, papers)
    topics_path = _write_topics(tmp_path, topics)

    cfg = SamplingConfig(
        duckdb_path=tmp_path / "papers.duckdb",
        topics_parquet=topics_path,
        n_sample=240,
        seed=20260522,
        topic_coverage_min=20,
    )
    out = stratified_sample(con, cfg)

    expected_cells = {(j, e) for j in JOURNALS for e in ERAS}
    seen_cells = set(map(tuple, out[["journal", "era"]].drop_duplicates().to_numpy()))
    assert seen_cells == expected_cells


def test_topic_coverage_assertion_fires(tmp_path: Path) -> None:
    # Only 5 distinct topic_ids -> coverage assertion w/ min=200 must raise.
    papers, _ = _build_papers(rows_per_cell=50, n_topics=100)
    topic_rows = [
        {"pmid": p, "topic_id": i % 5, "is_noise": False}
        for i, p in enumerate(papers["pmid"].tolist())
    ]
    topics = pd.DataFrame(topic_rows)
    con = _make_db(tmp_path, papers)
    topics_path = _write_topics(tmp_path, topics)

    cfg = SamplingConfig(
        duckdb_path=tmp_path / "papers.duckdb",
        topics_parquet=topics_path,
        n_sample=120,
        seed=20260522,
        topic_coverage_min=200,
    )
    with pytest.raises(AssertionError, match="topic coverage"):
        stratified_sample(con, cfg)


def test_papers_without_topic_id_are_kept(tmp_path: Path) -> None:
    papers, topics = _build_papers(rows_per_cell=50, n_topics=100)
    # Verify the topics fixture genuinely omits some PMIDs.
    omitted = set(papers["pmid"]) - set(topics["pmid"])
    assert len(omitted) > 0, "fixture must omit some PMIDs from topics"

    con = _make_db(tmp_path, papers)
    topics_path = _write_topics(tmp_path, topics)

    cfg = SamplingConfig(
        duckdb_path=tmp_path / "papers.duckdb",
        topics_parquet=topics_path,
        n_sample=300,
        seed=20260522,
        topic_coverage_min=20,
    )
    out = stratified_sample(con, cfg)

    # topic_id must be a nullable Int64 and contain some <NA> rows
    # (because the join is a left join, not an inner join).
    assert str(out["topic_id"].dtype) == "Int64"
    assert (
        out["topic_id"].isna().any()
    ), "expected some rows to have NULL topic_id (left-join semantics)"
    # Sanity: no row sneakily got dropped.
    assert len(out) == cfg.n_sample
