"""Unit tests for the V2 cartography journal-flow data model.

No corpus I/O: every fixture is a small hand-built pandas DataFrame whose volumes,
first-appearance years, and citation flows are known by construction. The module is
pure, so the builder script (``V2/scripts/build_flow.py``) is not exercised here.

The load-bearing test is :func:`test_journals_key_on_slug_not_display_name`: it
constructs a frame where one ``journal_slug`` carries two *display* names (the
jama_surg / Archives-of-surgery split) and confirms the flow collapses them to a
single journal — the one bug this module exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scifield.cartography.flow import (
    CANONICAL_JOURNAL_SLUGS,
    add_citation_flows,
    assert_canonical_journals,
    build_flow_table,
    first_appearance,
    topic_key_for_grain,
)

_SEED = 20260609


def _known_frame() -> pd.DataFrame:
    """Two topics, three canonical journals, with known first-appearance years.

    Topic 0: ann_surg first in 2000 (2 papers), surgery in 2003, spine in 2006.
    Topic 1: surgery first in 2001, ann_surg in 2004.
    A fixed-seed RNG adds strictly-later duplicate rows so volume > 1 in some cells
    while the FIRST year is unchanged.
    """
    rng = np.random.default_rng(_SEED)
    rows: list[dict[str, object]] = []
    plan = [
        (0, "ann_surg", 2000, 2),
        (0, "surgery", 2003, 1),
        (0, "spine", 2006, 1),
        (1, "surgery", 2001, 1),
        (1, "ann_surg", 2004, 1),
    ]
    pmid = 1
    for topic, slug, year, n_first_year in plan:
        for _ in range(n_first_year):
            rows.append({"pmid": pmid, "topic_id": topic, "journal_slug": slug, "year": year})
            pmid += 1
        for _ in range(int(rng.integers(0, 3))):
            later = year + int(rng.integers(1, 4))
            rows.append({"pmid": pmid, "topic_id": topic, "journal_slug": slug, "year": later})
            pmid += 1
    return pd.DataFrame(rows)


def test_topic_key_for_grain_maps_leaf_and_mid() -> None:
    """leaf -> topic_id, mid -> mid_level_id; unknown grain raises."""
    assert topic_key_for_grain("leaf") == "topic_id"
    assert topic_key_for_grain("mid") == "mid_level_id"
    with pytest.raises(ValueError):
        topic_key_for_grain("top")


def test_assert_canonical_journals_passes_clean_and_rejects_display_name() -> None:
    """The clean ten pass; a display string raises."""
    assert_canonical_journals(list(CANONICAL_JOURNAL_SLUGS))
    assert_canonical_journals(pd.Series(["ann_surg", np.nan, "spine"]))  # NaN ignored
    with pytest.raises(ValueError):
        assert_canonical_journals(["ann_surg", "JAMA surgery"])


def test_first_appearance_takes_earliest_year_ignoring_later_duplicates() -> None:
    """first_year is the earliest per (topic, journal_slug); later dups ignored."""
    df = _known_frame()
    fa = first_appearance(df, topic_key="topic_id")
    assert set(fa.columns) == {"topic_id", "journal_slug", "first_year"}

    t0 = fa[fa["topic_id"] == 0].set_index("journal_slug")["first_year"]
    assert t0["ann_surg"] == 2000
    assert t0["surgery"] == 2003
    assert t0["spine"] == 2006
    t1 = fa[fa["topic_id"] == 1].set_index("journal_slug")["first_year"]
    assert t1["surgery"] == 2001
    assert t1["ann_surg"] == 2004


def test_build_flow_table_volume_and_first_appearance_flag() -> None:
    """n_papers counts every paper in a cell; is_first_appearance marks the anchor."""
    df = _known_frame()
    flow = build_flow_table(df, grain="leaf")

    assert list(flow.columns) == [
        "topic_id",
        "journal_slug",
        "year",
        "n_papers",
        "is_first_appearance",
        "out_flow",
        "in_flow",
    ]
    # Volume: topic 0 / ann_surg / 2000 had exactly 2 first-year papers.
    cell = flow[
        (flow["topic_id"] == 0) & (flow["journal_slug"] == "ann_surg") & (flow["year"] == 2000)
    ]
    assert int(cell["n_papers"].iloc[0]) == 2
    assert bool(cell["is_first_appearance"].iloc[0]) is True

    # Total volume equals the row count of the source frame.
    assert int(flow["n_papers"].sum()) == len(df)

    # Exactly one first-appearance row per (topic, journal) pair.
    n_pairs = df.groupby(["topic_id", "journal_slug"]).ngroups
    assert int(flow["is_first_appearance"].sum()) == n_pairs


def test_no_citation_edges_yields_sentinel_zero_and_nan() -> None:
    """Without citation_edges, out_flow == 0 (int) and in_flow is NaN (float)."""
    df = _known_frame()
    flow = build_flow_table(df, grain="leaf")
    assert (flow["out_flow"] == 0).all()
    assert flow["in_flow"].isna().all()
    assert flow["in_flow"].dtype == np.dtype("float64")


def test_journals_key_on_slug_not_display_name() -> None:
    """Two display names sharing one slug collapse to a single journal (jama_surg).

    This is the canonical-journal-mapping guarantee: the flow keys on journal_slug,
    so 'JAMA surgery' and 'Archives of surgery ...' (both slug jama_surg) must NOT
    appear as two journals.
    """
    df = pd.DataFrame(
        {
            "pmid": [1, 2, 3, 4],
            "topic_id": [0, 0, 0, 0],
            "journal_slug": ["jama_surg", "jama_surg", "jama_surg", "ann_surg"],
            # Same slug, two display names (the rename split). Ignored by the flow.
            "journal": [
                "JAMA surgery",
                "Archives of surgery (Chicago, Ill. : 1960)",
                "JAMA surgery",
                "Annals of surgery",
            ],
            "year": [2010, 2008, 2014, 2009],
        }
    )
    flow = build_flow_table(df, grain="leaf")

    journals = set(flow["journal_slug"].unique())
    assert journals == {"jama_surg", "ann_surg"}  # exactly two, not three
    # jama_surg volume is the sum of both display names (3 papers across 3 years).
    assert int(flow.loc[flow["journal_slug"] == "jama_surg", "n_papers"].sum()) == 3
    # First appearance anchored to the earliest year (2008, an "Archives" paper).
    first_row = flow[(flow["journal_slug"] == "jama_surg") & flow["is_first_appearance"]]
    assert len(first_row) == 1
    assert int(first_row["year"].iloc[0]) == 2008


def test_mid_grain_uses_mid_level_id_column() -> None:
    """grain='mid' keys the flow on mid_level_id, not topic_id."""
    df = pd.DataFrame(
        {
            "pmid": [1, 2, 3],
            "mid_level_id": [5, 5, 5],
            "journal_slug": ["ann_surg", "ann_surg", "spine"],
            "year": [2000, 2001, 2002],
        }
    )
    flow = build_flow_table(df, grain="mid")
    assert "mid_level_id" in flow.columns
    assert "topic_id" not in flow.columns
    assert int(flow.loc[flow["journal_slug"] == "ann_surg", "n_papers"].sum()) == 2


def test_citation_flows_out_and_in_attribution() -> None:
    """add_citation_flows counts edges by citing (out) and cited (in) endpoint cell."""
    papers = pd.DataFrame(
        {
            "pmid": [10, 11, 20],
            "topic_id": [0, 0, 1],
            "journal_slug": ["ann_surg", "ann_surg", "spine"],
            "year": [2000, 2000, 1998],
        }
    )
    # Two edges: 10->20 and 11->20. Both cite the spine/topic1/1998 cell.
    edges = pd.DataFrame({"citing_pmid": [10, 11], "cited_pmid": [20, 20]})
    flow = build_flow_table(papers, citation_edges=edges, grain="leaf")

    ann = flow[(flow["topic_id"] == 0) & (flow["journal_slug"] == "ann_surg")]
    spine = flow[(flow["topic_id"] == 1) & (flow["journal_slug"] == "spine")]
    # ann_surg/2000 made 2 citations out, received 0.
    assert int(ann["out_flow"].iloc[0]) == 2
    assert int(ann["in_flow"].iloc[0]) == 0
    # spine/1998 received 2 citations in, made 0.
    assert int(spine["out_flow"].iloc[0]) == 0
    assert int(spine["in_flow"].iloc[0]) == 2
    # in_flow / out_flow are integer-typed when edges are supplied.
    assert flow["in_flow"].dtype == np.dtype("int64")
    assert flow["out_flow"].dtype == np.dtype("int64")


def test_add_citation_flows_pmid_dtype_mismatch_still_joins() -> None:
    """String vs int pmids across frames still match (both cast to string)."""
    flow = pd.DataFrame(
        {
            "topic_id": [0, 1],
            "journal_slug": ["ann_surg", "spine"],
            "year": [2000, 1998],
            "n_papers": [1, 1],
            "is_first_appearance": [True, True],
        }
    )
    papers = pd.DataFrame(
        {
            "pmid": ["10", "20"],  # strings
            "topic_id": [0, 1],
            "journal_slug": ["ann_surg", "spine"],
            "year": [2000, 1998],
        }
    )
    edges = pd.DataFrame({"citing_pmid": [10], "cited_pmid": [20]})  # ints
    out = add_citation_flows(flow, papers, edges, topic_key="topic_id")
    assert int(out.loc[out["journal_slug"] == "ann_surg", "out_flow"].iloc[0]) == 1
    assert int(out.loc[out["journal_slug"] == "spine", "in_flow"].iloc[0]) == 1


def test_non_canonical_journal_raises() -> None:
    """A display-name leak (11th journal) raises in build_flow_table."""
    df = pd.DataFrame(
        {
            "pmid": [1],
            "topic_id": [0],
            "journal_slug": ["JAMA surgery"],  # a display name, not a slug
            "year": [2010],
        }
    )
    with pytest.raises(ValueError):
        build_flow_table(df, grain="leaf")


def test_empty_input_yields_well_formed_empty_flow() -> None:
    """Empty input returns an empty frame with the full column set and dtypes."""
    empty = pd.DataFrame({"pmid": [], "topic_id": [], "journal_slug": [], "year": []})
    flow = build_flow_table(empty, grain="leaf")
    assert flow.empty
    assert list(flow.columns) == [
        "topic_id",
        "journal_slug",
        "year",
        "n_papers",
        "is_first_appearance",
        "out_flow",
        "in_flow",
    ]
