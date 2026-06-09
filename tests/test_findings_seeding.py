"""Unit tests for the V1-S15 cross-journal seeding bonus layer.

No network / no data files: every fixture is a small hand-built in-memory pandas
DataFrame whose first-publication order is known by construction, so the expected
seeding scores and edge weights are computable by hand. The module is pure, so
the driver notebook (``notebooks/14_bonus_cross_journal.ipynb``) is deliberately
NOT exercised here.

Any synthetic noise uses a FIXED seed for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scifield.findings.seeding import (
    SPECIALTY_GROUPS,
    directed_seeding_network,
    first_publication_year,
    normalize_journal,
    seeding_by_group,
    seeding_score,
    specialty_of,
)

_SEED = 20260608


def _known_order_frame() -> pd.DataFrame:
    """Five topics where A publishes before B before C, plus fixed jitter rows.

    A's first_year is always the earliest, then B, then C. A fixed-seed RNG adds a
    handful of *later* duplicate papers per (topic, journal) so the frame is not
    one row per pair; later duplicates must not change the FIRST year, which is the
    point of the noise.
    """
    rng = np.random.default_rng(_SEED)
    rows: list[dict[str, object]] = []
    for topic in range(5):
        base = 2000 + topic  # A's first year for this topic
        for offset, journal in [(0, "A"), (3, "B"), (6, "C")]:
            first_year = base + offset
            rows.append({"topic_id": topic, "journal": journal, "year": first_year})
            # A few strictly-later duplicate papers (noise): never earlier.
            for _ in range(int(rng.integers(0, 3))):
                later = first_year + int(rng.integers(1, 3))
                rows.append({"topic_id": topic, "journal": journal, "year": later})
    return pd.DataFrame(rows)


def test_first_publication_year_takes_earliest_ignoring_later_duplicates() -> None:
    """first_year is the earliest year per (topic, entity); later dups ignored."""
    df = _known_order_frame()
    fpy = first_publication_year(df, entity_col="journal")

    # Exactly one row per (topic, journal) pair: 5 topics x 3 journals.
    assert len(fpy) == 15
    assert set(fpy.columns) == {"topic_id", "journal", "first_year"}

    # Topic 0: A=2000, B=2003, C=2006 regardless of the later jitter rows.
    t0 = fpy[fpy["topic_id"] == 0].set_index("journal")["first_year"]
    assert t0["A"] == 2000
    assert t0["B"] == 2003
    assert t0["C"] == 2006


def test_known_order_yields_descending_seeding_scores() -> None:
    """A always first, then B, then C -> seeding_score A > B > C, with exact values."""
    df = _known_order_frame()
    scores = seeding_score(df, entity_col="journal").set_index("journal")

    # m == 3 every topic: ranks 0/1/2 -> normalized leads 1.0 / 0.5 / 0.0.
    assert scores.loc["A", "seeding_score"] == 1.0
    assert scores.loc["B", "seeding_score"] == 0.5
    assert scores.loc["C", "seeding_score"] == 0.0
    assert (
        scores.loc["A", "seeding_score"]
        > scores.loc["B", "seeding_score"]
        > scores.loc["C", "seeding_score"]
    )
    # Each journal participated in all five topics.
    assert scores.loc["A", "n_topics"] == 5
    assert scores.loc["B", "n_topics"] == 5
    assert scores.loc["C", "n_topics"] == 5


def test_seeding_score_rows_sorted_descending() -> None:
    """The returned frame is ordered by seeding_score descending (A, B, C)."""
    df = _known_order_frame()
    scores = seeding_score(df, entity_col="journal")
    assert list(scores["journal"]) == ["A", "B", "C"]


def test_directed_network_forward_edges_weight_one_reverses_zero() -> None:
    """Forward edges A->B, A->C, B->C carry weight ~1.0; the reverses ~0.0."""
    df = _known_order_frame()
    net = directed_seeding_network(df, entity_col="journal")

    assert list(net.columns) == ["src", "dst", "n_precedes", "n_shared", "weight"]
    # 3 entities, all co-occurring -> 6 ordered directed edges.
    assert len(net) == 6

    w = {(r.src, r.dst): r.weight for r in net.itertuples()}
    # Forward (earlier seeds later) -> precedes in every shared topic.
    assert w[("A", "B")] == 1.0
    assert w[("A", "C")] == 1.0
    assert w[("B", "C")] == 1.0
    # Reverse -> never precedes.
    assert w[("B", "A")] == 0.0
    assert w[("C", "A")] == 0.0
    assert w[("C", "B")] == 0.0

    # n_shared is symmetric and equals the 5 shared topics on every pair.
    shared = {(r.src, r.dst): r.n_shared for r in net.itertuples()}
    for pair in [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B")]:
        assert shared[pair] == 5


def test_tie_handling_min_rank_and_normalized_lead() -> None:
    """Two journals tying at the earliest year share rank 0; the last scores 0.0."""
    # Topic 0: X and Y both first in 2000 (tie), Z later in 2005.
    # Topic 1: same tie structure with a different late year.
    df = pd.DataFrame(
        {
            "topic_id": [0, 0, 1, 1, 0, 1],
            "journal": ["X", "Y", "X", "Y", "Z", "Z"],
            "year": [2000, 2000, 2010, 2010, 2005, 2015],
        }
    )
    scores = seeding_score(df, entity_col="journal").set_index("journal")

    # m == 3; min-rank: X,Y -> r=0 -> lead 1.0; Z -> r=2 -> lead 0.0.
    assert scores.loc["X", "seeding_score"] == 1.0
    assert scores.loc["Y", "seeding_score"] == 1.0
    assert scores.loc["Z", "seeding_score"] == 0.0

    # In a tie, neither X->Y nor Y->X strictly precedes -> both weights 0.
    net = directed_seeding_network(df, entity_col="journal")
    w = {(r.src, r.dst): r.weight for r in net.itertuples()}
    assert w[("X", "Y")] == 0.0
    assert w[("Y", "X")] == 0.0
    # But both tie-winners strictly precede Z in every shared topic.
    assert w[("X", "Z")] == 1.0
    assert w[("Y", "Z")] == 1.0


def test_single_entity_topic_scores_one() -> None:
    """An entity alone in a topic (m == 1) contributes a perfect 1.0 lead."""
    df = pd.DataFrame(
        {
            "topic_id": [0, 1, 1],
            "journal": ["solo", "solo", "other"],
            "year": [1999, 2001, 2000],
        }
    )
    scores = seeding_score(df, entity_col="journal").set_index("journal")
    # Topic 0: solo alone -> lead 1.0. Topic 1: other(2000) before solo(2001)
    # -> other lead 1.0, solo lead 0.0. solo mean = (1.0 + 0.0)/2 = 0.5.
    assert scores.loc["solo", "seeding_score"] == 0.5
    assert scores.loc["other", "seeding_score"] == 1.0
    assert scores.loc["solo", "n_topics"] == 2
    assert scores.loc["other", "n_topics"] == 1


def test_min_papers_threshold_filters_thin_year_buckets() -> None:
    """min_papers is evaluated per year: a 1-paper-per-year entity is excluded."""
    # entity "thin" never has 2 papers in one year; "thick" has 2 in 2001.
    df = pd.DataFrame(
        {
            "topic_id": [0, 0, 0, 0],
            "journal": ["thin", "thin", "thick", "thick"],
            "year": [2000, 2002, 2001, 2001],
        }
    )
    fpy = first_publication_year(df, entity_col="journal", min_papers=2)
    # Only "thick" qualifies, first_year == 2001.
    assert list(fpy["journal"]) == ["thick"]
    assert int(fpy.iloc[0]["first_year"]) == 2001


def test_specialty_of_resolves_slugs_display_names_and_unknowns() -> None:
    """specialty_of maps slugs AND display names to groups; unknown -> None."""
    # Slugs.
    assert specialty_of("spine") == "orthopedic"
    assert specialty_of("j_bone_joint_surg_am") == "orthopedic"
    assert specialty_of("ann_surg") == "general_surgery"
    assert specialty_of("jama_surg") == "general_surgery"

    # Display names (mixed case, spaces).
    assert specialty_of("J Bone Joint Surg Am") == "orthopedic"
    assert specialty_of("Clin Orthop Relat Res") == "orthopedic"
    assert specialty_of("Ann Surg") == "general_surgery"
    assert specialty_of("JAMA Surg") == "general_surgery"

    # Case / whitespace insensitivity and hyphen-vs-underscore agreement.
    assert specialty_of("  J ARTHROPLASTY  ") == "orthopedic"
    assert specialty_of("clin-orthop-relat-res") == "orthopedic"

    # Unknown journal and blank/None -> None.
    assert specialty_of("Nature") is None
    assert specialty_of("") is None
    assert specialty_of(None) is None
    assert specialty_of(np.nan) is None


def test_specialty_groups_partition_is_five_vs_five() -> None:
    """The locked grouping holds five distinct journals per specialty (by slug)."""
    ortho_slugs = {normalize_journal(n) for n in SPECIALTY_GROUPS["orthopedic"]}
    gen_slugs = {normalize_journal(n) for n in SPECIALTY_GROUPS["general_surgery"]}
    # Display names normalise onto the same five slugs in each group.
    assert len(ortho_slugs) == 5
    assert len(gen_slugs) == 5
    # The two specialties are disjoint.
    assert ortho_slugs.isdisjoint(gen_slugs)


def test_seeding_by_group_skips_blank_and_nan_entities() -> None:
    """Blank / NaN / whitespace country codes are dropped, not scored or imputed."""
    df = pd.DataFrame(
        {
            "topic_id": [0, 0, 0, 1, 1, 1],
            "country_code": ["US", "GB", "", "US", np.nan, "   "],
            "year": [2000, 2005, 2001, 2002, 2003, 2004],
        }
    )
    scores = seeding_by_group(df, group_col="country_code")
    entities = set(scores["country_code"])

    # Only real codes survive; blank "", NaN, and whitespace "   " are excluded.
    assert entities == {"US", "GB"}
    assert "" not in entities
    assert "   " not in entities

    # US is earliest in both topics it appears (after blanks removed):
    #   topic 0: US 2000 < GB 2005 -> US lead 1.0, GB lead 0.0
    #   topic 1: US is the only surviving entity -> US lead 1.0
    # US mean = 1.0; GB = 0.0.
    s = scores.set_index("country_code")
    assert s.loc["US", "seeding_score"] == 1.0
    assert s.loc["GB", "seeding_score"] == 0.0
    assert s.loc["US", "n_topics"] == 2
    assert s.loc["GB", "n_topics"] == 1


def test_seeding_by_group_all_blank_returns_empty() -> None:
    """If every entity is blank/NaN the result is an empty, well-formed frame."""
    df = pd.DataFrame(
        {
            "topic_id": [0, 1],
            "institution_canonical_id": ["", np.nan],
            "year": [2000, 2001],
        }
    )
    scores = seeding_by_group(df, group_col="institution_canonical_id")
    assert scores.empty
    assert list(scores.columns) == [
        "institution_canonical_id",
        "seeding_score",
        "n_topics",
    ]


def test_seeding_by_group_missing_column_returns_empty() -> None:
    """A group_col absent from the frame yields an empty frame, not a KeyError."""
    df = pd.DataFrame({"topic_id": [0], "journal": ["A"], "year": [2000]})
    scores = seeding_by_group(df, group_col="country_code")
    assert scores.empty
    assert list(scores.columns) == ["country_code", "seeding_score", "n_topics"]


def test_empty_input_frames_are_handled() -> None:
    """Empty input yields empty, well-formed frames from every public function."""
    empty = pd.DataFrame({"topic_id": [], "journal": [], "year": []})

    fpy = first_publication_year(empty)
    assert fpy.empty
    assert list(fpy.columns) == ["topic_id", "journal", "first_year"]

    scores = seeding_score(empty)
    assert scores.empty
    assert list(scores.columns) == ["journal", "seeding_score", "n_topics"]

    net = directed_seeding_network(empty)
    assert net.empty
    assert list(net.columns) == ["src", "dst", "n_precedes", "n_shared", "weight"]


def test_normalize_journal_collapses_separators() -> None:
    """Spaces, hyphens, and underscores all collapse to a single underscore key."""
    assert normalize_journal("J Bone Joint Surg Am") == "j_bone_joint_surg_am"
    assert normalize_journal("j_bone_joint_surg_am") == "j_bone_joint_surg_am"
    assert normalize_journal("clin-orthop-relat-res") == "clin_orthop_relat_res"
    assert normalize_journal("  Ann   Surg  ") == "ann_surg"
    assert normalize_journal(None) == ""
    assert normalize_journal(np.nan) == ""
