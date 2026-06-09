"""Unit tests for the V2 cartography cascade engine.

No corpus I/O: every fixture is a small hand-built pandas DataFrame whose
first-appearance order is known by construction, so origins, lead-lag signs, and
diffusion-curve shapes are computable by hand. The module is pure, so the builder
(``V2/scripts/build_cascade.py``) and the notebook are not exercised here.

The load-bearing tests are the **sign correctness** of ``lead_lag_matrix`` (a
frame where journal A always precedes B ⇒ A leads, positive ``mean_lag``), the
**origin / tie handling** of ``origin_attribution``, the **monotone-to-1.0**
property of ``diffusion_curve``, and **year-subset purity** (a filtered slice
recomputes anchors rather than trusting a precomputed flag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scifield.cartography.cascade import (
    cascade_edges,
    diffusion_curve,
    lead_lag_matrix,
    origin_attribution,
)

_SEED = 20260609


def _known_flow_slice() -> pd.DataFrame:
    """Flow-shaped slice: A precedes B precedes C in every topic, plus jitter.

    Three topics. In each, journal A first-appears in year ``2000 + topic``, B
    three years later, C six years later. A fixed-seed RNG adds strictly-later
    duplicate cell-rows (volume noise) so the slice is not one row per pair; the
    later rows must never change the recomputed first-appearance year.
    """
    rng = np.random.default_rng(_SEED)
    rows: list[dict[str, object]] = []
    for topic in range(3):
        base = 2000 + topic
        for offset, journal in [(0, "ann_surg"), (3, "spine"), (6, "surgery")]:
            first_year = base + offset
            rows.append(
                {"topic_id": topic, "journal_slug": journal, "year": first_year, "n_papers": 1}
            )
            for _ in range(int(rng.integers(0, 3))):
                later = first_year + int(rng.integers(1, 4))
                rows.append(
                    {"topic_id": topic, "journal_slug": journal, "year": later, "n_papers": 1}
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# origin_attribution
# --------------------------------------------------------------------------- #


def test_origin_attribution_picks_earliest_journal() -> None:
    """The origin is the earliest-publishing journal; no tie here."""
    df = _known_flow_slice()
    origins = origin_attribution(df, topic_key="topic_id").set_index("topic_id")

    # ann_surg is always first (offset 0).
    for topic in range(3):
        assert origins.loc[topic, "origin_journal_slug"] == "ann_surg"
        assert origins.loc[topic, "origin_year"] == 2000 + topic
        assert origins.loc[topic, "n_journals"] == 3
        assert bool(origins.loc[topic, "tie"]) is False
        assert origins.loc[topic, "n_co_earliest"] == 1


def test_origin_attribution_tie_break_is_alphabetical() -> None:
    """Co-earliest journals set tie=True; the winner is alphabetically first."""
    df = pd.DataFrame(
        {
            "topic_id": [0, 0, 0],
            # spine and ann_surg both first in 2000 (tie); surgery later.
            "journal_slug": ["spine", "ann_surg", "surgery"],
            "year": [2000, 2000, 2005],
        }
    )
    row = origin_attribution(df, topic_key="topic_id").iloc[0]
    assert row["tie"] is np.True_ or bool(row["tie"]) is True
    assert row["n_co_earliest"] == 2
    # Alphabetically-first co-earliest slug: ann_surg < spine.
    assert row["origin_journal_slug"] == "ann_surg"
    assert row["origin_year"] == 2000
    assert row["n_journals"] == 3


def test_origin_attribution_accepts_first_year_frame() -> None:
    """A precomputed first-appearance frame is accepted directly."""
    first = pd.DataFrame(
        {
            "topic_id": [0, 0, 1],
            "journal_slug": ["surgery", "ann_surg", "spine"],
            "first_year": [2002, 1999, 2010],
        }
    )
    origins = origin_attribution(first, topic_key="topic_id").set_index("topic_id")
    assert origins.loc[0, "origin_journal_slug"] == "ann_surg"
    assert origins.loc[0, "origin_year"] == 1999
    assert origins.loc[1, "origin_journal_slug"] == "spine"


# --------------------------------------------------------------------------- #
# lead_lag_matrix
# --------------------------------------------------------------------------- #


def test_lead_lag_sign_positive_when_i_leads() -> None:
    """A precedes B by 3 years ⇒ mean_lag(A, B) == +3 and A leads every topic."""
    df = _known_flow_slice()
    mat = lead_lag_matrix(df, topic_key="topic_id")

    by_pair = {(r.journal_i, r.journal_j): r for r in mat.itertuples()}
    # ann_surg leads spine by exactly 3 years in every shared topic.
    ab = by_pair[("ann_surg", "spine")]
    assert ab.mean_lag == 3.0
    assert ab.median_lag == 3.0
    assert ab.n_shared == 3
    assert ab.n_i_leads == 3

    # The reverse is the negation: spine "leads" ann_surg by -3 (i.e. follows).
    ba = by_pair[("spine", "ann_surg")]
    assert ba.mean_lag == -3.0
    assert ba.n_i_leads == 0

    # ann_surg vs surgery: 6-year lead.
    assert by_pair[("ann_surg", "surgery")].mean_lag == 6.0


def test_lead_lag_matrix_antisymmetric_and_symmetric_shared() -> None:
    """mean_lag is antisymmetric; n_shared is symmetric across the pair."""
    df = _known_flow_slice()
    mat = lead_lag_matrix(df, topic_key="topic_id")
    lag = {(r.journal_i, r.journal_j): r.mean_lag for r in mat.itertuples()}
    shared = {(r.journal_i, r.journal_j): r.n_shared for r in mat.itertuples()}
    for a, b in [("ann_surg", "spine"), ("ann_surg", "surgery"), ("spine", "surgery")]:
        assert lag[(a, b)] == -lag[(b, a)]
        assert shared[(a, b)] == shared[(b, a)]
    # No diagonal (a journal vs itself).
    assert all(r.journal_i != r.journal_j for r in mat.itertuples())


def test_lead_lag_ties_count_for_neither_direction() -> None:
    """When two journals tie, neither leads: lag 0 and n_i_leads 0 both ways."""
    df = pd.DataFrame(
        {
            "topic_id": [0, 0],
            "journal_slug": ["ann_surg", "spine"],
            "year": [2000, 2000],
        }
    )
    mat = lead_lag_matrix(df, topic_key="topic_id")
    by_pair = {(r.journal_i, r.journal_j): r for r in mat.itertuples()}
    assert by_pair[("ann_surg", "spine")].mean_lag == 0.0
    assert by_pair[("ann_surg", "spine")].n_i_leads == 0
    assert by_pair[("spine", "ann_surg")].n_i_leads == 0


# --------------------------------------------------------------------------- #
# diffusion_curve
# --------------------------------------------------------------------------- #


def test_diffusion_curve_monotone_reaches_full_panel() -> None:
    """A topic in all panel journals reaches reach_fraction 1.0; span is positive."""
    # Ten journals, one per year, all in one topic.
    journals = [
        "ann_surg",
        "arthroscopy",
        "br_j_surg",
        "clin_orthop_relat_res",
        "j_am_coll_surg",
        "j_arthroplasty",
        "j_bone_joint_surg_am",
        "jama_surg",
        "spine",
        "surgery",
    ]
    df = pd.DataFrame(
        {
            "topic_id": [0] * 10,
            "journal_slug": journals,
            "year": list(range(2000, 2010)),
        }
    )
    curve = diffusion_curve(df, topic_key="topic_id").iloc[0]
    assert curve["n_journals"] == 10
    assert curve["reach_fraction"] == 1.0
    assert curve["origin_year"] == 2000
    assert curve["span_years"] == 9
    # Half of ten journals reached after ~4-5 years (linear 1/yr adoption).
    assert 3.5 <= curve["t50_empirical"] <= 5.5
    # A clean monotone curve with 10 distinct offsets fits a logistic.
    assert bool(curve["curve_fitted"]) is True
    assert np.isfinite(curve["t50_logistic"])
    assert curve["logistic_rate"] > 0


def test_diffusion_curve_single_journal_topic_has_no_spread() -> None:
    """A topic reaching one journal has zero span and an unfittable curve."""
    df = pd.DataFrame({"topic_id": [0], "journal_slug": ["ann_surg"], "year": [2005]})
    curve = diffusion_curve(df, topic_key="topic_id").iloc[0]
    assert curve["n_journals"] == 1
    assert curve["reach_fraction"] == 0.1
    assert curve["span_years"] == 0
    assert curve["t50_empirical"] == 0.0
    assert bool(curve["curve_fitted"]) is False
    assert np.isnan(curve["t50_logistic"])


def test_diffusion_reach_fraction_below_one_for_partial_panel() -> None:
    """A topic in 4 of 10 journals reaches reach_fraction 0.4 (< 1.0)."""
    df = pd.DataFrame(
        {
            "topic_id": [0, 0, 0, 0],
            "journal_slug": ["ann_surg", "spine", "surgery", "br_j_surg"],
            "year": [2000, 2001, 2002, 2003],
        }
    )
    curve = diffusion_curve(df, topic_key="topic_id").iloc[0]
    assert curve["n_journals"] == 4
    assert curve["reach_fraction"] == 0.4
    assert curve["span_years"] == 3


# --------------------------------------------------------------------------- #
# cascade_edges (wrapper)
# --------------------------------------------------------------------------- #


def test_cascade_edges_match_directed_seeding_network() -> None:
    """Forward edges (earlier seeds later) carry weight 1.0; reverses 0.0."""
    df = _known_flow_slice()
    edges = cascade_edges(df, topic_key="topic_id")
    assert list(edges.columns) == ["src", "dst", "n_precedes", "n_shared", "weight"]
    w = {(r.src, r.dst): r.weight for r in edges.itertuples()}
    assert w[("ann_surg", "spine")] == 1.0
    assert w[("ann_surg", "surgery")] == 1.0
    assert w[("spine", "ann_surg")] == 0.0


# --------------------------------------------------------------------------- #
# year-subset purity (the V2-S04 contract)
# --------------------------------------------------------------------------- #


def test_year_subset_recomputes_anchors_not_precomputed_flag() -> None:
    """Filtering the slice changes the origin; a precomputed flag is ignored.

    Build a slice where journal spine has the panel-earliest year (1990) but is
    DELETED from the held-out (year >= 2000) view; an ``is_first_appearance``
    flag is also present and deliberately *wrong* for the subset to prove the
    engine recomputes from ``year`` rather than trusting the flag.
    """
    full = pd.DataFrame(
        {
            "topic_id": [0, 0, 0],
            "journal_slug": ["spine", "ann_surg", "surgery"],
            "year": [1990, 2001, 2003],
            # Flag marks spine as the first appearance for the FULL window.
            "is_first_appearance": [True, True, True],
        }
    )
    # On the full slice, spine (1990) is the origin.
    full_origin = origin_attribution(full, topic_key="topic_id").iloc[0]
    assert full_origin["origin_journal_slug"] == "spine"
    assert full_origin["origin_year"] == 1990

    # Held-out window year >= 2000 drops spine entirely → origin is now ann_surg.
    held = full[full["year"] >= 2000]
    held_origin = origin_attribution(held, topic_key="topic_id").iloc[0]
    assert held_origin["origin_journal_slug"] == "ann_surg"
    assert held_origin["origin_year"] == 2001
    assert held_origin["n_journals"] == 2


def test_mid_grain_topic_key_is_honoured() -> None:
    """topic_key='mid_level_id' keys every output on the mid column."""
    df = pd.DataFrame(
        {
            "mid_level_id": [7, 7, 7],
            "journal_slug": ["ann_surg", "spine", "surgery"],
            "year": [2000, 2002, 2004],
        }
    )
    origins = origin_attribution(df, topic_key="mid_level_id")
    assert "mid_level_id" in origins.columns
    assert "topic_id" not in origins.columns
    assert origins.iloc[0]["origin_journal_slug"] == "ann_surg"

    curve = diffusion_curve(df, topic_key="mid_level_id")
    assert "mid_level_id" in curve.columns


# --------------------------------------------------------------------------- #
# empty-input hygiene
# --------------------------------------------------------------------------- #


def test_empty_input_yields_well_formed_empty_frames() -> None:
    """Every public function returns an empty, well-formed frame on empty input."""
    empty = pd.DataFrame({"topic_id": [], "journal_slug": [], "year": []})

    origins = origin_attribution(empty)
    assert origins.empty
    assert list(origins.columns) == [
        "topic_id",
        "origin_journal_slug",
        "origin_year",
        "n_journals",
        "tie",
        "n_co_earliest",
    ]

    mat = lead_lag_matrix(empty)
    assert mat.empty
    assert list(mat.columns) == [
        "journal_i",
        "journal_j",
        "mean_lag",
        "median_lag",
        "n_shared",
        "n_i_leads",
    ]

    curve = diffusion_curve(empty)
    assert curve.empty
    assert list(curve.columns) == [
        "topic_id",
        "origin_year",
        "n_journals",
        "reach_fraction",
        "span_years",
        "t50_empirical",
        "t50_logistic",
        "logistic_rate",
        "curve_fitted",
    ]
