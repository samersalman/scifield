"""Unit tests for the V2-S05 journal-role taxonomy + citational velocity layer.

No network / no data files: every fixture is a small hand-built in-memory pandas
DataFrame whose role structure or velocity is known by construction, so expected
labels, scores, and stability values are computable by hand. The module is pure, so
the driver script / notebook are deliberately NOT exercised here.

Any synthetic noise uses a FIXED seed for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scifield.cartography.nulls import rank_stability
from scifield.cartography.roles import (
    ROLE_COMPONENTS,
    assign_roles,
    bridge_betweenness,
    citational_velocity,
    role_scores,
    role_scores_jackknife,
)
from scifield.findings.seeding import directed_seeding_network

_SEED = 20260609


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _ordered_flow() -> pd.DataFrame:
    """A→B→C→D path: A always first (source), D always last (terminal).

    Across ``n_topics`` topics each journal publishes the same topic, with first
    years A < B < C < D. Citation flows are set so A is a pure exporter
    (out_flow >> in_flow) and D a pure importer (in_flow >> out_flow), reinforcing
    A=source / D=terminal on BOTH the temporal and the citation axes. B and C sit
    in the middle, where B is also placed on the strong lead→follow path so it
    scores high betweenness (a bridge).
    """
    rows: list[dict[str, object]] = []
    n_topics = 8
    first_year = {"A": 2000, "B": 2002, "C": 2004, "D": 2006}
    out_in = {
        "A": (40, 2),  # net exporter
        "B": (10, 10),  # neutral
        "C": (10, 10),  # neutral
        "D": (2, 40),  # net importer
    }
    for topic in range(n_topics):
        for j in ("A", "B", "C", "D"):
            o, i = out_in[j]
            rows.append(
                {
                    "topic_id": topic,
                    "journal_slug": j,
                    "year": first_year[j],
                    "out_flow": o,
                    "in_flow": i,
                }
            )
    return pd.DataFrame(rows)


def _net_for(flow: pd.DataFrame) -> pd.DataFrame:
    """Directed seeding network for a flow frame (the role_scores second input)."""
    return directed_seeding_network(
        flow[["topic_id", "journal_slug", "year"]], entity_col="journal_slug"
    )


# --------------------------------------------------------------------------- #
# role_scores — source / terminal extremes
# --------------------------------------------------------------------------- #
def test_first_publisher_scores_source_late_adopter_scores_terminal() -> None:
    """A (always first + net exporter) -> source; D (always last + importer) -> terminal."""
    flow = _ordered_flow()
    rs = role_scores(flow, _net_for(flow), topic_key="topic_id").set_index("journal_slug")

    # A leads every topic -> highest seeding; net exporter -> highest net_outflow.
    assert rs.loc["A", "seeding_score"] == 1.0
    assert rs.loc["D", "seeding_score"] == 0.0
    assert rs.loc["A", "net_outflow_share"] > 0
    assert rs.loc["D", "net_outflow_share"] < 0

    # Labels by the documented argmax-of-z rule.
    assert rs.loc["A", "role"] == "source"
    assert rs.loc["D", "role"] == "terminal"

    # Source score orders A > D; terminal score orders D > A.
    assert rs.loc["A", "source"] > rs.loc["D", "source"]
    assert rs.loc["D", "terminal"] > rs.loc["A", "terminal"]


def test_role_columns_and_canonical_panel() -> None:
    """role_scores returns the documented columns, one row per journal in the slice."""
    flow = _ordered_flow()
    rs = role_scores(flow, _net_for(flow), topic_key="topic_id")
    expected = {
        "journal_slug",
        "seeding_score",
        "net_outflow_share",
        "betweenness",
        "source",
        "bridge",
        "terminal",
        "source_z",
        "bridge_z",
        "terminal_z",
        "role",
    }
    assert set(rs.columns) == expected
    assert sorted(rs["journal_slug"]) == ["A", "B", "C", "D"]
    # Every role label is one of the three components.
    assert set(rs["role"]).issubset(set(ROLE_COMPONENTS))


# --------------------------------------------------------------------------- #
# betweenness picks the obvious bridge in a hand-built path network
# --------------------------------------------------------------------------- #
def test_betweenness_picks_the_obvious_bridge() -> None:
    """On a path A->M->Z, the middle node M carries all the betweenness."""
    # Strong lead edges A->M and M->Z (weight 1.0); the only A->Z path runs via M.
    net = pd.DataFrame(
        {
            "src": ["A", "M"],
            "dst": ["M", "Z"],
            "n_precedes": [5, 5],
            "n_shared": [5, 5],
            "weight": [1.0, 1.0],
        }
    )
    br = bridge_betweenness(net, weight_threshold=0.5).set_index("journal_slug")
    # M sits on the single A->Z shortest path; endpoints have zero betweenness.
    assert br.loc["M", "bridge"] > 0
    assert br.loc["A", "bridge"] == 0.0
    assert br.loc["Z", "bridge"] == 0.0
    # M is strictly the most central node.
    assert br["bridge"].idxmax() == "M"


def test_betweenness_weight_threshold_drops_weak_edges() -> None:
    """An edge below the weight threshold is excluded from the graph."""
    net = pd.DataFrame(
        {
            "src": ["A", "M"],
            "dst": ["M", "Z"],
            "n_precedes": [5, 1],
            "n_shared": [5, 5],
            "weight": [1.0, 0.2],  # M->Z weak (0.2) -> dropped at threshold 0.5
        }
    )
    br = bridge_betweenness(net, weight_threshold=0.5)
    # Only the A->M edge survives -> no node is *between* two others -> all zero.
    assert (br["bridge"] == 0.0).all()


# --------------------------------------------------------------------------- #
# assign_roles — the documented argmax + tie-break rule
# --------------------------------------------------------------------------- #
def test_assign_roles_argmax_of_zscores() -> None:
    """role = argmax over (source_z, bridge_z, terminal_z)."""
    df = pd.DataFrame(
        {
            "journal_slug": ["s", "b", "t"],
            "source_z": [2.0, 0.0, -1.0],
            "bridge_z": [0.0, 2.0, 0.0],
            "terminal_z": [-1.0, 0.0, 2.0],
        }
    )
    out = assign_roles(df).set_index("journal_slug")
    assert out.loc["s", "role"] == "source"
    assert out.loc["b", "role"] == "bridge"
    assert out.loc["t", "role"] == "terminal"


def test_assign_roles_tie_breaks_source_over_bridge_over_terminal() -> None:
    """Equal maxima break in the fixed priority order source > bridge > terminal."""
    df = pd.DataFrame(
        {
            "journal_slug": ["all_equal", "bridge_terminal_tie"],
            "source_z": [1.0, -5.0],
            "bridge_z": [1.0, 1.0],
            "terminal_z": [1.0, 1.0],
        }
    )
    out = assign_roles(df).set_index("journal_slug")
    # First row: all three tie -> source wins (highest priority).
    assert out.loc["all_equal", "role"] == "source"
    # Second row: bridge and terminal tie at the max -> bridge wins.
    assert out.loc["bridge_terminal_tie", "role"] == "bridge"


# --------------------------------------------------------------------------- #
# citational velocity
# --------------------------------------------------------------------------- #
def test_citational_velocity_summaries_are_correct() -> None:
    """Lag = citing_year - pub_year; per-journal median/IQR computed correctly."""
    # Journal "fast": one paper (2010) cited at +1, +1, +2 -> lags [1,1,2].
    # Journal "slow": one paper (2010) cited at +8, +9, +10 -> lags [8,9,10].
    paper_meta = pd.DataFrame(
        {
            "openalex_id": ["Wfast", "Wslow"],
            "journal_slug": ["fast", "slow"],
            "year": [2010, 2010],
        }
    )
    citations = pd.DataFrame(
        {
            "openalex_id": ["Wfast", "Wfast", "Wfast", "Wslow", "Wslow", "Wslow"],
            "citing_year": [2011, 2011, 2012, 2018, 2019, 2020],
        }
    )
    vel = citational_velocity(citations, paper_meta).set_index("journal_slug")

    assert vel.loc["fast", "median_lag"] == 1.0
    assert vel.loc["fast", "mean_lag"] == (1 + 1 + 2) / 3
    assert vel.loc["slow", "median_lag"] == 9.0
    assert vel.loc["fast", "n_citations"] == 3
    assert vel.loc["fast", "n_papers_cited"] == 1
    # frac_within_2y: fast = all 3 within 2y -> 1.0; slow = 0.
    assert vel.loc["fast", "frac_within_2y"] == 1.0
    assert vel.loc["slow", "frac_within_2y"] == 0.0
    # Relative indicator: the below-panel-median journal is "fast".
    assert vel.loc["fast", "velocity"] == "fast"
    assert vel.loc["slow", "velocity"] == "slow"


def test_citational_velocity_drops_negative_and_huge_lags() -> None:
    """A citation before publication (negative lag) or a >max_lag lag is dropped."""
    paper_meta = pd.DataFrame({"openalex_id": ["W1"], "journal_slug": ["j"], "year": [2010]})
    citations = pd.DataFrame(
        {
            "openalex_id": ["W1", "W1", "W1"],
            "citing_year": [2005, 2012, 2200],  # -5 (drop), +2 (keep), +190 (drop)
        }
    )
    vel = citational_velocity(citations, paper_meta, max_lag=60).set_index("journal_slug")
    assert vel.loc["j", "n_citations"] == 1
    assert vel.loc["j", "median_lag"] == 2.0


# --------------------------------------------------------------------------- #
# jackknife rank stability == 1.0 when scores are robust
# --------------------------------------------------------------------------- #
def test_jackknife_rank_stability_is_one_when_order_is_robust() -> None:
    """A clean A>B>C>D ordering survives dropping any single journal -> stability 1.0."""
    flow = _ordered_flow()
    jk = role_scores_jackknife(flow, topic_key="topic_id", score_col="source_z")
    # 4 leave-one-out runs, each with the 3 surviving journals.
    assert jk["held_out"].nunique() == 4
    rrc = rank_stability(jk, key_col="journal_slug", score_col="score", run_col="held_out")
    # The source ordering A>B>C>D is monotone and preserved on every shared triple.
    assert rrc == pytest.approx(1.0)


def test_jackknife_columns_and_runs() -> None:
    """role_scores_jackknife returns the documented columns + one run per journal."""
    flow = _ordered_flow()
    jk = role_scores_jackknife(flow, topic_key="topic_id")
    assert set(jk.columns) == {
        "held_out",
        "journal_slug",
        "score",
        "source",
        "bridge",
        "terminal",
        "role",
    }
    # The held-out journal never appears among the surviving rows of its own run.
    for held in jk["held_out"].unique():
        run = jk[jk["held_out"] == held]
        assert held not in set(run["journal_slug"])


# --------------------------------------------------------------------------- #
# empty / robustness
# --------------------------------------------------------------------------- #
def test_empty_inputs_yield_well_formed_empty_frames() -> None:
    """Every public function tolerates empty input without raising."""
    empty_flow = pd.DataFrame(
        {
            "topic_id": pd.Series([], dtype="int64"),
            "journal_slug": pd.Series([], dtype="object"),
            "year": pd.Series([], dtype="int64"),
            "out_flow": pd.Series([], dtype="int64"),
            "in_flow": pd.Series([], dtype="int64"),
        }
    )
    empty_net = pd.DataFrame({"src": [], "dst": [], "n_precedes": [], "n_shared": [], "weight": []})
    rs = role_scores(empty_flow, empty_net, topic_key="topic_id")
    assert rs.empty
    assert "role" in rs.columns

    br = bridge_betweenness(empty_net)
    assert br.empty
    assert list(br.columns) == ["journal_slug", "bridge"]

    jk = role_scores_jackknife(empty_flow, topic_key="topic_id")
    assert jk.empty

    empty_meta = pd.DataFrame({"openalex_id": [], "journal_slug": [], "year": []})
    empty_cit = pd.DataFrame({"openalex_id": [], "citing_year": []})
    vel = citational_velocity(empty_cit, empty_meta)
    assert vel.empty
    assert "velocity" in vel.columns


def test_velocity_with_fixed_noise_is_deterministic() -> None:
    """A fixed-seed synthetic citation set gives a stable, reproducible summary."""
    rng = np.random.default_rng(_SEED)
    # 50 papers in one journal, each cited a few times at small positive lags.
    pmeta_rows = []
    cit_rows = []
    for k in range(50):
        oa = f"W{k}"
        pmeta_rows.append({"openalex_id": oa, "journal_slug": "j", "year": 2000})
        for _ in range(int(rng.integers(1, 5))):
            cit_rows.append({"openalex_id": oa, "citing_year": 2000 + int(rng.integers(1, 6))})
    vel = citational_velocity(pd.DataFrame(cit_rows), pd.DataFrame(pmeta_rows))
    # All lags lie in [1, 5] -> median within that range, no NaNs.
    assert 1.0 <= float(vel.iloc[0]["median_lag"]) <= 5.0
    assert vel.iloc[0]["n_papers_cited"] == 50
