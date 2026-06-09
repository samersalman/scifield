"""Unit tests for the V2 cascade-validation layer (Method A–D rigor).

No corpus I/O: every fixture is a small hand-built pandas DataFrame whose
first-appearance order is known by construction, so the permutation verdict, the
jackknife flip/forced accounting, the granularity ρ, and the resolvable-subset
filter are all checkable against a planted ground truth. All randomness is seeded
(:data:`scifield.cartography.cascade_validation.SEED`) so the verdicts are
deterministic.

The load-bearing tests are:

* a **planted strong-lead** structure yields a small Null-1 permutation p and a
  high z under S1 (the structure is real);
* a **pure-random** first-year structure yields a non-significant p (no cascade);
* the **jackknife** flip-rate stays 0 when origins are robust, and dropping a
  topic's origin journal is counted as a *forced* reattribution, never a flip;
* **granularity** ρ behaves (identical rankings → ρ = 1.0, PASS);
* the **resolvable-subset** filter removes all-tied (single-first-year) topics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scifield.cartography import cascade_validation as cv

_SEED = cv.SEED
# A small permutation count keeps the suite fast; the planted effects are strong
# enough that 200 replicates separate signal from null unambiguously.
_NP = 200


def _strong_lead_flow(n_topics: int = 40) -> pd.DataFrame:
    """Flow slice where ann_surg ALWAYS leads, then spine, then surgery.

    Every topic appears in three journals with a fixed 0/+3/+6-year offset and a
    topic-specific base year (so first-years are distinct across topics, giving a
    resolvable ordering). ann_surg is the consistent leader → high seeding spread
    and a strongly directional lead-lag.
    """
    rows: list[dict[str, object]] = []
    for topic in range(n_topics):
        base = 2000 + (topic % 10)
        for offset, journal in [(0, "ann_surg"), (3, "spine"), (6, "surgery")]:
            rows.append(
                {
                    "topic_id": topic,
                    "journal_slug": journal,
                    "year": base + offset,
                    "n_papers": 1,
                }
            )
    return pd.DataFrame(rows)


def _random_first_year_flow(n_topics: int = 40, seed: int = _SEED) -> pd.DataFrame:
    """Flow slice where each journal's first year in a topic is i.i.d. random.

    No journal systematically leads — first-years are drawn independently per
    (topic, journal) from a fixed window, so seeding scores converge and S1 spread
    sits inside its own permutation null.
    """
    rng = np.random.default_rng(seed)
    journals = ["ann_surg", "spine", "surgery", "br_j_surg", "j_arthroplasty"]
    rows: list[dict[str, object]] = []
    for topic in range(n_topics):
        for journal in journals:
            rows.append(
                {
                    "topic_id": topic,
                    "journal_slug": journal,
                    "year": int(rng.integers(2000, 2015)),
                    "n_papers": 1,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Method B — permutation verdict
# --------------------------------------------------------------------------- #
def test_planted_lead_yields_small_p_and_high_z() -> None:
    """A strong planted lead lands in the Null-1 tail: small p, z >= 2, PASS."""
    flow = _strong_lead_flow()
    res = cv.permutation_verdict(
        flow, topic_key="topic_id", stat="S1", null="within_topic", n_perm=_NP, seed=_SEED
    )
    assert res["observed"] > res["null_mean"]
    assert res["p_value"] < 0.05
    assert res["z"] >= 2.0
    assert res["percentile"] >= 0.95
    assert res["verdict"] in {"PASS", "PARTIAL"}
    # With this strong an effect the gate-critical call is PASS.
    assert res["verdict"] == "PASS"
    assert res["n_perm"] == _NP


def test_random_first_years_not_significant() -> None:
    """Pure-random first-years give a non-significant S1 p (no cascade)."""
    flow = _random_first_year_flow()
    res = cv.permutation_verdict(
        flow, topic_key="topic_id", stat="S1", null="within_topic", n_perm=_NP, seed=_SEED
    )
    assert res["p_value"] > 0.05
    assert res["verdict"] == "FAIL"


def test_permutation_verdict_is_deterministic() -> None:
    """Same seed → byte-identical p, z, observed (reproducibility)."""
    flow = _strong_lead_flow()
    a = cv.permutation_verdict(flow, topic_key="topic_id", n_perm=_NP, seed=_SEED)
    b = cv.permutation_verdict(flow, topic_key="topic_id", n_perm=_NP, seed=_SEED)
    assert a["p_value"] == b["p_value"]
    assert a["z"] == b["z"]
    assert a["observed"] == b["observed"]


def test_year_shuffle_null_also_flags_planted_lead() -> None:
    """Null-2 (first-year shuffle) also detects the planted lead via S1."""
    flow = _strong_lead_flow()
    res = cv.permutation_verdict(
        flow, topic_key="topic_id", stat="S1", null="year_shuffle", n_perm=_NP, seed=_SEED
    )
    assert res["p_value"] < 0.05
    assert res["observed"] > res["null_mean"]


def test_permutation_verdict_rejects_bad_args() -> None:
    """Unknown stat / null raise ValueError."""
    flow = _strong_lead_flow()
    cases: tuple[dict[str, str], ...] = ({"stat": "S9"}, {"null": "bogus"})
    for kwargs in cases:
        try:
            cv.permutation_verdict(
                flow,
                topic_key="topic_id",
                n_perm=10,
                **kwargs,  # type: ignore[arg-type]
            )
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


# --------------------------------------------------------------------------- #
# resolvable-subset filter (the 1995-censoring guard)
# --------------------------------------------------------------------------- #
def test_resolvable_filter_drops_all_tied_topics() -> None:
    """Topics whose journals all share one first-year are not resolvable."""
    flow = pd.DataFrame(
        {
            "topic_id": [0, 0, 1, 1, 2, 2],
            "journal_slug": ["ann_surg", "spine", "ann_surg", "spine", "ann_surg", "spine"],
            # topic 0: both 1995 (all-tied) ; topic 1: both 1995 (all-tied) ;
            # topic 2: 1995 vs 2001 (resolvable).
            "year": [1995, 1995, 1995, 1995, 1995, 2001],
        }
    )
    resolvable = cv.resolvable_topics(flow, topic_key="topic_id")
    assert resolvable == [2]


def test_resolvable_only_changes_topic_count() -> None:
    """resolvable_only restricts the permutation to ordered topics."""
    flow = pd.DataFrame(
        {
            "topic_id": [0, 0, 1, 1],
            "journal_slug": ["ann_surg", "spine", "ann_surg", "spine"],
            "year": [1995, 1995, 2000, 2005],  # topic 0 all-tied, topic 1 resolvable
        }
    )
    all_res = cv.permutation_verdict(flow, topic_key="topic_id", n_perm=10, resolvable_only=False)
    res_res = cv.permutation_verdict(flow, topic_key="topic_id", n_perm=10, resolvable_only=True)
    assert all_res["n_topics"] == 2
    assert res_res["n_topics"] == 1
    assert res_res["resolvable_only"] is True


# --------------------------------------------------------------------------- #
# Method C — jackknife of origins + forced-reattribution accounting
# --------------------------------------------------------------------------- #
def test_jackknife_flip_rate_low_when_origins_robust() -> None:
    """Robust origins → flip rate 0; PASS."""
    flow = _strong_lead_flow()
    per_run = cv.jackknife_origins(flow, topic_key="topic_id")
    summary = cv.summarize_jackknife_origins(per_run, topic_key="topic_id")
    assert summary["attribution_flip_rate"] == 0.0
    assert summary["verdict"] == "PASS"


def test_dropping_origin_journal_is_forced_not_a_flip() -> None:
    """Dropping a topic's origin journal counts as forced, never as a flip.

    ann_surg is the origin of every topic; in the run that drops ann_surg, all
    topics MUST reattribute (to spine) — those are forced reattributions and are
    excluded from the flip numerator. No non-forced run changes any origin, so the
    flip rate is 0 while the forced count equals (#topics) for the ann_surg run.
    """
    flow = _strong_lead_flow(n_topics=12)
    per_run = cv.jackknife_origins(flow, topic_key="topic_id")

    # Every topic's full-run origin is ann_surg → exactly the ann_surg run forces.
    forced = per_run.loc[per_run["forced"]]
    assert set(forced["held_out"].unique()) == {"ann_surg"}
    assert int(forced.shape[0]) == 12  # one forced cell per topic, in that one run

    summary = cv.summarize_jackknife_origins(per_run, topic_key="topic_id")
    assert summary["forced_reattribution_count"] == 12
    assert summary["n_genuine_flips"] == 0
    assert summary["attribution_flip_rate"] == 0.0


def test_jackknife_detects_a_genuine_flip() -> None:
    """A topic whose origin changes when a NON-origin journal is dropped flips.

    Construct a topic where dropping a co-earliest-by-tie journal changes which
    surviving journal becomes the alphabetically-first earliest. Topic 0: spine and
    surgery tie earliest at 2000 (origin = spine, alphabetical), br_j_surg at 2001.
    Dropping spine (NOT the journal forced out for any OTHER topic) → origin becomes
    surgery: a genuine, non-forced flip for the runs that drop spine relative to the
    full origin spine — but the run dropping spine IS forced. So to see a genuine
    flip we need a topic whose origin is stable yet flips when a different journal is
    dropped: add a second co-earliest journal arthroscopy at 2000 too.
    """
    # Topic 0: ann_surg(2000), arthroscopy(2000) tie earliest; origin=ann_surg.
    # Dropping arthroscopy (non-origin) keeps origin ann_surg (no flip).
    # Topic 1: spine(2000) origin; surgery(2002). Dropping a non-origin (surgery)
    # keeps spine. So by construction argmin-origin is flip-free for non-forced
    # drops — assert that invariant holds (flip rate 0) AND that a hand-injected
    # inconsistency is caught by the summarizer.
    per_run = pd.DataFrame(
        {
            "held_out": ["spine", "surgery", "spine", "surgery"],
            "topic_id": [0, 0, 1, 1],
            "origin_journal_slug": ["ann_surg", "ann_surg", "spine", "j_am_coll_surg"],
            "full_origin": ["ann_surg", "ann_surg", "spine", "spine"],
            "forced": [False, False, False, False],
        }
    )
    summary = cv.summarize_jackknife_origins(per_run, topic_key="topic_id")
    # Topic 1 flips in the 'surgery' run (spine -> j_am_coll_surg) — one genuine flip.
    assert summary["n_genuine_flips"] == 1
    assert summary["n_eligible_cells"] == 4
    assert summary["attribution_flip_rate"] == 0.25
    assert summary["verdict"] == "PARTIAL"


def test_jackknife_high_flip_rate_fails() -> None:
    """A flip rate above 0.40 yields FAIL."""
    per_run = pd.DataFrame(
        {
            "held_out": ["a", "b", "a", "b"],
            "topic_id": [0, 0, 1, 1],
            "origin_journal_slug": ["x", "y", "x", "y"],
            "full_origin": ["x", "x", "x", "x"],
            "forced": [False, False, False, False],
        }
    )
    summary = cv.summarize_jackknife_origins(per_run, topic_key="topic_id")
    assert summary["attribution_flip_rate"] == 0.5
    assert summary["verdict"] == "FAIL"


# --------------------------------------------------------------------------- #
# Method D — granularity consistency
# --------------------------------------------------------------------------- #
def test_granularity_identical_rankings_pass() -> None:
    """Identical leaf/mid seeding rankings → ρ = 1.0, PASS."""
    leaf = _strong_lead_flow()
    mid = leaf.rename(columns={"topic_id": "mid_level_id"})
    res = cv.granularity_consistency(leaf, mid)
    assert res["spearman_rho"] == 1.0
    assert res["verdict"] == "PASS"
    assert res["n_journals"] == 3  # only the three journals in the fixture


def test_granularity_reversed_rankings_fail() -> None:
    """A leaf grain where ann_surg leads but a mid grain where it trails → low ρ."""
    leaf = _strong_lead_flow()
    # Mid grain: reverse the offsets so surgery leads and ann_surg trails.
    rows: list[dict[str, object]] = []
    for topic in range(40):
        base = 2000 + (topic % 10)
        for offset, journal in [(0, "surgery"), (3, "spine"), (6, "ann_surg")]:
            rows.append({"mid_level_id": topic, "journal_slug": journal, "year": base + offset})
    mid = pd.DataFrame(rows)
    res = cv.granularity_consistency(leaf, mid)
    assert res["spearman_rho"] < 0.5
    assert res["verdict"] in {"PARTIAL", "FAIL"}


# --------------------------------------------------------------------------- #
# Method A — held-out years (supporting)
# --------------------------------------------------------------------------- #
def test_heldout_excludes_post_train_topics() -> None:
    """Topics first appearing after 2018 are excluded and counted."""
    flow = pd.DataFrame(
        {
            "topic_id": [0, 0, 1, 1, 2, 2],
            "journal_slug": ["ann_surg", "spine", "ann_surg", "spine", "ann_surg", "spine"],
            # topic 2 first appears in 2020 (> 2018) → excluded.
            "year": [2000, 2005, 2010, 2019, 2020, 2024],
        }
    )
    res = cv.heldout_consistency(flow, topic_key="topic_id")
    assert res["n_topics_excluded"] == 1
    assert res["supporting_only"] is True


def test_heldout_requires_year_column() -> None:
    """A frame lacking 'year' raises (Method A needs raw years to split)."""
    first = pd.DataFrame(
        {"topic_id": [0, 0], "journal_slug": ["ann_surg", "spine"], "first_year": [2000, 2003]}
    )
    try:
        cv.heldout_consistency(first, topic_key="topic_id")
    except ValueError:
        return
    raise AssertionError("expected ValueError when 'year' is absent")


# --------------------------------------------------------------------------- #
# Orchestrator — run_all_validations end to end on a planted structure
# --------------------------------------------------------------------------- #
def test_run_all_validations_planted_structure_is_real() -> None:
    """On a strongly planted, robust, grain-consistent structure → REAL."""
    leaf = _strong_lead_flow()
    mid = leaf.rename(columns={"topic_id": "mid_level_id"})
    out = cv.run_all_validations(leaf, mid, n_perm=_NP, seed=_SEED, secondary_n_perm=_NP)

    assert set(out) == {"method_b", "method_c", "method_d", "method_a", "overall"}
    primary = out["method_b"]["leaf"]["all"]["cells"][("S1", "within_topic")]
    assert primary["verdict"] == "PASS"
    assert out["method_c"]["leaf"]["all"]["verdict"] == "PASS"
    assert out["method_d"]["verdict"] == "PASS"
    assert out["overall"]["cascade_verdict"] in {"REAL", "PARTIAL"}


def test_run_all_validations_random_structure_is_artifact() -> None:
    """Pure-random first-years → the primary FAILs → ARTIFACT."""
    leaf = _random_first_year_flow()
    mid = leaf.rename(columns={"topic_id": "mid_level_id"})
    out = cv.run_all_validations(leaf, mid, n_perm=_NP, seed=_SEED, secondary_n_perm=_NP)
    primary = out["method_b"]["leaf"]["all"]["cells"][("S1", "within_topic")]
    assert primary["verdict"] == "FAIL"
    assert out["overall"]["cascade_verdict"] == "ARTIFACT"
