"""Unit tests for the V2-S06 novelty-origins aggregations.

No network / no data files: every fixture is a small hand-built in-memory pandas
DataFrame whose sector / country / recombination structure is known by
construction, so the expected contrasts are computable by hand. The module is
pure, so the builder/notebook are not exercised here.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from scifield.cartography.origins import (
    COMPANY_TYPE,
    bootstrap_delta_ci,
    novelty_by_country,
    novelty_by_institution_type,
    paper_sector_tags,
    recombination_by_topic,
    recombination_events,
    split_half_stability,
)

SEED = 20260609


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _institutions() -> pd.DataFrame:
    """Four institutions: one company (US), one education (US), one healthcare
    (GB), one with a blank type and blank country."""
    return pd.DataFrame(
        {
            "institution_canonical_id": ["I_co", "I_edu", "I_hosp", "I_blank"],
            "type": ["company", "education", "healthcare", ""],
            "country_code": ["US", "US", "GB", ""],
        }
    )


def _paper_inst() -> pd.DataFrame:
    """Paper→institution rows.

    p1: company + education author (=> is_company True, spans US sectors).
    p2: education only (=> is_company False).
    p3: healthcare (GB) + the blank-metadata institution.
    """
    return pd.DataFrame(
        {
            "pmid": ["p1", "p1", "p2", "p3", "p3"],
            "author_position": [0, 1, 0, 0, 1],
            "institution_canonical_id": ["I_co", "I_edu", "I_edu", "I_hosp", "I_blank"],
        }
    )


def _paper_novelty() -> pd.DataFrame:
    """Company-affiliated p1 is the most novel; p2 (education) least."""
    return pd.DataFrame(
        {
            "pmid": ["p1", "p2", "p3"],
            "sem_nov_mean": [0.9, 0.1, 0.5],
            "cd5": [0.8, 0.2, 0.4],
        }
    )


# --------------------------------------------------------------------------- #
# Sector × novelty
# --------------------------------------------------------------------------- #
def test_company_vs_noncompany_novelty_contrast() -> None:
    """The company row's mean novelty exceeds the education/healthcare rows."""
    out = novelty_by_institution_type(
        _paper_novelty(),
        _paper_inst(),
        _institutions(),
        novelty_cols=["sem_nov_mean", "cd5"],
    )
    by_type = out.set_index("type")

    # company sees only p1 (0.9); education sees p1 and p2 (mean 0.5);
    # healthcare sees only p3 (0.5).
    assert by_type.loc[COMPANY_TYPE, "sem_nov_mean_mean"] == 0.9
    assert by_type.loc["education", "sem_nov_mean_mean"] == 0.5
    assert by_type.loc["healthcare", "sem_nov_mean_mean"] == 0.5
    # Company is strictly more novel than the non-company sectors here.
    assert (
        by_type.loc[COMPANY_TYPE, "sem_nov_mean_mean"]
        > by_type.loc["education", "sem_nov_mean_mean"]
    )
    # cd5 contrast is computed independently and matches the hand value.
    assert by_type.loc[COMPANY_TYPE, "cd5_mean"] == 0.8

    # Blank institution type never becomes a phantom sector row.
    assert "" not in by_type.index


def test_any_author_affiliation_rule_company_flag() -> None:
    """A paper with one company author counts as company (any-affiliation rule)."""
    tags = paper_sector_tags(_paper_inst(), _institutions())
    # p1 has a company author => is_company True and appears under 'company'.
    p1 = tags[tags["pmid"] == "p1"]
    assert p1["is_company"].all()
    assert COMPANY_TYPE in set(p1["type"])
    # p1 also spans 'education' (its other author) — multi-sector tagging.
    assert "education" in set(p1["type"])
    # p2 has no company author => is_company False everywhere.
    p2 = tags[tags["pmid"] == "p2"]
    assert not p2["is_company"].any()
    # Blank-type institution contributes no tag row.
    assert (tags["type"].str.strip() != "").all()


def test_paper_count_sums_above_paper_total_for_multisector() -> None:
    """A multi-sector paper is counted in each sector it spans (documented rule)."""
    out = novelty_by_institution_type(
        _paper_novelty(),
        _paper_inst(),
        _institutions(),
        novelty_cols=["sem_nov_mean"],
    )
    # 3 distinct papers, but p1 spans company+education => total counts > 3.
    assert int(out["n_papers"].sum()) > 3


# --------------------------------------------------------------------------- #
# Geography × novelty
# --------------------------------------------------------------------------- #
def test_blank_country_codes_excluded() -> None:
    """Blank country codes are dropped, never imputed."""
    out = novelty_by_country(
        _paper_novelty(),
        _paper_inst(),
        _institutions(),
        novelty_cols=["sem_nov_mean"],
        low_n_threshold=2,
    )
    countries = set(out["country_code"])
    assert countries == {"US", "GB"}
    assert "" not in countries
    # GB sees only p3 (n=1) => flagged low_n at threshold 2.
    gb = out.set_index("country_code").loc["GB"]
    assert bool(gb["low_n"]) is True
    assert gb["sem_nov_mean_mean"] == 0.5


def test_low_n_flag_threshold() -> None:
    """Countries below the threshold are flagged; above are not."""
    out = novelty_by_country(
        _paper_novelty(),
        _paper_inst(),
        _institutions(),
        novelty_cols=["sem_nov_mean"],
        low_n_threshold=2,
    )
    by_country = out.set_index("country_code")
    # US sees p1 and p2 (n=2) => not low_n at threshold 2.
    assert bool(by_country.loc["US", "low_n"]) is False


# --------------------------------------------------------------------------- #
# Topic recombination
# --------------------------------------------------------------------------- #
def test_recombination_score_higher_for_more_topics() -> None:
    """A paper citing 3 topics scores higher than one citing 1 topic."""
    refs = pd.DataFrame(
        {
            "citing_pmid": ["multi", "multi", "multi", "single", "single"],
            "ref_topic_id": [10, 20, 30, 7, 7],
        }
    )
    ev = recombination_events(refs).set_index("citing_pmid")

    assert ev.loc["multi", "n_distinct_topics"] == 3
    assert ev.loc["single", "n_distinct_topics"] == 1
    assert ev.loc["multi", "n_distinct_topics"] > ev.loc["single", "n_distinct_topics"]
    # Entropy: single-topic paper has zero entropy; multi-topic is positive.
    assert ev.loc["single", "topic_entropy"] == 0.0
    assert ev.loc["multi", "topic_entropy"] > 0.0
    # is_recombination boolean.
    assert bool(ev.loc["multi", "is_recombination"]) is True
    assert bool(ev.loc["single", "is_recombination"]) is False
    # Resolved-reference count counts every resolved row.
    assert ev.loc["single", "n_references_resolved"] == 2


def test_recombination_entropy_even_spread_exceeds_skewed() -> None:
    """Even spread over 2 topics has higher entropy than a skewed spread."""
    even = pd.DataFrame({"citing_pmid": ["e", "e"], "ref_topic_id": [1, 2]})
    skew = pd.DataFrame({"citing_pmid": ["s", "s", "s", "s"], "ref_topic_id": [1, 1, 1, 2]})
    e_ent = recombination_events(even).set_index("citing_pmid").loc["e", "topic_entropy"]
    s_ent = recombination_events(skew).set_index("citing_pmid").loc["s", "topic_entropy"]
    # Even 50/50 split => entropy ln(2); skewed 3/1 split < ln(2).
    assert np.isclose(e_ent, np.log(2))
    assert s_ent < e_ent


def test_recombination_by_topic_summary() -> None:
    """Per-topic summary aggregates recombination over each focal topic."""
    refs = pd.DataFrame(
        {
            "citing_pmid": ["a", "a", "b", "c", "c", "c"],
            "ref_topic_id": [1, 2, 5, 1, 2, 3],
        }
    )
    events = recombination_events(refs)
    paper_topic = pd.DataFrame({"pmid": ["a", "b", "c"], "topic_id": [100, 100, 200]})
    summary = recombination_by_topic(events, paper_topic).set_index("topic_id")
    # Topic 100: paper a (2 topics) + paper b (1 topic) => mean 1.5, rate 0.5.
    assert summary.loc[100, "n_papers"] == 2
    assert np.isclose(summary.loc[100, "mean_distinct_topics"], 1.5)
    assert np.isclose(summary.loc[100, "recombination_rate"], 0.5)
    # Topic 200: paper c (3 topics) => mean 3, rate 1.0.
    assert np.isclose(summary.loc[200, "mean_distinct_topics"], 3.0)
    assert np.isclose(summary.loc[200, "recombination_rate"], 1.0)


def test_empty_inputs_return_empty_frames() -> None:
    """Empty inputs yield empty, correctly-columned frames (no crash)."""
    empty_refs = pd.DataFrame({"citing_pmid": [], "ref_topic_id": []})
    ev = recombination_events(empty_refs)
    assert list(ev.columns) == [
        "citing_pmid",
        "n_references_resolved",
        "n_distinct_topics",
        "topic_entropy",
        "is_recombination",
    ]
    assert ev.empty


# --------------------------------------------------------------------------- #
# Split-half stability (grain-independent Method-D analogue, geography axis)
# --------------------------------------------------------------------------- #
def _stable_per_key_frame() -> pd.DataFrame:
    """Per-key frame where the per-key ORDERING is identical in both halves.

    Five keys with cleanly separated, non-overlapping novelty bands (A < B < C < D <
    E). Whatever the random 50/50 split, each key's mean lands in its band, so the
    two halves' rankings are identical => Spearman ρ == 1.0. 60 papers per key keeps
    every key well above n_min in both halves.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i, key in enumerate(["A", "B", "C", "D", "E"]):
        base = float(i)  # bands 0,1,2,3,4 — non-overlapping with width-0.1 jitter
        for _ in range(60):
            rows.append({"country_code": key, "sem_nov_mean": base + rng.uniform(0, 0.1)})
    return pd.DataFrame(rows)


def test_split_half_stability_perfect_when_ordering_identical() -> None:
    """ρ == 1.0 when the per-key ordering is preserved in both halves."""
    out = split_half_stability(
        _stable_per_key_frame(),
        key_col="country_code",
        novelty_col="sem_nov_mean",
        seed=SEED,
        n_min=20,
        mode="random",
    )
    assert out["mode"] == "random"
    assert out["n_keys"] == 5
    assert np.isclose(out["spearman_rho"], 1.0)


def test_split_half_stability_low_when_pure_noise() -> None:
    """ρ is well below 1 when per-key means are pure noise (no real ordering)."""
    rng = np.random.default_rng(1)
    # 12 keys, all drawn from the SAME distribution => no stable ordering.
    rows = []
    for k in range(12):
        for _ in range(80):
            rows.append({"country_code": f"K{k}", "sem_nov_mean": rng.normal(0.0, 1.0)})
    out = split_half_stability(
        pd.DataFrame(rows),
        key_col="country_code",
        novelty_col="sem_nov_mean",
        seed=SEED,
        n_min=20,
        mode="random",
    )
    assert out["n_keys"] == 12
    # No real signal => the two halves' rankings are essentially uncorrelated.
    assert abs(cast(float, out["spearman_rho"])) < 0.6


def test_split_half_stability_n_min_filters_thin_keys() -> None:
    """Keys without >= n_min papers in BOTH halves are excluded."""
    frame = _stable_per_key_frame()
    # Add a thin key with only 4 papers — cannot reach n_min in both halves.
    thin = pd.DataFrame({"country_code": ["Z"] * 4, "sem_nov_mean": [9.0, 9.1, 9.2, 9.3]})
    out = split_half_stability(
        pd.concat([frame, thin], ignore_index=True),
        key_col="country_code",
        novelty_col="sem_nov_mean",
        seed=SEED,
        n_min=20,
        mode="random",
    )
    # Only the 5 well-sampled keys survive; the thin key Z is dropped.
    assert out["n_keys"] == 5


def test_split_half_stability_temporal_mode() -> None:
    """Temporal mode splits early-vs-late and preserves a stable ordering."""
    rng = np.random.default_rng(2)
    rows = []
    for i, key in enumerate(["A", "B", "C", "D"]):
        base = float(i)
        for yr in range(2000, 2020):  # 20 years => both halves well-sampled
            for _ in range(3):
                rows.append(
                    {
                        "country_code": key,
                        "sem_nov_mean": base + rng.uniform(0, 0.1),
                        "year": yr,
                    }
                )
    out = split_half_stability(
        pd.DataFrame(rows),
        key_col="country_code",
        novelty_col="sem_nov_mean",
        seed=SEED,
        n_min=20,
        mode="temporal",
    )
    assert out["mode"] == "temporal"
    assert out["n_keys"] == 4
    assert np.isclose(out["spearman_rho"], 1.0)


# --------------------------------------------------------------------------- #
# Bootstrap delta CI (sector near-null vs real-gap robustness)
# --------------------------------------------------------------------------- #
def test_bootstrap_delta_ci_spans_zero_for_true_null() -> None:
    """A true null (both groups same distribution) => CI spans zero."""
    rng = np.random.default_rng(3)
    n = 2000
    df = pd.DataFrame(
        {
            "is_company": [True] * n + [False] * n,
            "sem_nov_mean": np.concatenate([rng.normal(0.4, 0.1, n), rng.normal(0.4, 0.1, n)]),
        }
    )
    out = bootstrap_delta_ci(
        df, flag_col="is_company", novelty_col="sem_nov_mean", seed=SEED, n_boot=500
    )
    assert out["spans_zero"] is True
    assert cast(float, out["ci_low"]) < 0.0 < cast(float, out["ci_high"])
    assert abs(cast(float, out["delta"])) < 0.02


def test_bootstrap_delta_ci_excludes_zero_for_real_gap() -> None:
    """A planted real gap (+0.2) => CI excludes zero and brackets the gap."""
    rng = np.random.default_rng(4)
    n = 2000
    df = pd.DataFrame(
        {
            "is_company": [True] * n + [False] * n,
            "sem_nov_mean": np.concatenate([rng.normal(0.6, 0.1, n), rng.normal(0.4, 0.1, n)]),
        }
    )
    out = bootstrap_delta_ci(
        df, flag_col="is_company", novelty_col="sem_nov_mean", seed=SEED, n_boot=500
    )
    assert out["spans_zero"] is False
    assert cast(float, out["ci_low"]) > 0.0
    assert cast(float, out["ci_low"]) < 0.2 < cast(float, out["ci_high"])


def test_bootstrap_delta_ci_deterministic_with_seed() -> None:
    """Same seed => identical bounds (locked-seed determinism)."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "is_company": [True, False] * 500,
            "sem_nov_mean": rng.normal(0.4, 0.1, 1000),
        }
    )
    a = bootstrap_delta_ci(
        df, flag_col="is_company", novelty_col="sem_nov_mean", seed=SEED, n_boot=300
    )
    b = bootstrap_delta_ci(
        df, flag_col="is_company", novelty_col="sem_nov_mean", seed=SEED, n_boot=300
    )
    assert a == b
