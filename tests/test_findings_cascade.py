"""Unit tests for the V1-S15 F1 epistemic-cascade analysis (pure logic).

No network / no data files: every fixture is a small hand-built in-memory pandas
DataFrame (or a deterministic ``np.random.default_rng(0)`` synthetic series) with
hand-computable expectations and explicit assertions. The five contract fixtures
are covered: injected-lead recovery, known-null, BH-FDR truth test,
decision-rule boundary truth test (28/138 vs 27/138 + panel-disagree), and
non-evaluable-topic handling. Granger/panel are exercised on tiny synthetic
panels; the I/O notebook is deliberately NOT exercised here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scifield.findings.cascade import (
    TIER_MAP,
    bh_fdr,
    build_topic_year_series,
    classify_direction,
    cross_correlation,
    decide_f1,
    evidence_tier,
    granger_pair,
    panel_granger,
    prepare_series,
    qualifying_topics,
)

# --------------------------------------------------------------------------- #
# Helpers for building synthetic tidy paper frames
# --------------------------------------------------------------------------- #


def _papers(rows: list[tuple[int, int, str, int]]) -> pd.DataFrame:
    """Expand ``(topic, year, design, count)`` tuples into a tidy paper frame."""
    records: list[dict] = []
    for topic, year, design, count in rows:
        for _ in range(count):
            records.append(
                {"pmid": len(records), "topic_id": topic, "year": year, "study_design": design}
            )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# evidence_tier + TIER_MAP
# --------------------------------------------------------------------------- #


def test_evidence_tier_map_and_excluded_designs() -> None:
    """Graded designs map to their ordinal; review/other/unknown/NaN -> NaN."""
    assert evidence_tier("RCT") == 4.0
    assert evidence_tier("cohort") == 3.0
    assert evidence_tier("case_control") == 2.0
    assert evidence_tier("case_series") == 1.0
    # Excluded / ungraded -> NaN tier (carry no tier).
    assert np.isnan(evidence_tier("review"))
    assert np.isnan(evidence_tier("other"))
    assert np.isnan(evidence_tier("not_a_design"))
    assert np.isnan(evidence_tier(None))
    assert np.isnan(evidence_tier(float("nan")))
    # The pinned map excludes review/other by absence.
    assert "review" not in TIER_MAP and "other" not in TIER_MAP
    assert set(TIER_MAP) == {"RCT", "cohort", "case_control", "case_series"}


# --------------------------------------------------------------------------- #
# build_topic_year_series
# --------------------------------------------------------------------------- #


def test_build_topic_year_series_volume_meantier_rctshare() -> None:
    """volume counts all designs; mean_tier NaN-skips review/other; rct_share /all."""
    # Topic 1, year 2000: 2 RCT (tier 4), 1 cohort (tier 3), 1 review (NaN), 1 other (NaN).
    # volume = 5 (all designs). graded = 3 -> mean_tier = (4+4+3)/3 = 11/3.
    # rct_share = 2 / 5 = 0.4.
    df = _papers(
        [
            (1, 2000, "RCT", 2),
            (1, 2000, "cohort", 1),
            (1, 2000, "review", 1),
            (1, 2000, "other", 1),
        ]
    )
    series = build_topic_year_series(df)
    row = series.loc[(series["topic_id"] == 1) & (series["year"] == 2000)].iloc[0]
    assert row["volume"] == 5
    assert row["n_graded"] == 3
    assert abs(row["mean_tier"] - (11.0 / 3.0)) < 1e-12
    assert abs(row["rct_share"] - 0.4) < 1e-12
    assert list(series.columns) == [
        "topic_id",
        "year",
        "volume",
        "mean_tier",
        "rct_share",
        "n_graded",
    ]


def test_build_topic_year_series_zero_graded_year_is_nan_tier() -> None:
    """A topic-year with only review/other papers -> mean_tier NaN, rct_share 0."""
    df = _papers([(2, 2010, "review", 3), (2, 2010, "other", 2)])
    series = build_topic_year_series(df)
    row = series.iloc[0]
    assert row["volume"] == 5
    assert row["n_graded"] == 0
    assert np.isnan(row["mean_tier"])
    assert row["rct_share"] == 0.0


# --------------------------------------------------------------------------- #
# qualifying_topics
# --------------------------------------------------------------------------- #


def test_qualifying_topics_total_and_dense_year_rules() -> None:
    """Qualify iff >= v_min total AND >= min_years years with >= min_count papers."""
    # Topic A: 8 years x 5 papers = 40 total, 8 dense years -> qualifies.
    rows_a = [(1, 2000 + y, "RCT", 5) for y in range(8)]
    # Topic B: 40 total but spread so only 7 years have >=5 -> fails year rule.
    rows_b = [(2, 2000 + y, "RCT", 5) for y in range(7)] + [
        (2, 2007, "RCT", 2),
        (2, 2008, "RCT", 3),
    ]
    # Topic C: 8 dense years but only with 3 papers each = 24 total -> fails v_min.
    rows_c = [(3, 2000 + y, "RCT", 3) for y in range(8)]
    series = build_topic_year_series(_papers(rows_a + rows_b + rows_c))

    qual = qualifying_topics(series, v_min=30, min_years=8, min_count=5)
    assert qual == [1]
    # Topic B total is 40 (>=30) but only 7 dense years.
    assert int(series.loc[series["topic_id"] == 2, "volume"].sum()) == 40
    # Topic C has 8 dense years? no -- 3 < 5, so 0 dense years.
    assert int((series.loc[series["topic_id"] == 3, "volume"] >= 5).sum()) == 0


# --------------------------------------------------------------------------- #
# prepare_series  (span trim, interpolation, non-evaluable flag)
# --------------------------------------------------------------------------- #


def test_prepare_series_trims_thin_edges_and_interpolates_internal_gap() -> None:
    """Span = [first dense, last dense]; internal NaN-tier year is interpolated."""
    # year:    1998 1999 2000 2001 2002 2003
    # volume:    2    6    7    8    6    1   (dense >=5 -> 1999..2002)
    # mean_tier: 4    4   NaN    2    2    4
    topic_series = pd.DataFrame(
        {
            "year": [1998, 1999, 2000, 2001, 2002, 2003],
            "volume": [2, 6, 7, 8, 6, 1],
            "mean_tier": [4.0, 4.0, np.nan, 2.0, 2.0, 4.0],
            "rct_share": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    quality, volume, status = prepare_series(topic_series, quality_col="mean_tier")
    # Span trimmed to 1999..2002.
    assert list(volume.index) == [1999, 2000, 2001, 2002]
    assert status == "ok"
    # Internal NaN at 2000 linearly interpolated between 4.0 (1999) and 2.0 (2001) = 3.0.
    assert abs(quality.loc[2000] - 3.0) < 1e-12
    assert quality.notna().all()


def test_prepare_series_flags_non_evaluable_when_too_many_nan_tier_years() -> None:
    """FIXTURE 5: >25% in-span NaN-tier years -> status 'non_evaluable', not dropped."""
    # 8-year dense span; 3 of 8 (37.5% > 25%) years have no graded papers (NaN tier).
    topic_series = pd.DataFrame(
        {
            "year": list(range(2000, 2008)),
            "volume": [6, 6, 6, 6, 6, 6, 6, 6],
            "mean_tier": [4.0, np.nan, np.nan, np.nan, 3.0, 3.0, 2.0, 2.0],
            "rct_share": [0.1] * 8,
        }
    )
    quality, volume, status = prepare_series(topic_series, quality_col="mean_tier")
    assert status == "non_evaluable"
    # Still returned (not dropped) and interpolated for transparency.
    assert len(volume) == 8
    assert quality.notna().all()


def test_prepare_series_too_short_when_fewer_than_two_dense_years() -> None:
    """A topic with <2 dense years yields status 'too_short' and empty series."""
    topic_series = pd.DataFrame(
        {
            "year": [2000, 2001, 2002],
            "volume": [6, 2, 1],
            "mean_tier": [4.0, 3.0, 2.0],
            "rct_share": [0.0, 0.0, 0.0],
        }
    )
    quality, volume, status = prepare_series(topic_series, quality_col="mean_tier")
    assert status == "too_short"
    assert len(quality) == 0 and len(volume) == 0


def test_prepare_series_rct_share_never_non_evaluable() -> None:
    """The robustness series has no structural NaNs -> 'ok' even with sparse tiers."""
    topic_series = pd.DataFrame(
        {
            "year": list(range(2000, 2006)),
            "volume": [6, 6, 6, 6, 6, 6],
            "mean_tier": [np.nan, np.nan, np.nan, np.nan, np.nan, 4.0],
            "rct_share": [0.0, 0.2, 0.1, 0.3, 0.0, 0.5],
        }
    )
    quality, volume, status = prepare_series(topic_series, quality_col="rct_share")
    assert status == "ok"
    assert quality.notna().all()


# --------------------------------------------------------------------------- #
# cross_correlation  (sign convention + injected lead)
# --------------------------------------------------------------------------- #


def test_cross_correlation_negative_peak_when_quality_leads() -> None:
    """FIXTURE 1 (CCF leg): volume is quality shifted later -> NEGATIVE peak lag."""
    rng = np.random.default_rng(0)
    n = 60
    quality = np.cumsum(rng.standard_normal(n))
    # volume[t] tracks quality[t-2] (quality leads by 2 years).
    shift = 2
    volume = np.empty(n)
    volume[:shift] = 0.0
    volume[shift:] = quality[: n - shift]
    volume = volume + 0.01 * rng.standard_normal(n)

    peak_lag, peak_corr, ccf = cross_correlation(quality, volume, max_lag=5)
    # quality LEADS -> negative peak lag (rigor precedes the surge).
    assert peak_lag < 0
    assert abs(peak_corr) > 0.5
    assert set(ccf.keys()) == set(range(-5, 6))


def test_cross_correlation_degenerate_series_reports_lag_zero() -> None:
    """A constant (zero-variance) series gives all-NaN CCF -> peak_lag 0, NaN corr."""
    quality = np.ones(30)
    volume = np.arange(30, dtype="float64")
    peak_lag, peak_corr, ccf = cross_correlation(quality, volume, max_lag=5)
    assert peak_lag == 0
    assert np.isnan(peak_corr)


# --------------------------------------------------------------------------- #
# granger_pair  (injected lead + known null)
# --------------------------------------------------------------------------- #


def test_granger_pair_injected_lead_small_p_quality_to_volume() -> None:
    """FIXTURE 1 (Granger leg): quality drives future volume -> small p_q_to_v."""
    rng = np.random.default_rng(0)
    n = 80
    quality = np.cumsum(rng.standard_normal(n))
    volume = np.zeros(n)
    for t in range(1, n):
        volume[t] = 0.5 * volume[t - 1] + 1.5 * quality[t - 1] + 0.2 * rng.standard_normal()

    p_q_to_v, p_v_to_q = granger_pair(quality, volume, lag=3)
    assert np.isfinite(p_q_to_v) and np.isfinite(p_v_to_q)
    # quality -> volume is the true mechanism: that p must be tiny.
    assert p_q_to_v < 0.01
    # ... and clearly smaller than the reverse direction.
    assert p_q_to_v < p_v_to_q


def test_granger_pair_known_null_independent_random_walks() -> None:
    """FIXTURE 2: two independent random walks -> neither direction significant."""
    rng = np.random.default_rng(0)
    n = 80
    quality = np.cumsum(rng.standard_normal(n))
    volume = np.cumsum(rng.standard_normal(n))
    p_q_to_v, p_v_to_q = granger_pair(quality, volume, lag=3)
    assert p_q_to_v > 0.05
    assert p_v_to_q > 0.05


def test_granger_pair_too_short_returns_nan_both_directions() -> None:
    """A series too short to fit lag-3 -> NaN p in both directions (no raise)."""
    quality = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    volume = np.array([2.0, 1.0, 2.0, 3.0, 4.0])
    p_q_to_v, p_v_to_q = granger_pair(quality, volume, lag=3)
    assert np.isnan(p_q_to_v) and np.isnan(p_v_to_q)


# --------------------------------------------------------------------------- #
# bh_fdr  (truth test)
# --------------------------------------------------------------------------- #


def test_bh_fdr_known_reject_set() -> None:
    """FIXTURE 3: hand-built p-vector with an analytically known BH reject set."""
    # m = 5, q = 0.05. Sorted p: [0.001, 0.008, 0.039, 0.04, 0.9].
    # BH thresholds k/m*q: [0.01, 0.02, 0.03, 0.04, 0.05].
    #   p(1)=0.001 <= 0.01   ok
    #   p(2)=0.008 <= 0.02   ok
    #   p(3)=0.039 <= 0.03   NO
    #   p(4)=0.04  <= 0.04   ok  <- largest passing rank = 4
    #   p(5)=0.9   <= 0.05   NO
    # Reject every p <= p(4)=0.04 -> ranks 1..4 rejected, rank 5 not.
    pvals = np.array([0.04, 0.9, 0.001, 0.039, 0.008])
    reject = bh_fdr(pvals, q=0.05)
    # Indices: 0->0.04 reject, 1->0.9 no, 2->0.001 reject, 3->0.039 reject, 4->0.008 reject.
    assert reject.tolist() == [True, False, True, True, True]


def test_bh_fdr_nan_pvalues_not_rejected_and_excluded_from_m() -> None:
    """NaN p-values are carried as not-rejected and do not inflate m."""
    # Finite set is just [0.001] with m=1, threshold 1/1*0.05 = 0.05 -> reject.
    pvals = np.array([0.001, np.nan, np.nan])
    reject = bh_fdr(pvals, q=0.05)
    assert reject.tolist() == [True, False, False]
    # All-NaN -> all False.
    assert bh_fdr(np.array([np.nan, np.nan]), q=0.05).tolist() == [False, False]


def test_bh_fdr_no_rejections_when_all_large() -> None:
    """All large p-values -> empty reject set."""
    reject = bh_fdr(np.array([0.6, 0.7, 0.8]), q=0.05)
    assert not reject.any()


# --------------------------------------------------------------------------- #
# classify_direction
# --------------------------------------------------------------------------- #


def test_classify_direction_all_four_cases() -> None:
    """exactly-one -> lead/lag; both -> coupled; neither -> none."""
    assert classify_direction(leads_sig=True, lags_sig=False) == "lead"
    assert classify_direction(leads_sig=False, lags_sig=True) == "lag"
    assert classify_direction(leads_sig=True, lags_sig=True) == "coupled"
    assert classify_direction(leads_sig=False, lags_sig=False) == "none"


# --------------------------------------------------------------------------- #
# panel_granger  (pooled fixed-effects)
# --------------------------------------------------------------------------- #


def test_panel_granger_recovers_dominant_direction() -> None:
    """Pooled topics where quality leads -> panel p_q_to_v << p_v_to_q."""
    rng = np.random.default_rng(0)
    differenced: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(6):
        n = 45
        q = np.cumsum(rng.standard_normal(n))
        v = np.zeros(n)
        for t in range(1, n):
            v[t] = 0.4 * v[t - 1] + 1.2 * q[t - 1] + 0.3 * rng.standard_normal()
        differenced.append((np.diff(q), np.diff(v)))

    p_q_to_v, p_v_to_q = panel_granger(differenced, lag=3)
    assert np.isfinite(p_q_to_v) and np.isfinite(p_v_to_q)
    assert p_q_to_v < 0.05
    assert p_q_to_v < p_v_to_q


def test_panel_granger_empty_input_returns_nan() -> None:
    """No usable topic series -> NaN panel p-values (no raise)."""
    p_q_to_v, p_v_to_q = panel_granger([], lag=3)
    assert np.isnan(p_q_to_v) and np.isnan(p_v_to_q)


# --------------------------------------------------------------------------- #
# decide_f1  (decision-rule boundary + panel-disagree)
# --------------------------------------------------------------------------- #


def _results(n_lead: int, n_lag: int, n_qualifying: int) -> list[dict]:
    """Build a per-topic results list: n_lead leads, n_lag lags, rest 'none'."""
    out = [{"direction": "lead"} for _ in range(n_lead)]
    out += [{"direction": "lag"} for _ in range(n_lag)]
    out += [{"direction": "none"} for _ in range(n_qualifying - n_lead - n_lag)]
    return out


def test_decide_f1_holds_at_28_of_138_with_panel_agreement() -> None:
    """FIXTURE 4a: exactly 28/138 directional + panel agrees -> HOLDS."""
    results = _results(n_lead=28, n_lag=0, n_qualifying=138)
    # Panel agrees on quality_leads (p_q_to_v significant).
    verdict = decide_f1(results, (1e-4, 0.6), n_qualifying=138)
    assert verdict["n_directional"] == 28
    assert abs(verdict["frac_directional"] - 28 / 138) < 1e-12
    assert verdict["frac_directional"] >= 0.20  # 0.2029 >= 0.20
    assert verdict["dominant_direction"] == "quality_leads"
    assert verdict["panel_agrees"] is True
    assert verdict["holds"] is True


def test_decide_f1_does_not_hold_at_27_of_138() -> None:
    """FIXTURE 4b: 27/138 is below the 20% boundary -> does NOT hold."""
    results = _results(n_lead=27, n_lag=0, n_qualifying=138)
    verdict = decide_f1(results, (1e-4, 0.6), n_qualifying=138)
    assert verdict["n_directional"] == 27
    assert verdict["frac_directional"] < 0.20  # 0.1957 < 0.20
    assert verdict["holds"] is False


def test_decide_f1_does_not_hold_when_panel_disagrees() -> None:
    """FIXTURE 4c: 28/138 directional but panel NOT significant -> does NOT hold."""
    results = _results(n_lead=28, n_lag=0, n_qualifying=138)
    # Panel p_q_to_v = 0.40 (not < 0.05) -> panel does not agree.
    verdict = decide_f1(results, (0.40, 0.6), n_qualifying=138)
    assert verdict["frac_directional"] >= 0.20
    assert verdict["panel_agrees"] is False
    assert verdict["holds"] is False


def test_decide_f1_panel_significant_but_wrong_direction_does_not_agree() -> None:
    """Panel significant only in the volume->quality leg while topics say leads -> no hold."""
    results = _results(n_lead=30, n_lag=0, n_qualifying=138)
    # Dominant per-topic direction is quality_leads, but the panel is significant
    # only for volume->quality (the WRONG direction) -> panel does not agree.
    verdict = decide_f1(results, (0.8, 1e-4), n_qualifying=138)
    assert verdict["dominant_direction"] == "quality_leads"
    assert verdict["panel_agrees"] is False
    assert verdict["holds"] is False


def test_decide_f1_direction_split_and_none_count() -> None:
    """The full lead/lag/coupled/none split is reported and sums to the input."""
    results = (
        [{"direction": "lead"} for _ in range(10)]
        + [{"direction": "lag"} for _ in range(4)]
        + [{"direction": "coupled"} for _ in range(3)]
        + [{"direction": "none"} for _ in range(121)]
    )
    verdict = decide_f1(results, (1e-3, 0.5), n_qualifying=138)
    assert verdict["n_lead"] == 10
    assert verdict["n_lag"] == 4
    assert verdict["n_coupled"] == 3
    assert verdict["n_none"] == 121
    assert verdict["n_lead"] + verdict["n_lag"] + verdict["n_coupled"] + verdict["n_none"] == 138
    # Dominant is quality_leads (10 > 4).
    assert verdict["dominant_direction"] == "quality_leads"


def test_decide_f1_dominant_lag_when_lags_outnumber_leads() -> None:
    """When lags outnumber leads, dominant is quality_lags and panel uses p_v_to_q."""
    results = _results(n_lead=5, n_lag=30, n_qualifying=138)
    verdict = decide_f1(results, (0.9, 1e-4), n_qualifying=138)
    assert verdict["dominant_direction"] == "quality_lags"
    assert verdict["panel_agrees"] is True  # p_v_to_q = 1e-4 < 0.05
    assert verdict["holds"] is True


def test_decide_f1_no_directional_topics_is_none_direction() -> None:
    """Zero directional topics -> dominant_direction 'none' and does not hold."""
    results = _results(n_lead=0, n_lag=0, n_qualifying=138)
    verdict = decide_f1(results, (1e-9, 1e-9), n_qualifying=138)
    assert verdict["dominant_direction"] == "none"
    assert verdict["holds"] is False


def test_decide_f1_exact_lead_lag_tie_is_none_even_with_strong_panel() -> None:
    """Exact lead/lag tie above the 20% gate -> 'none' direction, does NOT hold.

    A positive but EXACTLY tied per-topic lead/lag split (15 lead + 15 lag of
    138 = 21.7% >= 20%, clearing criterion (a)) yields NO dominant direction, so
    criterion (b) is undefined and F1 = NULL -- even though the panel is strongly
    significant in one direction. There is no panel tie-break (conservative).
    """
    results = _results(n_lead=15, n_lag=15, n_qualifying=138)
    # Directional fraction clears the 20% gate ...
    # ... and the panel screams quality_leads (p_q_to_v = 1e-6, p_v_to_q = 0.9).
    verdict = decide_f1(results, (1e-6, 0.9), n_qualifying=138)
    assert verdict["n_directional"] == 30
    assert verdict["frac_directional"] >= 0.20  # 30/138 = 0.2174 >= 0.20
    assert verdict["dominant_direction"] == "none"
    assert verdict["holds"] is False


# --------------------------------------------------------------------------- #
# End-to-end-ish: injected-lead pipeline -> decide holds (FIXTURE 1 full)
# --------------------------------------------------------------------------- #


def test_injected_lead_pipeline_holds_with_quality_leads() -> None:
    """FIXTURE 1 (full): many injected-lead topics flow to a HOLDS verdict.

    Builds several synthetic topics in which quality genuinely leads volume,
    runs each through ``granger_pair`` + ``bh_fdr`` + ``classify_direction``,
    pools them with ``panel_granger``, and confirms ``decide_f1`` -> holds with
    dominant_direction 'quality_leads'.
    """
    rng = np.random.default_rng(0)
    n_topics = 40
    pvals: list[float] = []
    differenced: list[tuple[np.ndarray, np.ndarray]] = []
    directions_inputs: list[tuple[float, float]] = []  # filled after FDR

    for _ in range(n_topics):
        n = 70
        q = np.cumsum(rng.standard_normal(n))
        v = np.zeros(n)
        for t in range(1, n):
            v[t] = 0.4 * v[t - 1] + 1.3 * q[t - 1] + 0.3 * rng.standard_normal()
        p_q_to_v, p_v_to_q = granger_pair(q, v, lag=3)
        pvals.extend([p_q_to_v, p_v_to_q])
        differenced.append((np.diff(q), np.diff(v)))
        directions_inputs.append((p_q_to_v, p_v_to_q))

    reject = bh_fdr(np.array(pvals), q=0.05)
    per_topic_results: list[dict] = []
    for i in range(n_topics):
        leads_sig = bool(reject[2 * i])
        lags_sig = bool(reject[2 * i + 1])
        per_topic_results.append(
            {"direction": classify_direction(leads_sig=leads_sig, lags_sig=lags_sig)}
        )

    panel_result = panel_granger(differenced, lag=3)
    verdict = decide_f1(per_topic_results, panel_result, n_qualifying=n_topics)

    # The mechanism is quality -> volume in every topic, so the verdict holds.
    assert verdict["n_lead"] >= int(0.20 * n_topics)
    assert verdict["dominant_direction"] == "quality_leads"
    assert verdict["panel_agrees"] is True
    assert verdict["holds"] is True


def test_known_null_pipeline_does_not_hold() -> None:
    """FIXTURE 2 (full): independent random-walk topics -> decide_f1 does NOT hold."""
    rng = np.random.default_rng(0)
    n_topics = 40
    pvals: list[float] = []
    differenced: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_topics):
        n = 70
        q = np.cumsum(rng.standard_normal(n))
        v = np.cumsum(rng.standard_normal(n))
        p_q_to_v, p_v_to_q = granger_pair(q, v, lag=3)
        pvals.extend([p_q_to_v, p_v_to_q])
        differenced.append((np.diff(q), np.diff(v)))

    reject = bh_fdr(np.array(pvals), q=0.05)
    per_topic_results = []
    for i in range(n_topics):
        per_topic_results.append(
            {
                "direction": classify_direction(
                    leads_sig=bool(reject[2 * i]), lags_sig=bool(reject[2 * i + 1])
                )
            }
        )
    panel_result = panel_granger(differenced, lag=3)
    verdict = decide_f1(per_topic_results, panel_result, n_qualifying=n_topics)

    # Independent series: well below the 20% directional threshold.
    assert verdict["frac_directional"] < 0.20
    assert verdict["holds"] is False
