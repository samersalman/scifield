"""Synthetic-only tests for V1-S09 internal-validity / agreement lenses.

No real labels or parquets are loaded; every fixture is hand-built
in-process. Mirrors the style of ``tests/test_epistemic_kappa.py``
(plain helper builders, synthetic frames, no API calls).
"""

from __future__ import annotations

import math

import pandas as pd

from scifield.epistemic.kappa import cohens_kappa
from scifield.epistemic.validate import (
    CheckResult,
    cross_tool_rct_agreement,
    internal_validity_checks,
    model_vs_model_agreement,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ds_row(
    pmid: int,
    study_design: str | None = "cohort",
    sample_size: int | None = 100,
    has_control: bool | None = False,
    effect_direction: str | None = "positive",
    statistical_claim_present: bool | None = True,
    coi_disclosed_in_abstract: bool | None = False,
    raw_response: str | None = "{}",
) -> dict[str, object]:
    """Build one DeepSeek-shaped row dict with sane defaults."""
    return {
        "pmid": pmid,
        "study_design": study_design,
        "sample_size": sample_size,
        "has_control": has_control,
        "effect_direction": effect_direction,
        "statistical_claim_present": statistical_claim_present,
        "coi_disclosed_in_abstract": coi_disclosed_in_abstract,
        "raw_response": raw_response,
    }


def _check_by_key(results: list[CheckResult], key: str) -> CheckResult:
    return next(r for r in results if r.key == key)


# --------------------------------------------------------------------------- #
# C1 — cross-tool RCT agreement
# --------------------------------------------------------------------------- #
def test_cross_tool_known_confusion_matrix() -> None:
    # Hand-built: 10 pmids, known TP/FP/FN/TN against PubMed RCT flag.
    #   pmid 1..3: DeepSeek RCT, PubMed RCT      -> TP x3
    #   pmid 4   : DeepSeek RCT, PubMed not RCT  -> FP x1
    #   pmid 5,6 : DeepSeek not RCT, PubMed RCT  -> FN x2
    #   pmid 7..10: DeepSeek not, PubMed not     -> TN x4
    rows = [
        _ds_row(1, study_design="RCT"),
        _ds_row(2, study_design="RCT"),
        _ds_row(3, study_design="RCT"),
        _ds_row(4, study_design="RCT"),
        _ds_row(5, study_design="cohort"),
        _ds_row(6, study_design="cohort"),
        _ds_row(7, study_design="cohort"),
        _ds_row(8, study_design="review"),
        _ds_row(9, study_design="other"),
        _ds_row(10, study_design="case_series"),
    ]
    deepseek_df = pd.DataFrame(rows)
    pubtype = {
        1: True,
        2: True,
        3: True,
        4: False,
        5: True,
        6: True,
        7: False,
        8: False,
        9: False,
        10: False,
    }

    out = cross_tool_rct_agreement(deepseek_df, pubtype)
    assert out["n"] == 10
    assert out["tp"] == 3
    assert out["fp"] == 1
    assert out["fn"] == 2
    assert out["tn"] == 4
    assert math.isclose(out["simple_agreement"], 7 / 10)
    assert math.isclose(out["sensitivity"], 3 / 5)  # tp/(tp+fn)
    assert math.isclose(out["precision"], 3 / 4)  # tp/(tp+fp)

    # κ must match a direct cohens_kappa call on the same binary vectors.
    ds_calls = [True, True, True, True, False, False, False, False, False, False]
    pm_flags = [True, True, True, False, True, True, False, False, False, False]
    assert math.isclose(out["cohens_kappa"], cohens_kappa(pm_flags, ds_calls), abs_tol=1e-12)


def test_cross_tool_only_scores_overlap_and_handles_none_study_design() -> None:
    rows = [
        _ds_row(1, study_design="RCT"),
        _ds_row(2, study_design=None),  # None design -> dropped
        _ds_row(3, study_design="cohort"),
        _ds_row(99, study_design="RCT"),  # pmid not in lookup -> dropped
    ]
    deepseek_df = pd.DataFrame(rows)
    pubtype = {1: True, 2: True, 3: False}  # pmid 99 absent
    out = cross_tool_rct_agreement(deepseek_df, pubtype)
    # Only pmids 1 and 3 score: (RCT,True)=TP, (cohort,False)=TN.
    assert out["n"] == 2
    assert out["tp"] == 1
    assert out["tn"] == 1
    assert out["fp"] == 0
    assert out["fn"] == 0


def test_cross_tool_accepts_series_lookup() -> None:
    deepseek_df = pd.DataFrame([_ds_row(1, study_design="RCT"), _ds_row(2, study_design="cohort")])
    lookup = pd.Series({1: True, 2: False})
    out = cross_tool_rct_agreement(deepseek_df, lookup)
    assert out["tp"] == 1 and out["tn"] == 1


# --------------------------------------------------------------------------- #
# C2 — model-vs-model agreement
# --------------------------------------------------------------------------- #
def test_model_vs_model_exact_match_and_kappa() -> None:
    paired = pd.DataFrame(
        {
            "study_design_deepseek": ["RCT", "cohort", "RCT", "review"],
            "study_design_claude": ["RCT", "cohort", "cohort", "review"],
            "has_control_deepseek": [True, False, True, None],
            "has_control_claude": [True, False, False, False],
            "sample_size_deepseek": [10, 20, 30, 40],
            "sample_size_claude": [11, 19, 33, 38],
        }
    )
    out = model_vs_model_agreement(paired)
    assert out["n"] == 4

    sd = out["fields"]["study_design"]
    assert sd["n"] == 4
    assert sd["n_agree"] == 3  # one RCT/cohort disagreement
    assert math.isclose(sd["exact_match"], 3 / 4)
    assert math.isclose(
        sd["cohens_kappa"],
        cohens_kappa(["RCT", "cohort", "RCT", "review"], ["RCT", "cohort", "cohort", "review"]),
        abs_tol=1e-12,
    )

    hc = out["fields"]["has_control"]
    # Row 4 has None on deepseek -> dropped; 3 usable, 2 agree.
    assert hc["n"] == 3
    assert hc["n_agree"] == 2
    assert math.isclose(hc["exact_match"], 2 / 3)


def test_model_vs_model_spearman_perfect_on_monotonic_nonlinear() -> None:
    # Strictly monotonic but non-linear relationship: Spearman ρ == 1.0
    # while Pearson would be < 1.0 (distinguishes rank from linear corr).
    paired = pd.DataFrame(
        {
            "study_design_deepseek": ["RCT", "RCT", "RCT", "RCT"],
            "study_design_claude": ["RCT", "RCT", "RCT", "RCT"],
            "has_control_deepseek": [True, True, True, True],
            "has_control_claude": [True, True, True, True],
            "sample_size_deepseek": [1, 2, 3, 4],
            "sample_size_claude": [1, 8, 27, 64],  # cube — monotonic, non-linear
        }
    )
    out = model_vs_model_agreement(paired)
    ss = out["fields"]["sample_size"]
    assert ss["n"] == 4
    assert math.isclose(ss["spearman_rho"], 1.0, abs_tol=1e-12)
    # Pearson on the raw (non-ranked) values would NOT be 1.0:
    assert not math.isclose(
        paired["sample_size_deepseek"].corr(paired["sample_size_claude"]),
        1.0,
        abs_tol=1e-6,
    )


def test_model_vs_model_spearman_drops_null_sample_pairs() -> None:
    paired = pd.DataFrame(
        {
            "sample_size_deepseek": [1, 2, None, 4],
            "sample_size_claude": [1, None, 30, 16],
        }
    )
    out = model_vs_model_agreement(paired)
    ss = out["fields"]["sample_size"]
    # Only rows 0 and 3 have both non-null.
    assert ss["n"] == 2
    assert math.isclose(ss["spearman_rho"], 1.0, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# C3 — internal-validity priors (PASS + FAIL per prior)
# --------------------------------------------------------------------------- #
def test_prior_a_rct_fraction_pass_and_fail() -> None:
    # PASS: 1 RCT out of 10 == 10% (in [2%, 15%]).
    rows = [_ds_row(i, study_design="RCT" if i == 0 else "cohort") for i in range(10)]
    res = internal_validity_checks(pd.DataFrame(rows))
    a = _check_by_key(res, "a_rct_fraction")
    assert math.isclose(a.value, 0.10)
    assert a.passed

    # FAIL: 5 RCT out of 10 == 50% (above 15%).
    rows = [_ds_row(i, study_design="RCT" if i < 5 else "cohort") for i in range(10)]
    a = _check_by_key(internal_validity_checks(pd.DataFrame(rows)), "a_rct_fraction")
    assert math.isclose(a.value, 0.50)
    assert not a.passed


def test_prior_b_statistical_claim_pass_and_fail() -> None:
    # PASS: 7/10 == 70% (in [45%, 85%]).
    rows = [_ds_row(i, statistical_claim_present=(i < 7)) for i in range(10)]
    b = _check_by_key(internal_validity_checks(pd.DataFrame(rows)), "b_statistical_claim_fraction")
    assert math.isclose(b.value, 0.70)
    assert b.passed

    # FAIL: 1/10 == 10% (below 45%).
    rows = [_ds_row(i, statistical_claim_present=(i < 1)) for i in range(10)]
    b = _check_by_key(internal_validity_checks(pd.DataFrame(rows)), "b_statistical_claim_fraction")
    assert not b.passed


def test_prior_c_coi_pass_and_fail() -> None:
    # PASS: 1/100 == 1% (in (0%, 5%]).
    rows = [_ds_row(i, coi_disclosed_in_abstract=(i == 0)) for i in range(100)]
    c = _check_by_key(internal_validity_checks(pd.DataFrame(rows)), "c_coi_fraction")
    assert math.isclose(c.value, 0.01)
    assert c.passed

    # FAIL (zero): exactly 0% must fail (field never fires -> suspicious).
    rows = [_ds_row(i, coi_disclosed_in_abstract=False) for i in range(100)]
    c = _check_by_key(internal_validity_checks(pd.DataFrame(rows)), "c_coi_fraction")
    assert math.isclose(c.value, 0.0)
    assert not c.passed

    # FAIL (too high): 50% > 5%.
    rows = [_ds_row(i, coi_disclosed_in_abstract=(i < 50)) for i in range(100)]
    c = _check_by_key(internal_validity_checks(pd.DataFrame(rows)), "c_coi_fraction")
    assert not c.passed


def test_prior_d_rct_implies_control_pass_and_fail() -> None:
    # PASS: 10 RCT rows, 10/10 controlled == 100% (in [90%, 100%]).
    rows = [_ds_row(i, study_design="RCT", has_control=True) for i in range(10)]
    # Add some non-RCT noise that must be ignored by the prior.
    rows += [_ds_row(100 + i, study_design="cohort", has_control=False) for i in range(5)]
    d = _check_by_key(internal_validity_checks(pd.DataFrame(rows)), "d_rct_implies_control")
    assert math.isclose(d.value, 1.0)
    assert d.passed

    # FAIL: only 5/10 RCT rows controlled == 50%.
    rows = [_ds_row(i, study_design="RCT", has_control=(i < 5)) for i in range(10)]
    d = _check_by_key(internal_validity_checks(pd.DataFrame(rows)), "d_rct_implies_control")
    assert math.isclose(d.value, 0.5)
    assert not d.passed


def test_prior_e_sample_size_median_and_max_list() -> None:
    # PASS: median 100 (in [10, 1000]); include a >10M outlier.
    rows = [_ds_row(i, sample_size=100) for i in range(10)]
    rows.append(_ds_row(999, sample_size=68_000_000))  # adjudicated outlier
    res = internal_validity_checks(pd.DataFrame(rows))
    e = _check_by_key(res, "e_sample_size_median")
    assert e.passed  # passes on median sanity, NOT the max
    # The >10M pmid must be surfaced in threshold_repr for adjudication.
    assert "999" in e.threshold_repr
    assert "68000000" in e.threshold_repr

    # FAIL: median 5 (below 10) -> median sanity fails.
    rows = [_ds_row(i, sample_size=5) for i in range(10)]
    e = _check_by_key(internal_validity_checks(pd.DataFrame(rows)), "e_sample_size_median")
    assert math.isclose(e.value, 5.0)
    assert not e.passed


def test_prior_f_effect_direction_na_pass_and_fail() -> None:
    # PASS: 4/10 == 40% na (in [20%, 55%]).
    rows = [_ds_row(i, effect_direction=("na" if i < 4 else "positive")) for i in range(10)]
    f = _check_by_key(
        internal_validity_checks(pd.DataFrame(rows)), "f_effect_direction_na_fraction"
    )
    assert math.isclose(f.value, 0.40)
    assert f.passed

    # FAIL: 9/10 == 90% na (above 55%).
    rows = [_ds_row(i, effect_direction=("na" if i < 9 else "positive")) for i in range(10)]
    f = _check_by_key(
        internal_validity_checks(pd.DataFrame(rows)), "f_effect_direction_na_fraction"
    )
    assert not f.passed


# --------------------------------------------------------------------------- #
# dedupe-on-pmid behavior
# --------------------------------------------------------------------------- #
def test_priors_dedupe_on_pmid() -> None:
    # 3 distinct pmids but pmid 1 appears 3 times (internal dup rows).
    # Without dedupe RCT fraction would be 3/5; with dedupe it's 1/3.
    rows = [
        _ds_row(1, study_design="RCT", raw_response="short"),
        _ds_row(1, study_design="RCT", raw_response="a much longer raw response wins"),
        _ds_row(1, study_design="RCT", raw_response="mid"),
        _ds_row(2, study_design="cohort"),
        _ds_row(3, study_design="cohort"),
    ]
    res = internal_validity_checks(pd.DataFrame(rows))
    a = _check_by_key(res, "a_rct_fraction")
    # Deduped to 3 rows -> 1 RCT / 3 == 33.3% (NOT 3/5).
    assert math.isclose(a.value, 1 / 3)


def test_dedupe_keeps_longest_raw_response() -> None:
    # The longest-raw_response row for pmid 1 says sample_size=500;
    # shorter dups say 1. After dedupe the median should reflect 500.
    rows = [
        _ds_row(1, sample_size=1, raw_response="x"),
        _ds_row(1, sample_size=500, raw_response="the longest raw response by far"),
    ]
    res = internal_validity_checks(pd.DataFrame(rows))
    e = _check_by_key(res, "e_sample_size_median")
    assert math.isclose(e.value, 500.0)


def test_e_returns_over_10m_pmid_list() -> None:
    rows = [
        _ds_row(1, sample_size=100),
        _ds_row(2, sample_size=20_000_000),
        _ds_row(3, sample_size=11_000_000),
    ]
    res = internal_validity_checks(pd.DataFrame(rows))
    e = _check_by_key(res, "e_sample_size_median")
    assert "2" in e.threshold_repr and "3" in e.threshold_repr


# --------------------------------------------------------------------------- #
# None / NaN safety
# --------------------------------------------------------------------------- #
def test_priors_none_safe_does_not_crash() -> None:
    rows = [
        _ds_row(1, study_design=None, has_control=None, sample_size=None, effect_direction=None),
        _ds_row(2, study_design="RCT", has_control=True, sample_size=200, effect_direction="na"),
        _ds_row(
            3, study_design="cohort", has_control=False, sample_size=80, effect_direction="positive"
        ),
    ]
    res = internal_validity_checks(pd.DataFrame(rows))
    # RCT fraction over non-null designs: 1 RCT / 2 non-null == 50%.
    a = _check_by_key(res, "a_rct_fraction")
    assert math.isclose(a.value, 0.5)
    # Median sample_size over non-null: median(200, 80) == 140.
    e = _check_by_key(res, "e_sample_size_median")
    assert math.isclose(e.value, 140.0)
    # na fraction over non-null effect_direction: 1/2 == 50%.
    f = _check_by_key(res, "f_effect_direction_na_fraction")
    assert math.isclose(f.value, 0.5)


def test_checkresult_is_frozen() -> None:
    cr = CheckResult(key="k", label="l", value=1.0, threshold_repr="r", passed=True)
    try:
        cr.value = 2.0  # type: ignore[misc]
    except Exception as exc:  # frozen dataclass raises FrozenInstanceError
        assert "frozen" in type(exc).__name__.lower() or "FrozenInstance" in type(exc).__name__
    else:
        raise AssertionError("CheckResult must be frozen")
