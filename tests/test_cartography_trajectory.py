"""Unit tests for the V2-S10 per-topic trajectory / forecast layer.

No network / no data files / no GPU: every fixture is a small hand-built in-memory
pandas DataFrame whose share/volume trajectory is known by construction, so the
expected direction, projection-horizon length, and band ordering are checkable by
hand. The module is pure (DataFrame-in / DataFrame-out), so the driver script and the
notebook are deliberately NOT exercised here.

Any synthetic content uses a FIXED seed for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scifield.cartography.trajectory import (
    build_topic_year_series,
    fit_trajectories,
)

_SEED = 20260609

# Locked output schemas (mirror the build-script reader; kept here so a schema drift
# fails a test rather than silently breaking the parquet contract downstream).
_SERIES_COLS = [
    "grain",
    "topic_id",
    "year",
    "kind",
    "share",
    "share_lo",
    "share_hi",
    "volume",
    "volume_lo",
    "volume_hi",
]
_SUMMARY_COLS = [
    "grain",
    "topic_id",
    "label",
    "size",
    "n_years_observed",
    "last_obs_year",
    "last_obs_share",
    "model",
    "slope_share_per_yr",
    "direction",
    "horizon_year",
    "proj_share",
    "proj_share_lo",
    "proj_share_hi",
    "proj_volume",
    "proj_volume_lo",
    "proj_volume_hi",
    "fit_ok",
]


# --------------------------------------------------------------------------- #
# Fixture builders — known share trajectories by construction
# --------------------------------------------------------------------------- #
def _papers_from_counts(counts: dict[int, dict[int, int]]) -> pd.DataFrame:
    """Explode a {topic_id: {year: n_papers}} spec into one row per paper.

    Each paper carries ``topic_id`` and ``year`` (the two columns the series builder
    needs). ``pmid`` is added as a throwaway id so the frame resembles the real
    ``archetypes.parquet`` shape without the module ever depending on it.
    """
    rows: list[dict[str, object]] = []
    pmid = 0
    for topic_id, by_year in counts.items():
        for year, n in by_year.items():
            for _ in range(int(n)):
                rows.append({"pmid": pmid, "topic_id": int(topic_id), "year": int(year)})
                pmid += 1
    return pd.DataFrame(rows, columns=["pmid", "topic_id", "year"])


def _rising_topic_counts(
    topic_id: int,
    *,
    start_year: int = 2014,
    last_year: int = 2025,
    base: int = 5,
    step: int = 4,
) -> dict[int, int]:
    """A monotonically rising per-year count series for one topic (12 years)."""
    return {y: base + step * (y - start_year) for y in range(start_year, last_year + 1)}


def _falling_topic_counts(
    topic_id: int,
    *,
    start_year: int = 2014,
    last_year: int = 2025,
    base: int = 60,
    step: int = 4,
) -> dict[int, int]:
    """A monotonically falling per-year count series for one topic (12 years)."""
    return {y: max(1, base - step * (y - start_year)) for y in range(start_year, last_year + 1)}


def _flat_topic_counts(
    topic_id: int,
    *,
    start_year: int = 2014,
    last_year: int = 2025,
    level: int = 30,
) -> dict[int, int]:
    """A constant per-year count series for one topic (12 years)."""
    return {y: level for y in range(start_year, last_year + 1)}


def _three_topic_papers() -> pd.DataFrame:
    """A corpus with a rising (0), flat (1), falling (2) topic + a -1 noise topic.

    The noise topic (-1) carries a big, distinctive per-year volume so a test can
    verify it is excluded from BOTH the numerator and the share denominator. A
    partial-final-year 2026 row is added to topic 0 so the drop-partial-year path is
    exercised by the same fixture.
    """
    counts = {
        0: _rising_topic_counts(0),
        1: _flat_topic_counts(1),
        2: _falling_topic_counts(2),
        -1: _flat_topic_counts(-1, level=1000),  # BERTopic noise — must be dropped
    }
    papers = _papers_from_counts(counts)
    # Partial 2026 year for topic 0 (must be dropped by max_complete_year=2025).
    partial = pd.DataFrame(
        {"pmid": [999_000, 999_001, 999_002], "topic_id": [0, 0, 0], "year": [2026, 2026, 2026]}
    )
    return pd.concat([papers, partial], ignore_index=True)


# --------------------------------------------------------------------------- #
# build_topic_year_series — share computation, noise + partial-year drops
# --------------------------------------------------------------------------- #
def test_series_share_excludes_noise_topic_from_numerator_and_denominator() -> None:
    """topic_id == -1 is dropped from the rows AND from the per-year share denominator."""
    papers = _three_topic_papers()
    series = build_topic_year_series(papers, topic_key="topic_id", year_col="year")

    # The noise topic never appears as a row.
    assert -1 not in set(series["topic_id"])

    # For 2014: rising=5, flat=30, falling=60 -> total_papers_year EXCLUDES the 1000
    # noise papers, so denominator is 5+30+60 = 95 (NOT 1095).
    y2014 = series.loc[series["year"] == 2014].set_index("topic_id")
    assert int(y2014.loc[0, "n_papers"]) == 5
    assert int(y2014.loc[0, "total_papers_year"]) == 95
    assert float(y2014.loc[0, "share"]) == pytest.approx(5 / 95)
    assert float(y2014.loc[1, "share"]) == pytest.approx(30 / 95)
    assert float(y2014.loc[2, "share"]) == pytest.approx(60 / 95)
    # Shares of the real topics in a year sum to 1.0 (denominator is noise-free).
    assert float(y2014["share"].sum()) == pytest.approx(1.0)


def test_series_drops_partial_final_year() -> None:
    """A year > max_complete_year (2026) is dropped entirely."""
    papers = _three_topic_papers()
    series = build_topic_year_series(papers, max_complete_year=2025)
    assert series["year"].max() == 2025
    assert 2026 not in set(series["year"])


def test_series_has_expected_columns_and_dtypes() -> None:
    """The long series carries exactly the documented columns/dtypes."""
    papers = _three_topic_papers()
    series = build_topic_year_series(papers)
    assert list(series.columns) == [
        "topic_id",
        "year",
        "n_papers",
        "total_papers_year",
        "share",
    ]
    assert series["topic_id"].dtype == np.int64
    assert series["year"].dtype == np.int64
    assert series["n_papers"].dtype == np.int64
    assert series["total_papers_year"].dtype == np.int64
    assert series["share"].dtype == np.float64


# --------------------------------------------------------------------------- #
# fit_trajectories — horizon, band ordering/widening, direction
# --------------------------------------------------------------------------- #
def test_projected_horizon_is_five_years_per_topic() -> None:
    """Each topic gets exactly 5 projected rows (2026..2030 when last_obs_year=2025)."""
    papers = _three_topic_papers()
    series = build_topic_year_series(papers)
    series_out, _ = fit_trajectories(series, horizon=5, last_obs_year=2025)

    proj = series_out.loc[series_out["kind"] == "projected"]
    for _tid, grp in proj.groupby("topic_id"):
        years = sorted(grp["year"].tolist())
        assert len(years) == 5
        assert years == [2026, 2027, 2028, 2029, 2030]


def test_series_out_schema_and_band_nullity() -> None:
    """series_out has the locked columns; bands are null on observed, present on projected."""
    papers = _three_topic_papers()
    series = build_topic_year_series(papers)
    series_out, _ = fit_trajectories(series)
    assert list(series_out.columns) == _SERIES_COLS
    assert (series_out["grain"] == "leaf").all()

    obs = series_out.loc[series_out["kind"] == "observed"]
    proj = series_out.loc[series_out["kind"] == "projected"]
    # Bands are null on every observed row.
    for col in ("share_lo", "share_hi", "volume_lo", "volume_hi"):
        assert obs[col].isna().all()
        # ... and populated (non-null) on every projected row.
        assert proj[col].notna().all()
    # Observed rows carry the real share/volume (non-null).
    assert obs["share"].notna().all()
    assert obs["volume"].notna().all()


def test_projected_bands_are_ordered_and_widen_with_horizon() -> None:
    """lo <= mean <= hi on projected rows, and the band widens out to the horizon."""
    papers = _three_topic_papers()
    series = build_topic_year_series(papers)
    series_out, summary = fit_trajectories(series, horizon=5, ci_level=0.80, last_obs_year=2025)

    proj = series_out.loc[series_out["kind"] == "projected"]
    # Band ordering on every projected row, for both share and volume.
    assert (proj["share_lo"] <= proj["share"] + 1e-9).all()
    assert (proj["share"] <= proj["share_hi"] + 1e-9).all()
    assert (proj["volume_lo"] <= proj["volume"] + 1e-9).all()
    assert (proj["volume"] <= proj["volume_hi"] + 1e-9).all()
    # Bands stay in valid ranges (share in [0,1], volume >= 0).
    assert (proj["share_lo"] >= -1e-9).all()
    assert (proj["share_hi"] <= 1.0 + 1e-9).all()
    assert (proj["volume_lo"] >= -1e-9).all()

    # Band WIDTH at the horizon year (2030) >= width at the first projected year (2026),
    # per topic, for the share band (the headline). Fixtures are built in the small-share
    # regime so this holds after the expit back-transform too.
    for tid, grp in proj.groupby("topic_id"):
        g = grp.set_index("year")
        w_first = float(g.loc[2026, "share_hi"] - g.loc[2026, "share_lo"])
        w_last = float(g.loc[2030, "share_hi"] - g.loc[2030, "share_lo"])
        assert w_last >= w_first - 1e-9, f"share band must widen for topic {tid}"


def test_rising_series_direction_and_slope() -> None:
    """A constructed rising share series -> direction == 'rising' and slope > 0."""
    papers = _three_topic_papers()
    series = build_topic_year_series(papers)
    _, summary = fit_trajectories(series)
    s = summary.set_index("topic_id")
    assert s.loc[0, "direction"] == "rising"
    assert float(s.loc[0, "slope_share_per_yr"]) > 0


def test_falling_series_direction_and_slope() -> None:
    """A constructed falling share series -> direction == 'falling' and slope < 0."""
    papers = _three_topic_papers()
    series = build_topic_year_series(papers)
    _, summary = fit_trajectories(series)
    s = summary.set_index("topic_id")
    assert s.loc[2, "direction"] == "falling"
    assert float(s.loc[2, "slope_share_per_yr"]) < 0


def test_flat_series_direction_is_flat() -> None:
    """A constant-count topic -> its SHARE drifts only slightly -> direction 'flat'.

    Topic 1 holds a constant 30 papers/yr while topics 0 and 2 move in opposite
    directions, so the denominator is near-constant and topic 1's share slope sits
    inside the flat threshold.
    """
    papers = _three_topic_papers()
    series = build_topic_year_series(papers)
    _, summary = fit_trajectories(series)
    s = summary.set_index("topic_id")
    assert s.loc[1, "direction"] == "flat"


def test_summary_schema_and_constant_columns() -> None:
    """summary has the locked columns; label/size are placeholders the build fills."""
    papers = _three_topic_papers()
    series = build_topic_year_series(papers)
    _, summary = fit_trajectories(series, horizon=5, last_obs_year=2025)
    assert list(summary.columns) == _SUMMARY_COLS
    assert (summary["grain"] == "leaf").all()
    # label/size are NOT available to fit_trajectories -> placeholders (build joins real).
    assert (summary["label"] == "").all()
    assert summary["size"].isna().all()
    # horizon_year = last_obs_year + horizon = 2030 for every topic.
    assert (summary["horizon_year"] == 2030).all()
    # last_obs_year recorded as 2025.
    assert (summary["last_obs_year"] == 2025).all()
    # direction is one of the three allowed labels.
    assert set(summary["direction"]).issubset({"rising", "flat", "falling"})
    # model is one of the two allowed paths.
    assert set(summary["model"]).issubset({"state_space_llt", "loglinear_fallback"})


def test_summary_proj_matches_series_horizon_row() -> None:
    """summary.proj_* equals the series_out projected row at the horizon year (2030)."""
    papers = _three_topic_papers()
    series = build_topic_year_series(papers)
    series_out, summary = fit_trajectories(series, horizon=5, last_obs_year=2025)
    proj2030 = series_out.loc[(series_out["kind"] == "projected") & (series_out["year"] == 2030)]
    proj2030 = proj2030.set_index("topic_id")
    s = summary.set_index("topic_id")
    for tid in s.index:
        assert float(s.loc[tid, "proj_share"]) == pytest.approx(float(proj2030.loc[tid, "share"]))
        assert float(s.loc[tid, "proj_share_lo"]) == pytest.approx(
            float(proj2030.loc[tid, "share_lo"])
        )
        assert float(s.loc[tid, "proj_share_hi"]) == pytest.approx(
            float(proj2030.loc[tid, "share_hi"])
        )
        assert float(s.loc[tid, "proj_volume"]) == pytest.approx(float(proj2030.loc[tid, "volume"]))


# --------------------------------------------------------------------------- #
# Robustness — zeros/epsilon, short-series fallback, empty input, determinism
# --------------------------------------------------------------------------- #
def test_leading_zero_share_and_volume_fits_without_nan_or_inf() -> None:
    """A topic with leading zero-share / zero-volume years fits with finite bands."""
    # Topic 0 has zero papers in early years (it simply doesn't appear those years in
    # ITS own rows; the builder fills the regular grid with 0). Topic 1 carries the
    # corpus volume so a denominator exists every year.
    counts = {
        0: {
            2014: 0,
            2015: 0,
            2016: 0,
            2017: 1,
            2018: 2,
            2019: 4,
            2020: 6,
            2021: 9,
            2022: 12,
            2023: 16,
            2024: 20,
            2025: 25,
        },
        1: _flat_topic_counts(1, level=100),
    }
    papers = _papers_from_counts(counts)
    series = build_topic_year_series(papers)
    series_out, summary = fit_trajectories(series)

    proj = series_out.loc[(series_out["kind"] == "projected") & (series_out["topic_id"] == 0)]
    for col in ("share", "share_lo", "share_hi", "volume", "volume_lo", "volume_hi"):
        assert np.isfinite(proj[col].to_numpy(dtype="float64")).all(), f"{col} has NaN/inf"
    # Summary projection is finite too.
    s0 = summary.set_index("topic_id").loc[0]
    for col in ("proj_share", "proj_share_lo", "proj_share_hi", "proj_volume"):
        assert np.isfinite(float(s0[col]))


def test_short_series_takes_loglinear_fallback() -> None:
    """A topic with < min_years observed points routes to the log-linear fallback."""
    # Only 4 observed years -> below the default min_years=8.
    counts = {
        0: {2022: 2, 2023: 4, 2024: 6, 2025: 9},
        1: {2022: 50, 2023: 50, 2024: 50, 2025: 50},  # gives a denominator each year
    }
    papers = _papers_from_counts(counts)
    series = build_topic_year_series(papers)
    _, summary = fit_trajectories(series, min_years=8, last_obs_year=2025)
    s = summary.set_index("topic_id")
    assert s.loc[0, "model"] == "loglinear_fallback"
    # The fallback still produces a 5-year projection with ordered, finite bands.
    assert s.loc[0, "horizon_year"] == 2030
    assert np.isfinite(float(s.loc[0, "proj_share"]))
    assert float(s.loc[0, "proj_share_lo"]) <= float(s.loc[0, "proj_share"]) + 1e-9
    assert float(s.loc[0, "proj_share"]) <= float(s.loc[0, "proj_share_hi"]) + 1e-9


def test_empty_input_yields_well_formed_empty_frames() -> None:
    """Empty input -> well-formed empty frames (correct columns, no exception)."""
    empty_papers = pd.DataFrame(
        {"topic_id": pd.Series([], dtype="int64"), "year": pd.Series([], dtype="int64")}
    )
    series = build_topic_year_series(empty_papers)
    assert series.empty
    assert list(series.columns) == [
        "topic_id",
        "year",
        "n_papers",
        "total_papers_year",
        "share",
    ]

    series_out, summary = fit_trajectories(series)
    assert series_out.empty
    assert list(series_out.columns) == _SERIES_COLS
    assert summary.empty
    assert list(summary.columns) == _SUMMARY_COLS


def test_pipeline_is_deterministic() -> None:
    """Running the whole pipeline twice on the same input yields identical frames."""
    papers = _three_topic_papers()
    series_a = build_topic_year_series(papers)
    series_b = build_topic_year_series(papers)
    pd.testing.assert_frame_equal(series_a, series_b)

    so_a, sm_a = fit_trajectories(series_a)
    so_b, sm_b = fit_trajectories(series_b)
    pd.testing.assert_frame_equal(so_a, so_b)
    pd.testing.assert_frame_equal(sm_a, sm_b)


def test_fit_trajectories_fills_grid_gaps_so_horizon_is_defined() -> None:
    """A topic with a GAP year still fits (the builder/fitter fill a regular grid).

    Topic 0 has no papers in 2018 and 2020 (gap years). The fitter must build a
    contiguous annual grid so the +5 horizon is well-defined and the projection
    exists.
    """
    counts = {
        0: {
            2014: 2,
            2015: 3,
            2016: 4,
            2017: 5,
            2019: 8,
            2021: 12,
            2022: 14,
            2023: 17,
            2024: 20,
            2025: 24,
        },
        1: _flat_topic_counts(1, level=80),
    }
    papers = _papers_from_counts(counts)
    series = build_topic_year_series(papers)
    series_out, summary = fit_trajectories(series, last_obs_year=2025)
    proj0 = series_out.loc[(series_out["kind"] == "projected") & (series_out["topic_id"] == 0)]
    assert sorted(proj0["year"].tolist()) == [2026, 2027, 2028, 2029, 2030]
    assert summary.set_index("topic_id").loc[0, "horizon_year"] == 2030


def test_observed_rows_round_trip_the_input_shares() -> None:
    """Observed series_out rows carry the same share values the builder computed."""
    papers = _three_topic_papers()
    series = build_topic_year_series(papers)
    series_out, _ = fit_trajectories(series)
    obs = series_out.loc[series_out["kind"] == "observed"].set_index(["topic_id", "year"])
    src = series.set_index(["topic_id", "year"])
    # Every observed (topic, year) share matches the builder's share to float tol.
    for idx in src.index:
        assert float(obs.loc[idx, "share"]) == pytest.approx(float(src.loc[idx, "share"]))
