"""Tests for the V2-S07 map loader/prep layer (``scifield.cartography.mapio``).

These are *integration-ish* tests against the **real** V2 parquet artifacts produced
by the earlier batches (cascade, roles, origins) and the V1 topic hierarchy. They
assert that each loader returns the expected columns and a non-empty, sanely-shaped
frame. If an artifact is missing the test ``skip``s gracefully (so the suite still runs
on a fresh checkout before the data builders have been run) — but in the live repo all
the inputs exist, so these tests exercise the real shaping.

A few small pure helpers (``journal_display``, ``_check_grain``, ``_join_words``) are
also tested with hand-built inputs so the prep logic has unit coverage independent of
the data files.
"""

from __future__ import annotations

import pytest

from scifield.cartography import mapio


def _has(*parts: str) -> bool:
    """True if a path under the inferred repo root exists."""
    return mapio._DEFAULT_REPO_ROOT.joinpath(*parts).exists()


# --------------------------------------------------------------------------- pure helpers


def test_journal_display_maps_known_and_passes_through_unknown() -> None:
    """Known slugs map to short labels; an unknown slug is returned verbatim."""
    assert mapio.journal_display("ann_surg") == "Ann Surg"
    assert mapio.journal_display("j_bone_joint_surg_am") == "J Bone Joint Surg"
    assert mapio.journal_display("not_a_journal") == "not_a_journal"


def test_check_grain_validates_and_maps() -> None:
    """Valid grains map to their topic key; an invalid grain raises ValueError."""
    assert mapio._check_grain("leaf") == "topic_id"
    assert mapio._check_grain("mid") == "mid_level_id"
    with pytest.raises(ValueError):
        mapio._check_grain("bogus")


def test_join_words_handles_list_and_nan() -> None:
    """``_join_words`` joins up to n words and tolerates a None/NaN cell."""
    assert mapio._join_words(["a", "b", "c"], n=2) == "a, b"
    assert mapio._join_words(None) == ""


def test_role_color_palette_is_complete() -> None:
    """The locked role palette covers exactly source/bridge/terminal."""
    assert set(mapio.ROLE_COLOR) == {"source", "bridge", "terminal"}


# --------------------------------------------------------------------------- topic labels


def test_topic_labels_columns_and_nonempty() -> None:
    """topic_labels returns id/label/size/mid columns, one row per leaf topic."""
    if not _has("data/v1/topic_hierarchy.parquet"):
        pytest.skip("topic_hierarchy.parquet not present")
    df = mapio.topic_labels()
    assert set(df.columns) == {"topic_id", "label", "size", "mid_level_id"}
    assert len(df) > 0
    # label is a non-empty string for the large topics.
    assert df["label"].map(lambda s: isinstance(s, str)).all()


# --------------------------------------------------------------------------- cascade loaders


def test_load_lag_matrix_columns_and_signed() -> None:
    """lag matrix is tidy long, has the signed mean_lag, and excludes the grain col."""
    if not _has("V2/data/cascade/lag_matrix.parquet"):
        pytest.skip("lag_matrix.parquet not present")
    df = mapio.load_lag_matrix("leaf")
    assert {"journal_i", "journal_j", "mean_lag", "n_shared"} <= set(df.columns)
    assert "grain" not in df.columns
    assert len(df) == 90  # 10 journals choose 2, ordered


def test_lag_matrix_wide_is_square_with_zero_diagonal() -> None:
    """The wide form is square over the journals with a zeroed diagonal."""
    if not _has("V2/data/cascade/lag_matrix.parquet"):
        pytest.skip("lag_matrix.parquet not present")
    wide = mapio.lag_matrix_wide("leaf")
    assert wide.shape[0] == wide.shape[1] == 10
    for j in wide.index:
        assert wide.loc[j, j] == 0.0


def test_load_origin_attribution_one_row_per_topic() -> None:
    """origin attribution has the expected columns and one row per topic."""
    if not _has("V2/data/cascade/origin_attribution.parquet"):
        pytest.skip("origin_attribution.parquet not present")
    df = mapio.load_origin_attribution("leaf")
    assert {"topic_id", "origin_journal_slug", "origin_year", "tie"} <= set(df.columns)
    assert df["topic_id"].is_unique
    assert len(df) > 0


def test_load_diffusion_curves_columns() -> None:
    """diffusion curves carry reach/span/logistic columns, one row per topic."""
    if not _has("V2/data/cascade/diffusion_curves.parquet"):
        pytest.skip("diffusion_curves.parquet not present")
    df = mapio.load_diffusion_curves("leaf")
    assert {"reach_fraction", "n_journals", "curve_fitted"} <= set(df.columns)
    assert len(df) > 0


def test_adoption_curve_is_monotone_non_increasing() -> None:
    """The corpus adoption curve is a non-increasing fraction over breadth 1..10."""
    if not _has("V2/data/cascade/diffusion_curves.parquet"):
        pytest.skip("diffusion_curves.parquet not present")
    ac = mapio.adoption_curve("leaf")
    assert list(ac["n_journals_reached"]) == list(range(1, 11))
    fracs = ac["frac_topics"].tolist()
    assert fracs[0] == pytest.approx(1.0)  # every topic reaches >= 1 journal
    assert all(a >= b - 1e-9 for a, b in zip(fracs, fracs[1:], strict=False))


# --------------------------------------------------------------------------- role loaders


def test_load_roles_columns_and_ten_journals() -> None:
    """role scores have the component columns, a role label, and 10 journals."""
    if not _has("V2/data/roles/role_scores.parquet"):
        pytest.skip("role_scores.parquet not present")
    df = mapio.load_roles("leaf")
    assert {"journal_slug", "source", "bridge", "terminal", "role", "display"} <= set(df.columns)
    assert len(df) == 10
    assert set(df["role"]) <= {"source", "bridge", "terminal"}


def test_load_velocity_columns_and_ten_journals() -> None:
    """velocity table has median_lag and the velocity label for 10 journals."""
    if not _has("V2/data/roles/velocity.parquet"):
        pytest.skip("velocity.parquet not present")
    df = mapio.load_velocity()
    assert {"journal_slug", "median_lag", "velocity", "display"} <= set(df.columns)
    assert len(df) == 10


# --------------------------------------------------------------------------- origin loaders


def test_load_origins_returns_four_frames() -> None:
    """load_origins returns the four novelty tables, each non-empty."""
    needed = [
        "V2/data/origins/sector_novelty.parquet",
        "V2/data/origins/geo_novelty.parquet",
        "V2/data/origins/recombination.parquet",
        "V2/data/origins/recombination_by_topic.parquet",
    ]
    if not all(_has(*p.split("/")) for p in needed):
        pytest.skip("origins parquets not all present")
    frames = mapio.load_origins()
    assert set(frames) == {"sector", "geo", "recombination", "recombination_by_topic"}
    for name, frame in frames.items():
        assert len(frame) > 0, f"{name} frame empty"


def test_geo_ranked_drops_low_n_and_sorts_descending() -> None:
    """geo_ranked drops low-n countries and is sorted descending by novelty."""
    if not _has("V2/data/origins/geo_novelty.parquet"):
        pytest.skip("geo_novelty.parquet not present")
    df = mapio.geo_ranked()
    assert {"country_code", "n_papers", "sem_nov_mean_mean"} <= set(df.columns)
    vals = df["sem_nov_mean_mean"].tolist()
    assert all(a >= b for a, b in zip(vals, vals[1:], strict=False))


# --------------------------------------------------------------------------- landscape frame


def test_topic_landscape_frame_joins_origin_and_recombination() -> None:
    """The landscape frame joins labels + origin + recombination, one row per topic."""
    needed = [
        "data/v1/topic_hierarchy.parquet",
        "V2/data/cascade/origin_attribution.parquet",
        "V2/data/origins/recombination_by_topic.parquet",
    ]
    if not all(_has(*p.split("/")) for p in needed):
        pytest.skip("inputs for landscape frame not all present")
    df = mapio.topic_landscape_frame("leaf")
    assert {
        "topic_id",
        "label",
        "size",
        "origin_journal_slug",
        "origin_display",
        "recombination_rate",
    } <= set(df.columns)
    assert df["topic_id"].is_unique
    assert len(df) > 0


def test_topic_landscape_frame_rejects_mid() -> None:
    """The landscape frame is leaf-only (topic words live at leaf grain)."""
    with pytest.raises(ValueError):
        mapio.topic_landscape_frame("mid")


# --------------------------------------------------------------------------- trajectory loaders
#
# The trajectory layer (V2-S10) is materialised ONLY for data version ``v2`` at
# ``V2/data_v2/trajectory/*.parquet`` (v1 has no trajectory layer). So every loader test
# below passes ``data_version="v2"`` explicitly and skip-guards on the v2 artifact path.

_TRAJ_SUMMARY = "V2/data_v2/trajectory/trajectory_summary.parquet"
_TRAJ_SERIES = "V2/data_v2/trajectory/trajectory_series.parquet"

#: Columns the trajectory-summary loader is contracted to return (grain dropped, label kept).
_TRAJ_SUMMARY_COLS = {
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
}

#: Columns the trajectory-series loader is contracted to return (grain dropped).
_TRAJ_SERIES_COLS = {
    "topic_id",
    "year",
    "kind",
    "share",
    "share_lo",
    "share_hi",
    "volume",
    "volume_lo",
    "volume_hi",
}


def test_load_trajectory_summary_columns_and_directions() -> None:
    """Summary has the contracted columns, drops grain, is unique-per-topic and bool fit_ok."""
    if not _has(*_TRAJ_SUMMARY.split("/")):
        pytest.skip("v2 trajectory_summary.parquet not present")
    df = mapio.load_trajectory_summary("leaf", data_version="v2")
    assert set(df.columns) >= _TRAJ_SUMMARY_COLS
    assert "grain" not in df.columns
    assert df["topic_id"].is_unique
    assert len(df) > 0
    assert set(df["direction"]) <= {"rising", "flat", "falling"}
    assert df["fit_ok"].dtype == bool


def test_load_trajectory_series_columns_and_kinds() -> None:
    """Series has the contracted columns; observed vs projected split + NaN bands hold."""
    if not _has(*_TRAJ_SERIES.split("/")):
        pytest.skip("v2 trajectory_series.parquet not present")
    df = mapio.load_trajectory_series("leaf", data_version="v2")
    assert set(df.columns) >= _TRAJ_SERIES_COLS
    assert "grain" not in df.columns
    assert len(df) > 0
    assert set(df["kind"]) == {"observed", "projected"}
    observed = df[df["kind"] == "observed"]
    projected = df[df["kind"] == "projected"]
    # The projection extends past the last observed year (to the 2030 horizon).
    assert int(projected["year"].max()) == 2030
    assert int(projected["year"].max()) > int(observed["year"].max())
    # Observed rows carry no uncertainty band (the band is a projection-only artefact).
    assert observed["share_lo"].isna().all()


def test_load_trajectory_summary_raises_when_absent(tmp_path) -> None:
    """An empty repo root (no artifact) raises FileNotFoundError, not a cryptic error."""
    with pytest.raises(FileNotFoundError):
        mapio.load_trajectory_summary("leaf", repo_root=tmp_path, data_version="v2")


def test_load_trajectory_series_raises_when_absent(tmp_path) -> None:
    """An empty repo root (no artifact) raises FileNotFoundError for the series loader too."""
    with pytest.raises(FileNotFoundError):
        mapio.load_trajectory_series("leaf", repo_root=tmp_path, data_version="v2")


def test_load_trajectory_summary_rejects_bad_grain() -> None:
    """A bogus grain token is rejected up front (before any I/O) with ValueError."""
    with pytest.raises(ValueError):
        mapio.load_trajectory_summary("bogus", data_version="v2")


def test_load_trajectory_series_rejects_bad_grain() -> None:
    """A bogus grain token is rejected up front (before any I/O) with ValueError."""
    with pytest.raises(ValueError):
        mapio.load_trajectory_series("bogus", data_version="v2")
