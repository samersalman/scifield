"""Unit tests for the V1-S11 dual-novelty 2x2 archetype binning.

No network / no data files: every fixture is a small hand-built in-memory
pandas DataFrame with values whose medians are computable by hand, and explicit
``thresholds`` dicts are passed to the pure :func:`bin_archetypes`. The I/O
driver :func:`compute_archetypes` is deliberately NOT exercised here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scifield.novelty.archetypes import (
    QUADRANT_LABELS,
    archetype_column_name,
    assign_quadrant,
    bin_archetypes,
)

# The four metric columns used throughout the suite.
_SEM_METRICS = ("sem_nov_mean", "sem_nov_min")
_STRUCT_METRICS = ("cd5", "cd10")
_ALL_PAIRINGS = [(s, t) for s in _SEM_METRICS for t in _STRUCT_METRICS]


def test_assign_quadrant_tie_and_around_threshold() -> None:
    """High iff value >= threshold: tie -> High, above -> High, below -> Low."""
    assert assign_quadrant(value=1.0, threshold=1.0) == "H"  # tie -> High
    assert assign_quadrant(value=1.5, threshold=1.0) == "H"  # above -> High
    assert assign_quadrant(value=0.5, threshold=1.0) == "L"  # below -> Low


def test_median_thresholds_drive_high_low_assignment() -> None:
    """bin_archetypes uses hand-computed medians; tie at the median is High."""
    # sem_nov_mean over these five rows: sorted [0.1,0.2,0.3,0.4,0.5] -> median 0.3.
    # cd5 over these five rows: sorted [-0.2,0.0,0.1,0.2,0.4] -> median 0.1.
    df = pd.DataFrame(
        {
            "sem_nov_mean": [0.3, 0.5, 0.2, 0.4, 0.1],
            "sem_nov_min": [0.3, 0.5, 0.2, 0.4, 0.1],
            "cd5": [0.1, 0.4, 0.0, 0.2, -0.2],
            "cd10": [0.1, 0.4, 0.0, 0.2, -0.2],
        }
    )
    thresholds = {
        "sem_nov_mean": 0.3,
        "sem_nov_min": 0.3,
        "cd5": 0.1,
        "cd10": 0.1,
    }
    out = bin_archetypes(df, pairings=[("sem_nov_mean", "cd5")], thresholds=thresholds)

    # Row 0 sits exactly on BOTH medians -> tie goes High on both -> HH.
    assert out.loc[0, "arch_mean_cd5"] == QUADRANT_LABELS["HH"]
    # Row 1 is above both medians -> HH.
    assert out.loc[1, "arch_mean_cd5"] == QUADRANT_LABELS["HH"]
    # Row 2 is below both medians -> LL.
    assert out.loc[2, "arch_mean_cd5"] == QUADRANT_LABELS["LL"]
    # Row 4 is below both medians -> LL.
    assert out.loc[4, "arch_mean_cd5"] == QUADRANT_LABELS["LL"]


def test_four_quadrant_label_mapping() -> None:
    """Each of HH/HL/LH/LL maps to its documented prose archetype label."""
    # Thresholds fixed at 0.0 on every axis; values chosen to land in each cell.
    thresholds = {m: 0.0 for m in (*_SEM_METRICS, *_STRUCT_METRICS)}
    df = pd.DataFrame(
        {
            # semantic-High (>=0) / structural-High (>=0) -> HH
            # semantic-High / structural-Low -> HL
            # semantic-Low / structural-High -> LH
            # semantic-Low / structural-Low -> LL
            "sem_nov_mean": [1.0, 1.0, -1.0, -1.0],
            "sem_nov_min": [1.0, 1.0, -1.0, -1.0],
            "cd5": [1.0, -1.0, 1.0, -1.0],
            "cd10": [1.0, -1.0, 1.0, -1.0],
        }
    )
    out = bin_archetypes(df, pairings=[("sem_nov_mean", "cd5")], thresholds=thresholds)

    assert out.loc[0, "arch_mean_cd5"] == "disruptive-novel"
    assert out.loc[1, "arch_mean_cd5"] == "novel-consolidating"
    assert out.loc[2, "arch_mean_cd5"] == "conventional-disruptive"
    assert out.loc[3, "arch_mean_cd5"] == "incremental"

    # Cross-check the prose labels against the module constant.
    assert out.loc[0, "arch_mean_cd5"] == QUADRANT_LABELS["HH"]
    assert out.loc[1, "arch_mean_cd5"] == QUADRANT_LABELS["HL"]
    assert out.loc[2, "arch_mean_cd5"] == QUADRANT_LABELS["LH"]
    assert out.loc[3, "arch_mean_cd5"] == QUADRANT_LABELS["LL"]


def test_null_axis_row_yields_null_archetype_without_crashing() -> None:
    """A NaN on either axis -> null label; the row is excluded, not a crash."""
    thresholds = {m: 0.0 for m in (*_SEM_METRICS, *_STRUCT_METRICS)}
    df = pd.DataFrame(
        {
            "sem_nov_mean": [1.0, np.nan, 1.0, -1.0],
            "sem_nov_min": [1.0, 1.0, 1.0, -1.0],
            "cd5": [1.0, 1.0, np.nan, -1.0],
            "cd10": [1.0, 1.0, 1.0, -1.0],
        }
    )
    out = bin_archetypes(df, pairings=[("sem_nov_mean", "cd5")], thresholds=thresholds)

    # Row 1 has NaN semantic; row 2 has NaN structural -> both null.
    assert pd.isna(out.loc[1, "arch_mean_cd5"])
    assert pd.isna(out.loc[2, "arch_mean_cd5"])
    # Non-null rows are unaffected.
    assert out.loc[0, "arch_mean_cd5"] == "disruptive-novel"
    assert out.loc[3, "arch_mean_cd5"] == "incremental"


def test_archetype_column_name_variants() -> None:
    """Column naming strips the sem_nov_ prefix and keeps the structural verbatim."""
    assert (
        archetype_column_name(semantic_metric="sem_nov_mean", structural_metric="cd5")
        == "arch_mean_cd5"
    )
    assert (
        archetype_column_name(semantic_metric="sem_nov_mean", structural_metric="cd10")
        == "arch_mean_cd10"
    )
    assert (
        archetype_column_name(semantic_metric="sem_nov_min", structural_metric="cd5")
        == "arch_min_cd5"
    )
    assert (
        archetype_column_name(semantic_metric="sem_nov_min", structural_metric="cd10")
        == "arch_min_cd10"
    )


def test_bin_archetypes_emits_exactly_the_four_pairing_columns() -> None:
    """All four pairings add exactly the four arch_* columns to the input frame."""
    thresholds = {m: 0.0 for m in (*_SEM_METRICS, *_STRUCT_METRICS)}
    input_cols = [*_SEM_METRICS, *_STRUCT_METRICS]
    df = pd.DataFrame(
        {
            "sem_nov_mean": [1.0, -1.0],
            "sem_nov_min": [1.0, -1.0],
            "cd5": [1.0, -1.0],
            "cd10": [1.0, -1.0],
        }
    )
    out = bin_archetypes(df, pairings=_ALL_PAIRINGS, thresholds=thresholds)

    expected_arch = {"arch_mean_cd5", "arch_mean_cd10", "arch_min_cd5", "arch_min_cd10"}
    new_cols = set(out.columns) - set(input_cols)
    assert new_cols == expected_arch
    # Input columns are preserved alongside the new label columns.
    assert set(input_cols).issubset(set(out.columns))


def test_pairings_are_independent_per_structural_axis() -> None:
    """One row can be HH on one pairing and LH on another via differing axes."""
    # Single row: semantic-Low on mean, but High on cd5 and Low on cd10.
    thresholds = {
        "sem_nov_mean": 0.0,
        "sem_nov_min": 0.0,
        "cd5": 0.0,
        "cd10": 0.0,
    }
    df = pd.DataFrame(
        {
            "sem_nov_mean": [-1.0],  # Low
            "sem_nov_min": [1.0],  # High
            "cd5": [1.0],  # High
            "cd10": [-1.0],  # Low
        }
    )
    out = bin_archetypes(df, pairings=_ALL_PAIRINGS, thresholds=thresholds)

    # mean (Low) x cd5 (High) -> LH -> conventional-disruptive
    assert out.loc[0, "arch_mean_cd5"] == "conventional-disruptive"
    # min (High) x cd5 (High) -> HH -> disruptive-novel
    assert out.loc[0, "arch_min_cd5"] == "disruptive-novel"
    # The two columns differ for the same row.
    assert out.loc[0, "arch_mean_cd5"] != out.loc[0, "arch_min_cd5"]
