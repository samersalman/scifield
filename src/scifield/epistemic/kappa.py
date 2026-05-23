"""Inter-rater agreement metrics (Cohen's kappa and Krippendorff's alpha) for V1-S07.

Thin wrappers around :func:`sklearn.metrics.cohen_kappa_score` and
:func:`krippendorff.alpha` plus a tidy :func:`per_field_summary`
DataFrame builder that V1-S08 will call once real label parquets land.

This module is intentionally narrow: paired ratings only (two raters),
None-safe (any pair where either side is None is dropped before the
metric runs), and defensive against degenerate inputs (single-class
columns, fewer than two usable pairs) — these return ``float("nan")``
rather than letting sklearn/krippendorff raise or warn mid-pipeline.

.. warning::
    **V1-S07 scope discipline.** This module is *not* to be run on real
    label data in V1-S07. Per the plan's risk-and-stop-condition
    section: "If you find yourself running ``kappa.py`` on anything
    except synthetic test data in this session, you've drifted into
    V1-S08 — stop." Synthetic test data is the only acceptable input
    during this session; real-label κ/α computation is V1-S08 work.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

__all__ = [
    "cohens_kappa",
    "krippendorffs_alpha",
    "per_field_summary",
]


def _drop_none_pairs(rater_a: Iterable[Any], rater_b: Iterable[Any]) -> tuple[list[Any], list[Any]]:
    """Return paired ratings with any (None-on-either-side) pair removed.

    Casting to ``list`` first lets us accept any iterable (generators,
    pandas Series, numpy arrays) without consuming the caller's input
    twice.
    """
    a = list(rater_a)
    b = list(rater_b)
    if len(a) != len(b):
        raise ValueError(f"rater_a and rater_b must have same length; got {len(a)} vs {len(b)}")
    paired = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not paired:
        return [], []
    a_clean, b_clean = zip(*paired, strict=True)
    return list(a_clean), list(b_clean)


def cohens_kappa(rater_a: Iterable[Any], rater_b: Iterable[Any]) -> float:
    """Cohen's kappa for two raters, None-safe and degenerate-input-safe.

    Pairs where either side is ``None`` are dropped before scoring.
    Returns ``float("nan")`` when fewer than 2 usable pairs survive or
    only one unique label is present across both raters (sklearn would
    otherwise raise/warn on a degenerate confusion matrix).
    """
    a, b = _drop_none_pairs(rater_a, rater_b)
    if len(a) < 2:
        return float("nan")
    unique_labels = set(a) | set(b)
    if len(unique_labels) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(cohen_kappa_score(a, b))


def _encode_for_krippendorff(a: Sequence[Any], b: Sequence[Any]) -> np.ndarray:
    """Map heterogeneous label values to integer codes for krippendorff.

    The ``krippendorff`` package can't infer a value domain from bool
    or object arrays, so we project both raters' values into a shared
    integer code space (NaN-free, since None pairs are already dropped)
    and return a ``(2, n)`` float array suitable for
    :func:`krippendorff.alpha`.
    """
    domain = sorted({*a, *b}, key=lambda v: (str(type(v)), str(v)))
    code = {v: i for i, v in enumerate(domain)}
    arr = np.array(
        [[code[x] for x in a], [code[y] for y in b]],
        dtype=float,
    )
    return arr


def krippendorffs_alpha(
    rater_a: Iterable[Any],
    rater_b: Iterable[Any],
    level_of_measurement: str = "nominal",
) -> float:
    """Krippendorff's alpha for two raters, None-safe.

    Builds the ``(n_raters=2, n_items)`` reliability array
    :func:`krippendorff.alpha` expects. Pairs where either side is
    ``None`` are dropped (rather than coded as ``np.nan``) to keep
    behaviour aligned with :func:`cohens_kappa`. Returns
    ``float("nan")`` for <2 usable pairs or a single-class input.
    """
    a, b = _drop_none_pairs(rater_a, rater_b)
    if len(a) < 2:
        return float("nan")
    unique_labels = set(a) | set(b)
    if len(unique_labels) < 2:
        return float("nan")
    reliability = _encode_for_krippendorff(a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(
            krippendorff.alpha(
                reliability_data=reliability,
                level_of_measurement=level_of_measurement,
            )
        )


def per_field_summary(
    label_pairs: Mapping[str, tuple[Sequence[Any], Sequence[Any]]],
) -> pd.DataFrame:
    """Tidy per-field κ/α summary for a paired set of labelers.

    Parameters
    ----------
    label_pairs:
        Mapping of ``field_name -> (rater_a_values, rater_b_values)``.
        Each list is the full per-row label sequence for that field;
        ``None`` entries mark unrated rows and are dropped pairwise.

    Returns
    -------
    pandas.DataFrame
        Columns ``field, n, n_agree, kappa, alpha``, one row per input
        field, preserving insertion order. ``n`` counts comparable
        (non-None) pairs; ``n_agree`` counts those where the two
        raters' values matched. Constant-field cases (single unique
        value across both raters) report ``kappa=nan, alpha=nan``
        rather than raising.
    """
    rows: list[dict[str, Any]] = []
    for field, (a_raw, b_raw) in label_pairs.items():
        a, b = _drop_none_pairs(a_raw, b_raw)
        n = len(a)
        n_agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
        kappa = cohens_kappa(a, b)
        alpha = krippendorffs_alpha(a, b)
        rows.append(
            {
                "field": field,
                "n": n,
                "n_agree": n_agree,
                "kappa": kappa,
                "alpha": alpha,
            }
        )
    return pd.DataFrame(rows, columns=["field", "n", "n_agree", "kappa", "alpha"])
