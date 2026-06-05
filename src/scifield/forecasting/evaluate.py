"""V1-S14 sealed-test-set evaluation orchestration + pre-registered Gate G4 stats.

This module is the integrity-critical adjudication layer for V1-S14: it runs the
ONE-TIME evaluation of the four V1-S12 baselines and the frozen V1-S13 HGT on the
sealed TEST slice, and computes the pure, pre-registered statistics that decide
Gate G4 (OSF DOI 10.17605/OSF.IO/XP94F). It deliberately mirrors the val-time
orchestration in :mod:`scifield.forecasting.baselines.evaluate` — same baselines,
same fit-on-TRAIN/score-on-held-out shape, same locked slice masks — with ``val``
swapped for ``test`` and the HGT appended as a fifth model.

It computes whatever the inputs say. There is no path that favours one model: a
null finding (HGT failing to clear the pre-registered bar over ``no_graph``) is
the expected, honest outcome, and :func:`gate_g4_verdict` reports it as such.

Slice definitions (LOCKED, contract §7 — reused verbatim from the baseline harness)
-----------------------------------------------------------------------------------
* TRAIN slice = ``split == "train" & volume_ok & label_complete`` (the four
  baselines are fit here, identically to validation/model-selection).
* TEST slice  = ``split == "test"  & volume_ok & label_complete`` (every model is
  scored here, and ONLY here, exactly once).

Both metrics flow through the pre-registered guards in
:mod:`scifield.forecasting.baselines.base`: :func:`emergence_auc` returns ``nan``
on a single-class slice and :func:`share_mape` is computed only over rows with a
strictly positive ``forward_share`` target.

The HGT is INJECTED, never retrained
------------------------------------
The ``hgt`` argument is an ALREADY-FITTED
:class:`scifield.forecasting.gnn.hgt.HGTForecaster` restored from a frozen
checkpoint by the caller. This module calls ``hgt.predict(test_slice)`` directly
and NEVER calls ``hgt.fit`` — re-fitting on the test session would void the
pre-registration. Consequently ``torch`` / ``torch_geometric`` / HGT symbols are
kept out of this module's top level entirely; only the injected, opaque
``predict``-able object crosses the boundary.

The pre-registered hypotheses
-----------------------------
* **H1 (AUC margin):** ``test emergence-AUC(hgt) - AUC(no_graph) > margin_pp``
  (default 5 percentage points). The "graph adds nothing unless it clears a real
  margin" bar.
* **H2 (paired significance):** a per-row **Brier-loss** Wilcoxon signed-rank test
  of HGT vs ``no_graph`` over the identical ``(topic_id, origin_year)`` test rows,
  ``diff = brier(hgt) - brier(no_graph)`` (negative => HGT lower loss => better),
  significant at ``alpha`` (default 0.05). The Brier comparison is meaningful only
  because both models emit ``emergence_score`` as a sigmoid probability in
  ``[0, 1]`` (verified for ``no_graph`` and ``hgt``).

:func:`gate_g4_verdict` ANDs H1 and H2 into the mechanical recommendation; the
human adjudication (the actual Gate G4 sign-off) is recorded elsewhere.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

from scifield.forecasting.baselines.base import emergence_auc, share_mape
from scifield.forecasting.baselines.evaluate import (
    _build_predictors,
    _labeled_slice,
    _pin_torch_single_threaded,
)

__all__ = [
    "calibration_table",
    "gate_g4_verdict",
    "paired_brier_wilcoxon",
    "paired_score_wilcoxon",
    "test_metrics_table",
    "test_per_unit_scores",
    "test_sensitivity_table",
]

#: LOCKED — the test metrics-table columns, one row per model, in this order.
_TEST_METRIC_COLUMNS = (
    "model",
    "emergence_auc",
    "share_mape",
    "n_train",
    "n_test",
    "n_test_pos",
    "fallback_frac",
)

#: The per-unit score-staging keys (test slice columns + the two model outputs).
_PER_UNIT_KEYS = (
    "topic_id",
    "origin_year",
    "emergence_score",
    "emergent",
    "emergent_additive",
    "emergent_count_surge",
    "share_forecast",
    "forward_share",
)

#: LOCKED — the calibration (reliability-diagram) table columns, in this order.
_CALIBRATION_COLUMNS = (
    "bin_index",
    "bin_lo",
    "bin_hi",
    "n",
    "mean_predicted",
    "observed_freq",
)

#: LOCKED — the sensitivity table columns, in this order.
_SENSITIVITY_COLUMNS = (
    "variant",
    "label_column",
    "emergence_auc",
    "n_test",
    "n_test_pos",
    "delta_vs_primary_pp",
)

#: The three pre-registered emergence-label variants: (variant, label column).
_SENSITIVITY_VARIANTS = (
    ("primary", "emergent"),
    ("additive_jump", "emergent_additive"),
    ("count_surge", "emergent_count_surge"),
)


def _arima_fallback_frac(predictor: Any) -> float:
    """Read ``n_fallback_ / n_total_`` off a fitted arima, else ``nan``.

    Only meaningful for :class:`~scifield.forecasting.baselines.arima.ArimaPerTopic`
    and only AFTER ``predict`` has populated the counters (identical to the
    val-harness diagnostic in
    :func:`scifield.forecasting.baselines.evaluate.run_baselines`).
    """
    n_total = getattr(predictor, "n_total_", 0)
    if predictor.name == "arima" and n_total:
        return float(predictor.n_fallback_) / float(n_total)
    return float("nan")


def test_metrics_table(features: pd.DataFrame, cfg: Any, hgt: Any) -> pd.DataFrame:
    """Fit the four baselines on TRAIN, score them + the injected HGT on TEST.

    The four baselines are fit on the TRAIN labeled slice
    (``split=="train" & volume_ok & label_complete``) — byte-identical to the
    validation/model-selection fit — and then every model is scored ONCE on the
    TEST labeled slice (``split=="test" & ...``). The injected ``hgt`` is scored
    via ``hgt.predict`` only; it is NEVER fit here. ``fallback_frac`` is read off
    the fitted :class:`ArimaPerTopic` AFTER ``predict`` and is ``nan`` for every
    other model (including the HGT).

    Parameters
    ----------
    features:
        The materialized ``(topic_id, origin_year, NODE_FEATURES..., label cols,
        split)`` frame from :func:`scifield.forecasting.data.materialize`.
    cfg:
        The forecasting ``DictConfig`` (``conf/forecasting/v1.yaml``).
    hgt:
        An ALREADY-FITTED forecaster with ``name == "hgt"`` and
        ``predict(features) -> ForecastPrediction`` (restored from a frozen
        checkpoint by the caller; never retrained).

    Returns
    -------
    pandas.DataFrame
        Exactly five rows in the order ``naive, arima, mlp, no_graph, hgt`` with
        the LOCKED columns ``model, emergence_auc, share_mape, n_train, n_test,
        n_test_pos, fallback_frac`` (in that order).
    """
    train = _labeled_slice(features, "train")
    test = _labeled_slice(features, "test")

    n_train = int(len(train))
    n_test = int(len(test))

    test_emergent = test["emergent"].to_numpy() if n_test else np.empty(0)
    test_forward = test["forward_share"].to_numpy() if n_test else np.empty(0)
    n_test_pos = int((test_emergent == 1).sum()) if n_test else 0

    train_labels = train[["emergent", "forward_share"]]

    _pin_torch_single_threaded()

    rows: list[dict[str, Any]] = []

    def _score(name: str, pred: Any, fallback_frac: float) -> None:
        rows.append(
            {
                "model": str(name),
                "emergence_auc": float(emergence_auc(pred.emergence_score, test_emergent)),
                "share_mape": float(share_mape(pred.share_forecast, test_forward)),
                "n_train": n_train,
                "n_test": n_test,
                "n_test_pos": n_test_pos,
                "fallback_frac": fallback_frac,
            }
        )

    for predictor in _build_predictors(cfg):
        predictor.fit(train, train_labels)
        pred = predictor.predict(test)
        _score(predictor.name, pred, _arima_fallback_frac(predictor))

    # The HGT is scored from the injected, already-fitted object — NEVER fit here.
    _score("hgt", hgt.predict(test), float("nan"))

    return pd.DataFrame(rows, columns=list(_TEST_METRIC_COLUMNS))


def test_per_unit_scores(
    features: pd.DataFrame, cfg: Any, hgt: Any
) -> dict[str, dict[str, np.ndarray]]:
    """Stage per-test-row score arrays for every model (baselines + injected HGT).

    The TEST-slice analogue of
    :func:`scifield.forecasting.baselines.evaluate.per_topic_scores`, plus the
    HGT. Each baseline is fit on TRAIN and scored on TEST; the injected ``hgt`` is
    scored via ``hgt.predict`` only. The three label arrays
    (``emergent, emergent_additive, emergent_count_surge``) come straight off the
    TEST slice columns and are identical across every model; the two score arrays
    (``emergence_score, share_forecast``) come from each model's own ``predict``.
    All arrays for a model are the same length and in the same row order as the
    TEST labeled slice, so a later step can pair models on identical
    ``(topic_id, origin_year)`` keys.

    Parameters
    ----------
    features:
        The materialized frame (see :func:`test_metrics_table`).
    cfg:
        The forecasting ``DictConfig``.
    hgt:
        The already-fitted injected HGT (never retrained).

    Returns
    -------
    dict
        Maps each model name (``naive, arima, mlp, no_graph, hgt``) to a dict of
        equal-length, same-order ``np.ndarray`` with EXACTLY the keys
        ``topic_id, origin_year, emergence_score, emergent, emergent_additive,
        emergent_count_surge, share_forecast, forward_share``.
    """
    train = _labeled_slice(features, "train")
    test = _labeled_slice(features, "test")
    train_labels = train[["emergent", "forward_share"]]

    n_test = len(test)
    topic_id = test["topic_id"].to_numpy() if n_test else np.empty(0, dtype=np.int64)
    origin_year = test["origin_year"].to_numpy() if n_test else np.empty(0, dtype=np.int64)
    emergent = test["emergent"].to_numpy() if n_test else np.empty(0)
    emergent_additive = test["emergent_additive"].to_numpy() if n_test else np.empty(0)
    emergent_count_surge = test["emergent_count_surge"].to_numpy() if n_test else np.empty(0)
    forward_share = test["forward_share"].to_numpy() if n_test else np.empty(0)

    def _stage(pred: Any) -> dict[str, np.ndarray]:
        return {
            "topic_id": np.asarray(topic_id),
            "origin_year": np.asarray(origin_year),
            "emergence_score": np.asarray(pred.emergence_score, dtype=float),
            "emergent": np.asarray(emergent),
            "emergent_additive": np.asarray(emergent_additive),
            "emergent_count_surge": np.asarray(emergent_count_surge),
            "share_forecast": np.asarray(pred.share_forecast, dtype=float),
            "forward_share": np.asarray(forward_share, dtype=float),
        }

    _pin_torch_single_threaded()

    out: dict[str, dict[str, np.ndarray]] = {}
    for predictor in _build_predictors(cfg):
        predictor.fit(train, train_labels)
        out[str(predictor.name)] = _stage(predictor.predict(test))

    # Injected HGT — scored, never fit.
    out["hgt"] = _stage(hgt.predict(test))
    return out


def _aligned_diff(per_unit: dict, model: str, baseline: str, field: str) -> tuple[np.ndarray, int]:
    """Align ``model`` vs ``baseline`` on identical ``(topic_id, origin_year)`` keys.

    Builds a ``key -> row index`` map for each side, intersects the key sets, and
    iterates in DETERMINISTIC sorted-key order (never assuming the two staged
    arrays are positionally aligned). For the ``"brier"`` field the per-row Brier
    score ``(emergence_score - emergent) ** 2`` is formed on each side using the
    SHARED ``emergent`` at that key (and the two sides' ``emergent`` are asserted
    to agree); for any other ``field`` the raw per-side arrays at that key are
    differenced.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(diff, n_pairs)`` where ``diff = value_model - value_baseline`` over the
        aligned keys in sorted order and ``n_pairs`` is the number of aligned keys
        (before any zero-dropping).
    """
    m = per_unit[model]
    b = per_unit[baseline]

    def _key_index(side: dict) -> dict[tuple[int, int], int]:
        tids = np.asarray(side["topic_id"])
        years = np.asarray(side["origin_year"])
        return {(int(t), int(y)): i for i, (t, y) in enumerate(zip(tids, years, strict=True))}

    m_idx = _key_index(m)
    b_idx = _key_index(b)
    keys = sorted(set(m_idx) & set(b_idx))
    n_pairs = len(keys)

    m_emergent = np.asarray(m["emergent"])
    b_emergent = np.asarray(b["emergent"])
    m_field = np.asarray(m["emergence_score"]) if field == "brier" else np.asarray(m[field])
    b_field = np.asarray(b["emergence_score"]) if field == "brier" else np.asarray(b[field])

    diff = np.empty(n_pairs, dtype=np.float64)
    for j, key in enumerate(keys):
        i_m = m_idx[key]
        i_b = b_idx[key]
        if field == "brier":
            y = float(m_emergent[i_m])
            assert float(b_emergent[i_b]) == y, f"emergent disagrees at key {key}"
            diff[j] = (float(m_field[i_m]) - y) ** 2 - (float(b_field[i_b]) - y) ** 2
        else:
            diff[j] = float(m_field[i_m]) - float(b_field[i_b])
    return diff, n_pairs


def _wilcoxon_result(diff: np.ndarray, n_pairs: int, favors_neg: str, favors_pos: str) -> dict:
    """Run the signed-rank test on ``diff`` and package the pre-registered dict.

    Shared core for :func:`paired_brier_wilcoxon` and
    :func:`paired_score_wilcoxon`: the only thing that differs between them is the
    DIRECTION vocabulary, passed as ``favors_neg`` (the label when the median diff
    is ``< 0``) and ``favors_pos`` (the label when it is ``> 0``).

    Edge cases (handled WITHOUT raising): if ``n_pairs < 1`` OR every ``diff`` is
    exactly zero, the test is undefined / degenerate, so ``statistic`` and
    ``pvalue`` are ``nan``, ``median_diff`` is ``0.0`` and ``direction`` is
    ``"tie"``. (We pre-check the all-zero case explicitly because some scipy
    versions return ``(0.0, 1.0)`` rather than raising on an all-zero input; the
    ``scipy.stats.wilcoxon`` call is additionally wrapped in ``try/except
    ValueError -> nan`` as a belt-and-suspenders for other degenerate inputs.)

    Returns
    -------
    dict
        ``{"statistic": float, "pvalue": float, "n_pairs": int,
        "median_diff": float, "direction": str}``.
    """
    if n_pairs < 1 or bool(np.all(diff == 0.0)):
        return {
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "n_pairs": int(n_pairs),
            "median_diff": 0.0,
            "direction": "tie",
        }

    median_diff = float(np.median(diff))
    try:
        res = scipy.stats.wilcoxon(diff)
        statistic = float(res.statistic)
        pvalue = float(res.pvalue)
    except ValueError:
        statistic = float("nan")
        pvalue = float("nan")

    if median_diff < 0:
        direction = favors_neg
    elif median_diff > 0:
        direction = favors_pos
    else:
        direction = "tie"

    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "n_pairs": int(n_pairs),
        "median_diff": median_diff,
        "direction": direction,
    }


def paired_brier_wilcoxon(per_unit: dict, model: str = "hgt", baseline: str = "no_graph") -> dict:
    """Per-row Brier-loss Wilcoxon signed-rank test, ``model`` vs ``baseline`` (H2).

    Aligns ``model`` and ``baseline`` on identical ``(topic_id, origin_year)`` test
    rows (sorted-key order, NOT positional), forms each side's per-row Brier score
    ``(emergence_score - emergent) ** 2`` against the shared ``emergent`` label,
    and runs ``scipy.stats.wilcoxon`` on ``diff = brier_model - brier_baseline``. A
    negative median diff means ``model`` carries the LOWER Brier loss (it is the
    better-calibrated forecaster), so ``direction`` is ``"favors_model"``.

    Parameters
    ----------
    per_unit:
        The :func:`test_per_unit_scores` mapping (must carry ``model`` and
        ``baseline``).
    model, baseline:
        The two model names to compare. Default ``"hgt"`` vs ``"no_graph"`` (the
        pre-registered H2 comparison).

    Returns
    -------
    dict
        ``{"statistic": float, "pvalue": float, "n_pairs": int,
        "median_diff": float, "direction": str}`` where ``n_pairs`` is the number
        of aligned keys (before zero-dropping), ``median_diff`` is
        ``float(np.median(diff))``, and ``direction`` is ``"favors_model"`` if
        ``median_diff < 0``, ``"favors_baseline"`` if ``> 0``, else ``"tie"``.
        Degenerate inputs (no pairs / all-zero diff) yield ``nan`` stats and
        ``"tie"`` (see :func:`_wilcoxon_result`).
    """
    diff, n_pairs = _aligned_diff(per_unit, model, baseline, "brier")
    return _wilcoxon_result(diff, n_pairs, favors_neg="favors_model", favors_pos="favors_baseline")


def paired_score_wilcoxon(per_unit: dict, model: str = "hgt", baseline: str = "no_graph") -> dict:
    """Per-row raw-score Wilcoxon signed-rank test, ``model`` vs ``baseline``.

    The secondary robustness line for H2: identical ``(topic_id, origin_year)``
    alignment as :func:`paired_brier_wilcoxon`, but on the RAW emergence scores —
    ``diff = emergence_score_model - emergence_score_baseline`` — rather than the
    Brier loss. A positive median diff means ``model`` assigns systematically
    higher emergence scores, so ``direction`` is ``"model_higher"``.

    Parameters
    ----------
    per_unit:
        The :func:`test_per_unit_scores` mapping.
    model, baseline:
        The two model names to compare. Default ``"hgt"`` vs ``"no_graph"``.

    Returns
    -------
    dict
        Same shape as :func:`paired_brier_wilcoxon`, but ``direction`` is
        ``"model_higher"`` if ``median_diff > 0``, ``"baseline_higher"`` if
        ``< 0``, else ``"tie"``.
    """
    diff, n_pairs = _aligned_diff(per_unit, model, baseline, "emergence_score")
    return _wilcoxon_result(diff, n_pairs, favors_neg="baseline_higher", favors_pos="model_higher")


def calibration_table(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width reliability-diagram table over ``[0, 1]`` (always ``n_bins`` rows).

    Bins the (clipped-to-``[0, 1]``) emergence ``scores`` into ``n_bins`` equal-width
    bins and, per bin, reports the count, the mean predicted score, and the observed
    positive frequency of ``labels``. A score exactly equal to ``1.0`` is assigned
    to the LAST bin (so the right edge is inclusive). Empty bins report ``n=0`` and
    ``nan`` for both means; an empty input returns ``n_bins`` all-empty rows.

    Parameters
    ----------
    scores:
        Predicted emergence scores (probabilities; clipped into ``[0, 1]`` for
        binning robustness).
    labels:
        Binary ground-truth labels, aligned 1:1 with ``scores``.
    n_bins:
        Number of equal-width bins over ``[0, 1]``. Default 10.

    Returns
    -------
    pandas.DataFrame
        Exactly ``n_bins`` rows with the LOCKED columns ``bin_index, bin_lo,
        bin_hi, n, mean_predicted, observed_freq`` (in that order).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    if scores.shape[0]:
        clipped = np.clip(scores, 0.0, 1.0)
        # Right-open bins [lo, hi) via searchsorted; clamp into [0, n_bins-1] so a
        # score of exactly 1.0 falls in the last bin (inclusive right edge).
        idx = np.searchsorted(edges, clipped, side="right") - 1
        idx = np.clip(idx, 0, n_bins - 1)
    else:
        clipped = scores
        idx = np.empty(0, dtype=np.int64)

    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        in_bin = idx == b
        n = int(in_bin.sum())
        if n:
            mean_predicted = float(np.mean(clipped[in_bin]))
            observed_freq = float(np.mean(labels[in_bin]))
        else:
            mean_predicted = float("nan")
            observed_freq = float("nan")
        rows.append(
            {
                "bin_index": int(b),
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "n": n,
                "mean_predicted": mean_predicted,
                "observed_freq": observed_freq,
            }
        )
    return pd.DataFrame(rows, columns=list(_CALIBRATION_COLUMNS))


def gate_g4_verdict(
    metrics: pd.DataFrame,
    wilcoxon_res: dict,
    *,
    comparator: str = "no_graph",
    margin_pp: float = 5.0,
    alpha: float = 0.05,
) -> dict:
    """Mechanically adjudicate Gate G4 (H1 AUC margin AND H2 significance).

    Reads the HGT and ``comparator`` test emergence-AUCs off ``metrics`` and the
    paired-test p-value off ``wilcoxon_res`` (typically
    :func:`paired_brier_wilcoxon`'s output), and ANDs the two pre-registered
    hypotheses. It NEVER favours the HGT: a missing margin or a non-significant
    p-value flips the recommendation to a reported null finding.

    * ``h1_pass`` = both AUCs are non-nan AND ``delta_pp > margin_pp`` (where
      ``delta_pp = (auc_hgt - auc_comparator) * 100``).
    * ``h2_pass`` = the p-value is non-nan AND ``< alpha``.
    * ``overall_pass`` = ``h1_pass and h2_pass``.

    Parameters
    ----------
    metrics:
        A :func:`test_metrics_table`-shaped frame with at least the ``model`` and
        ``emergence_auc`` columns and rows for ``"hgt"`` and ``comparator``.
    wilcoxon_res:
        The paired-test result dict (uses its ``"pvalue"`` and ``"direction"``).
    comparator:
        The baseline the HGT must beat. Default ``"no_graph"``.
    margin_pp:
        The H1 AUC margin in percentage points. Default 5.0.
    alpha:
        The H2 significance level. Default 0.05.

    Returns
    -------
    dict
        EXACTLY ``{"auc_hgt", "auc_comparator", "comparator", "delta_pp",
        "margin_pp", "h1_pass", "pvalue", "alpha", "h2_pass", "h2_direction",
        "overall_pass", "mechanical_recommendation"}``. ``mechanical_recommendation``
        is ``"PASS (F3 success)"`` when ``overall_pass`` else
        ``"NULL FINDING (F3 reported as null)"``.
    """
    auc_hgt = float(metrics.loc[metrics["model"] == "hgt", "emergence_auc"].iloc[0])
    auc_comparator = float(metrics.loc[metrics["model"] == comparator, "emergence_auc"].iloc[0])

    delta_pp = (auc_hgt - auc_comparator) * 100.0
    aucs_ok = not (math.isnan(auc_hgt) or math.isnan(auc_comparator))
    h1_pass = bool(aucs_ok and delta_pp > margin_pp)

    pvalue = float(wilcoxon_res.get("pvalue", float("nan")))
    h2_pass = bool((not math.isnan(pvalue)) and pvalue < alpha)

    overall_pass = bool(h1_pass and h2_pass)

    return {
        "auc_hgt": auc_hgt,
        "auc_comparator": auc_comparator,
        "comparator": str(comparator),
        "delta_pp": float(delta_pp),
        "margin_pp": float(margin_pp),
        "h1_pass": h1_pass,
        "pvalue": pvalue,
        "alpha": float(alpha),
        "h2_pass": h2_pass,
        "h2_direction": str(wilcoxon_res.get("direction", "")),
        "overall_pass": overall_pass,
        "mechanical_recommendation": (
            "PASS (F3 success)" if overall_pass else "NULL FINDING (F3 reported as null)"
        ),
    }


def test_sensitivity_table(per_unit: dict, *, model: str = "hgt") -> pd.DataFrame:
    """Re-score one model's STORED test scores against the three label variants.

    A pure re-scoring (no re-fit, no re-predict): the SAME stored
    ``per_unit[model]["emergence_score"]`` is scored with :func:`emergence_auc`
    against each of the three pre-registered emergence labels already staged in
    ``per_unit[model]`` — the primary ``emergent``, the additive-jump
    ``emergent_additive``, and the count-surge ``emergent_count_surge``. This shows
    whether the AUC conclusion is robust to the emergence-label definition.

    Parameters
    ----------
    per_unit:
        The :func:`test_per_unit_scores` mapping (must carry ``model`` with the
        ``emergence_score`` and three label arrays).
    model:
        Which model's stored scores to re-score. Default ``"hgt"``.

    Returns
    -------
    pandas.DataFrame
        Exactly three rows (``primary, additive_jump, count_surge``) with the
        LOCKED columns ``variant, label_column, emergence_auc, n_test, n_test_pos,
        delta_vs_primary_pp`` (in that order). ``delta_vs_primary_pp`` is
        ``(auc_variant - auc_primary) * 100`` (``0.0`` for the primary row, and
        ``nan``-safe if either AUC is undefined).
    """
    scores = np.asarray(per_unit[model]["emergence_score"], dtype=np.float64)
    n_test = int(scores.shape[0])

    primary_auc = float(emergence_auc(scores, np.asarray(per_unit[model]["emergent"])))

    rows: list[dict[str, Any]] = []
    for variant, label_column in _SENSITIVITY_VARIANTS:
        labels = np.asarray(per_unit[model][label_column])
        auc = float(emergence_auc(scores, labels))
        n_test_pos = int((labels == 1).sum())
        if variant == "primary":
            delta = 0.0
        elif math.isnan(auc) or math.isnan(primary_auc):
            delta = float("nan")
        else:
            delta = (auc - primary_auc) * 100.0
        rows.append(
            {
                "variant": variant,
                "label_column": label_column,
                "emergence_auc": auc,
                "n_test": n_test,
                "n_test_pos": n_test_pos,
                "delta_vs_primary_pp": float(delta),
            }
        )
    return pd.DataFrame(rows, columns=list(_SENSITIVITY_COLUMNS))
