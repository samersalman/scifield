"""V1-S12 topic-level forecasting features, emergence labels, and the temporal
train/val/test split — the leakage-safe data layer the baselines (and the future
GNN) consume.

Sample granularity
------------------
One row per **(topic_id, origin_year = t)**, leaf topics only
(``topic_id != noise_topic_id``). Each row's *features* describe the topic's
history at or before ``t`` and its *label* describes what happens just after:

* **Features** use only years ``<= t`` (strictly trailing). The trailing 3-yr
  window is ``[t-2, t]`` inclusive.
* **Labels** use only years ``{t+1, t+2, t+3}`` (strictly leading; ``horizon=3``).

That asymmetry is the whole point: a model trained on origin year ``t`` can never
see a future it is being asked to predict. :func:`assert_no_leakage` enforces it.

Share denominator (leaf-only)
-----------------------------
For a year ``y``, ``N(y)`` counts papers whose ``topic_id != noise_topic_id``;
``n(g, y)`` counts papers in topic ``g``; ``share(g, y) = n / N``. Noise papers
(``topic_id == -1``, ~24% of the corpus) are excluded from BOTH numerator and
denominator, so ``sum_g share(g, y) == 1`` for every year (leakage assertion #7).
Noise volume is never silently dropped — :func:`materialize` records
``n_noise_total`` / ``noise_frac`` in its ``info`` dict.

Emergence label (primary = multiplicative share growth)
------------------------------------------------------
A ``(topic, t)`` sample is **emergent** iff
``forward_3yr_mean_share >= gamma * trailing_3yr_mean_share`` (``gamma=1.5``),
AND it clears the volume guard ``trailing_3yr_volume >= v_min`` (``v_min=30``).
The volume guard and the "the forward window must fit inside the corpus"
completeness guard are carried as the boolean columns ``volume_ok`` /
``label_complete``; the **labeled set** is the rows where both are True. Rows that
fail ``volume_ok`` are recorded (counted in ``info``), never silently dropped.
``gamma`` and ``v_min`` are Samer's scientific call (config-tunable); two
sensitivity variants — additive jump and count surge — are computed alongside the
primary so the pre-registration can report them.

NaN handling
------------
The paper-level novelty columns (``sem_nov_mean``, ``cd5``, ``cd10``,
``cited_by_pctile_within_year``) carry NaNs (~3-6% missing). Every trailing
novelty aggregation skips NaN (pandas ``.mean()`` default), so a topic-year with
some missing novelty still yields a mean over its observed papers; a window with
*no* observed value yields NaN for that feature (left for the downstream scaler /
model to impute, never silently zero-filled).

Feature schema
--------------
``MLP_FEATURES`` / ``NODE_FEATURES`` are kept byte-identical to
``cfg.features.mlp_columns`` / ``cfg.features.node_columns`` in
``conf/forecasting/v1.yaml``; ``MLP_FEATURES == NODE_FEATURES[:9]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = [
    "MLP_FEATURES",
    "NODE_FEATURES",
    "assert_no_leakage",
    "assign_split",
    "build_features",
    "build_labels",
    "compute_yearly_topic_counts",
    "materialize",
]

#: LOCKED — the 9 "block A" structural/temporal features (count/share dynamics),
#: byte-identical to ``cfg.features.mlp_columns`` and in the same order.
MLP_FEATURES: tuple[str, ...] = (
    "count_3yr",
    "share_3yr_mean",
    "share_last",
    "share_growth_3yr",
    "share_momentum",
    "share_accel",
    "share_volatility_3yr",
    "topic_age",
    "share_of_max",
)

#: LOCKED — the 16 GNN node features = ``MLP_FEATURES`` (block A, first, in order)
#: followed by the 7 trailing novelty / epistemic / journal features (block B).
#: Byte-identical to ``cfg.features.node_columns``.
NODE_FEATURES: tuple[str, ...] = (
    *MLP_FEATURES,
    "sem_nov_mean_3yr",
    "cd5_3yr",
    "cd10_3yr",
    "cited_by_pctile_3yr",
    "rct_share_3yr",
    "review_share_3yr",
    "n_journals_3yr",
)

#: The four split-name labels :func:`assign_split` emits.
_SPLIT_NAMES = ("train", "val", "test", "excluded")


# --------------------------------------------------------------------------- #
# Pure numeric core (pandas in, pandas out — unit-testable without I/O)
# --------------------------------------------------------------------------- #


def compute_yearly_topic_counts(papers: pd.DataFrame, noise_topic_id: int = -1) -> pd.DataFrame:
    """Aggregate paper rows into per-(topic, year) leaf-only counts and shares.

    Noise papers (``topic_id == noise_topic_id``) are dropped before counting, so
    the year denominator ``N(y)`` is the count of *leaf* papers in year ``y`` and
    ``sum_g share(g, y) == 1`` for every year present.

    Parameters
    ----------
    papers:
        Paper-level frame with at least ``topic_id`` and ``year`` (one row per
        paper). Extra columns are ignored.
    noise_topic_id:
        Topic id flagging noise papers to exclude. Default ``-1``.

    Returns
    -------
    pandas.DataFrame
        Columns exactly ``topic_id, year, n, N, share`` (one row per
        leaf-topic-year), sorted by ``(topic_id, year)``. ``N`` and ``share`` are
        leaf-only. Empty input yields an empty frame with those columns.
    """
    cols = ["topic_id", "year", "n", "N", "share"]
    leaf = papers.loc[papers["topic_id"] != noise_topic_id, ["topic_id", "year"]]
    if leaf.empty:
        return pd.DataFrame({c: pd.Series(dtype="int64") for c in cols})

    counts = leaf.groupby(["topic_id", "year"], as_index=False).size().rename(columns={"size": "n"})
    yearly_total = counts.groupby("year", as_index=False)["n"].sum().rename(columns={"n": "N"})
    counts = counts.merge(yearly_total, on="year", how="left")
    counts["share"] = counts["n"] / counts["N"]
    counts = counts.sort_values(["topic_id", "year"]).reset_index(drop=True)
    return counts[cols]


def _trailing_window_share_stats(
    series_by_year: dict[int, float], t: int, window: int
) -> dict[str, float]:
    """Trailing share statistics for origin year ``t`` over ``[t-window+1, t]``.

    ``series_by_year`` maps year -> share for one topic; missing years are 0
    share (the topic simply had no papers that year). Returns the trailing share
    mean / std and the point-in-time momentum/acceleration/last-value features.
    """
    win_years = [t - i for i in range(window)]
    win_shares = np.array([series_by_year.get(y, 0.0) for y in win_years], dtype="float64")

    share_t = series_by_year.get(t, 0.0)
    share_tm1 = series_by_year.get(t - 1, 0.0)
    share_tm2 = series_by_year.get(t - 2, 0.0)

    return {
        "share_3yr_mean": float(win_shares.mean()),
        "share_last": float(share_t),
        "share_momentum": float(share_t - share_tm1),
        "share_accel": float((share_t - share_tm1) - (share_tm1 - share_tm2)),
        # Population std (ddof=0): deterministic, defined for a single point (0).
        "share_volatility_3yr": float(win_shares.std(ddof=0)),
    }


def build_features(
    counts: pd.DataFrame,
    novelty: pd.DataFrame,
    epistemic: pd.DataFrame,
    trailing_window: int = 3,
) -> pd.DataFrame:
    """Build the strictly-trailing per-(topic, origin_year) feature table.

    For every observed ``(topic_id, t)`` with ``t`` ranging over the years that
    topic appears in ``counts``, every feature is computed from years ``<= t``
    only. The 9 ``MLP_FEATURES`` come from the count/share series; the 7 extra
    ``NODE_FEATURES`` are trailing means/fractions over the paper-level
    ``novelty`` / ``epistemic`` frames (window ``[t-trailing_window+1, t]``,
    skipping NaN).

    Parameters
    ----------
    counts:
        Output of :func:`compute_yearly_topic_counts`
        (``topic_id, year, n, N, share``).
    novelty:
        Paper-level frame ``[pmid, topic_id, year, sem_nov_mean, cd5, cd10,
        cited_by_pctile_within_year]``. NaNs are skipped in trailing means.
    epistemic:
        Paper-level frame ``[pmid, topic_id, year, is_rct, is_review, journal]``.
    trailing_window:
        Width of the trailing window (default 3 ⇒ ``[t-2, t]``).

    Returns
    -------
    pandas.DataFrame
        Columns ``topic_id, origin_year`` + every name in :data:`NODE_FEATURES`
        + ``trailing_3yr_volume`` (sum of ``n`` over the trailing window; the
        ``v_min`` guard input). One row per observed ``(topic_id, origin_year)``,
        sorted by those keys.
    """
    out_cols = ["topic_id", "origin_year", *NODE_FEATURES, "trailing_3yr_volume"]
    if counts.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in out_cols})

    delta = 0.002  # growth-ratio / share-of-max denominator guard (matches label δ)

    # Pre-index the paper-level frames by (topic, year) for cheap window slices.
    nov = novelty.copy()
    epi = epistemic.copy()

    rows: list[dict[str, Any]] = []
    for topic_id, grp in counts.groupby("topic_id", sort=True):
        grp = grp.sort_values("year")
        share_by_year = dict(zip(grp["year"].to_numpy(), grp["share"].to_numpy(), strict=True))
        n_by_year = dict(zip(grp["year"].to_numpy(), grp["n"].to_numpy(), strict=True))
        first_active_year = int(grp.loc[grp["n"] > 0, "year"].min())
        years = [int(y) for y in grp["year"].to_numpy()]

        nov_topic = nov[nov["topic_id"] == topic_id]
        epi_topic = epi[epi["topic_id"] == topic_id]

        for t in years:
            win_years = [t - i for i in range(trailing_window)]
            win_lo = t - trailing_window + 1

            count_3yr = float(sum(n_by_year.get(y, 0) for y in win_years))

            # share_growth_3yr: trailing-window mean share vs the PRIOR window
            # mean share ([t-2*w+1 .. t-w]); δ guards the ratio denominator.
            prior_years = [t - trailing_window - i for i in range(trailing_window)]
            cur_mean = float(np.mean([share_by_year.get(y, 0.0) for y in win_years]))
            prior_mean = float(np.mean([share_by_year.get(y, 0.0) for y in prior_years]))
            share_growth_3yr = (cur_mean + delta) / (prior_mean + delta)

            # share_of_max: current share vs the topic's best share so far (≤ t).
            past_shares = [share_by_year.get(y, 0.0) for y in years if y <= t]
            max_share = max(past_shares) if past_shares else 0.0
            share_t = share_by_year.get(t, 0.0)
            share_of_max = (share_t + delta) / (max_share + delta)

            stats = _trailing_window_share_stats(share_by_year, t, trailing_window)

            # Trailing paper windows (skipna for novelty; fractions over papers).
            nov_win = nov_topic[(nov_topic["year"] >= win_lo) & (nov_topic["year"] <= t)]
            epi_win = epi_topic[(epi_topic["year"] >= win_lo) & (epi_topic["year"] <= t)]

            rows.append(
                {
                    "topic_id": int(topic_id),
                    "origin_year": t,
                    "count_3yr": count_3yr,
                    "share_3yr_mean": stats["share_3yr_mean"],
                    "share_last": stats["share_last"],
                    "share_growth_3yr": share_growth_3yr,
                    "share_momentum": stats["share_momentum"],
                    "share_accel": stats["share_accel"],
                    "share_volatility_3yr": stats["share_volatility_3yr"],
                    "topic_age": float(t - first_active_year),
                    "share_of_max": share_of_max,
                    "sem_nov_mean_3yr": _safe_mean(nov_win, "sem_nov_mean"),
                    "cd5_3yr": _safe_mean(nov_win, "cd5"),
                    "cd10_3yr": _safe_mean(nov_win, "cd10"),
                    "cited_by_pctile_3yr": _safe_mean(nov_win, "cited_by_pctile_within_year"),
                    "rct_share_3yr": _safe_frac(epi_win, "is_rct"),
                    "review_share_3yr": _safe_frac(epi_win, "is_review"),
                    "n_journals_3yr": (
                        float(epi_win["journal"].nunique()) if "journal" in epi_win.columns else 0.0
                    ),
                    "trailing_3yr_volume": count_3yr,
                }
            )

    features = pd.DataFrame(rows, columns=out_cols)
    features = features.sort_values(["topic_id", "origin_year"]).reset_index(drop=True)
    return features


def _safe_mean(window: pd.DataFrame, col: str) -> float:
    """Mean of ``window[col]`` skipping NaN; NaN if the column is empty/all-NaN."""
    if col not in window.columns or window.empty:
        return float("nan")
    return float(window[col].mean())  # pandas .mean() skips NaN by default


def _safe_frac(window: pd.DataFrame, col: str) -> float:
    """Fraction of True in a boolean ``window[col]``; NaN if window is empty."""
    if col not in window.columns or window.empty:
        return float("nan")
    return float(window[col].astype("float64").mean())


def build_labels(
    counts: pd.DataFrame,
    horizon: int = 3,
    gamma: float = 1.5,
    v_min: int = 30,
    mode: str = "multiplicative",
    delta: float = 0.002,
) -> pd.DataFrame:
    """Build the strictly-leading per-(topic, origin_year) emergence labels.

    For each ``(topic_id, t)`` the trailing aggregates cover ``[t-horizon+1, t]``
    and the forward aggregates cover ``{t+1, .., t+horizon}``. The primary label
    ``emergent`` is multiplicative share growth; two sensitivity variants
    (additive jump, count surge) are computed alongside.

    Parameters
    ----------
    counts:
        Output of :func:`compute_yearly_topic_counts`.
    horizon:
        Forward window width (default 3 ⇒ ``{t+1, t+2, t+3}``).
    gamma:
        Multiplicative growth threshold for the primary / count-surge labels.
    v_min:
        Minimum trailing volume for ``volume_ok``.
    mode:
        Label mode; only ``"multiplicative"`` is wired (the contract's primary).
        Other values still return the frame (primary stays multiplicative) so the
        config is the single source of truth.
    delta:
        Additive-jump threshold for ``emergent_additive``.

    Returns
    -------
    pandas.DataFrame
        Columns ``topic_id, origin_year`` + ``trailing_3yr_mean_share,
        forward_3yr_mean_share, forward_share, trailing_3yr_volume, volume_ok,
        label_complete, emergent, emergent_additive, emergent_count_surge``.
        ``forward_share == forward_3yr_mean_share`` (the MAPE target). The
        **labeled set** is rows with ``volume_ok & label_complete``; ``emergent``
        is computed for all rows but only consumed on that subset.
    """
    out_cols = [
        "topic_id",
        "origin_year",
        "trailing_3yr_mean_share",
        "forward_3yr_mean_share",
        "forward_share",
        "trailing_3yr_volume",
        "volume_ok",
        "label_complete",
        "emergent",
        "emergent_additive",
        "emergent_count_surge",
    ]
    if counts.empty:
        empty = {c: pd.Series(dtype="float64") for c in out_cols}
        empty["volume_ok"] = pd.Series(dtype="bool")
        empty["label_complete"] = pd.Series(dtype="bool")
        for c in ("emergent", "emergent_additive", "emergent_count_surge"):
            empty[c] = pd.Series(dtype="int64")
        return pd.DataFrame(empty)

    max_year = int(counts["year"].max())

    rows: list[dict[str, Any]] = []
    for topic_id, grp in counts.groupby("topic_id", sort=True):
        grp = grp.sort_values("year")
        share_by_year = dict(zip(grp["year"].to_numpy(), grp["share"].to_numpy(), strict=True))
        n_by_year = dict(zip(grp["year"].to_numpy(), grp["n"].to_numpy(), strict=True))
        years = [int(y) for y in grp["year"].to_numpy()]

        for t in years:
            trail_years = [t - i for i in range(horizon)]
            fwd_years = [t + 1 + i for i in range(horizon)]

            trailing_mean = float(np.mean([share_by_year.get(y, 0.0) for y in trail_years]))
            forward_mean = float(np.mean([share_by_year.get(y, 0.0) for y in fwd_years]))
            trailing_vol = float(sum(n_by_year.get(y, 0) for y in trail_years))
            forward_vol = float(sum(n_by_year.get(y, 0) for y in fwd_years))

            volume_ok = trailing_vol >= v_min
            label_complete = (t + horizon) <= max_year

            emergent = int(forward_mean >= gamma * trailing_mean)
            emergent_additive = int((forward_mean - trailing_mean) >= delta)
            emergent_count_surge = int(forward_vol >= gamma * trailing_vol)

            rows.append(
                {
                    "topic_id": int(topic_id),
                    "origin_year": t,
                    "trailing_3yr_mean_share": trailing_mean,
                    "forward_3yr_mean_share": forward_mean,
                    "forward_share": forward_mean,
                    "trailing_3yr_volume": trailing_vol,
                    "volume_ok": bool(volume_ok),
                    "label_complete": bool(label_complete),
                    "emergent": emergent,
                    "emergent_additive": emergent_additive,
                    "emergent_count_surge": emergent_count_surge,
                }
            )

    labels = pd.DataFrame(rows, columns=out_cols)
    labels = labels.sort_values(["topic_id", "origin_year"]).reset_index(drop=True)
    return labels


def assign_split(origin_year: Any, split_cfg: Any) -> pd.Series | str:
    """Map forecast-origin year(s) onto ``{train, val, test, excluded}``.

    Uses the inclusive ``[lo, hi]`` ranges in ``split_cfg`` for ``train`` /
    ``val`` / ``test``. Years outside every range map to ``"excluded"``. Ranges
    are checked train→val→test (they are required to be disjoint by
    :func:`assert_no_leakage`, so order does not change the result).

    Parameters
    ----------
    origin_year:
        A scalar int (returns ``str``) OR an array / Series of ints (returns a
        ``pd.Series`` aligned to the input).
    split_cfg:
        Mapping / DictConfig with ``train``, ``val``, ``test`` keys, each a
        2-element ``[lo, hi]`` sequence. Accessed via attribute or item lookup.

    Returns
    -------
    pandas.Series | str
        The split name(s).
    """
    bounds = {name: _get_range(split_cfg, name) for name in ("train", "val", "test")}

    def _one(year: int) -> str:
        for name in ("train", "val", "test"):
            lo, hi = bounds[name]
            if lo <= year <= hi:
                return name
        return "excluded"

    if np.isscalar(origin_year):
        return _one(int(origin_year))

    values = pd.Series(origin_year)
    out = values.astype("int64").map(_one)
    out.index = values.index
    return out


def _get_range(split_cfg: Any, name: str) -> tuple[int, int]:
    """Read a ``[lo, hi]`` 2-tuple from ``split_cfg`` by attribute or item."""
    rng = getattr(split_cfg, name, None)
    if rng is None:
        rng = split_cfg[name]
    return int(rng[0]), int(rng[1])


def assert_no_leakage(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split: pd.Series,
    *,
    horizon: int,
    allowed_origins: set[int],
) -> None:
    """Assert the seven leakage invariants; raise ``AssertionError`` on any.

    Each failure carries a SHORT keyword in the message so tests/callers can
    assert on it. The seven checks (keywords in parentheses):

    1. Every ``origin_year`` present is in ``allowed_origins``
       (``leakage:allowed_origins``).
    2. ``split`` values ⊆ the four split names AND no row is ``"test"``
       (``leakage:test_present``).
    3. The origin-year SETS of the splits are pairwise disjoint — no single
       origin year is labeled both ``train`` and ``val`` (or val/test, etc.)
       (``leakage:disjoint``).
    4. Boundary ordering holds — every ``train`` origin precedes every ``val``
       origin, which precedes every ``test`` origin:
       ``max(train) < min(val) < min(test)`` (``leakage:boundary``).
    5. Every volume-qualified row (``volume_ok``) is also ``label_complete`` —
       i.e. no row that clears the volume guard is allowed into the materialized
       frame with a forward window running past the corpus
       (``leakage:label_complete``). In ``materialize`` this holds because every
       kept origin year is ≤ ``corpus_max - horizon``; a test trips it by passing
       a ``volume_ok`` row with ``label_complete=False``.
    6. ``features`` and ``labels`` share the same ``(topic_id, origin_year)``
       keys — no orphan future rows (``leakage:key_align``).
    7. Per-year leaf shares sum to ~1, if ``labels`` carries the share columns
       needed to reconstruct them (``leakage:share_sum``).

    Parameters
    ----------
    features, labels:
        The (topic_id, origin_year)-keyed feature / label frames. Must align 1:1
        on those keys.
    split:
        A ``pd.Series`` of split names aligned row-for-row with ``features``.
    horizon:
        Forecast horizon (used only for the docstringed completeness semantics;
        the completeness flag itself lives in ``labels``).
    allowed_origins:
        The set of origin years the materialized frame is permitted to contain
        (e.g. train ∪ val years; never test years when ``allow_test=False``).
    """
    # (1) origins ⊆ allowed
    present = set(int(y) for y in features["origin_year"].unique())
    bad = present - set(int(y) for y in allowed_origins)
    assert not bad, f"leakage:allowed_origins origins {sorted(bad)} outside allowed set"

    # (2) split names valid + no test rows materialized
    split_values = set(split.unique())
    assert split_values <= set(
        _SPLIT_NAMES
    ), f"leakage:test_present unknown split labels {split_values - set(_SPLIT_NAMES)}"
    assert (
        "test" not in split_values
    ), "leakage:test_present test rows present in materialized frame"

    # (3) origin-year SETS per split are pairwise disjoint (no year is in two
    #     splits — that would only happen if assign_split were inconsistent).
    aligned = features.assign(_split=split.to_numpy())
    year_sets: dict[str, set[int]] = {}
    for name in ("train", "val", "test"):
        yrs = aligned.loc[aligned["_split"] == name, "origin_year"].to_numpy()
        year_sets[name] = {int(y) for y in yrs}
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for a, b in pairs:
        overlap = year_sets[a] & year_sets[b]
        assert not overlap, f"leakage:disjoint origin year(s) {sorted(overlap)} in both {a} and {b}"

    # (4) boundary ordering: max(train) < min(val) < min(test) for whichever
    #     splits are present (test usually dropped before this call).
    present_splits = [(n, year_sets[n]) for n in ("train", "val", "test") if year_sets[n]]
    for (na, sa), (nb, sb) in zip(present_splits, present_splits[1:], strict=False):
        assert max(sa) < min(
            sb
        ), f"leakage:boundary {na} origin {max(sa)} not strictly before {nb} origin {min(sb)}"

    # (5) every volume-qualified row is also complete (no incomplete forward
    #     window survives into the materialized frame).
    incomplete_volume = labels[labels["volume_ok"] & ~labels["label_complete"]]
    assert incomplete_volume.empty, (
        f"leakage:label_complete {len(incomplete_volume)} volume-ok row(s) with "
        "incomplete forward window"
    )

    # (6) feature/label keys align exactly
    fkeys = set(map(tuple, features[["topic_id", "origin_year"]].to_numpy().tolist()))
    lkeys = set(map(tuple, labels[["topic_id", "origin_year"]].to_numpy().tolist()))
    assert fkeys == lkeys, "leakage:key_align features and labels have mismatched keys"

    # (7) per-year leaf shares sum to ~1 (only checkable if shares present)
    if {"forward_share", "origin_year"}.issubset(labels.columns) and "share" in labels.columns:
        sums = labels.groupby("origin_year")["share"].sum()
        assert np.allclose(
            sums.to_numpy(), 1.0, atol=1e-6
        ), "leakage:share_sum per-year leaf shares do not sum to 1"


def check_share_sum(counts: pd.DataFrame, atol: float = 1e-6) -> None:
    """Assert leaf shares sum to ~1 per year on a ``counts`` frame.

    Companion to :func:`assert_no_leakage` check #7 — :func:`materialize` calls
    it directly on the counts frame (where the per-year shares actually live).
    Raises ``AssertionError`` with ``leakage:share_sum`` on violation.
    """
    if counts.empty:
        return
    sums = counts.groupby("year")["share"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=atol), (
        f"leakage:share_sum per-year leaf shares do not sum to 1 (max dev "
        f"{float(np.abs(sums.to_numpy() - 1.0).max()):.2e})"
    )


# --------------------------------------------------------------------------- #
# I/O entry point (real parquet + DuckDB; the only function that touches disk)
# --------------------------------------------------------------------------- #


def _merge_features_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Inner-join features + labels on ``(topic_id, origin_year)``, deduped.

    ``trailing_3yr_volume`` is a LOCKED output column of BOTH :func:`build_features`
    and :func:`build_labels`. A plain merge would emit ``trailing_3yr_volume_x`` /
    ``trailing_3yr_volume_y`` and leave no clean ``trailing_3yr_volume`` — breaking
    the §4 output contract. We drop the features-side copy before merging so the
    result carries exactly one ``trailing_3yr_volume``: the LABELS-side ``[t-2, t]``
    value the ``v_min`` guard (``volume_ok``) is derived from.
    """
    feats = features.drop(columns=["trailing_3yr_volume"])
    return feats.merge(labels, on=["topic_id", "origin_year"], how="inner")


def materialize(cfg: DictConfig, *, allow_test: bool = False) -> tuple[pd.DataFrame, dict]:
    """Read the corpus, build features + labels, split, and seal the test set.

    Reads ``cfg.input.archetypes_parquet`` (paper-level novelty + journal + year
    + topic_id) and ``cfg.input.duckdb_path`` ``papers.publication_types`` (deduped
    on pmid, ``CAST(pmid AS BIGINT)``). Derives ``is_rct`` (publication_types
    contains ``"Randomized Controlled Trial"``) and ``is_review`` (contains
    ``"Review"``). Builds counts/features/labels, joins on
    ``(topic_id, origin_year)``, assigns ``split``, runs :func:`assert_no_leakage`
    + :func:`check_share_sum`, and **drops ``split == "test"`` rows unless
    ``allow_test=True``** (S12 never passes True).

    Parameters
    ----------
    cfg:
        The forecasting ``DictConfig`` (``conf/forecasting/v1.yaml``).
    allow_test:
        If False (default + S12 invariant), test-origin rows are dropped and
        never returned; the assertion also fails loudly if any survive.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        ``(df, info)``. ``df`` has one row per ``(topic_id, origin_year)`` with
        ``NODE_FEATURES`` + the label columns + a ``split`` column. ``info``
        carries per-split class balance, exclusion / noise diagnostics, the
        denominator policy, and echoed label thresholds (see HANDOFF NOTES).
    """
    archetypes_path = Path(str(cfg.input.archetypes_parquet))
    duckdb_path = Path(str(cfg.input.duckdb_path))

    noise_topic_id = int(cfg.noise.noise_topic_id)
    horizon = int(cfg.split.horizon)
    trailing_window = int(cfg.split.trailing_window)
    gamma = float(cfg.label.gamma)
    v_min = int(cfg.label.v_min)
    mode = str(cfg.label.mode)
    delta = float(cfg.label.delta)

    # --- Paper-level frame (novelty + journal + year + topic_id). ----------
    papers = pd.read_parquet(
        archetypes_path,
        columns=[
            "pmid",
            "year",
            "journal",
            "topic_id",
            "sem_nov_mean",
            "cd5",
            "cd10",
            "cited_by_pctile_within_year",
        ],
    )
    papers["pmid"] = papers["pmid"].astype(np.int64)
    papers["topic_id"] = papers["topic_id"].astype(np.int64)
    papers["year"] = papers["year"].astype(np.int64)

    # --- publication_types -> is_rct / is_review (dedup pmid, BIGINT cast). -
    pub_types = _load_publication_types(duckdb_path)
    papers = papers.merge(pub_types, on="pmid", how="left")
    papers["is_rct"] = papers["is_rct"].fillna(False).astype(bool)
    papers["is_review"] = papers["is_review"].fillna(False).astype(bool)

    n_noise_total = int((papers["topic_id"] == noise_topic_id).sum())
    noise_frac = float((papers["topic_id"] == noise_topic_id).mean()) if len(papers) else 0.0

    # --- Counts / features / labels (leaf-only denominator). ----------------
    counts = compute_yearly_topic_counts(papers, noise_topic_id=noise_topic_id)
    check_share_sum(counts)

    novelty = papers[
        ["pmid", "topic_id", "year", "sem_nov_mean", "cd5", "cd10", "cited_by_pctile_within_year"]
    ]
    epistemic = papers[["pmid", "topic_id", "year", "is_rct", "is_review", "journal"]]

    features = build_features(counts, novelty, epistemic, trailing_window=trailing_window)
    labels = build_labels(counts, horizon=horizon, gamma=gamma, v_min=v_min, mode=mode, delta=delta)

    # --- Join features + labels on (topic_id, origin_year). -----------------
    # ``trailing_3yr_volume`` is a LOCKED output of BOTH build_features and
    # build_labels, so a plain merge would yield ``trailing_3yr_volume_x/_y`` and
    # no clean column (violating the §4 contract). Drop the features-side copy and
    # keep the LABELS-side value as canonical — it's the ``[t-2, t]`` volume the
    # ``v_min`` guard (``volume_ok``) is computed from.
    df = _merge_features_labels(features, labels)

    # --- Split, then enforce no-leakage, then seal test. --------------------
    split = cast(pd.Series, assign_split(df["origin_year"], cfg.split))
    df["split"] = split.to_numpy()

    train_lo, train_hi = _get_range(cfg.split, "train")
    val_lo, val_hi = _get_range(cfg.split, "val")
    test_lo, test_hi = _get_range(cfg.split, "test")
    allowed_origins: set[int] = set(range(train_lo, train_hi + 1)) | set(range(val_lo, val_hi + 1))
    if allow_test:
        allowed_origins |= set(range(test_lo, test_hi + 1))

    # Keep only origins this run is permitted to see (drop "excluded" + sealed
    # test rows). The post-drop frame is what we assert no-leakage over.
    keep_splits = {"train", "val"} | ({"test"} if allow_test else set())
    kept = df[df["split"].isin(keep_splits)].reset_index(drop=True)

    # The leakage net (which includes "no test rows present") guards the DEFAULT
    # S12 path. ``allow_test=True`` is an explicit, unguarded escape hatch (never
    # used by S12) for V1-S14's final test-set evaluation, so the seal-check is
    # skipped there — the caller has deliberately opted in to test rows.
    if not allow_test:
        assert_no_leakage(
            kept[["topic_id", "origin_year"]].assign(
                **{c: kept[c] for c in NODE_FEATURES if c in kept.columns}
            ),
            kept[
                [
                    "topic_id",
                    "origin_year",
                    "volume_ok",
                    "label_complete",
                    "forward_share",
                    "trailing_3yr_mean_share",
                    "forward_3yr_mean_share",
                ]
            ],
            kept["split"],
            horizon=horizon,
            allowed_origins=allowed_origins,
        )

    df = kept

    # --- info: class balance, exclusion / noise diagnostics, echoed config. -
    info = _build_info(
        df,
        n_noise_total=n_noise_total,
        noise_frac=noise_frac,
        denominator=str(cfg.noise.denominator),
        gamma=gamma,
        v_min=v_min,
        horizon=horizon,
        mode=mode,
        delta=delta,
    )
    return df, info


def _build_info(
    df: pd.DataFrame,
    *,
    n_noise_total: int,
    noise_frac: float,
    denominator: str,
    gamma: float,
    v_min: int,
    horizon: int,
    mode: str,
    delta: float,
) -> dict:
    """Assemble the ``materialize`` ``info`` dict (class balance + diagnostics)."""
    labeled_mask = df["volume_ok"] & df["label_complete"]
    per_split: dict[str, dict[str, float]] = {}
    for name in ("train", "val", "test"):
        sub = df[(df["split"] == name) & labeled_mask]
        n = int(len(sub))
        pos = float(sub["emergent"].mean()) if n else float("nan")
        per_split[name] = {"n": n, "positive_rate": pos}

    n_excluded_volume = int((~df["volume_ok"]).sum())

    return {
        "per_split": per_split,
        "n_excluded_volume": n_excluded_volume,
        "n_noise_total": n_noise_total,
        "noise_frac": noise_frac,
        "denominator": denominator,
        "thresholds": {
            "gamma": gamma,
            "v_min": v_min,
            "horizon": horizon,
            "mode": mode,
            "delta": delta,
        },
        "n_rows": int(len(df)),
        "n_labeled": int(labeled_mask.sum()),
    }


def _load_publication_types(duckdb_path: Path) -> pd.DataFrame:
    """Return a deduped ``pmid, is_rct, is_review`` frame from DuckDB.

    ``papers`` has DUPLICATE pmids, so we dedup on ``pmid`` (keep first) BEFORE
    deriving the flags. ``is_rct`` = ``publication_types`` contains
    ``"Randomized Controlled Trial"``; ``is_review`` = contains ``"Review"``. The
    DuckDB ``pmid`` is ``VARCHAR``; we ``CAST(pmid AS BIGINT)``.
    """
    import duckdb

    query = """
        SELECT
            CAST(pmid AS BIGINT) AS pmid,
            list_contains(publication_types, 'Randomized Controlled Trial') AS is_rct,
            list_contains(publication_types, 'Review') AS is_review
        FROM (
            SELECT pmid, publication_types,
                   row_number() OVER (PARTITION BY pmid ORDER BY pmid) AS rn
            FROM papers
        )
        WHERE rn = 1
    """
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        out = con.execute(query).df()
    finally:
        con.close()
    out["pmid"] = out["pmid"].astype(np.int64)
    out["is_rct"] = out["is_rct"].fillna(False).astype(bool)
    out["is_review"] = out["is_review"].fillna(False).astype(bool)
    return out
