"""V2 cartography — generic rigor primitives (permutation nulls + jackknife).

The V2 charter (``V2/CONTEXT.md`` §0) commits to "exploratory ≠ unrigorous": no
hypothesis gates, but heavy rigor — null/permutation models, drop-one-journal
jackknife, granularity sensitivity. Both the cascade-validation layer (V2-S04) and
the roles & velocity layer (V2-S05) need the *same* primitives, so they live here
once, grain-agnostic and pure, and are imported by both.

What lives here
---------------
* :func:`permute_labels` — build a permutation-null replicate by shuffling a label
  column, either globally or **within** a grouping column (the usual choice: shuffle
  topic→journal *first-appearance* labels within each topic so the marginal volume
  per topic is preserved while any systematic journal lead is destroyed).
* :func:`permutation_test` — turn an observed statistic plus a vector of
  null-replicate statistics into ``{p_value, z, n_perm}`` (two-sided by default).
* :func:`jackknife_drop_one` — leave-one-entity-out iterator that re-runs a caller
  ``compute_fn`` with each entity held out (drop-one-journal robustness).
* :func:`attribution_flip_rate` — fraction of keys whose attributed value *changes*
  across jackknife runs (stability of a per-entity claim).
* :func:`rank_stability` — Kendall-τ-style mean rank agreement of a ranking across
  jackknife runs (stability of an ordering).

Seeding discipline
-------------------
No global RNG state. Every stochastic function takes an explicit integer ``seed``
and constructs its own ``numpy.random.default_rng(seed)``; the same ``seed`` yields
byte-identical replicates, which is what reproducibility sidecars assert.

Conventions: ``from __future__ import annotations``; numpy-style docstrings; numpy +
pandas only; ruff/black line-length 100. Counts are COMPUTED, never hardcoded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

__all__ = [
    "attribution_flip_rate",
    "jackknife_drop_one",
    "permutation_test",
    "permute_labels",
    "rank_stability",
]


def permute_labels(
    df: pd.DataFrame,
    *,
    label_col: str,
    group_col: str | None = None,
    seed: int,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``label_col`` shuffled (null replicate).

    Construct one permutation-null replicate by randomly reassigning the values of
    ``label_col`` among the rows. The shuffle is the *only* change — every other
    column (and the row count, and each group's row count) is preserved — so any
    statistic computed on the replicate reflects the null in which ``label_col`` is
    independent of the rest of the row.

    Parameters
    ----------
    df :
        Source frame. Returned unmodified except for ``label_col``; a copy is made,
        the input is never mutated.
    label_col :
        The column whose values are permuted (e.g. ``"journal_slug"`` when testing
        whether journal identity is associated with leading a topic).
    group_col :
        If ``None`` (default), labels are shuffled **globally** across all rows. If
        given, labels are shuffled **independently within each group** — the multiset
        of labels inside every group is preserved, so group-level marginals (e.g. how
        many journals touch a topic) survive while within-group structure is broken.
        Use within-group when the per-group composition is a fixed feature of the
        design and only the *assignment* is under the null.
    seed :
        Integer seed for ``numpy.random.default_rng``. Deterministic: same seed →
        same replicate.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with a permuted ``label_col``, original index preserved.

    Raises
    ------
    KeyError
        If ``label_col`` (or ``group_col`` when supplied) is not a column of ``df``.

    Notes
    -----
    Within-group shuffling permutes positions *inside* each group, so a singleton
    group's label is unchanged (there is nothing to swap) — expected and harmless.
    """
    import numpy as np

    if label_col not in df.columns:
        raise KeyError(f"label_col {label_col!r} not in df")
    rng = np.random.default_rng(seed)
    out = df.copy()
    values = out[label_col].to_numpy()

    if group_col is None:
        perm = rng.permutation(len(values))
        out[label_col] = values[perm]
        return out

    if group_col not in df.columns:
        raise KeyError(f"group_col {group_col!r} not in df")

    new_values = values.copy()
    # Permute the label array in place within each group (row order preserved).
    for _, idx in out.groupby(group_col, dropna=False, sort=False).indices.items():
        idx = np.asarray(idx)
        if idx.size > 1:
            new_values[idx] = values[idx][rng.permutation(idx.size)]
    out[label_col] = new_values
    return out


def permutation_test(
    observed_stat: float,
    null_stats: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> dict[str, float | int]:
    """Empirical p-value and z-score of an observed statistic against a null sample.

    Parameters
    ----------
    observed_stat :
        The statistic computed on the real (unpermuted) data.
    null_stats :
        1-D array of the same statistic computed on permutation-null replicates
        (e.g. one per :func:`permute_labels` call). ``NaN`` replicates are dropped
        before tallying.
    alternative :
        ``"two-sided"`` (default), ``"greater"``, or ``"less"``. ``"greater"`` tests
        whether the observed value is unusually *large* (right tail), ``"less"``
        unusually *small* (left tail), ``"two-sided"`` either tail via ``|stat|``
        relative to the null mean.

    Returns
    -------
    dict
        ``{"p_value": float, "z": float, "n_perm": int}``.

        * ``p_value`` uses the **add-one** (biased, never-zero) estimator
          ``(1 + #{null as-or-more-extreme}) / (1 + n_perm)`` so a finite sample can
          never report exactly 0 — the conventional Monte-Carlo p-value.
        * ``z`` is ``(observed - mean(null)) / std(null)`` (population std,
          ``ddof=0``); ``NaN`` if the null has zero variance.
        * ``n_perm`` is the count of non-NaN null replicates used.

    Raises
    ------
    ValueError
        If ``alternative`` is not one of the three accepted values, or if
        ``null_stats`` has no non-NaN entries.

    Notes
    -----
    A clearly-structured observation lands in the null's tail → small ``p_value``,
    large ``|z|``. Pure noise sits inside the bulk → ``p_value`` near 0.5 (two-sided
    ~1.0), ``z`` near 0.
    """
    import numpy as np

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"alternative {alternative!r} must be 'two-sided', 'greater', or 'less'")
    nulls = np.asarray(null_stats, dtype="float64").ravel()
    nulls = nulls[~np.isnan(nulls)]
    n_perm = int(nulls.size)
    if n_perm == 0:
        raise ValueError("null_stats has no non-NaN replicates")

    mean = float(nulls.mean())
    std = float(nulls.std(ddof=0))
    z = float((observed_stat - mean) / std) if std > 0 else float("nan")

    if alternative == "greater":
        n_extreme = int(np.sum(nulls >= observed_stat))
    elif alternative == "less":
        n_extreme = int(np.sum(nulls <= observed_stat))
    else:  # two-sided: extremeness measured as distance from the null mean
        obs_dev = abs(observed_stat - mean)
        n_extreme = int(np.sum(np.abs(nulls - mean) >= obs_dev))

    p_value = (1 + n_extreme) / (1 + n_perm)
    return {"p_value": float(p_value), "z": z, "n_perm": n_perm}


def jackknife_drop_one(
    entities: Iterable,
    compute_fn: Callable[[object], pd.DataFrame],
) -> pd.DataFrame:
    """Leave-one-entity-out iterator collecting ``compute_fn`` results.

    For each unique entity in ``entities``, call ``compute_fn(held_out_entity)`` —
    which is expected to recompute whatever statistic/table while *excluding* that
    entity — and stack the returned frames, tagging each with the entity that was
    dropped. This is the drop-one-journal robustness check: re-run the cascade /
    roles computation ten times, each omitting one journal, and see whether the
    conclusion survives.

    Parameters
    ----------
    entities :
        Iterable of entity identifiers to hold out, one run each (e.g. the ten
        journal slugs). Order is preserved; **duplicates are de-duplicated** keeping
        first occurrence, so each entity is dropped exactly once.
    compute_fn :
        Callable taking a single held-out entity and returning a
        ``pandas.DataFrame`` of results for the run that excluded it. It owns the
        actual exclusion + recomputation; this function only orchestrates and tags.

    Returns
    -------
    pandas.DataFrame
        Concatenation of every run's frame with a prepended ``held_out`` column
        naming the entity dropped in that run. Index is reset. If ``entities`` is
        empty, an empty frame with just a ``held_out`` column is returned.

    Notes
    -----
    Pure orchestration: it adds no randomness and does not touch global state. The
    number of distinct runs equals the number of unique entities and is recoverable
    as ``result["held_out"].nunique()``.
    """
    import pandas as pd

    seen: list = []
    for ent in entities:
        if ent not in seen:
            seen.append(ent)

    if not seen:
        return pd.DataFrame({"held_out": pd.Series([], dtype="object")})

    frames: list[pd.DataFrame] = []
    for ent in seen:
        res = compute_fn(ent).copy()
        res.insert(0, "held_out", ent)
        frames.append(res)
    return pd.concat(frames, ignore_index=True)


def attribution_flip_rate(
    per_run: pd.DataFrame,
    *,
    key_col: str,
    value_col: str,
) -> float:
    """Fraction of keys whose attributed ``value_col`` is *not* stable across runs.

    Given the stacked output of :func:`jackknife_drop_one` (or any frame with one
    ``value_col`` per ``(held_out, key_col)``), measure stability of a per-key
    attribution: a key "flips" if it takes **more than one distinct** ``value_col``
    across the jackknife runs. The flip rate is the share of keys that flip.

    Parameters
    ----------
    per_run :
        Long frame with at least ``[key_col, value_col]`` and (typically) a
        ``held_out`` run column. One row per (run, key); the run column itself is
        not needed by this function — only the spread of ``value_col`` within each
        ``key_col``.
    key_col :
        The entity whose attribution stability is measured (e.g. ``"topic_id"`` —
        does each topic keep the same attributed origin journal when a journal is
        dropped?).
    value_col :
        The attributed value (e.g. ``"origin_journal"``). ``NaN`` values are treated
        as a distinct category so a key that gains/loses an attribution counts as a
        flip.

    Returns
    -------
    float
        A flip rate in ``[0, 1]``: ``#{keys with >1 distinct value} / #{keys}``.
        ``0.0`` when every key is perfectly stable; ``1.0`` when every key flips.
        Returns ``0.0`` for an empty frame (vacuously stable).

    Raises
    ------
    KeyError
        If ``key_col`` or ``value_col`` is absent from ``per_run``.
    """
    import pandas as pd

    for col in (key_col, value_col):
        if col not in per_run.columns:
            raise KeyError(f"{col!r} not in per_run")
    if per_run.empty:
        return 0.0

    vals = per_run[value_col]
    # Represent NaN as a sentinel so absent attributions register as their own value.
    sentinel = "__NA__"
    vals = vals.where(vals.notna(), sentinel).astype("string")
    work = pd.DataFrame({key_col: per_run[key_col].to_numpy(), "_v": vals.to_numpy()})
    distinct_per_key = work.groupby(key_col, dropna=False)["_v"].nunique()
    n_keys = int(distinct_per_key.size)
    if n_keys == 0:
        return 0.0
    n_flipped = int((distinct_per_key > 1).sum())
    return n_flipped / n_keys


def rank_stability(
    per_run: pd.DataFrame,
    *,
    key_col: str,
    score_col: str,
    run_col: str = "held_out",
) -> float:
    """Mean pairwise Spearman rank correlation of an ordering across jackknife runs.

    For a per-entity ranking (entities ranked by ``score_col``) recomputed in each
    jackknife run, measure how stable the *order* is: rank the entities within each
    run, then average the Spearman correlation over every pair of runs on their
    shared entities. ``1.0`` means the ordering is identical in every run; values
    near ``0`` mean the drop-one perturbation reshuffles the ranking.

    Parameters
    ----------
    per_run :
        Long frame with ``[run_col, key_col, score_col]``: one row per (run, entity)
        giving that entity's score in that run. Typically the output of
        :func:`jackknife_drop_one` where ``compute_fn`` returned per-entity scores.
    key_col :
        The ranked entity column.
    score_col :
        The numeric score the entities are ranked by (descending rank not required;
        Spearman is invariant to monotone transforms).
    run_col :
        The run identifier column. Default ``"held_out"`` (matching
        :func:`jackknife_drop_one`).

    Returns
    -------
    float
        Mean pairwise Spearman ρ in ``[-1, 1]`` over run pairs sharing ``>= 2``
        entities. Returns ``1.0`` when there is ``<= 1`` run (a single ordering is
        trivially self-consistent) or when no pair shares enough entities to
        correlate.

    Raises
    ------
    KeyError
        If any of ``run_col`` / ``key_col`` / ``score_col`` is absent.

    Notes
    -----
    Each run drops a different entity, so runs share only the entities common to
    both; the correlation is computed on that intersection. Pure / no randomness.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    for col in (run_col, key_col, score_col):
        if col not in per_run.columns:
            raise KeyError(f"{col!r} not in per_run")

    runs = list(pd.unique(per_run[run_col]))
    if len(runs) <= 1:
        return 1.0

    # run -> Series(score indexed by key)
    by_run: dict[object, pd.Series] = {}
    for r in runs:
        sub = per_run.loc[per_run[run_col] == r, [key_col, score_col]]
        by_run[r] = sub.set_index(key_col)[score_col]

    rhos: list[float] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = by_run[runs[i]], by_run[runs[j]]
            shared = a.index.intersection(b.index)
            if len(shared) < 2:
                continue
            rho, _ = spearmanr(a.loc[shared].to_numpy(), b.loc[shared].to_numpy())
            if not np.isnan(rho):
                rhos.append(float(rho))

    if not rhos:
        return 1.0
    return float(np.mean(rhos))
