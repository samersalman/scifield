"""V2 cartography — the topic **cascade engine**.

This module generalises the V1-S15 lead/follow prototype
(:mod:`scifield.findings.seeding`) into a reusable cascade engine over the
canonical journal-flow table (:mod:`scifield.cartography.flow`). It answers the
core cartography questions:

* **Where did a topic originate?** — :func:`origin_attribution` names the
  earliest-publishing journal per topic (the cascade *anchor*), with explicit
  tie handling.
* **Do some journals systematically lead others?** — :func:`lead_lag_matrix`
  builds the inter-journal mean/median signed year-lag matrix across every pair
  of journals that share a topic.
* **How fast does a topic diffuse across the panel?** —
  :func:`diffusion_curve` fits a saturating adoption curve (years-since-origin →
  cumulative fraction of the ten journals reached) per topic and a corpus
  aggregate, returning a half-saturation time ``t50`` and a logistic rate.
* **Who seeds whom?** — :func:`cascade_edges` wraps
  :func:`scifield.findings.seeding.directed_seeding_network` to emit a directed
  lead→follow edge list a notebook can draw.

Panel-conditional caveat (carry into every consumer)
----------------------------------------------------
"Topic *T* originated in journal *A*" means **"first within our ten-journal
panel"**, never "first in the world". The true origin may be a journal we never
harvested (``V2/CONTEXT.md`` §0.2). Every origin / lead-lag / diffusion result
inherits this caveat.

Input contract (pure, year-subset-friendly)
-------------------------------------------
Every public function accepts *either* a flow-table slice (the columns
``flow_leaf.parquet`` / ``flow_mid.parquet`` ship — at minimum ``[<topic_key>,
"journal_slug", "year"]``) *or* an already-computed first-appearance frame
(``[<topic_key>, "journal_slug", "first_year"]``). The frame is normalised
internally via :func:`_first_appearance_frame`, which **recomputes the
first-appearance year from whatever rows it is handed** rather than trusting a
precomputed ``is_first_appearance`` flag — so passing a ``year <= 2018`` slice
yields the held-out first-appearance structure for that window (what V2-S04
needs for temporal validation). All functions are therefore pure, I/O-free, and
operate on exactly the slice passed in.

Conventions: ``from __future__ import annotations``; numpy-style docstrings;
numpy + pandas + scipy only; ruff/black line-length 100. Real counts are
COMPUTED, never hardcoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "cascade_edges",
    "diffusion_curve",
    "lead_lag_matrix",
    "origin_attribution",
]

# Number of journals in the canonical panel (the cascade reaches at most this
# many). Used as the diffusion-curve denominator for the corpus aggregate.
N_PANEL_JOURNALS = 10


def _first_appearance_frame(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str,
) -> pd.DataFrame:
    """Normalise a flow slice *or* a first-appearance frame to first-appearance.

    Accepts either of the two shapes the cascade engine supports and returns the
    canonical first-appearance frame ``[topic_key, "journal_slug",
    "first_year"]`` — one row per (topic, journal) with the earliest year that
    pair appears **in the rows handed in**.

    Parameters
    ----------
    flow_or_first :
        Either a flow-table slice carrying at least ``[topic_key,
        "journal_slug", "year"]`` (the first-appearance year is recomputed from
        the ``year`` column, ignoring any ``is_first_appearance`` flag so a
        year-subset slice is honoured), or a frame that already carries
        ``"first_year"`` (taken as-is, only deduplicated to the earliest per
        pair).
    topic_key :
        The topic-grain column (``"topic_id"`` leaf or ``"mid_level_id"`` mid).

    Returns
    -------
    pandas.DataFrame
        ``[topic_key, "journal_slug", "first_year"]``, one row per (topic,
        journal), ``first_year`` ``int64``, sorted by ``topic_key`` then
        ``journal_slug``.

    Raises
    ------
    ValueError
        If neither ``"year"`` nor ``"first_year"`` is present, or if
        ``topic_key`` / ``"journal_slug"`` is missing.

    Notes
    -----
    The first-appearance year is **recomputed from the slice**, never read from a
    precomputed ``is_first_appearance`` flag, so the engine is year-subset-safe
    (a ``year <= 2018`` slice yields that window's anchors).
    """
    import pandas as pd

    base = [topic_key, "journal_slug"]
    missing = [c for c in base if c not in flow_or_first.columns]
    if missing:
        raise ValueError(f"input missing required column(s): {missing}")

    if "first_year" in flow_or_first.columns:
        work = flow_or_first.loc[:, [*base, "first_year"]].dropna(subset=base).copy()
        work["first_year"] = pd.to_numeric(work["first_year"], errors="coerce")
        work = work.dropna(subset=["first_year"])
        first = work.groupby(base, dropna=False)["first_year"].min().reset_index()
    elif "year" in flow_or_first.columns:
        work = flow_or_first.loc[:, [*base, "year"]].dropna(subset=base).copy()
        work["year"] = pd.to_numeric(work["year"], errors="coerce")
        work = work.dropna(subset=["year"])
        first = (
            work.groupby(base, dropna=False)["year"]
            .min()
            .reset_index()
            .rename(columns={"year": "first_year"})
        )
    else:
        raise ValueError("input must carry either a 'year' or a 'first_year' column")

    if first.empty:
        return pd.DataFrame(
            {topic_key: [], "journal_slug": [], "first_year": pd.Series([], dtype="int64")}
        )
    first["first_year"] = first["first_year"].astype("int64")
    return first.sort_values(base).reset_index(drop=True)


def origin_attribution(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
) -> pd.DataFrame:
    """Per-topic origin journal: the earliest-publishing journal in the panel.

    The *origin* of a topic is the journal whose first-appearance year is the
    minimum across all journals that ever publish the topic (panel-conditional —
    "first within our ten journals"). This is the cascade anchor V2-S04
    jackknifes.

    Tie-break rule
    --------------
    When two or more journals share the earliest ``first_year`` the topic has a
    *co-earliest* origin. ``tie`` is then ``True`` and the reported
    ``origin_journal_slug`` is the **alphabetically-first** co-earliest slug — a
    deterministic, jackknife-stable rule (re-running on any subset yields the
    same winner unless that very journal is dropped). ``n_co_earliest`` records
    how many journals tied so a caller can audit / list them all.

    Parameters
    ----------
    flow_or_first :
        A flow-table slice or a first-appearance frame; see
        :func:`_first_appearance_frame`. The origin is computed from the rows
        handed in, so a year-subset slice yields that window's origins.
    topic_key :
        Topic-grain column (``"topic_id"`` leaf / ``"mid_level_id"`` mid).
        Default ``"topic_id"``.

    Returns
    -------
    pandas.DataFrame
        One row per topic, sorted by ``topic_key``, with columns:

        ``<topic_key>``
            The topic id.
        ``origin_journal_slug`` : str
            The earliest-publishing journal (alphabetically-first on a tie).
        ``origin_year`` : int64
            That journal's first-appearance year (the topic's panel origin
            year).
        ``n_journals`` : int64
            How many of the ten journals ever publish the topic (in the slice).
        ``tie`` : bool
            ``True`` iff two or more journals share the earliest year.
        ``n_co_earliest`` : int64
            Number of journals tied at the earliest year (``1`` when no tie).

    Notes
    -----
    Empty input yields an empty, well-formed frame.
    """
    import pandas as pd

    first = _first_appearance_frame(flow_or_first, topic_key=topic_key)
    cols = [
        topic_key,
        "origin_journal_slug",
        "origin_year",
        "n_journals",
        "tie",
        "n_co_earliest",
    ]
    if first.empty:
        return pd.DataFrame({c: [] for c in cols})

    grp = first.groupby(topic_key, dropna=False)
    earliest_year = grp["first_year"].transform("min")
    n_journals = grp["journal_slug"].transform("size")

    co_earliest = first.loc[first["first_year"] == earliest_year].copy()
    co_earliest["n_journals"] = n_journals.loc[co_earliest.index]

    # Deterministic tie-break: alphabetically-first co-earliest journal slug.
    co_earliest = co_earliest.sort_values([topic_key, "journal_slug"])
    agg = co_earliest.groupby(topic_key, dropna=False).agg(
        origin_journal_slug=("journal_slug", "first"),
        origin_year=("first_year", "first"),
        n_journals=("n_journals", "first"),
        n_co_earliest=("journal_slug", "size"),
    )
    agg = agg.reset_index()
    agg["origin_year"] = agg["origin_year"].astype("int64")
    agg["n_journals"] = agg["n_journals"].astype("int64")
    agg["n_co_earliest"] = agg["n_co_earliest"].astype("int64")
    agg["tie"] = agg["n_co_earliest"] > 1
    agg = agg.sort_values(topic_key).reset_index(drop=True)
    return agg[cols]


def lead_lag_matrix(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
) -> pd.DataFrame:
    """Inter-journal signed year-lag matrix across shared topics.

    For each **ordered** journal pair ``(journal_i, journal_j)`` that shares
    ``n_shared`` topics, the signed lag on a shared topic is
    ``first_year[journal_j] - first_year[journal_i]`` — **positive when
    journal_i leads** (publishes the topic earlier). ``mean_lag`` /
    ``median_lag`` aggregate that signed lag across the shared topics, and
    ``n_i_leads`` counts the shared topics where journal_i strictly precedes
    journal_j (a strict lead; ties count for neither direction).

    The result is tidy long form; a notebook pivots
    ``(journal_i, journal_j) -> mean_lag`` to a 10×10 heatmap. The matrix is
    antisymmetric in the means (``mean_lag[i, j] == -mean_lag[j, i]``) and
    symmetric in ``n_shared``.

    Parameters
    ----------
    flow_or_first :
        A flow-table slice or a first-appearance frame; see
        :func:`_first_appearance_frame`. Lags are computed from the rows handed
        in (year-subset-safe).
    topic_key :
        Topic-grain column. Default ``"topic_id"``.

    Returns
    -------
    pandas.DataFrame
        One row per ordered pair of distinct journals sharing ``>= 1`` topic,
        sorted by ``journal_i`` then ``journal_j``, with columns:

        ``journal_i`` : str
            The "leader" side of the pair.
        ``journal_j`` : str
            The "follower" side of the pair.
        ``mean_lag`` : float
            Mean signed lag (``> 0`` ⇒ ``journal_i`` leads on average).
        ``median_lag`` : float
            Median signed lag.
        ``n_shared`` : int64
            Topics both journals publish (symmetric).
        ``n_i_leads`` : int64
            Shared topics where ``journal_i`` strictly precedes ``journal_j``.

    Notes
    -----
    Only pairs that co-occur appear (no all-zero self rows, no diagonal). Empty
    input — or a slice where no two journals share a topic — yields an empty,
    well-formed frame.
    """
    import pandas as pd

    first = _first_appearance_frame(flow_or_first, topic_key=topic_key)
    cols = ["journal_i", "journal_j", "mean_lag", "median_lag", "n_shared", "n_i_leads"]
    if first.empty:
        return pd.DataFrame({c: [] for c in cols})

    left = first.rename(columns={"journal_slug": "journal_i", "first_year": "year_i"})
    right = first.rename(columns={"journal_slug": "journal_j", "first_year": "year_j"})
    pairs = left.merge(right, on=topic_key)
    pairs = pairs.loc[pairs["journal_i"] != pairs["journal_j"]].copy()
    if pairs.empty:
        return pd.DataFrame({c: [] for c in cols})

    # Positive lag ⇒ journal_i appears earlier (leads) than journal_j.
    pairs["lag"] = (pairs["year_j"] - pairs["year_i"]).astype("float64")
    pairs["i_leads"] = (pairs["year_i"] < pairs["year_j"]).astype("int64")

    out = (
        pairs.groupby(["journal_i", "journal_j"], dropna=False)
        .agg(
            mean_lag=("lag", "mean"),
            median_lag=("lag", "median"),
            n_shared=("lag", "size"),
            n_i_leads=("i_leads", "sum"),
        )
        .reset_index()
    )
    out["n_shared"] = out["n_shared"].astype("int64")
    out["n_i_leads"] = out["n_i_leads"].astype("int64")
    out = out.sort_values(["journal_i", "journal_j"]).reset_index(drop=True)
    return out[cols]


def cascade_edges(
    papers_topics: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
    min_papers: int = 1,
) -> pd.DataFrame:
    """Directed lead→follow edge list (a thin wrapper, not a reimplementation).

    Delegates to :func:`scifield.findings.seeding.directed_seeding_network` on
    the ``journal_slug`` grain so the cascade overlay a notebook draws reuses the
    exact, tested lead/follow logic. A high-``weight`` edge ``src -> dst`` means
    ``src`` tends to publish a shared topic before ``dst``.

    Parameters
    ----------
    papers_topics :
        Either a per-paper frame (``[topic_key, "journal_slug", "year"]``, one
        row per paper) or a flow slice — anything
        :func:`~scifield.findings.seeding.first_publication_year` accepts after
        the topic-key rename. The underlying primitive recomputes first years
        from ``year``, so a flow slice (already deduplicated to cells) and a raw
        paper frame both yield the same edges.
    topic_key :
        Topic-grain column. Default ``"topic_id"``. Renamed to ``"topic_id"`` for
        the seeding primitive and back on the way out is unnecessary — the edge
        list is keyed on journals, not topics, so the topic column does not
        appear in the output.
    min_papers :
        Per-year publication threshold forwarded to the seeding primitive.
        Default ``1``.

    Returns
    -------
    pandas.DataFrame
        Edge list ``["src", "dst", "n_precedes", "n_shared", "weight"]`` exactly
        as :func:`~scifield.findings.seeding.directed_seeding_network`, with
        ``src`` / ``dst`` journal slugs.

    Notes
    -----
    This is deliberately a wrapper: the directed network already encodes the
    seeding structure, so duplicating it here would risk divergence.
    """
    from scifield.findings.seeding import directed_seeding_network

    work = papers_topics
    if topic_key != "topic_id":
        if topic_key not in work.columns:
            raise ValueError(f"input missing topic key column {topic_key!r}")
        work = work.rename(columns={topic_key: "topic_id"})
    return directed_seeding_network(work, entity_col="journal_slug", min_papers=min_papers)


def _fit_logistic_t50(
    years_since_origin: object,
    cumulative_fraction: object,
) -> tuple[float, float, bool]:
    """Fit a 2-parameter logistic to an adoption curve; return ``(t50, rate)``.

    Fits ``f(t) = 1 / (1 + exp(-rate * (t - t50)))`` (saturating at 1.0) to the
    empirical years-since-origin → cumulative-fraction points by least squares.
    ``t50`` is the half-saturation time (years to reach half the journals that
    are ever reached); ``rate`` is the logistic growth rate.

    Parameters
    ----------
    years_since_origin :
        1-D array of non-negative integer year offsets (``0`` at origin).
    cumulative_fraction :
        1-D array in ``[0, 1]``, the cumulative fraction of reached journals at
        each offset (monotonic non-decreasing).

    Returns
    -------
    t50 : float
        Half-saturation time in years (``NaN`` if unfittable).
    rate : float
        Logistic growth rate (``NaN`` if unfittable).
    fitted : bool
        ``True`` iff the curve fit converged; ``False`` falls back to ``NaN``
        params (the empirical ``t50`` is still reported separately by the
        caller).

    Notes
    -----
    Curves with fewer than three distinct ``t`` points, or that never vary, are
    not fittable and return ``(NaN, NaN, False)`` — the caller still reports the
    empirical (linear-interpolated) ``t50_empirical``.
    """
    import numpy as np

    t = np.asarray(years_since_origin, dtype="float64")
    y = np.asarray(cumulative_fraction, dtype="float64")
    if t.size < 3 or np.unique(t).size < 3 or np.ptp(y) == 0:
        return float("nan"), float("nan"), False

    try:
        from scipy.optimize import curve_fit
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        return float("nan"), float("nan"), False

    def _logistic(x: object, t50: float, rate: float) -> object:
        # Clip the exponent so a large trial rate during optimisation cannot
        # overflow float64 exp (the fit explores extreme params harmlessly).
        z = np.clip(-rate * (np.asarray(x) - t50), -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(z))

    # Seed t50 at the empirical median crossing, rate at a gentle positive slope.
    t50_0 = float(np.interp(0.5, y, t)) if y.max() >= 0.5 else float(t.max())
    import warnings

    from scipy.optimize import OptimizeWarning

    try:
        with np.errstate(over="ignore", invalid="ignore"), warnings.catch_warnings():
            # A clean, well-separated adoption curve fits "too well" — its
            # covariance is singular and SciPy warns. The point estimates are
            # still valid, so silence that specific, expected warning.
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                _logistic,
                t,
                y,
                p0=[t50_0, 0.5],
                maxfev=10000,
            )
    except (RuntimeError, ValueError):
        return float("nan"), float("nan"), False
    return float(popt[0]), float(popt[1]), True


def diffusion_curve(
    flow_or_first: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
    n_panel: int = N_PANEL_JOURNALS,
) -> pd.DataFrame:
    """Per-topic diffusion (adoption) curve: speed a topic spreads across journals.

    For each topic the journals are ordered by first-appearance year. With
    ``year_since_origin = first_year - origin_year`` (``0`` at the origin
    journal), the cumulative fraction of journals reached by offset ``t`` is the
    empirical adoption CDF. This function summarises that CDF per topic:

    * the empirical half-saturation time ``t50_empirical`` (years to reach half
      the journals the topic *ever* reaches in the slice), via linear
      interpolation on the step CDF;
    * the time to reach the last journal ``span_years`` (max offset);
    * a logistic fit (:func:`_fit_logistic_t50`) giving ``t50_logistic`` and
      ``logistic_rate`` when the curve has ≥ 3 distinct offsets;
    * ``reach_fraction = n_journals / n_panel`` — the share of the ten-journal
      panel the topic ever reaches (saturation ceiling).

    Parameters
    ----------
    flow_or_first :
        A flow-table slice or a first-appearance frame; see
        :func:`_first_appearance_frame`. Curves are computed from the rows
        handed in (year-subset-safe).
    topic_key :
        Topic-grain column. Default ``"topic_id"``.
    n_panel :
        Panel size used as the saturation denominator for ``reach_fraction``.
        Default :data:`N_PANEL_JOURNALS` (10).

    Returns
    -------
    pandas.DataFrame
        One row per topic, sorted by ``topic_key``, with columns:

        ``<topic_key>``
            Topic id.
        ``origin_year`` : int64
            The topic's panel origin year (min first-appearance year).
        ``n_journals`` : int64
            Journals the topic ever reaches in the slice.
        ``reach_fraction`` : float
            ``n_journals / n_panel`` — fraction of the panel reached (≤ 1.0).
        ``span_years`` : int64
            Years from origin to the last journal reached (max offset).
        ``t50_empirical`` : float
            Years to reach half the reached journals (empirical, interpolated).
            ``0.0`` when the topic reaches only one journal (instant half).
        ``t50_logistic`` : float
            Half-saturation time from the logistic fit (``NaN`` if unfittable).
        ``logistic_rate`` : float
            Logistic growth rate (``NaN`` if unfittable).
        ``curve_fitted`` : bool
            Whether the logistic fit converged.

    Notes
    -----
    *Single-journal topics* (``n_journals == 1``) cannot define a diffusion speed
    — there is no spread — so ``span_years == 0``, ``t50_empirical == 0.0``, and
    the logistic columns are ``NaN`` with ``curve_fitted == False``. Topics with
    only two distinct offsets are likewise too thin to fit a logistic and report
    ``NaN`` logistic params while still carrying a valid empirical ``t50``. The
    cumulative fraction is normalised to the journals **reached in the slice**
    (so it always saturates at 1.0); ``reach_fraction`` separately records how
    much of the full panel that represents.
    """
    import numpy as np
    import pandas as pd

    first = _first_appearance_frame(flow_or_first, topic_key=topic_key)
    cols = [
        topic_key,
        "origin_year",
        "n_journals",
        "reach_fraction",
        "span_years",
        "t50_empirical",
        "t50_logistic",
        "logistic_rate",
        "curve_fitted",
    ]
    if first.empty:
        return pd.DataFrame({c: [] for c in cols})

    rows: list[dict[str, object]] = []
    for tid, grp in first.groupby(topic_key, dropna=False):
        years = np.sort(grp["first_year"].to_numpy())
        origin_year = int(years[0])
        n_j = int(years.size)
        offsets = (years - origin_year).astype("float64")
        span = int(offsets[-1])

        # Cumulative fraction of the journals REACHED IN THE SLICE, by offset.
        # Steps at each distinct offset; collapse duplicates to the running count.
        uniq_offsets, counts = np.unique(offsets, return_counts=True)
        cum = np.cumsum(counts).astype("float64") / float(n_j)

        # Single-journal topics have no spread; otherwise interpolate the offset
        # at which the step CDF first reaches 0.5.
        t50_emp = 0.0 if n_j == 1 else float(np.interp(0.5, cum, uniq_offsets))

        t50_log, rate, fitted = _fit_logistic_t50(uniq_offsets, cum)

        rows.append(
            {
                topic_key: tid,
                "origin_year": origin_year,
                "n_journals": n_j,
                "reach_fraction": n_j / float(n_panel),
                "span_years": span,
                "t50_empirical": t50_emp,
                "t50_logistic": t50_log,
                "logistic_rate": rate,
                "curve_fitted": fitted,
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    out["origin_year"] = out["origin_year"].astype("int64")
    out["n_journals"] = out["n_journals"].astype("int64")
    out["span_years"] = out["span_years"].astype("int64")
    out["reach_fraction"] = out["reach_fraction"].astype("float64")
    out["t50_empirical"] = out["t50_empirical"].astype("float64")
    out["t50_logistic"] = out["t50_logistic"].astype("float64")
    out["logistic_rate"] = out["logistic_rate"].astype("float64")
    out["curve_fitted"] = out["curve_fitted"].astype("bool")
    out = out.sort_values(topic_key).reset_index(drop=True)
    return out
