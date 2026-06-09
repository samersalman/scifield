"""V2 cartography — journal **role taxonomy** + **citational velocity**.

This module classifies each canonical journal's role in topic flow and
characterises how fast its papers accrue citations. It is the V2-S05 layer on top
of the flow table (:mod:`scifield.cartography.flow`) and the directed seeding
network (:func:`scifield.findings.seeding.directed_seeding_network`).

The three roles (source / bridge / terminal)
---------------------------------------------
Every journal gets three continuous component scores, then a single discrete
``role`` label by a documented rule:

* **source** — propensity to *publish topics first and send influence outward*. It
  is the mean of two z-scored sub-signals: (i) the journal's **seeding score**
  (:func:`scifield.findings.seeding.seeding_score` — mean normalized first-publication
  lead across the topics it touches) and (ii) its **net citation out-flow** share
  (``out_flow − in_flow`` summed over its flow cells, normalised by the panel total
  absolute net-flow). A source publishes early *and* is a net exporter of citations.
* **bridge** — **betweenness centrality** of the journal in the directed topic-flow
  (seeding) network. A bridge sits on many shortest lead→follow paths: topics pass
  *through* it on the way from early to late journals. Computed with networkx on the
  ``weight``-filtered directed edge list.
* **terminal** — propensity to *receive rather than send*: high citation **in-flow**
  relative to out-flow (a net importer / late adopter). It is the mean of two
  z-scored sub-signals: (i) the journal's **net citation in-flow** share (the
  negative of the source out-flow share) and (ii) ``1 − seeding_score`` (arrives
  late across topics). Source and terminal are deliberately near-mirror images on
  the temporal/flow axes; the *bridge* axis is orthogonal (a journal can be a strong
  bridge while being neither a clear source nor terminal).

**Label rule.** The three components are each z-scored across the ten journals, and
the journal's ``role`` is the **argmax** of ``{source_z, bridge_z, terminal_z}``.
Ties (numerically equal maxima) break in the fixed order source > bridge > terminal.
This is a deliberately simple, documented decision rule — the *scores* are the
nuanced output; the label is a one-word summary for the map.

1995 left-censoring caveat
--------------------------
The corpus starts in 1995, so "publishes a topic first" is dominated by topics
already present at panel start (S03 hand-off: 79% of leaf topics tie at a 1995
origin). The seeding-based sub-signals therefore inherit a 1995-tie bias. To stay
robust we (a) weight the *citation-flow* sub-signals equally with the seeding ones
in source/terminal, and (b) make **bridge** purely betweenness on the directed
network — both far less censoring-sensitive than a raw "first in 1995" count. The
module documents this on every affected function; the script reports the seeding-only
vs. flow-only decomposition so a reader can see neither sub-signal alone drives the
label.

Citational velocity
--------------------
:func:`citational_velocity` summarises, per journal, the distribution of
*years-from-publication-to-citation* (``citing_year − publication_year``) over the
external inbound citations of that journal's papers (``cited_by.parquet`` joined via
``openalex_id``). A *fast* journal accrues citations soon after publication (small
median lag); a *slow* one accrues them over a longer tail. The fast/slow indicator is
relative to the panel median.

Purity
------
Every function takes in-memory frames and returns a frame. Reading parquet / Kùzu /
DuckDB and drawing live in ``V2/scripts/build_roles.py`` and the notebook. networkx
is used *here* (for betweenness) — that is allowed for the roles layer; the seeding
primitive stays networkx-free.

Conventions: ``from __future__ import annotations``; numpy-style docstrings; numpy +
pandas + scipy + networkx; ruff/black line-length 100. Real counts are COMPUTED,
never hardcoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "ROLE_COMPONENTS",
    "assign_roles",
    "bridge_betweenness",
    "citational_velocity",
    "role_scores",
    "role_scores_jackknife",
]

# The three role component score columns, in label tie-break priority order.
ROLE_COMPONENTS: tuple[str, ...] = ("source", "bridge", "terminal")


def _zscore(series: pd.Series) -> pd.Series:
    """Population z-score of a numeric series (zeros if it has no spread).

    Parameters
    ----------
    series :
        Numeric series to standardise.

    Returns
    -------
    pandas.Series
        ``(series - mean) / std`` (population std, ``ddof=0``), index preserved.
        If the standard deviation is ``0`` (every value identical) the result is all
        zeros, so a degenerate component never dominates the argmax label.
    """
    import numpy as np

    vals = series.astype("float64")
    std = float(vals.std(ddof=0))
    if std == 0 or np.isnan(std):
        return (vals * 0.0).astype("float64")
    return ((vals - float(vals.mean())) / std).astype("float64")


def bridge_betweenness(
    directed_net: pd.DataFrame,
    *,
    weight_threshold: float = 0.5,
) -> pd.DataFrame:
    """Betweenness centrality of each journal in the directed topic-flow network.

    Builds a directed graph from the seeding edge list (``src`` leads ``dst`` when
    ``weight = n_precedes / n_shared`` is high) keeping only edges whose ``weight``
    meets ``weight_threshold`` (so only genuine lead→follow links carry flow), then
    computes networkx **betweenness centrality**: the fraction of all-pairs shortest
    paths that pass *through* each node. A high score marks a journal that topics
    routinely traverse on the way from early to late journals — a bridge.

    Edges are weighted by ``1 / weight`` as the shortest-path *distance* (a stronger
    lead = a shorter hop), so betweenness favours paths along strong lead→follow
    links.

    Parameters
    ----------
    directed_net :
        Edge list ``["src", "dst", "n_precedes", "n_shared", "weight"]`` from
        :func:`scifield.findings.seeding.directed_seeding_network` (journal-slug
        nodes). Extra columns are ignored.
    weight_threshold :
        Minimum ``weight`` for an edge to be kept (default ``0.5`` — ``src`` leads
        ``dst`` on a majority of their shared topics). Lowering it densifies the
        graph; raising it keeps only dominant leads.

    Returns
    -------
    pandas.DataFrame
        ``["journal_slug", "bridge"]`` for every node appearing in the retained
        edges (both endpoints), sorted by ``bridge`` descending then slug. ``bridge``
        is the networkx betweenness in ``[0, 1]``. An empty / no-edge input yields an
        empty frame with these columns.

    Notes
    -----
    Betweenness is far less sensitive to the 1995 left-censoring than a raw
    "first-in-1995" count: it depends on the *relative ordering* of journals across
    many topics, not on the absolute origin year. Isolated nodes (no retained edge)
    do not appear; the caller fills them with ``bridge = 0`` against the canonical
    panel.
    """
    import networkx as nx
    import pandas as pd

    cols = ["journal_slug", "bridge"]
    if directed_net.empty:
        return pd.DataFrame({"journal_slug": pd.Series([], dtype="object"), "bridge": []})

    kept = directed_net.loc[directed_net["weight"] >= float(weight_threshold)].copy()
    if kept.empty:
        return pd.DataFrame({"journal_slug": pd.Series([], dtype="object"), "bridge": []})

    graph = nx.DiGraph()
    for row in kept.itertuples():
        # Distance = 1 / weight: a stronger lead is a shorter hop, so shortest paths
        # run along dominant lead->follow links.
        graph.add_edge(row.src, row.dst, distance=1.0 / float(row.weight))

    bc = nx.betweenness_centrality(graph, weight="distance", normalized=True)
    out = pd.DataFrame({"journal_slug": list(bc.keys()), "bridge": list(bc.values())})
    out["bridge"] = out["bridge"].astype("float64")
    out = out.sort_values(["bridge", "journal_slug"], ascending=[False, True]).reset_index(
        drop=True
    )
    return out[cols]


def role_scores(
    flow: pd.DataFrame,
    directed_net: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
    weight_threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-journal source / bridge / terminal scores + a discrete ``role`` label.

    Combines three signals into the role taxonomy (see the module docstring):

    * **source** — mean of z-scored ``seeding_score`` and z-scored *net out-flow
      share*. Publishes early and exports citations.
    * **bridge** — betweenness centrality (:func:`bridge_betweenness`) on the
      directed topic-flow network.
    * **terminal** — mean of z-scored *net in-flow share* and z-scored
      ``1 − seeding_score``. Receives more than it sends; arrives late.

    The discrete ``role`` is the argmax of the three *z-scored* components
    (``source_z``/``bridge_z``/``terminal_z``), ties broken source > bridge >
    terminal (:func:`assign_roles`).

    Parameters
    ----------
    flow :
        A flow-table slice carrying at least ``[topic_key, "journal_slug", "year"]``
        for the seeding sub-signal and (for the flow sub-signals) ``"out_flow"`` /
        ``"in_flow"`` integer columns — exactly the shipped ``flow_leaf.parquet`` /
        ``flow_mid.parquet`` schema. If ``out_flow`` / ``in_flow`` are absent (or all
        NaN), the citation sub-signals are treated as zero and the source/terminal
        scores reduce to the seeding axis alone (logged by the caller).
    directed_net :
        The directed seeding edge list for the *same* slice, from
        :func:`scifield.findings.seeding.directed_seeding_network` (journal-slug
        nodes). The caller builds this from the flow slice so a jackknife drop is
        consistent across both inputs.
    topic_key :
        Topic-grain column (``"topic_id"`` leaf / ``"mid_level_id"`` mid). Default
        ``"topic_id"``.
    weight_threshold :
        Forwarded to :func:`bridge_betweenness`. Default ``0.5``.

    Returns
    -------
    pandas.DataFrame
        One row per ``journal_slug`` that appears in ``flow``, sorted by
        ``journal_slug``, with columns:

        ``journal_slug`` : str
        ``seeding_score`` : float — mean normalized first-publication lead.
        ``net_outflow_share`` : float — ``(out − in) / Σ|out − in|`` over the panel.
        ``betweenness`` : float — raw betweenness from :func:`bridge_betweenness`.
        ``source`` : float — mean of ``z(seeding_score)`` and ``z(net_outflow_share)``.
        ``bridge`` : float — ``betweenness`` (carried through as the bridge score).
        ``terminal`` : float — mean of ``z(−net_outflow_share)`` and ``z(1−seeding)``.
        ``source_z`` / ``bridge_z`` / ``terminal_z`` : float — the components
            z-scored across journals (the quantities the label argmaxes over).
        ``role`` : str — ``"source"`` / ``"bridge"`` / ``"terminal"``.

    Notes
    -----
    The seeding sub-signals inherit the 1995 left-censoring bias (S03 hand-off);
    the citation-flow sub-signals and betweenness do not, which is why they are given
    equal weight. Empty input yields an empty, well-formed frame.
    """
    import numpy as np
    import pandas as pd

    from scifield.findings.seeding import directed_seeding_network, seeding_score

    out_cols = [
        "journal_slug",
        "seeding_score",
        "net_outflow_share",
        "betweenness",
        "source",
        "bridge",
        "terminal",
        "source_z",
        "bridge_z",
        "terminal_z",
        "role",
    ]
    base = [topic_key, "journal_slug"]
    missing = [c for c in [*base, "year"] if c not in flow.columns]
    if missing:
        raise ValueError(f"flow missing required column(s): {missing}")

    work = flow.loc[:, [c for c in flow.columns]].copy()
    if work.empty:
        return pd.DataFrame({c: [] for c in out_cols})

    # --- seeding sub-signal (rename topic key to topic_id for the primitive) ---
    seed_in = work[[topic_key, "journal_slug", "year"]].copy()
    if topic_key != "topic_id":
        seed_in = seed_in.rename(columns={topic_key: "topic_id"})
    seed = seeding_score(seed_in, entity_col="journal_slug")
    if seed.empty:
        return pd.DataFrame({c: [] for c in out_cols})
    seed = seed[["journal_slug", "seeding_score"]]

    journals = sorted(seed["journal_slug"].unique())

    # --- citation net-flow sub-signal (panel-normalised) ---
    have_flows = ("out_flow" in work.columns) and ("in_flow" in work.columns)
    if have_flows:
        fl = work[["journal_slug", "out_flow", "in_flow"]].copy()
        fl["out_flow"] = pd.to_numeric(fl["out_flow"], errors="coerce").fillna(0.0)
        fl["in_flow"] = pd.to_numeric(fl["in_flow"], errors="coerce").fillna(0.0)
        per_j = fl.groupby("journal_slug", dropna=False)[["out_flow", "in_flow"]].sum()
        net = (per_j["out_flow"] - per_j["in_flow"]).reindex(journals).fillna(0.0)
        denom = float(net.abs().sum())
        net_share = (net / denom) if denom > 0 else net * 0.0
    else:
        net_share = pd.Series(0.0, index=journals, dtype="float64")

    # --- bridge sub-signal (betweenness) ---
    net_for_bridge = directed_net
    if net_for_bridge is None:
        net_for_bridge = directed_seeding_network(seed_in, entity_col="journal_slug")
    bridge = bridge_betweenness(net_for_bridge, weight_threshold=weight_threshold)
    bridge_map = dict(zip(bridge["journal_slug"], bridge["bridge"], strict=False))

    # --- assemble per-journal frame on the canonical journal set in this slice ---
    df = pd.DataFrame({"journal_slug": journals})
    df = df.merge(seed, on="journal_slug", how="left")
    df["seeding_score"] = df["seeding_score"].astype("float64")
    df["net_outflow_share"] = df["journal_slug"].map(net_share).astype("float64")
    df["betweenness"] = df["journal_slug"].map(bridge_map).fillna(0.0).astype("float64")

    # Raw role components.
    df["source"] = (_zscore(df["seeding_score"]) + _zscore(df["net_outflow_share"])) / 2.0
    df["bridge"] = df["betweenness"]
    late = 1.0 - df["seeding_score"]
    df["terminal"] = (_zscore(-df["net_outflow_share"]) + _zscore(late)) / 2.0

    # z-scored components — the quantities the label argmaxes over.
    df["source_z"] = _zscore(df["source"])
    df["bridge_z"] = _zscore(df["bridge"])
    df["terminal_z"] = _zscore(df["terminal"])

    df = assign_roles(df)

    df = df.sort_values("journal_slug").reset_index(drop=True)
    for c in ("seeding_score", "net_outflow_share", "betweenness"):
        df[c] = df[c].astype("float64")
    # Guard against all-NaN seeding (e.g. a journal with no scored topic).
    df["seeding_score"] = df["seeding_score"].fillna(np.nan)
    return df[out_cols]


def assign_roles(scores: pd.DataFrame) -> pd.DataFrame:
    """Attach a discrete ``role`` label = argmax of the z-scored components.

    Given a frame already carrying ``source_z`` / ``bridge_z`` / ``terminal_z``, set
    each row's ``role`` to the component with the largest z-score. Numerically equal
    maxima break in the fixed priority order :data:`ROLE_COMPONENTS`
    (``source`` > ``bridge`` > ``terminal``) so the rule is deterministic.

    Parameters
    ----------
    scores :
        Frame with float columns ``source_z``, ``bridge_z``, ``terminal_z`` (one row
        per journal).

    Returns
    -------
    pandas.DataFrame
        ``scores`` with an added/overwritten string ``role`` column.

    Raises
    ------
    KeyError
        If any of the three z-score columns is absent.

    Notes
    -----
    The label is intentionally coarse — the continuous components are the nuanced
    output. Argmax-of-z makes "which axis is this journal *most* extreme on" the
    label, regardless of the components' different natural scales.
    """
    import numpy as np

    zcols = [f"{c}_z" for c in ROLE_COMPONENTS]
    for c in zcols:
        if c not in scores.columns:
            raise KeyError(f"{c!r} not in scores")
    out = scores.copy()
    if out.empty:
        out["role"] = []
        return out
    mat = out[zcols].to_numpy(dtype="float64")
    # argmax with the fixed priority tie-break: ROLE_COMPONENTS is the column order,
    # and numpy.argmax returns the FIRST max on ties -> source > bridge > terminal.
    idx = np.argmax(mat, axis=1)
    out["role"] = [ROLE_COMPONENTS[i] for i in idx]
    return out


def role_scores_jackknife(
    flow: pd.DataFrame,
    *,
    topic_key: str = "topic_id",
    weight_threshold: float = 0.5,
    score_col: str = "source_z",
) -> pd.DataFrame:
    """Drop-one-journal jackknife of :func:`role_scores`, stacked for rank-stability.

    Runs the role-scoring pipeline ten times, each excluding one of the journals in
    ``flow``, and stacks the surviving-journal scores with a ``held_out`` run column
    — the long frame :func:`scifield.cartography.nulls.rank_stability` consumes to
    produce the protocol's ``role_rank_correlation``. The directed seeding network is
    rebuilt from the *held-out* flow slice each run so the bridge axis is consistent
    with the dropped panel.

    Parameters
    ----------
    flow :
        The full flow slice (``[topic_key, "journal_slug", "year", "out_flow",
        "in_flow"]``). Each run drops one ``journal_slug``.
    topic_key :
        Topic-grain column. Default ``"topic_id"``.
    weight_threshold :
        Forwarded to :func:`role_scores`. Default ``0.5``.
    score_col :
        Which role-score column to retain for the rank-stability check. Default
        ``"source_z"`` (the source axis). The returned frame carries this column
        renamed to ``"score"`` plus the raw ``source``/``bridge``/``terminal`` and
        ``role`` for auditing.

    Returns
    -------
    pandas.DataFrame
        ``["held_out", "journal_slug", "score", "source", "bridge", "terminal",
        "role"]``: one row per (run, surviving journal). Feed it to
        :func:`scifield.cartography.nulls.rank_stability` with
        ``key_col="journal_slug", score_col="score"`` to get
        ``role_rank_correlation``. Empty input yields an empty, well-formed frame.

    Notes
    -----
    This helper owns the *role-score* jackknife (V2-S05); the origin-attribution
    jackknife is V2-S04's. The directed network is recomputed per run from the
    held-out slice, so removing a hub journal genuinely perturbs the betweenness axis
    — the point of the stability check.
    """
    import pandas as pd

    from scifield.cartography.nulls import jackknife_drop_one
    from scifield.findings.seeding import directed_seeding_network

    keep = [
        "held_out",
        "journal_slug",
        "score",
        "source",
        "bridge",
        "terminal",
        "role",
    ]
    if flow.empty or "journal_slug" not in flow.columns:
        return pd.DataFrame({c: pd.Series([], dtype="object") for c in keep})

    journals = sorted(flow["journal_slug"].dropna().unique())

    def _compute(held_out: object) -> pd.DataFrame:
        sub = flow.loc[flow["journal_slug"] != held_out].copy()
        seed_in = sub[[topic_key, "journal_slug", "year"]].copy()
        if topic_key != "topic_id":
            seed_in = seed_in.rename(columns={topic_key: "topic_id"})
        net = directed_seeding_network(seed_in, entity_col="journal_slug")
        scored = role_scores(sub, net, topic_key=topic_key, weight_threshold=weight_threshold)
        scored = scored.rename(columns={score_col: "score"})
        return scored[["journal_slug", "score", "source", "bridge", "terminal", "role"]]

    stacked = jackknife_drop_one(journals, _compute)
    if stacked.empty:
        return pd.DataFrame({c: pd.Series([], dtype="object") for c in keep})
    return stacked[keep]


def citational_velocity(
    citations: pd.DataFrame,
    paper_meta: pd.DataFrame,
    *,
    journal_col: str = "journal_slug",
    focal_id_col: str = "openalex_id",
    citing_year_col: str = "citing_year",
    pub_year_col: str = "year",
    max_lag: int = 60,
) -> pd.DataFrame:
    """Per-journal distribution of years-from-publication-to-citation.

    Joins each external inbound citation (``citations``: one row per citing event,
    keyed on the focal paper's ``openalex_id`` with the ``citing_year``) to the focal
    paper's journal + publication year (``paper_meta``), forms the citation **lag**
    ``citing_year − publication_year``, and summarises the lag distribution per
    journal.

    Parameters
    ----------
    citations :
        One row per inbound citation, carrying ``[focal_id_col, citing_year_col]``
        (e.g. ``cited_by.parquet``: ``focal_oa_id`` renamed to ``openalex_id`` by the
        caller, plus ``citing_year``). Extra columns ignored.
    paper_meta :
        Per-paper metadata carrying ``[focal_id_col, journal_col, pub_year_col]``
        (e.g. archetypes' ``openalex_id`` + the resolved ``journal_slug`` + ``year``).
        One row per focal paper; duplicate ids are de-duplicated keeping the first.
    journal_col :
        The journal column in ``paper_meta``. Default ``"journal_slug"``.
    focal_id_col :
        The shared focal-paper id used to join. Default ``"openalex_id"``.
    citing_year_col :
        The citing-event year column in ``citations``. Default ``"citing_year"``.
    pub_year_col :
        The focal-paper publication-year column in ``paper_meta``. Default ``"year"``.
    max_lag :
        Upper bound (years) on a valid lag; lags ``> max_lag`` or ``< 0`` are dropped
        as data errors (a citation cannot precede publication, and a >60y lag is
        almost surely a bad ``citing_year``). Default ``60``.

    Returns
    -------
    pandas.DataFrame
        One row per journal (sorted by ``median_lag`` ascending then journal), with:

        ``journal_slug`` : str
        ``n_citations`` : int64 — valid inbound citations used.
        ``n_papers_cited`` : int64 — distinct focal papers with ≥1 valid citation.
        ``median_lag`` : float — median years to citation (the headline speed).
        ``mean_lag`` : float
        ``iqr_lag`` : float — ``q75 − q25`` of the lag (spread / tail).
        ``q25_lag`` / ``q75_lag`` : float
        ``frac_within_2y`` : float — share of citations arriving within 2 years
            (the "early-impact" fraction).
        ``velocity`` : str — ``"fast"`` if the journal's ``median_lag`` is below the
            panel median of the per-journal medians, else ``"slow"`` (a relative
            indicator; ties → ``"slow"``).

    Notes
    -----
    The lag is *receive-side* (how long after publication the paper is cited),
    independent of the cascade's first-appearance year, so it is not affected by the
    1995 origin censoring. Papers with no inbound citation contribute no rows and are
    absent from the table (report coverage separately). Empty input yields an empty,
    well-formed frame.
    """
    import numpy as np
    import pandas as pd

    cols = [
        "journal_slug",
        "n_citations",
        "n_papers_cited",
        "median_lag",
        "mean_lag",
        "iqr_lag",
        "q25_lag",
        "q75_lag",
        "frac_within_2y",
        "velocity",
    ]
    need_meta = [focal_id_col, journal_col, pub_year_col]
    need_cit = [focal_id_col, citing_year_col]
    checks = ((paper_meta, need_meta, "paper_meta"), (citations, need_cit, "citations"))
    for frame, need, name in checks:
        miss = [c for c in need if c not in frame.columns]
        if miss:
            raise ValueError(f"{name} missing required column(s): {miss}")

    if citations.empty or paper_meta.empty:
        return pd.DataFrame({c: [] for c in cols})

    meta = paper_meta.loc[:, need_meta].dropna(subset=[focal_id_col]).copy()
    meta = meta.drop_duplicates(subset=[focal_id_col], keep="first")
    meta[pub_year_col] = pd.to_numeric(meta[pub_year_col], errors="coerce")
    meta = meta.dropna(subset=[journal_col, pub_year_col])

    cit = citations.loc[:, need_cit].dropna(subset=[focal_id_col]).copy()
    cit[citing_year_col] = pd.to_numeric(cit[citing_year_col], errors="coerce")
    cit = cit.dropna(subset=[citing_year_col])

    joined = cit.merge(meta, on=focal_id_col, how="inner")
    if joined.empty:
        return pd.DataFrame({c: [] for c in cols})

    joined["lag"] = (joined[citing_year_col] - joined[pub_year_col]).astype("float64")
    joined = joined.loc[(joined["lag"] >= 0) & (joined["lag"] <= float(max_lag))]
    if joined.empty:
        return pd.DataFrame({c: [] for c in cols})

    grp = joined.groupby(journal_col, dropna=False)
    summary = grp["lag"].agg(
        n_citations="size",
        median_lag="median",
        mean_lag="mean",
        q25_lag=lambda s: float(np.percentile(s, 25)),
        q75_lag=lambda s: float(np.percentile(s, 75)),
    )
    summary["iqr_lag"] = summary["q75_lag"] - summary["q25_lag"]
    summary["frac_within_2y"] = grp["lag"].apply(lambda s: float((s <= 2).mean()))
    summary["n_papers_cited"] = grp[focal_id_col].nunique()
    summary = summary.reset_index().rename(columns={journal_col: "journal_slug"})

    panel_median = float(summary["median_lag"].median())
    summary["velocity"] = np.where(summary["median_lag"] < panel_median, "fast", "slow")

    summary["n_citations"] = summary["n_citations"].astype("int64")
    summary["n_papers_cited"] = summary["n_papers_cited"].astype("int64")
    for c in ("median_lag", "mean_lag", "iqr_lag", "q25_lag", "q75_lag", "frac_within_2y"):
        summary[c] = summary[c].astype("float64")

    summary = summary.sort_values(["median_lag", "journal_slug"]).reset_index(drop=True)
    return summary[cols]
