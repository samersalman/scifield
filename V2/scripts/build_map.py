"""Assemble the V2 cartography **map v0** — a self-contained static HTML (V2-S07 / V2-S09).

This is the first artifact a domain reader can actually *explore*: it stitches the
cascade, role and novelty-origin tables produced by the earlier batches into a single
double-click-able HTML page. **No server.** streamlit and dash are deliberately not used
(not installed); we emit a standalone page of Plotly panels via ``fig.to_html`` assembled
into a tabbed template. The only network reference is the Plotly CDN ``<script src>`` the
*viewer's* browser fetches — there is no build-time network call.

Data-version aware (V2-S09): ``--data-version v1`` (default) builds the frozen 10-journal
prove-phase map to ``V2/map_v0/``; ``--data-version v2`` builds the 78-journal expansion
map to ``V2/map_v0_v2/`` (v1 left untouched). The mapio loaders read
``$SCIFIELD_DATA_VERSION``, which ``main()`` sets before any figure is built, so they pull
the matching version's tables under ``V2/data[_v]/{cascade,roles,origins}``.

Sections (labeled tabs)
-----------------------
a. **Topic landscape** — every leaf topic as a sized scatter coloured by its origin
   journal's **role** (size = topic size, x = origin year, y = cross-journal reach).
b. **Cascade flows** — the inter-journal lead-lag heatmap, the directed seeding network
   (nodes coloured by role), and the corpus adoption-by-breadth curve.
c. **Journal roles** — source/bridge/terminal component bars + citational-velocity bars.
d. **Novelty origins** — sector novelty, geography ranking, and the top recombinant topics.
e. **Trajectories** (v2 only) — descriptive per-topic state-space +5y share projections:
   direction counts, the biggest rising/falling movers, and an all-topic dropdown explorer.

Usage
-----
``.venv/bin/python V2/scripts/build_map.py [--data-version v2]``

$0 / read-only (reads version parquets + ``data/<v>/topic_hierarchy.parquet``; writes the
HTML + ``.run.json`` sidecar). No GPU, no DeepSeek, no build-time network.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from scifield.cartography import mapio
from scifield.repro import record_run

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAIN = "leaf"  # the map's primary grain (mid is the robustness check, not the display)

# Routing globals, set in main() from --data-version. v1 keeps its historical unversioned
# paths (V2/data, V2/map_v0) so the frozen prove-phase artifact stays put; other versions
# are namespaced (V2/data_<v>, V2/map_v0_<v>). N_JOURNALS is recomputed from the data.
OUT_DIR = REPO_ROOT / "V2/map_v0"
OUT_HTML = OUT_DIR / "index.html"
N_JOURNALS = 10

PLOTLY_KW = {"full_html": False, "include_plotlyjs": False, "default_height": "560px"}


def _inputs(version: str) -> dict[str, Path]:
    """Provenance inputs for the record_run sidecar, routed to the active version.

    The two trajectory parquets (V2-S10) are added only when present on disk so the v1
    prove-phase build (no trajectory layer) records the same input set as before.
    """
    out = REPO_ROOT / ("V2/data" if version == "v1" else f"V2/data_{version}")
    inputs = {
        "topic_hierarchy": REPO_ROOT / "data" / version / "topic_hierarchy.parquet",
        "lag_matrix": out / "cascade/lag_matrix.parquet",
        "origin_attribution": out / "cascade/origin_attribution.parquet",
        "diffusion_curves": out / "cascade/diffusion_curves.parquet",
        "role_scores": out / "roles/role_scores.parquet",
        "velocity": out / "roles/velocity.parquet",
        "sector_novelty": out / "origins/sector_novelty.parquet",
        "geo_novelty": out / "origins/geo_novelty.parquet",
        "recombination_by_topic": out / "origins/recombination_by_topic.parquet",
    }
    traj_summary = out / "trajectory/trajectory_summary.parquet"
    traj_series = out / "trajectory/trajectory_series.parquet"
    if traj_summary.exists() and traj_series.exists():
        inputs["trajectory_summary"] = traj_summary
        inputs["trajectory_series"] = traj_series
    return inputs


# --------------------------------------------------------------------------- (a) landscape


def fig_topic_landscape() -> go.Figure:
    """Sized scatter of every leaf topic, coloured by its origin journal's **role**.

    x = origin year (panel-conditional first appearance), y = cross-journal reach
    (``n_journals``), marker size = topic paper count, colour = origin journal's
    source/bridge/terminal role (legible at 78 journals, unlike a per-journal legend),
    hover = top-words + origin journal + recombination rate.
    """
    df = mapio.topic_landscape_frame(GRAIN)
    roles = mapio.load_roles(GRAIN).set_index("journal_slug")["role"].to_dict()
    role_color = mapio.ROLE_COLOR
    sizes = df["size"].fillna(1.0) ** 0.5
    sizes = 6 + 34 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)

    fig = go.Figure()
    df = df.rename(columns={"size": "n_papers_topic"})
    df["_role"] = df["origin_journal_slug"].map(lambda s: roles.get(s, "unknown"))
    for role in ("source", "bridge", "terminal", "unknown"):
        sub = df[df["_role"] == role]
        if sub.empty:
            continue
        hover = [
            f"<b>T{int(r.topic_id)}</b>: {r.label}"
            f"<br>origin: {mapio.journal_display(r.origin_journal_slug)} ({int(r.origin_year)})"
            f" — {role}"
            f"<br>reach: {int(r.n_journals)}/{N_JOURNALS} journals"
            f"<br>size: {int(r.n_papers_topic)} papers"
            f"<br>recombination rate: "
            f"{0.0 if pd.isna(r.recombination_rate) else r.recombination_rate:.2f}"
            for r in sub.itertuples()
        ]
        fig.add_trace(
            go.Scatter(
                x=sub["origin_year"],
                y=sub["n_journals"],
                mode="markers",
                name=role,
                marker={
                    "size": sizes.loc[sub.index],
                    "color": role_color.get(role, "#999999"),
                    "line": {"width": 0.5, "color": "white"},
                    "opacity": 0.8,
                },
                text=hover,
                hoverinfo="text",
            )
        )
    fig.update_layout(
        title=f"Topic landscape — {len(df)} leaf topics, coloured by origin-journal role",
        xaxis_title=f"origin year (first appearance in the {N_JOURNALS}-journal panel)",
        yaxis_title="cross-journal reach (n journals)",
        legend_title="origin role",
        template="plotly_white",
        margin={"t": 60, "b": 50, "l": 60, "r": 20},
    )
    return fig


# --------------------------------------------------------------------------- (b) cascade


def fig_lag_heatmap() -> go.Figure:
    """Heatmap of the signed inter-journal mean lead-lag (journal_i leads journal_j > 0)."""
    wide = mapio.lag_matrix_wide(GRAIN)
    disp = [mapio.journal_display(s) for s in wide.index]
    fig = go.Figure(
        go.Heatmap(
            z=wide.values,
            x=disp,
            y=disp,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "mean lag (yr)"},
            hovertemplate="%{y} vs %{x}<br>mean lag: %{z:.2f} yr<extra></extra>",
        )
    )
    fig.update_layout(
        title="Inter-journal lead-lag (row leads column when blue / positive)",
        template="plotly_white",
        xaxis_title="follower (journal_j)",
        yaxis_title="leader (journal_i)",
        margin={"t": 60, "b": 80, "l": 120, "r": 20},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_cascade_network() -> go.Figure:
    """Directed seeding network: nodes = journals (coloured by role), edges = lead→follow.

    Edge weight is the net directed lead asymmetry from the lag matrix, drawn only for the
    dominant direction of each pair; node colour is the role label, node size scales with
    the journal's net-outflow share.
    """
    lag = mapio.load_lag_matrix(GRAIN)
    roles = mapio.load_roles(GRAIN).set_index("journal_slug")

    g = nx.DiGraph()
    for slug in roles.index:
        g.add_node(slug)
    for r in lag.itertuples():
        if r.mean_lag > 0 and r.n_shared >= 5:
            g.add_edge(r.journal_i, r.journal_j, weight=float(r.mean_lag))

    pos = nx.circular_layout(sorted(g.nodes()))

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v, _d in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 0.8, "color": "rgba(120,120,120,0.45)"},
        hoverinfo="none",
        showlegend=False,
    )

    role_color = mapio.ROLE_COLOR
    node_traces: list[go.Scatter] = []
    for role in ("source", "bridge", "terminal"):
        members = [s for s in g.nodes() if roles.loc[s, "role"] == role]
        if not members:
            continue
        share = roles.loc[members, "net_outflow_share"].abs()
        size = 18 + 40 * (share - share.min()) / (share.max() - share.min() + 1e-9)
        node_traces.append(
            go.Scatter(
                x=[pos[s][0] for s in members],
                y=[pos[s][1] for s in members],
                mode="markers+text",
                name=role,
                marker={
                    "size": size,
                    "color": role_color[role],
                    "line": {"width": 1, "color": "white"},
                },
                text=[mapio.journal_display(s) for s in members],
                textposition="bottom center",
                textfont={"size": 9},
                hovertext=[
                    f"{mapio.journal_display(s)}<br>role: {role}"
                    f"<br>net-outflow share: {roles.loc[s, 'net_outflow_share']:.3f}"
                    f"<br>betweenness: {roles.loc[s, 'betweenness']:.3f}"
                    for s in members
                ],
                hoverinfo="text",
            )
        )

    fig = go.Figure([edge_trace, *node_traces])
    fig.update_layout(
        title="Directed seeding network (edge = lead→follow; node colour = role)",
        template="plotly_white",
        legend_title="role",
        showlegend=True,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"t": 60, "b": 30, "l": 20, "r": 20},
    )
    return fig


def fig_adoption_curve() -> go.Figure:
    """Corpus adoption-by-breadth curve: fraction of topics reaching >= k journals."""
    ac = mapio.adoption_curve(GRAIN)
    fig = go.Figure(
        go.Scatter(
            x=ac["n_journals_reached"],
            y=ac["frac_topics"],
            mode="lines+markers",
            line={"color": "#4285f4", "width": 2},
            marker={"size": 7},
            hovertemplate="reach >= %{x} journals<br>fraction of topics: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Cross-journal adoption — fraction of topics reaching at least k journals",
        xaxis_title="journals reached (k)",
        yaxis_title="fraction of topics",
        template="plotly_white",
        yaxis={"range": [0, 1.02]},
        margin={"t": 60, "b": 50, "l": 60, "r": 20},
    )
    return fig


# --------------------------------------------------------------------------- (c) roles


def fig_role_components() -> go.Figure:
    """Grouped bars of the source/bridge/terminal component scores per journal."""
    roles = mapio.load_roles(GRAIN).sort_values("source", ascending=False)
    fig = go.Figure()
    for comp in ("source", "bridge", "terminal"):
        fig.add_trace(
            go.Bar(
                name=comp,
                x=roles["display"],
                y=roles[comp],
                marker_color=mapio.ROLE_COLOR[comp],
                hovertemplate="%{x}<br>" + comp + ": %{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Journal role components (sorted by source score; argmax drives the label)",
        barmode="group",
        template="plotly_white",
        xaxis_title="journal",
        yaxis_title="component score",
        legend_title="component",
        margin={"t": 60, "b": 140, "l": 60, "r": 20},
    )
    fig.update_xaxes(tickangle=-60)
    return fig


def fig_velocity() -> go.Figure:
    """Per-journal median time-to-citation (years), coloured by fast/slow."""
    vel = mapio.load_velocity().sort_values("median_lag")
    color = ["#34a853" if v == "fast" else "#9aa0a6" for v in vel["velocity"]]
    fig = go.Figure(
        go.Bar(
            x=vel["display"],
            y=vel["median_lag"],
            marker_color=color,
            customdata=vel[["frac_within_2y", "n_citations"]].values,
            hovertemplate=(
                "%{x}<br>median lag: %{y:.0f} yr"
                "<br>within 2y: %{customdata[0]:.1%}"
                "<br>citations: %{customdata[1]:,}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Citational velocity — median years to citation (green = fast)",
        template="plotly_white",
        xaxis_title="journal",
        yaxis_title="median time-to-citation (yr)",
        margin={"t": 60, "b": 140, "l": 60, "r": 20},
    )
    fig.update_xaxes(tickangle=-60)
    return fig


# --------------------------------------------------------------------------- (d) origins


def fig_sector_novelty() -> go.Figure:
    """Sector semantic-novelty bars — honestly presented as a near-null (company mid-pack)."""
    sector = mapio.load_origins()["sector"].sort_values("sem_nov_mean_mean", ascending=False)
    color = ["#ea4335" if t == "company" else "#9aa0a6" for t in sector["type"]]
    fig = go.Figure(
        go.Bar(
            x=sector["type"],
            y=sector["sem_nov_mean_mean"],
            marker_color=color,
            customdata=sector[["n_papers"]].values,
            hovertemplate="%{x}<br>mean sem novelty: %{y:.4f}"
            "<br>n papers: %{customdata[0]:,}<extra></extra>",
        )
    )
    company = sector[sector["type"] == "company"]["sem_nov_mean_mean"]
    fig.update_layout(
        title="Novelty by sector — company (red) sits mid-pack: a near-null contrast",
        template="plotly_white",
        xaxis_title="institution type",
        yaxis_title="mean semantic novelty",
        margin={"t": 60, "b": 70, "l": 60, "r": 20},
    )
    if len(company):
        lo = float(sector["sem_nov_mean_mean"].min())
        hi = float(sector["sem_nov_mean_mean"].max())
        fig.update_yaxes(range=[lo - 0.005, hi + 0.005])  # zoom to show the flatness honestly
    return fig


def fig_geo_novelty(top_n: int = 15) -> go.Figure:
    """Top + bottom well-sampled countries by semantic novelty (most-differentiated axis)."""
    geo = mapio.geo_ranked()
    top = geo.head(top_n)
    bottom = geo.tail(top_n)
    band = pd.concat([top, bottom]).drop_duplicates("country_code")
    band = band.sort_values("sem_nov_mean_mean", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=band["sem_nov_mean_mean"],
            y=band["country_code"],
            orientation="h",
            marker_color="#4285f4",
            customdata=band[["n_papers"]].values,
            hovertemplate="%{y}<br>mean sem novelty: %{x:.3f}"
            "<br>n papers: %{customdata[0]:,}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Novelty by geography — top {top_n} vs bottom {top_n} well-sampled countries",
        template="plotly_white",
        xaxis_title="mean semantic novelty",
        yaxis_title="country",
        height=620,
        margin={"t": 60, "b": 50, "l": 60, "r": 20},
    )
    return fig


def fig_recombination_topics(top_n: int = 15) -> go.Figure:
    """Top recombinant topics by recombination rate (interdisciplinary bridges)."""
    rbt = mapio.load_origins()["recombination_by_topic"]
    top = rbt.nlargest(top_n, "recombination_rate").sort_values("recombination_rate")
    labels = [
        f"T{int(t)}: {mapio._join_words(w, n=4)}"
        for t, w in zip(top["topic_id"], top["top_words"], strict=False)
    ]
    fig = go.Figure(
        go.Bar(
            x=top["recombination_rate"],
            y=labels,
            orientation="h",
            marker_color="#7b1fa2",
            customdata=top[["mean_distinct_topics"]].values,
            hovertemplate="%{y}<br>recombination rate: %{x:.2f}"
            "<br>mean distinct ref-topics: %{customdata[0]:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} recombinant topics (fraction of papers citing >= 2 topics)",
        template="plotly_white",
        xaxis_title="recombination rate",
        yaxis_title="topic",
        height=620,
        margin={"t": 60, "b": 50, "l": 60, "r": 20},
    )
    return fig


# ------------------------------------------------------------------- (e) trajectories


def fig_trajectory_directions() -> go.Figure:
    """Bar chart of leaf-topic trajectory direction counts (rising / flat / falling)."""
    summ = mapio.load_trajectory_summary(GRAIN)
    order = ["rising", "flat", "falling"]
    color = {"rising": "#34a853", "flat": "#9aa0a6", "falling": "#ea4335"}
    counts = summ["direction"].value_counts()
    x = [d for d in order if d in counts.index] or list(counts.index)
    y = [int(counts.get(d, 0)) for d in x]
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            marker_color=[color.get(d, "#9aa0a6") for d in x],
            hovertemplate="%{x}<br>topics: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Projected direction of {len(summ)} leaf topics (+5y share slope sign)",
        template="plotly_white",
        xaxis_title="projected direction",
        yaxis_title="number of topics",
        margin={"t": 60, "b": 50, "l": 60, "r": 20},
    )
    return fig


def _traj_label(topic_id: int, label: str, *, n: int = 40) -> str:
    """Compact ``T<id>: <truncated label>`` tag for dropdowns / legends."""
    lab = label if isinstance(label, str) else ""
    if len(lab) > n:
        lab = lab[: n - 1].rstrip() + "…"
    return f"T{int(topic_id)}: {lab}"


def fig_trajectory_movers(top_n: int = 8) -> go.Figure:
    """Fan charts for the top-N rising + top-N falling leaf topics by share slope.

    Each topic gets a solid observed-share line and a dashed projected-share line with an
    80% band (``share_lo``..``share_hi`` fill). Rising movers are greens, falling movers
    are reds (legend grouped) so the two regimes read at a glance.
    """
    summ = mapio.load_trajectory_summary(GRAIN)
    series = mapio.load_trajectory_series(GRAIN)
    risers = summ.nlargest(top_n, "slope_share_per_yr")
    fallers = summ.nsmallest(top_n, "slope_share_per_yr")
    greens = ["#0b6e2e", "#1b9e4b", "#34a853", "#5cb87a", "#7fc99a", "#0f7a36", "#2bb05f"]
    reds = ["#8b1a12", "#b3261b", "#ea4335", "#f06b60", "#c9342a", "#a01f16", "#d84236"]

    fig = go.Figure()
    for movers, palette, kind_lbl in ((risers, greens, "rising"), (fallers, reds, "falling")):
        for i, row in enumerate(movers.itertuples()):
            col = palette[i % len(palette)]
            tser = series[series["topic_id"] == row.topic_id].sort_values("year")
            obs = tser[tser["kind"] == "observed"]
            proj = tser[tser["kind"] == "projected"]
            name = _traj_label(row.topic_id, row.label)
            gid = f"{kind_lbl}-{int(row.topic_id)}"
            # Observed share: solid line.
            fig.add_trace(
                go.Scatter(
                    x=obs["year"],
                    y=obs["share"],
                    mode="lines",
                    name=name,
                    legendgroup=gid,
                    line={"color": col, "width": 1.8},
                    hovertemplate=f"{name}<br>%{{x}}: share %{{y:.3f}}<extra></extra>",
                )
            )
            # Projected share: dashed line, bridged from the last observed point.
            if not proj.empty:
                bridge = pd.concat([obs.tail(1), proj], ignore_index=True)
                fig.add_trace(
                    go.Scatter(
                        x=bridge["year"],
                        y=bridge["share"],
                        mode="lines",
                        name=name,
                        legendgroup=gid,
                        showlegend=False,
                        line={"color": col, "width": 1.8, "dash": "dash"},
                        hovertemplate=f"{name} (proj)<br>%{{x}}: share %{{y:.3f}}<extra></extra>",
                    )
                )
                # 80% band: hi then lo with fill='tonexty' over the projected horizon.
                fig.add_trace(
                    go.Scatter(
                        x=proj["year"],
                        y=proj["share_hi"],
                        mode="lines",
                        name=name,
                        legendgroup=gid,
                        showlegend=False,
                        line={"color": col, "width": 0},
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=proj["year"],
                        y=proj["share_lo"],
                        mode="lines",
                        name=name,
                        legendgroup=gid,
                        showlegend=False,
                        line={"color": col, "width": 0},
                        fill="tonexty",
                        fillcolor=_rgba(col, 0.15),
                        hoverinfo="skip",
                    )
                )
    fig.update_layout(
        title=f"Biggest movers — top {top_n} rising (green) + top {top_n} falling (red) topics",
        template="plotly_white",
        xaxis_title="year (solid = observed, dashed = projected +5y, band = 80%)",
        yaxis_title="panel-conditional share",
        legend_title="topic (click to toggle)",
        height=640,
        margin={"t": 60, "b": 50, "l": 60, "r": 20},
    )
    return fig


def fig_trajectory_explorer() -> go.Figure:
    """One figure with an ``updatemenus`` dropdown over ALL leaf topics (by topic_id).

    Selecting a topic shows its observed+projected share with the 80% band. To keep the
    HTML small (~149 topics), each topic is rendered as just THREE traces — observed line,
    projected dashed line (bridged), and a single filled band trace built as a closed
    polygon (hi forward then lo reversed) — and the dropdown toggles the ``visible`` array
    three-at-a-time. The first topic is shown by default.
    """
    summ = mapio.load_trajectory_summary(GRAIN).sort_values("topic_id")
    series = mapio.load_trajectory_series(GRAIN)
    color = "#1a73e8"
    band_color = _rgba(color, 0.15)

    traces_per_topic = 3
    topic_ids = list(summ["topic_id"])
    labels = {int(r.topic_id): r.label for r in summ.itertuples()}

    for idx, tid in enumerate(topic_ids):
        visible = idx == 0
        tser = series[series["topic_id"] == tid].sort_values("year")
        obs = tser[tser["kind"] == "observed"]
        proj = tser[tser["kind"] == "projected"]
        name = _traj_label(tid, labels[int(tid)])
        # Band as a single closed polygon trace (hi forward, lo reversed) — 1 trace/topic.
        band_x: list = []
        band_y: list = []
        if not proj.empty:
            band_x = list(proj["year"]) + list(proj["year"][::-1])
            band_y = list(proj["share_hi"]) + list(proj["share_lo"][::-1])
        fig_band = go.Scatter(
            x=band_x,
            y=band_y,
            mode="lines",
            fill="toself",
            fillcolor=band_color,
            line={"width": 0},
            name=f"{name} 80% band",
            showlegend=False,
            hoverinfo="skip",
            visible=visible,
        )
        obs_trace = go.Scatter(
            x=obs["year"],
            y=obs["share"],
            mode="lines",
            line={"color": color, "width": 2},
            name="observed",
            showlegend=False,
            hovertemplate="%{x}: share %{y:.3f}<extra></extra>",
            visible=visible,
        )
        bridge = pd.concat([obs.tail(1), proj], ignore_index=True) if not proj.empty else proj
        proj_trace = go.Scatter(
            x=bridge["year"] if not bridge.empty else [],
            y=bridge["share"] if not bridge.empty else [],
            mode="lines",
            line={"color": color, "width": 2, "dash": "dash"},
            name="projected",
            showlegend=False,
            hovertemplate="%{x}: proj share %{y:.3f}<extra></extra>",
            visible=visible,
        )
        if idx == 0:
            fig = go.Figure([fig_band, obs_trace, proj_trace])
        else:
            fig.add_traces([fig_band, obs_trace, proj_trace])

    n_topics = len(topic_ids)
    buttons = []
    for idx, tid in enumerate(topic_ids):
        vis = [False] * (n_topics * traces_per_topic)
        for j in range(traces_per_topic):
            vis[idx * traces_per_topic + j] = True
        buttons.append(
            {
                "label": _traj_label(tid, labels[int(tid)], n=34),
                "method": "update",
                "args": [
                    {"visible": vis},
                    {"title": f"Trajectory — {_traj_label(tid, labels[int(tid)], n=60)}"},
                ],
            }
        )

    first = topic_ids[0]
    fig.update_layout(
        title=f"Trajectory — {_traj_label(first, labels[int(first)], n=60)}",
        template="plotly_white",
        xaxis_title="year (solid = observed, dashed = projected +5y, band = 80%)",
        yaxis_title="panel-conditional share",
        height=560,
        margin={"t": 90, "b": 50, "l": 60, "r": 20},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.16,
                "yanchor": "top",
            }
        ],
    )
    return fig


# --------------------------------------------------------------------------- HTML assembly


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba(r,g,b,alpha)`` string for translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _div(fig: go.Figure) -> str:
    """Render a figure to an embeddable Plotly div (no full html, shared CDN)."""
    return str(fig.to_html(**PLOTLY_KW))


# Version-generic caveats (drop version-specific numbers so the text is correct for any
# panel size; the precise statistics live in V2/docs/cartography/*.md).
_CASCADE_CAVEAT = (
    "Cascade caveats: origins are <b>panel-conditional</b> ('first in this journal panel', "
    "not first in the world) and the corpus is <b>1995 left-censored</b> (early topics read "
    "as 'present at panel start', not 'born 1995'). The cascade is <b>validation-REAL</b>: "
    "the gate-critical S1/Null-1 seeding-spread permutation PASSes at both grains and on the "
    "resolvable subset, the drop-one-journal origin jackknife flip-rate = 0, and the leaf-vs-"
    "mid granularity check holds. The held-out-years check is supporting-only."
)
_ROLE_CAVEAT = (
    "Roles are jackknife-stable (role_rank_correlation well above the 0.7 bar) and robust "
    "across grain. Caveat: <b>'terminal' = within-panel citation/flow sink, not a global "
    "dead-end</b>; high-impact clinical journals consolidate work (net inflow) while the "
    "source role belongs to basic-science + specialty venues. The source/bridge/terminal "
    "label is a coarse argmax; the continuous component scores are the nuanced output."
)
_ORIGIN_CAVEAT = (
    "Sector novelty is essentially a <b>null</b> (company sits mid-pack; novelty is not a "
    "tech/industry phenomenon here). <b>Geography is the most differentiated axis</b> "
    "(descriptive, likely topic-mix driven). Recombination is a corpus-internal <b>lower "
    "bound</b>. <b>Funding and citation-intent layers are deferred</b> (coverage gaps; $0 "
    "session). Institution type/country blanks are excluded, never imputed."
)
_LANDSCAPE_CAVEAT = (
    "Each point is a leaf topic; x = panel-conditional origin year, y = cross-journal reach, "
    "size = paper count, colour = the origin journal's source/bridge/terminal role. Hover "
    "for top-words + origin journal + recombination rate. Same panel-conditional + "
    "1995-censoring caveats as the cascade tab."
)
_TRAJECTORY_CAVEAT = (
    "Trajectories are a <b>descriptive</b> per-topic state-space model (a local-linear-trend "
    "fitted on <b>logit(share)</b> / <b>log(volume)</b>) projected <b>+5y → 2030</b> with "
    "<b>80% uncertainty bands</b>; all <b>149/149</b> topics fit OK. The fit window is "
    "<b>1995&ndash;2025</b> with <b>2026 excluded</b> (a partial harvest year). 'Share' is "
    "<b>panel-conditional</b> — the share of the 78-journal annual assigned output, not a "
    "global field share. This is a state-space extrapolation of past trend, <b>explicitly "
    "NOT a predictive/causal forecast</b> and <b>NOT the F3 emergence GNN</b> (which was "
    "signed NULL at Gate G4). Read the bands, not the point line."
)


def _section(title: str, caveat: str, *divs: str) -> str:
    """One labeled section: heading, caveat note, then the stacked figure divs."""
    body = "\n".join(f'<div class="panel">{d}</div>' for d in divs)
    return (
        f'<section class="tabpanel">\n'
        f"<h2>{title}</h2>\n"
        f'<p class="caveat">{caveat}</p>\n'
        f"{body}\n"
        f"</section>"
    )


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SciField {version_upper} — Literature Cartography Map v0</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          margin: 0; color: #202124; background: #f7f8fa; }}
  header {{ background: #1a237e; color: white; padding: 18px 28px; }}
  header h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
  header p {{ margin: 0; font-size: 13px; opacity: 0.9; max-width: 1100px; }}
  nav {{ display: flex; gap: 4px; background: #283593; padding: 0 20px; }}
  nav button {{ background: transparent; border: none; color: #c5cae9; padding: 12px 18px;
                font-size: 14px; cursor: pointer; border-bottom: 3px solid transparent; }}
  nav button.active {{ color: white; border-bottom-color: #fbbc04; font-weight: 600; }}
  main {{ padding: 18px 28px 60px 28px; max-width: 1180px; margin: 0 auto; }}
  .tabpanel {{ display: none; }}
  .tabpanel.active {{ display: block; }}
  .tabpanel h2 {{ font-size: 19px; margin: 8px 0 4px 0; }}
  .caveat {{ font-size: 13px; color: #5f6368; background: #fff8e1; border-left: 4px solid
             #fbbc04; padding: 10px 14px; border-radius: 4px; line-height: 1.45; }}
  .panel {{ background: white; border: 1px solid #e0e0e0; border-radius: 6px;
            margin: 16px 0; padding: 6px; }}
  footer {{ font-size: 12px; color: #80868b; padding: 18px 28px; border-top: 1px solid #e0e0e0;
            max-width: 1180px; margin: 0 auto; }}
</style>
</head>
<body>
<header>
  <h1>SciField {version_upper} — Literature Cartography Map v0</h1>
  <p>A <b>$0</b> exploratory cartography on the <b>{n_journals}-journal</b> corpus
     (1995&ndash;2025) — not a hypothesis test. Every origin and cascade claim is
     <b>panel-conditional</b> (first in this {n_journals}-journal panel, not first in the
     world) and the corpus is <b>1995 left-censored</b>. Built {date} ({task}).</p>
</header>
<nav>
  <button class="tab-btn active" data-tab="t-landscape">a. Topic landscape</button>
  <button class="tab-btn" data-tab="t-cascade">b. Cascade flows</button>
  <button class="tab-btn" data-tab="t-roles">c. Journal roles</button>
  <button class="tab-btn" data-tab="t-origins">d. Novelty origins</button>
  {traj_nav_button}
</nav>
<main>
  <div id="t-landscape" class="tab-wrap">{sec_landscape}</div>
  <div id="t-cascade" class="tab-wrap" hidden>{sec_cascade}</div>
  <div id="t-roles" class="tab-wrap" hidden>{sec_roles}</div>
  <div id="t-origins" class="tab-wrap" hidden>{sec_origins}</div>
  {traj_wrap}
</main>
<footer>
  SciField {version_upper} cartography &middot; map v0 &middot; {n_journals} journals. Data:
  V2/data[_v]/&#123;cascade,roles,origins&#125; + data/&#123;v&#125;/topic_hierarchy.
  Reproducibility sidecar: index.html.run.json. Full statistics + caveats:
  V2/docs/cartography/.</footer>
<script>
  const wraps = document.querySelectorAll('.tab-wrap');
  const panels = document.querySelectorAll('.tabpanel');
  panels.forEach(p => p.classList.add('active'));  // panels visible within their wrap
  document.querySelectorAll('.tab-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      wraps.forEach(w => {{ w.hidden = (w.id !== btn.dataset.tab); }});
      window.dispatchEvent(new Event('resize'));  // re-flow Plotly on tab show
    }});
  }});
</script>
</body>
</html>
"""


def build_html(version: str) -> str:
    """Assemble all four sections into the single self-contained HTML string."""
    sec_landscape = _section("a. Topic landscape", _LANDSCAPE_CAVEAT, _div(fig_topic_landscape()))
    sec_cascade = _section(
        "b. Cascade flows",
        _CASCADE_CAVEAT,
        _div(fig_lag_heatmap()),
        _div(fig_cascade_network()),
        _div(fig_adoption_curve()),
    )
    sec_roles = _section(
        "c. Journal roles",
        _ROLE_CAVEAT,
        _div(fig_role_components()),
        _div(fig_velocity()),
    )
    sec_origins = _section(
        "d. Novelty origins",
        _ORIGIN_CAVEAT,
        _div(fig_sector_novelty()),
        _div(fig_geo_novelty()),
        _div(fig_recombination_topics()),
    )

    # 5th tab is conditional: v1 has no trajectory layer, so the loaders raise
    # FileNotFoundError — in that case omit the button + wrap entirely (4-tab page).
    traj_nav_button = ""
    traj_wrap = ""
    try:
        sec_trajectory = _section(
            "e. Trajectories",
            _TRAJECTORY_CAVEAT,
            _div(fig_trajectory_directions()),
            _div(fig_trajectory_movers()),
            _div(fig_trajectory_explorer()),
        )
        traj_nav_button = '<button class="tab-btn" data-tab="t-trajectory">e. Trajectories</button>'
        traj_wrap = f'<div id="t-trajectory" class="tab-wrap" hidden>{sec_trajectory}</div>'
    except FileNotFoundError:
        pass  # no trajectory parquets (e.g. v1) → stay at 4 tabs

    return _PAGE_TEMPLATE.format(
        version_upper="V2" if version == "v1" else version.upper(),
        n_journals=N_JOURNALS,
        task="V2-S07" if version == "v1" else "V2-S09",
        date=pd.Timestamp.today().strftime("%Y-%m-%d"),
        sec_landscape=sec_landscape,
        sec_cascade=sec_cascade,
        sec_roles=sec_roles,
        sec_origins=sec_origins,
        traj_nav_button=traj_nav_button,
        traj_wrap=traj_wrap,
    )


def main(argv: list[str] | None = None) -> None:
    """Build the map for a data version, write the HTML + sidecar, print the path.

    With ``--publish-to PATH`` the identical HTML is also written to ``PATH/index.html``
    (plus a ``record_run`` sidecar there) — e.g. the public ``docs/cartography/map`` site.
    """
    ap = argparse.ArgumentParser(description="Assemble the V2 cartography map v0 (static HTML).")
    ap.add_argument("--data-version", default="v1", help="v1 = frozen prove-phase; e.g. v2.")
    ap.add_argument(
        "--publish-to",
        default=None,
        help="optional dir to ALSO write index.html + sidecar to (e.g. docs/cartography/map).",
    )
    args = ap.parse_args(argv)
    version = args.data_version
    os.environ["SCIFIELD_DATA_VERSION"] = version  # mapio loaders read this

    global OUT_DIR, OUT_HTML, N_JOURNALS
    OUT_DIR = REPO_ROOT / ("V2/map_v0" if version == "v1" else f"V2/map_v0_{version}")
    OUT_HTML = OUT_DIR / "index.html"
    N_JOURNALS = int(len(mapio.load_roles(GRAIN)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html = build_html(version)
    OUT_HTML.write_text(html, encoding="utf-8")

    config = {
        "task": "V2-S07" if version == "v1" else "V2-S09",
        "data_version": version,
        "artifact": f"{OUT_DIR.name}/index.html",
        "grain": GRAIN,
        "n_journals": N_JOURNALS,
        "renderer": "plotly static HTML (CDN script), no server (streamlit/dash absent)",
        "panel_conditional": "origins are first-in-panel, not first in world",
        "left_censoring_1995": "early topics originate at the 1995 panel start",
    }
    record_run(artifact_path=OUT_HTML, inputs=_inputs(version), config=config)

    size_kb = OUT_HTML.stat().st_size / 1024
    n_divs = html.count("plotly-graph-div")
    print(f"\nWROTE map ({version}): {OUT_HTML}")
    print(f"  size: {size_kb:.1f} KB  |  plotly divs: {n_divs}  |  journals: {N_JOURNALS}")
    print(f"  sidecar: {OUT_HTML}.run.json")
    print(f"  open: double-click {OUT_HTML.relative_to(REPO_ROOT)} (no server needed)")

    if args.publish_to is not None:
        pub_dir = Path(args.publish_to)
        if not pub_dir.is_absolute():
            pub_dir = REPO_ROOT / pub_dir
        pub_dir.mkdir(parents=True, exist_ok=True)
        pub_html = pub_dir / "index.html"
        pub_html.write_text(html, encoding="utf-8")
        pub_config = dict(config)
        pub_config["published_to"] = str(pub_dir.relative_to(REPO_ROOT))
        record_run(artifact_path=pub_html, inputs=_inputs(version), config=pub_config)
        print(f"  PUBLISHED copy: {pub_html}")
        print(f"  sidecar: {pub_html}.run.json")

    assert not math.isnan(size_kb)


if __name__ == "__main__":
    main()
