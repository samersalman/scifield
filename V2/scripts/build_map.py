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
    """Provenance inputs for the record_run sidecar, routed to the active version."""
    out = REPO_ROOT / ("V2/data" if version == "v1" else f"V2/data_{version}")
    return {
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


# --------------------------------------------------------------------------- HTML assembly


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
</nav>
<main>
  <div id="t-landscape" class="tab-wrap">{sec_landscape}</div>
  <div id="t-cascade" class="tab-wrap" hidden>{sec_cascade}</div>
  <div id="t-roles" class="tab-wrap" hidden>{sec_roles}</div>
  <div id="t-origins" class="tab-wrap" hidden>{sec_origins}</div>
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
    return _PAGE_TEMPLATE.format(
        version_upper="V2" if version == "v1" else version.upper(),
        n_journals=N_JOURNALS,
        task="V2-S07" if version == "v1" else "V2-S09",
        date=pd.Timestamp.today().strftime("%Y-%m-%d"),
        sec_landscape=sec_landscape,
        sec_cascade=sec_cascade,
        sec_roles=sec_roles,
        sec_origins=sec_origins,
    )


def main(argv: list[str] | None = None) -> None:
    """Build the map for a data version, write the HTML + sidecar, print the path."""
    ap = argparse.ArgumentParser(description="Assemble the V2 cartography map v0 (static HTML).")
    ap.add_argument("--data-version", default="v1", help="v1 = frozen prove-phase; e.g. v2.")
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

    record_run(
        artifact_path=OUT_HTML,
        inputs=_inputs(version),
        config={
            "task": "V2-S07" if version == "v1" else "V2-S09",
            "data_version": version,
            "artifact": f"{OUT_DIR.name}/index.html",
            "grain": GRAIN,
            "n_journals": N_JOURNALS,
            "renderer": "plotly static HTML (CDN script), no server (streamlit/dash absent)",
            "panel_conditional": "origins are first-in-panel, not first in world",
            "left_censoring_1995": "early topics originate at the 1995 panel start",
        },
    )

    size_kb = OUT_HTML.stat().st_size / 1024
    n_divs = html.count("plotly-graph-div")
    print(f"\nWROTE map ({version}): {OUT_HTML}")
    print(f"  size: {size_kb:.1f} KB  |  plotly divs: {n_divs}  |  journals: {N_JOURNALS}")
    print(f"  sidecar: {OUT_HTML}.run.json")
    print(f"  open: double-click {OUT_HTML.relative_to(REPO_ROOT)} (no server needed)")
    assert not math.isnan(size_kb)


if __name__ == "__main__":
    main()
