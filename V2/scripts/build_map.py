"""Assemble the V2 cartography **map v0** — a self-contained static HTML (V2-S07).

This is the first artifact a domain reader can actually *explore*: it stitches the
cascade, role and novelty-origin tables produced by the earlier batches into a single
double-click-able HTML page. **No server.** streamlit and dash are deliberately not used
(not installed); we follow the ``docs/figures/topic_landscape.html`` precedent and emit a
standalone page of Plotly panels via ``fig.to_html(full_html=False,
include_plotlyjs="cdn")`` assembled into a tabbed template. The only network reference is
the Plotly CDN ``<script src>`` the *viewer's* browser fetches — there is no build-time
network call.

Sections (labeled tabs)
-----------------------
a. **Topic landscape** — every leaf topic as a sized, origin-coloured scatter
   (size = topic size, x = origin year, y = cross-journal reach), hover = top-words +
   recombination rate.
b. **Cascade flows** — the inter-journal lead-lag heatmap (signed mean lag), the directed
   seeding network (nodes coloured by S05 role), and the corpus adoption-by-breadth curve.
c. **Journal roles** — the 10 journals' source/bridge/terminal component bars + the
   citational-velocity (median time-to-citation) bars.
d. **Novelty origins** — sector novelty (honest near-null), geography (ranked bar, the
   most-differentiated axis) and the top recombinant topics.

Every panel carries the locked caveats (panel-conditional origin, 1995 left-censoring,
terminal = within-panel citation sink, sector ≈ null, funding/intent deferred) and the
cascade's S04 REAL-with-qualification verdict.

Data prep lives in :mod:`scifield.cartography.mapio` (pure, tested); this script owns the
figure assembly + HTML write + the ``record_run`` sidecar.

Usage
-----
``.venv/bin/python V2/scripts/build_map.py``

$0 / read-only: reads only the V2 ``data/`` parquets + ``data/v1/topic_hierarchy.parquet``;
writes ``V2/map_v0/index.html`` + its ``.run.json`` sidecar. No GPU, no DeepSeek, no
build-time network.
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from scifield.cartography import mapio
from scifield.repro import record_run

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "V2/map_v0"
OUT_HTML = OUT_DIR / "index.html"

GRAIN = "leaf"  # the map's primary grain (mid is the robustness check, not the display)

# Inputs (for the record_run sidecar + the README data-provenance).
INPUTS: dict[str, Path] = {
    "topic_hierarchy": REPO_ROOT / "data/v1/topic_hierarchy.parquet",
    "lag_matrix": REPO_ROOT / "V2/data/cascade/lag_matrix.parquet",
    "origin_attribution": REPO_ROOT / "V2/data/cascade/origin_attribution.parquet",
    "diffusion_curves": REPO_ROOT / "V2/data/cascade/diffusion_curves.parquet",
    "role_scores": REPO_ROOT / "V2/data/roles/role_scores.parquet",
    "velocity": REPO_ROOT / "V2/data/roles/velocity.parquet",
    "sector_novelty": REPO_ROOT / "V2/data/origins/sector_novelty.parquet",
    "geo_novelty": REPO_ROOT / "V2/data/origins/geo_novelty.parquet",
    "recombination_by_topic": REPO_ROOT / "V2/data/origins/recombination_by_topic.parquet",
}

PLOTLY_KW = {"full_html": False, "include_plotlyjs": False, "default_height": "560px"}

# A stable, qualitative colour map for the 10 journals (origin colouring on the landscape).
_JOURNAL_PALETTE = [
    "#4285f4",
    "#ea4335",
    "#fbbc04",
    "#34a853",
    "#ff6d01",
    "#46bdc6",
    "#7b1fa2",
    "#a52714",
    "#0097a7",
    "#616161",
]


def _journal_colors() -> dict[str, str]:
    """Stable slug -> hex colour mapping over the 10 canonical journals."""
    slugs = sorted(mapio.JOURNAL_DISPLAY)
    return {slug: _JOURNAL_PALETTE[i % len(_JOURNAL_PALETTE)] for i, slug in enumerate(slugs)}


# --------------------------------------------------------------------------- (a) landscape


def fig_topic_landscape() -> go.Figure:
    """Sized, origin-coloured scatter of every leaf topic.

    x = origin year (panel-conditional first appearance), y = cross-journal reach
    (``n_journals``), marker size = topic paper count, colour = origin journal, hover =
    top-words + recombination rate.
    """
    df = mapio.topic_landscape_frame(GRAIN)
    colors = _journal_colors()
    sizes = df["size"].fillna(1.0) ** 0.5
    sizes = 6 + 34 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)

    fig = go.Figure()
    # Rename ``size`` (collides with the namedtuple ``.size`` attr) for hover building.
    df = df.rename(columns={"size": "n_papers_topic"})
    for slug, sub in df.groupby("origin_journal_slug", dropna=False):
        disp = mapio.journal_display(slug) if isinstance(slug, str) else "unknown"
        idx = sub.index
        hover = [
            f"<b>T{int(r.topic_id)}</b>: {r.label}"
            f"<br>origin: {disp} ({int(r.origin_year)})"
            f"<br>reach: {int(r.n_journals)}/10 journals"
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
                name=disp,
                marker={
                    "size": sizes.loc[idx],
                    "color": colors.get(slug, "#999999"),
                    "line": {"width": 0.5, "color": "white"},
                    "opacity": 0.8,
                },
                text=hover,
                hoverinfo="text",
            )
        )
    fig.update_layout(
        title="Topic landscape — 149 leaf topics by panel-conditional origin",
        xaxis_title="origin year (first appearance in the 10-journal panel)",
        yaxis_title="cross-journal reach (n journals)",
        legend_title="origin journal",
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
    """Directed seeding network: nodes = 10 journals (coloured by role), edges = lead→follow.

    Edge weight is the net directed lead asymmetry from the lag matrix
    (``n_i_leads`` − reciprocal), drawn only for the dominant direction of each pair; node
    colour is the S05 role label, node size scales with the journal's net-outflow share.
    """
    lag = mapio.load_lag_matrix(GRAIN)
    roles = mapio.load_roles(GRAIN).set_index("journal_slug")

    # Build a directed graph: i -> j when journal_i leads journal_j on average (mean_lag>0).
    g = nx.DiGraph()
    for slug in roles.index:
        g.add_node(slug)
    for r in lag.itertuples():
        if r.mean_lag > 0 and r.n_shared >= 5:
            g.add_edge(r.journal_i, r.journal_j, weight=float(r.mean_lag))

    pos = nx.circular_layout(sorted(g.nodes()))

    # Edges as line segments (width ~ lead magnitude).
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

    # Nodes coloured by role, sized by |net outflow share|.
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
                textfont={"size": 10},
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
        title="Directed seeding network (edge = lead→follow; node colour = S05 role)",
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
    roles = mapio.load_roles(GRAIN).sort_values("role")
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
        title="Journal role components (argmax label drives source / bridge / terminal)",
        barmode="group",
        template="plotly_white",
        xaxis_title="journal",
        yaxis_title="component score",
        legend_title="component",
        margin={"t": 60, "b": 90, "l": 60, "r": 20},
    )
    fig.update_xaxes(tickangle=-30)
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
        margin={"t": 60, "b": 90, "l": 60, "r": 20},
    )
    fig.update_xaxes(tickangle=-30)
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


_CASCADE_CAVEAT = (
    "Cascade caveats: origins are <b>panel-conditional</b> ('first in our 10 journals', "
    "not first in the world); <b>1995 left-censoring</b> means ~139/149 topics 'originate' "
    "at the 1995 panel start (read as 'present at panel start', not 'born 1995'). The "
    "cascade is <b>S04-validated REAL-with-qualification</b>: the gate-critical S1/Null-1 "
    "seeding-spread permutation PASSes at both grains and on the resolvable subset "
    "(z≈+3.2, p≈0.002), origin jackknife flip-rate = 0, granularity ρ = 0.855; "
    "qualifications = S2/Null-1 non-corroboration and a partial held-out check."
)
_ROLE_CAVEAT = (
    "Roles are jackknife-stable (role_rank_correlation 0.945 leaf / 0.962 mid, bar 0.7) "
    "and robust across grain. Caveat: <b>'terminal' = within-panel citation sink, not a "
    "global dead-end</b> — with no true generalist source (Nature/NEJM/Lancet) in the "
    "panel, the most-cited prestige journals surface on the receive axis. The label is a "
    "coarse argmax; the continuous component scores are the nuanced output."
)
_ORIGIN_CAVEAT = (
    "Sector novelty is essentially a <b>null</b> (company sits mid-pack; novelty is not a "
    "tech/industry phenomenon here) — itself a clean result. <b>Geography is the most "
    "differentiated axis</b> (descriptive, likely topic-mix driven). Recombination is a "
    "corpus-internal <b>lower bound</b>. <b>Funding and citation-intent layers are "
    "deferred</b> ($0 session: grants parser-only, intents table empty). Type blanks ≈35% "
    "/ country blanks ≈44% of institutions are excluded, never imputed."
)
_LANDSCAPE_CAVEAT = (
    "Each point is a leaf topic; x = panel-conditional origin year, y = cross-journal "
    "reach, size = paper count, colour = origin journal. Hover for top-words + "
    "recombination rate. Same panel-conditional + 1995-censoring caveats as the cascade tab."
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
<title>SciField V2 — Literature Cartography Map v0</title>
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
  <h1>SciField V2 — Literature Cartography Map v0</h1>
  <p>A <b>$0 proof-of-concept</b> on the current <b>10-journal</b> orthopedic + general-surgery
     corpus (1995&ndash;2025). Exploratory cartography, not a hypothesis test. Every origin and
     cascade claim is <b>panel-conditional</b> (first in our 10 journals, not first in the world)
     and the corpus is <b>1995 left-censored</b>. Built {date} (V2-S07).</p>
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
  SciField V2 cartography &middot; map v0 &middot; sections: {section_list}. Data:
  V2/data/&#123;cascade,roles,origins&#125; + data/v1/topic_hierarchy. Reproducibility sidecar:
  index.html.run.json. This is a prove-phase artifact; the corpus expansion that would correct
  the panel-conditional bias is gated behind the human Gate G6.
</footer>
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


def build_html() -> str:
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
        date=pd.Timestamp.today().strftime("%Y-%m-%d"),
        sec_landscape=sec_landscape,
        sec_cascade=sec_cascade,
        sec_roles=sec_roles,
        sec_origins=sec_origins,
        section_list="a. Topic landscape | b. Cascade flows | c. Journal roles | "
        "d. Novelty origins",
    )


def main() -> None:
    """Build the map, write the HTML + sidecar, print the path and section list."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html = build_html()
    OUT_HTML.write_text(html, encoding="utf-8")

    record_run(
        artifact_path=OUT_HTML,
        inputs=INPUTS,
        config={
            "task": "V2-S07",
            "artifact": "map_v0/index.html",
            "grain": GRAIN,
            "sections": [
                "a. Topic landscape",
                "b. Cascade flows (lag heatmap + seeding network + adoption curve)",
                "c. Journal roles (component bars + citational velocity)",
                "d. Novelty origins (sector near-null + geography + recombinant topics)",
            ],
            "renderer": "plotly static HTML (CDN script), no server (streamlit/dash absent)",
            "panel_conditional": "origins are first-in-10-journal-panel, not first in world",
            "left_censoring_1995": "≈139/149 leaf topics originate at the 1995 panel start",
            "cascade_verdict": "S04 REAL-with-qualification (S1/Null-1 PASS both grains)",
            "roles_verdict": "S05 jackknife-stable (role_rank_correlation 0.945/0.962)",
            "sector_verdict": "near-null (company mid-pack); geography most differentiated",
            "deferred": "funding (grants parser-only) + citation-intent (table empty)",
        },
    )

    size_kb = OUT_HTML.stat().st_size / 1024
    n_divs = html.count("plotly-graph-div")
    print(f"\nWROTE map v0: {OUT_HTML}")
    print(f"  size: {size_kb:.1f} KB  |  plotly divs: {n_divs}")
    print("  sections:")
    for s in (
        "a. Topic landscape (149 leaf topics, origin-coloured scatter)",
        "b. Cascade flows (lead-lag heatmap + directed seeding network + adoption curve)",
        "c. Journal roles (source/bridge/terminal component bars + citational velocity)",
        "d. Novelty origins (sector near-null + geography ranking + top recombinant topics)",
    ):
        print(f"    - {s}")
    print(f"  sidecar: {OUT_HTML}.run.json")
    print("  open: double-click V2/map_v0/index.html (no server needed)")
    assert not math.isnan(size_kb)


if __name__ == "__main__":
    main()
