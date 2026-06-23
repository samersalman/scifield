# Using the map

The cartography is a single self-contained page —
[**open it here**](map/index.html) — with **five tabs** along the top. This page
explains what each tab shows and how to read it. Before drawing conclusions,
read the [headline caveats](index.md#read-this-first-headline-caveats): origins
are panel-conditional, the corpus is 1995 left-censored, and the trajectory
layer is a descriptive projection, not a forecast.

Every panel is built from the derived tables documented in the
[Data dictionary](data-dictionary.md); each tab below links to the relevant
section there.

---

## 1. Topic landscape

A scatter of **every leaf topic** (149 of them) — one point per topic, **sized**
by how many corpus papers it contains and **coloured** by its origin journal's
diffusion role (**source** / **bridge** / **terminal**). This is the base map:
it shows the shape of the topic space and which topics were first observed
(panel-conditionally) in source-type vs terminal-type journals.

- **Read it as:** a bird's-eye view of *what the panel is about* and *where each
  topic entered the panel*.
- **Watch out:** colour reflects the *origin journal's* role, and origin is
  panel-conditional and left-censored — a topic coloured as "source-origin" was
  first *observed in this panel* in a source-type journal, not necessarily first
  in the world.
- Backing tables: `topic_hierarchy.parquet`, `cascade/origin_attribution.parquet`,
  `roles/role_scores.parquet`
  ([dictionary](data-dictionary.md#topic-label-dictionary)).

## 2. Cascade flows

How topics move **between journals**. Three views:

- a **lead–lag heatmap** of inter-journal time offsets (who tends to enter a
  shared topic before whom);
- a **directed seeding network** (arrows from seeding journals to following
  journals);
- an **adoption-by-breadth curve** (how a topic's reach across journals grows
  over time).

- **Read it as:** the diffusion structure of the panel — the evidence behind the
  "cascade is real" claim (journal roles are stable under jackknife,
  rank-correlation ≈ 0.9993).
- **Watch out:** lead–lag and seeding are onset-timing signals and so are
  sensitive to the 1995 left-censoring (caveat 2).
- Backing tables: `cascade/lag_matrix.parquet`, `flow/flow_leaf.parquet`,
  `cascade/diffusion_curves.parquet`
  ([dictionary](data-dictionary.md#cascade-journal-leadlag-diffusion-v2-s04s09)).

## 3. Journal roles

Where each of the 78 journals sits in the diffusion network. Two bar views:

- **source / bridge / terminal** component bars (the z-scored sub-scores that
  decide each journal's role);
- **citational-velocity** bars (how fast each journal's papers accrue
  citations).

- **Read it as:** the journal-level summary of the cascade. This is where the
  **headline finding** is visible: the generalist source layer **splits** —
  **PNAS is the #1 source**, while **NEJM** is the most terminal and *Lancet /
  JAMA / Nature / Science* are also terminal sinks.
- **Watch out:** roles are *within-panel*; the seeding sub-signal is
  left-censoring-sensitive while flow/betweenness are robust. A journal labelled
  `source` is a source *relative to this panel*.
- Backing tables: `roles/role_scores.parquet`, `roles/velocity.parquet`
  ([dictionary](data-dictionary.md#roles-journal-roles-velocity-v2-s05s09)).

## 4. Novelty origins

Where novelty comes from. Three views:

- **sector novelty** — novelty by institution sector (this signal is
  **near-null** across sectors);
- a **geography ranking** — novelty by country;
- **top recombinant topics** — the topics whose papers most combine references
  from across distinct topics (combinatorial novelty).

- **Read it as:** a descriptive breakdown of novelty by *who* and *where*; the
  recombination list highlights the most combinatorially novel topic areas.
- **Watch out:** the sector signal is near-null — do not over-read small
  differences. Country rows flagged low-sample should be filtered out for
  headline reads. All origin signals are panel-conditional and left-censored.
- Backing tables: `origins/sector_novelty.parquet`, `origins/geo_novelty.parquet`,
  `origins/recombination_by_topic.parquet`
  ([dictionary](data-dictionary.md#origins-novelty-origins-v2-s06s09)).

## 5. Trajectories

The **descriptive "future" layer**. For each topic, the model fits the observed
1995–2025 series of its panel **share** (and raw **volume**) and extrapolates
**+5 years to 2030** with an **80% band**. The tab has:

- a **direction distribution** — across all 149 topics the split is **flat 81 /
  falling 38 / rising 30**;
- **top rising and top falling movers** drawn as **fan charts** (the projected
  point with its widening 80% band);
- a **per-topic dropdown selector** (a Plotly `updatemenus` control): pick any
  one of the 149 topics and the chart redraws to its **observed + projected
  share** with the 80% band. **Use this to inspect a specific topic** rather
  than only the headline movers.

!!! warning "Trajectories are descriptive, not a forecast"
    This layer is a *descriptive* state-space projection (local-linear-trend on
    `logit(share)` / `log(volume)`), extrapolated forward. It is **not** causal,
    **not** a validated predictive model, and is explicitly **not** the V1 F3
    emergence GNN — that predictive model was **signed NULL at Gate G4**. The
    80% bands are extrapolation uncertainty, **not** a tested forecast. `share`
    is panel-conditional. See [caveat 4](index.md#read-this-first-headline-caveats).

- Backing tables: `trajectory/trajectory_summary.parquet`,
  `trajectory/trajectory_series.parquet`
  ([dictionary](data-dictionary.md#trajectory-descriptive-topic-trajectories-v2-s10)).

---

## General reading tips

- **Hover** for the underlying numbers (topic labels, journal slugs, lags,
  shares); the map is interactive Plotly.
- **Join key.** Almost everything is keyed on `topic_id` (0–148), resolved to a
  human-readable label via `topic_hierarchy.parquet`.
- **Grain.** The headline cartography is the `leaf` grain (149 topics). Some
  tables also carry a coarser `mid` grain; the map shows `leaf`.
- When in doubt about what a field means, the
  [Data dictionary](data-dictionary.md) is authoritative; to rebuild the map
  from source, see [Reproduce](reproduce.md).
