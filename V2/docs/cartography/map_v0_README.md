# Literature Cartography — Map v0 (proof-of-concept)

**Status:** Proof-of-concept artifact (2026-06-09)
**Author:** Samer Salman
**Program:** V2 — Literature Cartography (Phase E, V2-S07)
**Artifact:** [`../../map_v0/index.html`](../../map_v0/index.html) (self-contained static HTML)
**Code:** `scifield.cartography.mapio` (pure loader/prep) · `V2/scripts/build_map.py` (builder)
**Companion docs:** [`charter.md`](charter.md) · [`validation_protocol.md`](validation_protocol.md)
· [`cascade_validation.md`](cascade_validation.md) · [`roles_velocity.md`](roles_velocity.md)
· [`infra_decision.md`](infra_decision.md) · [`../../CONTEXT.md`](../../CONTEXT.md)

This is the **first artifact a domain reader can actually explore** — it assembles the
cascade, journal-role and novelty-origin layers from the earlier V2 batches into a single
double-click-able HTML page. It is the last cartographic deliverable before the human
**Gate G6** synthesis.

---

## This is a $0 proof-of-concept on the 10-journal corpus

Read this framing first. Map v0 is built **at $0** on the **current 10-journal**
orthopedic + general-surgery corpus (`ann_surg, arthroscopy, br_j_surg,
clin_orthop_relat_res, j_am_coll_surg, j_arthroplasty, j_bone_joint_surg_am, jama_surg,
spine, surgery`, 1995–2025). It is **exploratory cartography, not a hypothesis test**. Two
limitations are elevated to first-class, on-page caveats and apply to every origin/cascade
panel:

1. **Panel-conditional.** "Topic T originated in journal A" means only "T appeared first
   *in our 10 journals*." The true origin may be a generalist we never harvested
   (Nature / NEJM / Lancet / JAMA / PNAS). Origin attributions are therefore systematically
   biased toward the specialty journals **by construction** — the Phase-F hybrid corpus
   expansion (gated behind Gate G6) exists to correct this.
2. **1995 left-censoring.** The corpus starts in 1995, so ≈139/149 leaf topics "originate"
   at the panel start. Read a 1995 origin as "present at panel start," **not** "born 1995."

Proving the *method* on this panel is cheap and reversible; expanding the *corpus* is the
expensive thing Gate G6 decides. Map v0 is the proof that the method produces structure a
domain reader can read.

---

## How to open it

**Just open the file** — no server, no install, no build step:

```
open V2/map_v0/index.html          # macOS
# or double-click V2/map_v0/index.html in any file browser
```

It is a single self-contained HTML page with a tab bar (sections a–d). The only network
reference is the Plotly library, loaded once from the public CDN by *your browser's*
`<script src>` — there is no build-time network call and no per-figure embedded copy of the
library, so the page stays small (~127 KB). If you are fully offline the panels still render
their data; only the interactive Plotly chrome needs the CDN.

To rebuild from the source parquets:

```
.venv/bin/python V2/scripts/build_map.py
```

This regenerates `V2/map_v0/index.html` and its `index.html.run.json` reproducibility
sidecar (records git SHA, config, and the SHA-256 of all 9 input parquets; `task="V2-S07"`).

---

## What the map shows — layer by layer

Each tab carries an on-page caveat box. The data source and key caveat for each layer:

### a. Topic landscape
- **What:** every one of the 149 leaf topics as a sized, origin-coloured scatter — x = the
  panel-conditional origin year, y = cross-journal reach (how many of the 10 journals the
  topic reached), marker size = topic paper count, colour = origin journal. Hover shows the
  topic's top-words and its recombination rate.
- **Data:** `data/v1/topic_hierarchy.parquet` (top-words, size) joined to
  `V2/data/cascade/origin_attribution.parquet` (origin journal/year, reach) and
  `V2/data/origins/recombination_by_topic.parquet` (recombination rate).
- **Caveat:** panel-conditional origin + 1995 left-censoring (the dense 1995 column is the
  censoring artifact, not a burst of topic births).

### b. Cascade flows
- **What:** three panels — (i) the inter-journal **lead-lag heatmap** (signed mean lag; a
  blue/positive cell means the row journal leads the column journal into shared topics),
  (ii) the **directed seeding network** (10 journal nodes coloured by their S05 role, edges
  drawn lead→follow for pairs sharing ≥5 topics, node size ∝ net-outflow share), and
  (iii) the corpus **adoption-by-breadth curve** (fraction of topics reaching at least *k*
  journals).
- **Data:** `V2/data/cascade/{lag_matrix,diffusion_curves}.parquet` +
  `V2/data/roles/role_scores.parquet` (for the node colours).
- **Caveat:** panel-conditional + 1995 left-censoring, **and** the cascade is
  **S04-validated REAL-with-qualification**: the gate-critical S1/Null-1 seeding-spread
  permutation PASSes at both grains and on the resolvable (non-1995-tie) subset (z ≈ +3.2,
  p ≈ 0.002); origin-attribution jackknife flip-rate = 0; granularity Spearman ρ = 0.855.
  Qualifications: the secondary S2/Null-1 lead-lag-asymmetry does not corroborate (a property
  of the within-topic label shuffle), and the held-out-years check is only partial.

### c. Journal roles
- **What:** two panels — (i) grouped bars of the **source / bridge / terminal** component
  scores for each of the 10 journals (the discrete role label is the argmax), and (ii) the
  per-journal **citational velocity** (median years to citation; green = the panel's "fast"
  journal).
- **Data:** `V2/data/roles/{role_scores,velocity}.parquet`.
- **Caveat:** roles are **jackknife-stable** (`role_rank_correlation` = 0.945 leaf / 0.962
  mid, bar 0.7) and robust across grain. **"Terminal" = within-panel citation sink, not a
  global dead-end** — with no true generalist source in the panel, the most-cited prestige
  journals (Ann Surg, J Bone Joint Surg, CORR, JAMA Surg) surface on the *receive* axis. The
  label is a coarse argmax; the continuous component scores are the nuanced output.

### d. Novelty origins
- **What:** three panels — (i) **sector** semantic-novelty bars by institution type (company
  highlighted in red), presented honestly as a **near-null**; (ii) **geography** — the top-
  vs-bottom well-sampled countries by semantic novelty (the most-differentiated origin axis);
  (iii) the top **recombinant topics** by recombination rate (interdisciplinary bridges).
- **Data:** `V2/data/origins/{sector_novelty,geo_novelty,recombination_by_topic}.parquet`.
- **Caveat:** sector novelty is essentially a null (company sits mid-pack — novelty is *not*
  a tech/industry-sector phenomenon here, itself a clean result); geography is the most
  differentiated axis but is descriptive and likely topic-mix driven; recombination is a
  corpus-internal **lower bound**. **The funding and citation-intent layers are deferred**
  (this $0 session: the OpenAlex `grants` parser exists but is not wired through the store,
  and `citation_intents.parquet` is empty). Institution-type blanks (≈35%) and country blanks
  (≈44%) are excluded, never imputed.

---

## Why static HTML (and the Streamlit/Dash alternative)

The build deliberately emits **self-contained static HTML** (Plotly panels assembled into a
tabbed page, the precedent set by `docs/figures/topic_landscape.html`). **streamlit and dash
are not installed**, and adding a server dependency for a prove-phase artifact would be
premature: a static page opens by double-click, ships in the repo, needs no runtime, and is
trivially reproducible from the parquets.

**Documented future option (not built):** once the corpus is expanded (Phase F, post-G6) and
the map needs live filtering / drill-down / cross-panel linking, a **Streamlit or Dash** app
over the same `scifield.cartography.mapio` loaders is the natural next step — the loader layer
is already factored to be the data seam such an app would consume, so moving to a server-backed
interactive map is additive, not a rewrite. The infra spike (`infra_decision.md`) confirms
every canonical query is sub-50 ms on the current corpus, so an interactive server is feasible
whenever it is worth the operational cost; it is simply not worth it at the $0 prove phase.

---

## Reproducibility

- **Builder:** `.venv/bin/python V2/scripts/build_map.py` (read-only on all inputs; no
  network at build time; no GPU; no DeepSeek).
- **Sidecar:** `V2/map_v0/index.html.run.json` (`scifield.repro.record_run`, `task="V2-S07"`)
  records the git SHA, git-dirty flag, config hash, software versions, and the SHA-256 of all
  nine input parquets.
- **Loader tests:** `tests/test_cartography_mapio.py` (16 tests) assert the loader frames have
  the expected columns and shape against the real artifacts (skipping gracefully if a parquet
  is absent).

---

## Gate G6 map-coherence input

Map v0 renders all three required layers — **cascades, roles, and origins** — on the current
corpus, each with its on-page caveat and its upstream validation verdict surfaced. The cascade
tab carries S04's REAL-with-qualification verdict, the roles tab carries S05's jackknife
stability, and the origins tab presents the geography signal alongside the honest sector null
and the deferred funding/intent layers. The assembled page is **coherent to a domain reader**:
the journal roles, lead-lag structure, and topic origins line up into a readable narrative
(generalist-surgery journals seed, subspecialty journals bridge, prestige journals are
within-panel citation sinks), and the limitations are foregrounded rather than buried. This is
the V2-S07 input to the human Gate G6 synthesis.
