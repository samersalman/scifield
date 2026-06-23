# Cartography (V2)

**A literature cartography of contemporary science.** SciField V2 maps how
research topics arise, move between journals, and trend over time across a
broad multi-disciplinary journal panel. It is **exploratory cartography** — a
descriptive atlas of the literature — **not a hypothesis test**.

[**Open the interactive map →**](map/index.html){ .md-button .md-button--primary }

The map is a single self-contained page with five tabs. A tab-by-tab walkthrough
is on the [Using the map](using-the-map.md) page; the underlying tables are
documented in the [Data dictionary](data-dictionary.md).

---

## What this is

The cartography is computed over an **expansion panel of 78 journals**, from a
harvest of roughly **1.49 million papers** (V2-S08). The published cartography
layers are computed on the assigned corpus and resolve into **149 leaf topics**.
On top of that topic landscape it derives four further views:

- **Cascade** — how a topic moves *between* journals (who leads, who follows).
- **Journal roles** — where each journal sits in that diffusion network
  (**source** / **bridge** / **terminal**) and how fast its papers get cited.
- **Novelty origins** — where novelty comes from, by institution sector and by
  country, plus a combinatorial-recombination signal.
- **Trajectories** — a *descriptive* projection of where each topic's share and
  volume have been trending, extrapolated five years to **2030** with 80% bands.

A **headline finding** of the scaled run (V2-S09) is that the generalist
**"source" layer splits by journal type**: **PNAS ranks #1 source**, while the
high-impact clinical-prestige journals — **NEJM** is the most terminal, with
*Lancet*, *JAMA*, *Nature*, and *Science* also terminal — behave as **sinks**
rather than seeds. The inter-journal cascade is **validation-real** (the journal
roles are highly stable under a leave-one-out jackknife, rank-correlation
≈ 0.9993).

---

## Read this first — headline caveats

These govern how every view on the map should be read. They are not optional
footnotes. The same caveats, per-table, are in the
[Data dictionary](data-dictionary.md#read-this-first-headline-caveats).

1. **Origins are PANEL-CONDITIONAL.** Every "origin", "first appearance",
   "source", or "seeding" signal means *first/source within the 78-journal
   panel* — **not** first in the world. Several true generalist sources of
   science (e.g. *Nature*, *NEJM*, *Lancet*, *Science*) are read here only as
   they appear *inside this panel*; they cannot be credited as world-origins.
   Read every origin claim as "earliest **observed in this panel**".

2. **The corpus is 1995 LEFT-CENSORED.** Coverage starts in 1995. Any topic or
   diffusion that began earlier has its onset clipped to the panel start, so
   onset-timing signals (origin year, seeding, time-to-50%) inherit that bias.
   Flow- and betweenness-based signals are comparatively robust to it.

3. **`share` is panel-conditional.** A topic's `share` is its fraction of
   *panel* papers in a topic-year — not a fraction of all of science.

4. **Trajectories are a DESCRIPTIVE projection, not a forecast.** The
   trajectory layer (V2-S10) extrapolates each topic's observed share/volume
   +5 years to 2030 with 80% uncertainty bands using a state-space
   local-linear-trend model (149/149 topics fit). It is **not** causal,
   **not** a validated predictive model, and is explicitly **not** the V1 F3
   emergence GNN — that *predictive* model was **signed NULL at Gate G4**.
   Do not read the bands as a tested forecast.

5. **This is exploratory cartography, not a confirmatory result.** The whole
   V2 program was gated as *publishable exploratory work* at **Gate G6 (PASS,
   2026-06-09)**. It describes structure in the panel; it does not test a
   pre-registered hypothesis.

---

## Where things live

- **Interactive map:** [`map/index.html`](map/index.html) (served at
  `…/cartography/map/`).
- **How to read each tab:** [Using the map](using-the-map.md).
- **Table-by-table schema:** [Data dictionary](data-dictionary.md).
- **Rebuild it yourself + release runbook:** [Reproduce](reproduce.md).
- **Methods write-ups** (charter, validation protocol, per-layer findings)
  live in
  [`V2/docs/cartography/`](https://github.com/samersalman/scifield/tree/main/V2/docs/cartography)
  on GitHub.

## Citation & license

The derived cartography tables (~1.7 MB) ship in the repository under
`V2/data_v2/` and are documented in the [Data dictionary](data-dictionary.md).
They are released under the repository's **Apache License 2.0**. The raw
1.49M-paper corpus is **not** redistributed; it is reproducible from the
pipeline in this repository. Please cite the Zenodo record (DOI minted on the
v0.2.0 release) and the software repository; see `CITATION.cff` at the repo root.
