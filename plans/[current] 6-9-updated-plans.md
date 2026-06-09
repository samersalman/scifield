# SciField V2 — Literature Cartography

**Author:** Samer Salman | **Date:** 2026-06-09 | **Supersedes:** the old "V2 = scaling run (Phase 7)" definition in `plans/Session-Objectives-MAP.md`

---

## Context

V1 finished as a confirmatory study: pre-registered hypotheses, five decision gates, pass/fail. Two of three findings (F1 epistemic cascade, F3 graph forecasting) came back as honest, pre-registered nulls; only F2 (dual novelty) held. Gate G5 was signed **DOWNSCOPE** — under the original plan, the V2 scale-up was *not* triggered, and V1 ships as a methods-and-resource paper.

That verdict is correct **for the question V1 asked**. But the project's real goal was never "prove three hypotheses" — it was to **build a usable map of the literature**: where topics originate, how they cascade across journals, which journals are sources vs. bridges vs. terminals, where novelty comes from (institution type, sector, funding, geography), and what the past/present/future of a field looks like.

This is a different *kind* of science than V1. V1 was confirmatory. This is **exploratory cartography**. So we retire the old "re-run the validated pipeline to test replication" definition of V2 and write a new program in its place.

**Three framing commitments that this plan is built on:**

1. **Exploratory ≠ unrigorous.** We drop pre-registered hypothesis gates (they were the right tool for V1, and they're done). We keep — and lean harder on — the rigor that makes a *map* trustworthy enough to use: held-out validation, null/permutation models, robustness to journal-panel choice, and sensitivity to topic granularity. A map nobody can trust is not usable.
2. **Cascades are panel-conditional, and that's the central threat.** With any finite journal set, "topic T originated in journal A" really means "first *in our set*." The true origin may be a journal we never harvested. This is why the corpus is sampled as a **hybrid** (high-impact "source" generalists + comprehensive coverage within a few specialties) and why we **prove the method on the current 10-journal corpus before spending a dollar to scale**.
3. **The journal cascade is orthogonal to the null F1 — not a revival of it.** F1 asked "does evidence *quality* lead *volume* within a topic" (null). The cascade asks "does topic T appear in journal A before journal B, and do some journals systematically lead." Different question; not foreclosed.

**Decisions locked with Samer (2026-06-09):** ship V1 now as the foundation paper; hybrid corpus design; **prove on the current corpus first ($0)**; spike infrastructure before any migration (we are on Kùzu + DuckDB + Parquet today — *not* Neo4j).

**Shape of the work.** The "next few weeks" = Track 0 (ship V1) + Phases A–E, **all on the existing corpus at $0**. A single new gate (**G6**) then decides whether the cartographic methods are real enough to justify spending money on Phases F–G (the funded expansion and public map release).

---

## Naming & versions

- **V2 — Literature Cartography.** This program. Sessions `V2-S01 …`. Code/configs/notebooks go in a `Version 2/`-style layout, reusing existing `scifield` modules.
- **V3 — Release + scaled map.** Folded into Phases F–G here (public usable map, Zenodo, docs site). The old "V3 = Phases 8–9" can be retired into this.
- The old Phase-7 "scaling run" V2 is **retired**; this document replaces it.

---

## Track 0 (parallel, non-blocking) — Ship V1 as the foundation paper

**Why first / why parallel.** V1 is ~75% to a preprint and is the citable base the map builds on. It does not block any cartography work, so it runs alongside Phase A.

**State (verified).** Done: `docs/results_draft.md`, 16 figures in `docs/figures/` (incl. interactive `topic_landscape.html`), all gates G1–G5 signed, `docs/gates/G5_scrutiny_audit.md`. Missing: Introduction, Discussion, Abstract, a cohesive Methods-as-prose section, author list / affiliations / COI, references, and submission formatting.

**Tasks.**
- Draft Introduction + Discussion with the `narrative-writer` agent; Methods prose with `methods-writer`; tighten Results with `results-writer` / `prose-polisher` — all from the existing `results_draft.md`, gate docs, and pre-registrations.
- Compile author list, affiliations, COI; assemble references; export to the target preprint format.
- Frame honestly: validated multi-axis pipeline + F2 finding + two characterized nulls (F1 internal pre-reg, F3 OSF DOI 10.17605/OSF.IO/XP94F).

**Deliverable.** `manuscript/v1/` (intro.md, methods.md, results.md, discussion.md, abstract.md, references) + a preprint-ready document. **Acceptance:** a co-author can read the manuscript end-to-end with no placeholders; all figure/citation references resolve.

---

## Phase A — Charter, exploratory-rigor protocol, infra spike (current corpus, $0)

### V2-S01 — Cartography charter + exploratory-methods lock + journal-flow data model

**Objective.** Replace "pre-registration of hypotheses" with an honest **exploratory-methods lock**: define the map's questions, the metrics, and the validation philosophy *before* computing, so results can't be quietly reverse-engineered.

**In scope.**
- `docs/cartography/charter.md`: the map's questions (origins, cascades, roles, novelty-by-sector, trajectory); explicit "exploratory, not confirmatory" stance; the explicit note that the journal cascade is orthogonal to null F1.
- `docs/cartography/validation_protocol.md`: the rigor that replaces gates — held-out years, null/permutation models, drop-one-journal jackknife, topic-granularity sensitivity. Locked before Phase B compute.
- `src/scifield/cartography/flow.py`: the canonical **journal-flow data model** — per `(topic, journal, year)`: first-appearance flag, volume, citation in-flow / out-flow. Built from existing `topics.parquet` + Kùzu CITES edges + enrichment.

**Out of scope.** Computing cascades (V2-S03); any corpus expansion.

**Acceptance.** Charter + protocol committed and dated; `flow.py` produces the per-(topic, journal, year) table for the current corpus with a sidecar JSON.

### V2-S02 — Infrastructure spike (decide, don't migrate)

**Objective.** Decide whether Kùzu + DuckDB + Parquet holds for the map's query patterns, with explicit migration thresholds — *not* a premature commitment to a graph DB.

**In scope.**
- Enumerate the 5–6 canonical map queries: temporal first-appearance ("when did topic T first appear in journal J"), cross-journal lag matrix, citation-cascade traversal, topic-recombination detection, sector/geo rollups.
- Benchmark each on the current corpus on Kùzu / DuckDB / Parquet. Record latency + ergonomics.
- `docs/cartography/infra_decision.md`: a documented go/stay decision with **explicit thresholds** for when to migrate (candidate targets: Memgraph or Neo4j for heavy graph traversal; a temporal columnar store or DuckDB-only for first-appearance/lag; the likely answer is *hybrid*). Recommendation for the prove phase: **stay on the current stack**; re-evaluate at Phase F scale.

**Out of scope.** Actually migrating; standing up a new DB.

**Acceptance.** Benchmark numbers + a decision doc with thresholds, committed.

---

## Phase B — Topic cascade / diffusion mapping (current corpus, $0)

### V2-S03 — Cascade engine (generalize the existing seeding code)

**Objective.** Turn the existing lead-follow prototype into a general cascade engine. **Build on `src/scifield/findings/seeding.py` and `notebooks/14_bonus_cross_journal.ipynb`** — these already compute journal/institution/country first-publication lead-follow; do not start from scratch.

**In scope.**
- `src/scifield/cartography/cascade.py`: per-topic first-appearance per journal; inter-journal **lead-lag matrix**; diffusion-curve fit (time-to-reach each journal); origin-journal attribution per topic.
- `notebooks/v2_01_cascades.ipynb`: cascade visualizations + the inter-journal lag matrix.

**Out of scope.** Validation/null models (V2-S04); journal-role classification (V2-S05).

**Acceptance.** Lag matrix + per-topic origin attribution produced for the current corpus; spot-check on 5–10 topics reads sensibly.

### V2-S04 — Cascade validation (the make-or-break)

**Objective.** Decide whether the cascade structure is **real or a panel artifact**. This is the single most important step in the prove phase and the primary input to Gate G6.

**In scope.**
- Null / permutation models: shuffle journal labels and years; show the observed lag structure exceeds the null.
- Drop-one-journal **jackknife**: do origin attributions and roles survive removing any single journal?
- Topic-granularity sensitivity: do conclusions hold across leaf vs. mid-level topics?
- `docs/cartography/cascade_validation.md`: verdict with figures.

**Out of scope.** Anything requiring the expanded corpus.

**Acceptance.** Each cascade claim is paired with a null comparison and a jackknife result; verdict (real / artifact / partial) documented.

---

## Phase C — Journal role taxonomy + citational velocity (current corpus, $0)

### V2-S05 — Source / bridge / terminal roles + velocity

**Objective.** Classify each journal's role in topic flow and characterize its citational velocity.

**In scope.**
- `src/scifield/cartography/roles.py`: from the flow network — **source** (high out-flow of novel topics), **bridge** (high betweenness in the topic-flow network), **terminal** (high in-flow / low out-flow). Per-journal role scores.
- Citational velocity per journal: time-to-citation-accrual distributions (from Kùzu CITES + publication dates).
- `notebooks/v2_02_roles_velocity.ipynb`.
- Validate against within-corpus priors (e.g., generalist surgery vs. subspecialty journals behaving as expected), since the true source generalists (Nature/NEJM/Lancet) aren't in the corpus yet.

**Out of scope.** Comparing to the expanded corpus (Phase F).

**Acceptance.** Role scores + velocity distributions per journal; roles are stable under the V2-S04 jackknife.

---

## Phase D — Origins of novelty: sector, geography, funding (current corpus, mostly $0)

### V2-S06 — Where does novelty arise?

**Objective.** Answer the "what is novelty and where does it come from" questions to the extent the data supports, honestly flagging coverage limits.

**In scope.**
- **Ready today (no re-harvest):** institution `type` × novelty (semantic + CD) — directly answers "is novelty from the tech/industry sector?" via `type == company`; geography (`country_code`) × novelty. Join path: `paper_institutions.parquet` → `institutions.parquet`.
- **Bounded $0 re-harvests (no DeepSeek, no GPU):**
  - OpenAlex `grants` field (currently ignored in `src/scifield/corpus/openalex.py::parse_openalex_work`) → enables funding-origin analysis. Add the field to the parser + a targeted re-pull.
  - Semantic Scholar citation intents (schema exists, `citation_intents.parquet` empty) → re-run with `SEMANTIC_SCHOLAR_API_KEY` set.
- **Topic recombination events:** detect when two topics combine into emergent work (recombinant novelty) — exploratory.
- `notebooks/v2_03_novelty_origins.ipynb`.

**Out of scope.** Any DeepSeek spend; corpus expansion.

**Spend note.** The two re-harvests use only free/polite-pool APIs ($0). No DeepSeek call is made in this phase; the `feedback-deepseek-spend-gating` rule is not triggered here but applies the moment Phase F begins.

**Acceptance.** Sector + geography novelty tables produced from current data; funding/intent analyses run if the re-harvests complete, else explicitly deferred with a coverage note.

---

## Phase E — Proof-of-concept map + Gate G6 (current corpus, $0)

### V2-S07 — Usable map v0

**Objective.** Assemble cascade + roles + origins into a **rough but real, usable map** — the first artifact someone could actually explore.

**In scope.**
- Build on existing tooling: `docs/figures/topic_landscape.html` (Plotly base layer), `seeding.py` edge-lists (force-directed cascade overlay), and a Kùzu query binding (the graph is materialized at `data/v1/kuzu_graph` but has no UI yet).
- A lightweight interface (Streamlit or Dash, or an enriched mkdocs page) exposing: topic landscape, cascade flows, journal roles, novelty-by-sector.
- `docs/cartography/map_v0/`.

**Out of scope.** Public deployment (Phase G); scaled data (Phase F).

**Acceptance.** A running local artifact that renders cascades, roles, and origins on the current corpus.

### 🚦 GATE G6 — Do the cartographic methods produce non-artifactual, interpretable structure?

**This is the spend gate.** It replaces the old "≥2-of-3 findings" bar with a cartography-appropriate one.

**Pass =** cascade structure survives the V2-S04 null/permutation + jackknife; journal roles are stable and interpretable; ≥1 novelty-origin finding is robust; the map v0 is coherent to a domain reader. **→ Fund Phases F–G.**

**Fail =** the structure is a panel artifact or too unstable to trust. **→ Do not spend.** Reconsider corpus design or method before scaling; the V1 paper (Track 0) still ships regardless.

*This gate is a human decision by Samer, like G1–G5.*

---

## Phase F — Corpus expansion (hybrid), GATED behind G6 (spend $)

### V2-S08 — Expansion harvest (sources + deep specialties)

**Objective.** Build the hybrid corpus once G6 passes.

**In scope.**
- **Source layer:** high-impact generalists (e.g., Nature, Science, NEJM, Lancet, JAMA, PNAS) — the hypothesized origin points for foundational papers.
- **Deep layer:** OpenAlex-complete coverage within 2–3 specialties already represented (e.g., orthopedics, general surgery, + one more) so cascades have true downstream depth.
- Re-harvest + re-enrich via the existing `harvest` / `enrich` pipeline.
- **Re-embed and re-cluster** — adding generalists *moves the topic model itself*; this is not "more rows through the same topics." Build an old↔new topic crosswalk.
- **DeepSeek spend gating applies:** dry-run cost estimate + explicit OK from Samer before any extraction (`feedback-deepseek-spend-gating`). Brev for embeddings (`scripts/brev_embed.sh`).

**Out of scope.** Re-running cartography (V2-S09).

**Acceptance.** Expanded corpus harvested + enriched + re-clustered with sidecars; spend logged in `docs/operations/api_costs.md` and within an approved budget.

### V2-S09 — Re-run cartography at scale

**Objective.** Re-run cascade / roles / origins on the expanded corpus and compare to the current-corpus prototype.

**In scope.**
- Re-run V2-S03–S06 pipelines at scale.
- Key test: **do the source generalists actually show up as sources?** Does adding them change origin attributions for the specialty topics (the panel-conditional correction in action)?
- `notebooks/v2_04_scaled_cartography.ipynb`.

**Acceptance.** Scaled cascade/role/origin tables; explicit current-vs-scaled comparison documenting what the source layer changed.

---

## Phase G — Trajectory layer + public usable map (release)

### V2-S10 — Trajectory / "future" layer

**Objective.** Add the past→present→**future** dimension as honest descriptive projection.

**In scope.**
- Descriptive topic-trajectory projection with uncertainty bands (trend extrapolation / state-space), **explicitly NOT the dead F3 GNN** — reframed as descriptive trend modeling, clearly labeled exploratory.
- `notebooks/v2_05_trajectories.ipynb`.

**Acceptance.** Per-topic trajectory projections with uncertainty; methods doc states the descriptive (not predictive-claim) framing.

### V2-S11 — Public usable map release

**Objective.** Ship the map as a usable public artifact — the goal of the whole program.

**In scope.**
- Polish map v0 → deployable (mkdocs/GitHub Pages or a hosted app); integrate cascades, roles, origins, trajectories.
- Zenodo deposit of the corpus + map artifacts; documentation; reproducibility sidecars.

**Acceptance.** A publicly reachable, queryable map; Zenodo DOI; docs explain how to use and reproduce it.

---

## Verification (how we know each phase worked)

- **Track 0:** manuscript reads end-to-end, no placeholders, all references resolve.
- **Phase A:** charter + protocol committed *before* compute; `flow.py` table + sidecar exists.
- **Phase B:** every cascade claim has a paired null + jackknife result (V2-S04 doc).
- **Phase C:** journal roles stable under jackknife; velocity distributions produced.
- **Phase D:** sector/geo novelty tables from current data; funding/intents run or explicitly deferred with coverage note; **$0 confirmed** (no DeepSeek).
- **Phase E:** running local map v0; **Gate G6 human decision recorded** before any spend.
- **Phase F:** expanded corpus with sidecars; DeepSeek dry-run + explicit OK logged; spend within approved budget.
- **Phase G:** public map reachable; Zenodo DOI; reproducible.

**Reproducibility (cross-cutting, unchanged from V1):** every artifact gets a `scifield.repro` sidecar JSON (git SHA, config hash, input hashes, versions). Notebooks import from `scifield`; logic lives in `src/scifield/cartography/`.

---

## What we are explicitly NOT doing

- Not reviving F1 or F3 as confirmatory claims (their pre-registered nulls stand; the cascade is a different, exploratory question).
- Not pre-registering new hypotheses — this program is exploratory by design; rigor comes from null models, jackknife, and held-out validation instead.
- Not migrating databases before the infra spike (V2-S02) justifies it.
- Not spending a dollar on corpus expansion before Gate G6 passes.
