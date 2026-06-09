# Cartography Charter — SciField V2

**Status:** LOCKED (2026-06-09)
**Author:** Samer Salman
**Program:** V2 — Literature Cartography (Phase A, V2-S01)
**Companion documents:** [`validation_protocol.md`](validation_protocol.md) (the rigor that replaces the V1 gates), [`../../CONTEXT.md`](../../CONTEXT.md) (shared ground truth), [`../../README.md`](../../README.md) (program orientation).

This charter is written *before any cascade compute* so that the map's questions, its
epistemic stance, and its hard limitations are fixed in advance and cannot be quietly
reverse-engineered from whatever the data happens to show. It is a commitment, not a draft.

---

## 1. What this program is

V1 was a **confirmatory** study: pre-registered hypotheses (F1 epistemic cascade, F2 dual
novelty, F3 graph forecasting), five decision gates G1–G5, pass/fail. It closed honestly —
F2 holds, F1 and F3 are pre-registered nulls, Gate G5 = DOWNSCOPE (signed 2026-06-08). V1
ships as a methods-and-resource paper. That verdict is correct **for the question V1 asked**.

V2 is a **different kind of science**: **exploratory cartography**. The goal is to build a
*usable map of the literature* — not to test three more hypotheses. The map answers five
families of questions (Section 3). The artifact we are proving in this session is the
**method**, on the **current 10-journal corpus, at $0**, so that we earn the right to spend
money expanding the corpus only if the cartography is real (Gate G6).

---

## 2. The map's questions

These are the five layers the map must answer. Each is exploratory: we characterize structure
and bound its trustworthiness; we do not pre-register a hypothesis about its sign or magnitude.

1. **Topic origins** — For each topic, *which journal in our panel published it first*, and how
   confident can we be in that attribution? (See the panel-conditional caveat, Section 5 — this
   is "first in our 10-journal set," not "first in the world.")
2. **Cross-journal cascades / diffusion** — Once a topic appears, how does it spread across the
   other journals? What is the inter-journal lead-lag structure, the time-to-reach each journal,
   and the shape of the diffusion curve? Do *some journals systematically lead others*?
3. **Journal roles (source / bridge / terminal)** — From the topic-flow network, classify each
   journal: **source** (high out-flow of topics it published first), **bridge** (high
   betweenness — topics pass through it on the way between others), **terminal** (high in-flow,
   low out-flow — topics arrive here late). Roles are *relative within the panel*.
4. **Novelty by sector / geography / funding** — Where does novel work arise? Using institution
   `type == company` as the tech/industry-sector proxy, `country_code` for geography, and (if a
   $0 re-harvest completes) OpenAlex `grants` for funding, characterize how semantic and
   structural (CD-index) novelty distribute across these strata.
5. **Trajectory / future layer** — A past→present→**future** dimension as *honest descriptive
   projection* with uncertainty bands. Explicitly **not** a predictive claim and explicitly
   **not** the dead F3 GNN (Section 4); reframed as descriptive trend modeling, clearly labeled
   exploratory. (Compute is Phase G, behind Gate G6 — out of scope for this session.)

---

## 3. Stance: exploratory, not confirmatory

**We retire the pre-registered hypothesis gates.** Gates G1–G5 were the right tool for V1's
confirmatory questions, and they are done. Pre-registering a hypothesis about, say, the
direction of a cascade and then running one test against a fixed bar is the correct discipline
when you have a single sharp claim. A map is not a single sharp claim — it is a structure to be
characterized, and pre-registering its shape would be either vacuous or dishonest.

**Exploratory ≠ unrigorous.** Retiring the hypothesis gates does *not* loosen the bar. It
*moves* the rigor from "did the pre-specified statistic clear a pre-specified threshold" to "is
the structure the map reports real, or an artifact of how we sampled the corpus and chose its
parameters." That rigor is specified concretely and **locked before any Phase B compute** in
the companion [`validation_protocol.md`](validation_protocol.md):

- **Held-out years** — fit cascade structure on early years, check it holds on later years.
- **Null / permutation models** — shuffle the labels the structure depends on, show the
  observed structure exceeds the permutation null.
- **Drop-one-journal jackknife** — re-run with each of the 10 journals removed in turn; report
  how stable the attributions and roles are.
- **Topic-granularity sensitivity** — a conclusion must hold at both leaf (149 topics) and
  mid-level (96 `mid_level_id`) grains to be called robust.

A map nobody can trust is not usable. These four checks are how a map earns trust without a
hypothesis gate, and they are the primary inputs to Gate G6.

---

## 4. The cascade is orthogonal to the null F1 (and is NOT a revival of F1 or F3)

This is a load-bearing commitment, stated in the shared ground truth and quoted here verbatim
so no downstream task or reader can mistake the cascade for a back-door revival of a signed null.

From `V2/CONTEXT.md` §0.3:

> The journal cascade is **orthogonal to the null F1** (F1 = does evidence *quality* lead
> *volume* within a topic — null). The cascade asks *which journal publishes topic T first*
> and *do some journals systematically lead*. Different question; not a revival of F1.

To make the distinction unmistakable:

- **F1** asked, *within a single topic's time series*, whether the **evidence quality** of that
  topic leads or lags its **research volume**. It was tested on once-differenced series, found a
  clean pre-registered null (0/138 directional; panel p = 0.277 / 0.739), and a self-pre-registered
  Toda-Yamamoto **levels** diagnostic (PR3D) subsequently confirmed the level-null is genuine, not
  an artifact of differencing. F1 is about **quality vs. volume**. It is closed.
- **The V2 cascade** asks, *across journals*, **which journal publishes a topic first** and
  whether some journals **systematically lead** others into topics. It is about **journal-to-journal
  ordering of first appearance**, not about quality vs. volume at all. It uses neither the evidence
  tier nor the volume time-series that F1 used.

**F3** was a graph-augmented forecaster of topic emergence; its sealed-test null stands
(test AUC hgt 0.781 < no_graph 0.804; paired Brier favors baseline, p = 2.6e-11). The V2
**trajectory layer** (Section 3, layer 5) is descriptive trend modeling with uncertainty bands,
**not** a re-run of the F3 HGT and **not** a predictive claim. It is reframed precisely so it
cannot be read as F3 reopened.

**Commitment:** nothing in V2 retroactively edits, re-scores, or reopens signed Gate G5, PR2,
or PR3/PR3D. The V2 questions are *different questions*. If V2 ever produces a finding that
would change a V1 conclusion, it routes through a fresh additive gate (the integrity model in
`docs/gates/G5_scrutiny_audit.md` §5), never a retroactive edit.

---

## 5. The panel-conditional caveat (first-class limitation)

Every origin and cascade claim in this map carries a structural limitation that we elevate to
a first-class commitment rather than a footnote:

> With a finite 10-journal panel, **"topic T originated in journal A" means only "T appeared
> first *in our set of 10 journals*."** The true origin may be a journal we never harvested —
> very plausibly a high-impact generalist (Nature / Science / NEJM / Lancet / JAMA / PNAS) that
> is *not* in the current corpus.

This is not a minor measurement caveat; it is **the central threat to the map's validity**, and
it drives two design decisions:

1. **Why the corpus is a hybrid design.** A trustworthy origin map needs both the hypothesized
   *source* generalists (where foundational work plausibly first appears) and *deep, near-complete
   coverage within a few specialties* (so cascades have real downstream depth). The current
   corpus has the deep specialty layer (orthopedics + general surgery, 10 journals) but **not**
   the source-generalist layer. So origin attributions today are *systematically biased toward
   the specialty journals* — by construction, because the generalists that might actually be the
   source are absent. The hybrid expansion (Phase F) exists precisely to correct this bias.
2. **Why we prove the method at $0 first.** We do **not** spend a dollar expanding the corpus
   until we have shown, on the current 10-journal panel, that (a) the cascade machinery produces
   structure that survives the null/permutation and jackknife checks, and (b) that structure is
   interpretable to a domain reader. Proving the *method* is cheap and reversible; expanding the
   *corpus* is expensive and is the thing Gate G6 decides. The panel-conditional caveat is *why*
   the prove phase is the right place to spend our effort before any money.

Concretely, every cascade/origin claim produced in V2-S03–S07 must be reported with this caveat
attached, and the V2-S04 jackknife (drop-one-journal) is the operational proxy for "how much
would this attribution move if the panel were different" — the closest thing we can compute to
the panel-conditional sensitivity *before* the corpus is actually expanded.

---

## 6. Locked decisions (2026-06-09, with Samer)

These were locked in conversation with Samer on 2026-06-09 and are binding for the V2 program:

1. **Ship V1 now** as the foundation/methods-and-resource paper. V1's nulls stand; it is the
   citable base the map builds on. Track 0 runs parallel to and does not block cartography.
2. **Hybrid corpus design** — source generalists + deep specialty coverage — is the target for
   any expansion, *because of* the panel-conditional caveat (Section 5).
3. **Prove-first-at-$0** — prove the cartographic method on the current 10-journal corpus
   before spending any money. No DeepSeek, no GPU, no paid APIs this session.
4. **Spike infrastructure before migrating.** We are on **Kùzu + DuckDB + Parquet** today and
   stay there for the prove phase. We do **not** migrate to Neo4j (or any graph DB) before the
   V2-S02 infra spike documents an explicit, threshold-based justification. The likely
   recommendation is *stay on the current stack; re-evaluate at Phase F scale.*

---

## 7. Scope of this session

This session executes **Phases A–E only** (V2-S01 → V2-S07) on the current 10-journal corpus at
$0, and **stops at the human Gate G6**. Specifically in scope:

- **Phase A** — this charter, the validation protocol, the journal-flow data model, the infra spike.
- **Phase B** — the cascade engine and its make-or-break validation.
- **Phase C** — journal source/bridge/terminal roles + citational velocity.
- **Phase D** — origins of novelty by sector / geography / (bounded $0 re-harvest) funding.
- **Phase E** — the proof-of-concept map v0, then **Gate G6** (a human decision by Samer).

Explicitly **out of scope** this session: corpus expansion (Phase F), the scaled re-run
(V2-S09), the public release (V2-S11), any DeepSeek/GPU/paid-API spend, and migrating databases.
Those are gated behind Samer's G6 decision. We characterize coverage limits honestly and defer
anything that needs the expanded corpus or money — we do not overclaim. This is a prove-phase on
a 10-journal panel, and every deliverable says so.
