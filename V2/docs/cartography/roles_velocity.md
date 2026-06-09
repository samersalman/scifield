# Journal Roles + Citational Velocity — SciField V2 Cartography

**Status:** Exploratory result (2026-06-09)
**Author:** Samer Salman
**Program:** V2 — Literature Cartography (Phase C, V2-S05)
**Companion documents:** [`validation_protocol.md`](validation_protocol.md) (the jackknife
stability bar this task must meet), [`charter.md`](charter.md), [`../../CONTEXT.md`](../../CONTEXT.md).
**Code:** `scifield.cartography.roles` (pure logic) · `V2/scripts/build_roles.py` (builder)
· `V2/notebooks/v2_02_roles_velocity.ipynb` (figures).
**Data:** `V2/data/roles/role_scores.parquet` (both grains) · `V2/data/roles/velocity.parquet`.

This document records the journal **role taxonomy** (source / bridge / terminal) and the
per-journal **citational velocity**, validates the roles against the within-corpus prior, and
reports the drop-one-journal `role_rank_correlation` against the protocol bar — the V2-S05
input to Gate G6.

---

## 1. Method (what each score means)

Every journal gets three continuous component scores; the discrete `role` label is the argmax
of the three z-scored components (ties break source > bridge > terminal).

| Component | Definition | Censoring-sensitive? |
|---|---|---|
| **source** | mean of z-scored `seeding_score` (mean normalized first-publication lead, from `seeding.seeding_score`) **and** z-scored **net citation out-flow share** (`Σ(out_flow − in_flow)` over the journal's flow cells, normalised by the panel total absolute net-flow) | partly (seeding half) |
| **bridge** | **betweenness centrality** of the journal in the directed topic-flow (seeding) network, networkx on edges with `weight = n_precedes/n_shared ≥ 0.5`, distance `1/weight` | no |
| **terminal** | mean of z-scored **net in-flow share** (`−net_outflow_share`) **and** z-scored `1 − seeding_score` (arrives late) | partly (seeding half) |

**Why this construction.** The V2-S05 spec defines a *source* as "high out-flow of novel
topics," a *bridge* as "high betweenness in the topic-flow network," and a *terminal* as "high
in-flow / low out-flow (late adopter)." We operationalise *source* and *terminal* as
near-mirror images on a temporal axis (publishes first vs. last) **and** a citation-flow axis
(net exporter vs. net importer), given equal weight; *bridge* is the orthogonal betweenness
axis. A journal can be a strong bridge while being neither a clear source nor terminal.

**1995 left-censoring (S03 hand-off).** The corpus starts in 1995, so "publishes a topic
first" is dominated by topics already present at panel start (79% of leaf topics tie at a 1995
origin). The seeding sub-signal therefore inherits that bias. We mitigate two ways: (a) the
citation-flow sub-signal is weighted equally with seeding in source/terminal, and (b) the
bridge axis is *purely* betweenness — both depend on the *relative ordering* of journals across
many topics, not on the absolute origin year, so neither is censoring-driven. The role heatmap
(notebook panel a) shows that net-outflow and betweenness, not seeding alone, separate the
labels.

**Citational velocity.** Per journal, the distribution of years-from-publication-to-citation
(`citing_year − publication_year`) over external inbound citations (`cited_by.parquet` joined
to the focal paper's `journal_slug` + pub year via `archetypes.openalex_id →
papers_distinct.journal_slug`). Receive-side, so unaffected by the 1995 origin censoring.
Lags < 0 or > 60 years dropped as data errors. `velocity` = "fast" if the journal's median lag
is below the panel median of medians, else "slow."

---

## 2. Role assignments (leaf grain)

| Journal | Specialty | seeding | net out-flow | betweenness | source | bridge | terminal | **role** |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Br J Surg | general_surgery | 0.739 | −0.008 | 0.000 | **0.731** | 0.000 | −0.731 | **source** |
| Surgery | general_surgery | 0.679 | 0.076 | 0.014 | **0.737** | 0.014 | −0.737 | **source** |
| J Am Coll Surg | general_surgery | 0.659 | −0.022 | 0.000 | **0.236** | 0.000 | −0.236 | **source** |
| J Arthroplasty | orthopedic | 0.480 | 0.220 | 0.000 | **0.212** | 0.000 | −0.212 | **source** |
| Arthroscopy | orthopedic | 0.463 | 0.057 | **0.111** | −0.530 | 0.111 | 0.530 | **bridge** |
| Spine | orthopedic | 0.538 | −0.010 | **0.069** | −0.379 | 0.069 | 0.379 | **bridge** |
| Ann Surg | general_surgery | 0.632 | −0.153 | 0.042 | −0.430 | 0.042 | **0.430** | **terminal** |
| CORR | orthopedic | 0.692 | −0.162 | 0.000 | −0.132 | 0.000 | **0.132** | **terminal** |
| J Bone Joint Surg | orthopedic | 0.702 | −0.237 | 0.000 | −0.380 | 0.000 | **0.380** | **terminal** |
| JAMA Surg | general_surgery | 0.627 | −0.053 | 0.000 | −0.066 | 0.000 | **0.066** | **terminal** |

(Mid grain: same three sources Br J Surg / Surgery / J Am Coll Surg plus JAMA Surg flips to
source; the same three terminals Ann Surg / J Bone Joint Surg / CORR plus Spine flips to
terminal; Arthroscopy stays bridge. Full table in `role_scores.parquet`, `grain=="mid"`.)

---

## 3. Validation against the within-corpus prior

The true source generalists (Nature / NEJM / Lancet) are **not** in this 10-journal
surgical/orthopedic corpus, so we validate against the within-corpus expectation: generalist
*surgery* journals should look more source/bridge-like, subspecialty journals more
terminal-like.

**The prior largely holds, with one honest, defensible twist.**

- **Sources are the broad general-surgery journals.** Br J Surg, Surgery, and J Am Coll Surg
  publish topics early and export citations — they sit upstream in the within-panel flow,
  exactly as the prior predicts for generalist surgery journals. (J Arthroplasty also scores
  source: it is the corpus's single largest net citation *exporter*, a high-volume
  subspecialty engine.)
- **Bridges are subspecialty journals.** Arthroscopy and Spine carry the highest betweenness —
  topics pass *through* them on the lead→follow paths between early and late journals.
- **Terminals are the heavily-cited prestige journals.** Ann Surg, J Bone Joint Surg, CORR,
  and JAMA Surg score terminal. This is the twist worth stating plainly: by the V2-S05
  definition a *terminal* has high citation *in-flow* relative to out-flow, and these four are
  the corpus's biggest net citation **importers** (their papers are cited *within* the corpus
  far more than they cite out — J Bone Joint Surg nets −27,100 citations, Ann Surg −17,513).
  In a panel with **no true generalist source**, the most authoritative journals surface on the
  *receive* axis, not the *publish-first* axis. A "terminal" here is a within-panel net citation
  sink, **not** a global research dead-end — this is the panel-conditional caveat in action, and
  it is precisely the kind of attribution the expanded-corpus source layer (Phase F) is designed
  to correct.

So the roles are interpretable to a domain reader: generalist-surgery → source, subspecialty →
bridge, prestige-authority → terminal — with the caveat that "terminal" reads as "citation
authority within the panel" until real generalists are added.

---

## 4. Citational velocity

| Journal | Specialty | n citations | median lag (y) | IQR (y) | within-2y | velocity |
|---|---|---:|---:|---:|---:|---|
| J Arthroplasty | orthopedic | 357,116 | **4.0** | 5.0 | 0.277 | **fast** |
| Arthroscopy | orthopedic | 310,712 | 5.0 | 5.0 | 0.238 | slow |
| Br J Surg | general_surgery | 391,221 | 5.0 | 5.0 | 0.213 | slow |
| J Am Coll Surg | general_surgery | 257,733 | 5.0 | 5.0 | 0.224 | slow |
| JAMA Surg | general_surgery | 317,708 | 5.0 | 5.0 | 0.239 | slow |
| Surgery | general_surgery | 312,895 | 5.0 | 5.0 | 0.233 | slow |
| Ann Surg | general_surgery | 881,729 | 6.0 | 7.0 | 0.199 | slow |
| CORR | orthopedic | 505,617 | 6.0 | 6.0 | 0.162 | slow |
| J Bone Joint Surg | orthopedic | 631,951 | 7.0 | 6.0 | 0.154 | slow |
| Spine | orthopedic | 754,411 | 7.0 | 6.0 | 0.144 | slow |

**Headline.** Median time-to-citation runs **4–7 years**. **J Arthroplasty is the fastest**
(4-year median, 28% of citations within 2 years). The prestige *terminals* — Ann Surg, J Bone
Joint Surg, Spine — are the **slowest** (6–7-year median) yet accrue the **most** citations
overall, with the longest tails (IQR up to 7 years). Velocity and role-terminal status are
distinct axes: a journal can be heavily cited *and* slow to be cited. The relative fast/slow
indicator splits at the panel median (≈5 y), which puts only J Arthroplasty in "fast"; the
finer story is in the continuous median/IQR columns.

---

## 5. Jackknife stability vs. the protocol bar (Method C)

Drop-one-journal jackknife: re-run the role scoring ten times, each omitting one journal, and
measure the mean pairwise Spearman correlation of the source-axis ranking across runs
(`nulls.rank_stability` → `role_rank_correlation`). Protocol bar (validation_protocol §3):
PASS at `role_rank_correlation ≥ 0.7`.

| Grain | `role_rank_correlation` | Bar | Verdict |
|---|---:|---:|---|
| leaf | **0.9450** | ≥ 0.7 | **PASS** |
| mid | **0.9619** | ≥ 0.7 | **PASS** |

**Granularity sensitivity (Method D, supporting).** Leaf-vs-mid role-label agreement = **0.80**
(8/10 journals keep their label); component Spearman ρ ≈ 0.90 (source/terminal), 0.85 (bridge)
— well above the 0.5 robustness bar. The two journals that flip (JAMA Surg, Spine) sit near the
source/terminal boundary at both grains, so the flip is a boundary effect, not a reversal.

The roles are therefore **stable to the choice of panel** (no single journal's removal
reshuffles the ranking) and **robust across topic granularity**.

---

## 6. Gate G6 roles input

Gate G6 requires "journal source/bridge/terminal role scores earn **PASS** on the Method C
`role_rank_correlation` rubric (roles survive the jackknife) and are interpretable to a domain
reader" (validation_protocol §5.2).

**Verdict: roles PASS the Gate G6 stability input.** `role_rank_correlation` = 0.945 (leaf) /
0.962 (mid), both ≥ the 0.7 PASS bar; roles are robust across granularity (0.80 label agreement,
component ρ ≈ 0.85–0.90); and the assignments are interpretable against the within-corpus prior
(generalist-surgery → source, subspecialty → bridge, prestige-authority → terminal).

**Honest limits carried forward.** (1) *Panel-conditional* — no true generalist sources yet, so
"terminal" reads as "within-panel citation authority/importer," a label the Phase-F source layer
is expected to correct. (2) *1995 left-censoring* — the seeding sub-signal is biased by the panel
start; mitigated by the equally-weighted, censoring-free citation-flow and betweenness signals
(the heatmap shows these, not seeding, separate the labels). (3) The role *labels* are a coarse
argmax; the continuous component scores are the nuanced output the map (S07) should carry.

This feeds the G6 synthesis alongside S04's cascade null/jackknife verdict: roles are stable and
interpretable; the cascade-structure make-or-break verdict is S04's to report.
