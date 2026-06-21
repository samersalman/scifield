# V2-S09 Cartography — 78-Journal Expansion Results

**Status:** EXECUTED 2026-06-21 (the Phase-F expansion run; the headline test of the prove-phase)
**Author:** Samer Salman (executed by Claude Code session)
**Corpus:** 78 journals, 1,493,891 papers (1,391,306 distinct PMIDs), 1995–2025 — the expansion of the v1 10-journal panel
**Topics:** `data/v2/topics.parquet` — 149 leaf topics over 802,139 deduped research abstracts (BERTopic, `nr_topics=150`, seed 42), **subsample-assign fit (200k)**, 35.5% noise, 517,712 papers assigned
**Code:** `src/scifield/{thematic,novelty,cartography}/*` (unchanged since `b12de2f`) · `V2/scripts/{build_flow,build_cascade,build_roles,build_origins,run_cascade_validation,build_crosswalk}.py`
**Tables:** `V2/data_v2/{flow,cascade,roles,origins}/*.parquet`; `data/v2/{archetypes,cd_index,novelty_semantic}.parquet`; `data/v2/kuzu_graph/`; `V2/data/crosswalk/topic_crosswalk.parquet`
**Spend:** $0 (OpenAlex polite pool; clustering on Colab Pro; local compute; DeepSeek untouched)

> **One-line verdict:** the v2 cartography **reproduces and strengthens** the prove-phase. The inter-journal
> cascade is **REAL** and now **un-qualified** — the secondary lead-lag statistic (S2) that *failed* under the
> primary null in v1 **passes strongly at scale** (z≈+25–28). Roles are jackknife-stable (ρ=0.9993). And the
> headline source-layer test resolves into a sharp, defensible structure: **the generalist "source" expectation
> splits by journal type** — PNAS is the corpus's #1 source, but the high-impact clinical-prestige journals
> (NEJM, Lancet, JAMA, Nature, Science) are **terminal sinks**, confirming and sharpening v1's panel-bound finding.

---

## 0. Corpus, topics, and the upstream gate

| | v1 (prove-phase) | **v2 (expansion)** |
|---|---|---|
| Journals | 10 | **78** |
| Papers (distinct) | ~122k | **1,391,306** |
| Research abstracts embedded | — | **884,780 → 802,139 distinct** |
| Leaf topics | 149 | **149** |
| Topic noise | ~15–25% | **35.5%** (broad-corpus characteristic) |
| Papers assigned | — | **517,712** |

**Topic-model note.** A second Colab run with `--assign-mode full` (v1's exact method) was performed to test
whether the elevated noise was a subsample artifact. It was **not**: the full fit gave *worse* noise (38.8%) and
*coarser* topics (all orthopedics merged). The noise is intrinsic to a corpus this broad; the **subsample model
was kept** (finer topics, lower noise). 149 topics keeps the v1↔v2 crosswalk directly comparable.

**Enrichment gate.** OpenAlex match-rate = **88.1%** (1,226,073 matched / 1,391,306; 165,233 papers OpenAlex
does not index) — ~2 pp under the v1 ≥90% guideline. cited-by harvest = **3,210,276** forward-citation edges.
Reported as a coverage qualification, not a blocker: the cartography consumes the matched papers' refs/institutions.

---

## 1. Cascade — REAL, and now un-qualified (Method B/C/D)

Protocol = the locked `validation_protocol.md` (n_perm 1000; Null-1 = within-topic journal-label shuffle [primary];
Null-2 = first-year shuffle; S1 = seeding-score spread [primary]; S2 = net lead-lag asymmetry [secondary]). Rubric:
**PASS** = `p<0.01 ∧ z≥2`. Run: `V2/data_v2/cascade/{null,jackknife,granularity}_results.parquet` (elapsed 333 s).

### Method B — null/permutation (gate-critical)

| grain | subset | stat | null | z | p | verdict |
|---|---|---|---|---:|---:|---|
| leaf | all/resolvable | **S1** | within_topic | **+3.69** | 0.001 | **PASS** |
| leaf | all/resolvable | S1 | year_shuffle | +5.56 | 0.001 | PASS |
| leaf | all/resolvable | **S2** | within_topic | **+27.66** | 0.001 | **PASS** |
| leaf | all/resolvable | S2 | year_shuffle | +52.47 | 0.001 | PASS |
| mid | all/resolvable | S1 | within_topic | +3.74 | 0.001 | PASS |
| mid | all/resolvable | S2 | within_topic | +25.09 | 0.001 | PASS |

**The headline cascade result:** in v1, S2 under the primary within-topic null **failed** (z≈+0.4) — the
10-journal/1995-tied panel conserved S2's magnitude under the shuffle. In v2 the shuffle no longer conserves it,
and **S2 passes strongly (z≈+25–28)**. The corpus expansion did exactly what the prove-phase predicted: it
**removed v1's main qualification.** Note the all-subset and resolvable-subset z are identical → not a
left-censoring artifact.

### Method C — drop-one-journal jackknife of origin attributions

| grain | subset | flip_rate | forced | genuine flips | verdict |
|---|---|---:|---:|---:|---|
| leaf | all / resolvable | **0.0000** | 3 | 0 | **PASS** |
| mid | all / resolvable | **0.0000** | 2 | 0 | **PASS** |

Origin attributions are perfectly stable to dropping any journal (forced-reattribution count 3/2 — far below v1's
149/96, reflecting much less 1995-tie dominance with the broader, longer-history corpus). *Same honest caveat as
v1:* a 0.0 flip-rate is partly structural (origin = `argmin(first_year)` with deterministic tie-break).

### Method D — granularity & Method A — held-out (supporting)

- **Granularity:** leaf-vs-mid seeding-rank Spearman ρ = **0.9879** (bar 0.5) → **PASS** (up from v1's 0.855).
- **Held-out years (1995–2018 → 2019–2025), supporting-only:** leaf ρ=0.115 (FAIL), mid ρ=0.382 (PARTIAL) — still
  the weakest line, attributable to the short 7-yr window; does not sink the gate (B/C/D hold).

**Cascade verdict: REAL** — S1/Null-1 PASS at both grains and on the resolvable subset, **S2 now corroborates**,
origin jackknife flip-rate 0, granularity ρ=0.99. Stronger than v1's "REAL-with-qualification."

---

## 2. Journal roles — the source/terminal split (the V2-S09 headline)

`V2/data_v2/roles/role_scores.parquet` (78 journals × leaf grain). **Jackknife role_rank_correlation = 0.9993**
(leaf & mid; bar 0.7 → PASS — tighter than v1's 0.945/0.962). 78-journal distribution: **41 source / 29 terminal
/ 8 bridge.**

### The 8 Tier-1 generalists (the "Phase-F source layer" test)

| journal | role | source_z | net_outflow_share |
|---|---|---:|---:|
| **PNAS** | **source** | **+2.86** (#1 in corpus) | +0.041 |
| Ann Intern Med | source | +0.84 | −0.015 |
| BMJ | source | +0.43 | −0.018 |
| Lancet | terminal | −0.76 | −0.063 |
| Science | terminal | −0.88 | −0.060 |
| JAMA | terminal | −0.91 | −0.064 |
| Nature | terminal | −1.90 | −0.072 |
| **NEJM** | **terminal** | **−3.79** (most terminal in corpus) | −0.141 |

**Read.** The prove-phase hypothesis — "adding generalists as a source layer makes them emerge as sources" — is
**partially confirmed and sharpened, not uniformly true:**

1. **Basic-science generalists source.** PNAS is the single strongest source in the 78-journal corpus; Ann Intern
   Med and BMJ also source. (3/8.)
2. **Clinical-prestige generalists are genuine terminals.** NEJM, Lancet, JAMA (and Nature, Science) have **high
   seeding scores but negative net outflow** — work flows *into* them. NEJM is the most terminal journal in the
   whole corpus. This is **not** the v1 "within-panel sink" artifact (v1 had no true source layer to compare
   against); with 78 journals including specialty + basic-science sources, the prestige clinical journals are
   *still* terminal → a real structural property, not a panel composition effect.
3. **The actual source layer = PNAS + the specialty/surgical journals** (am_j_surg, br_j_surg, surg_endosc,
   ann_thorac_surg, world_j_surg, ann_surg_oncol …). The map's "source" is where work *originates and flows out*,
   which is the working-specialty literature and broad basic science — not the high-impact clinical journals that
   *consolidate* it.

This refines v1's roles finding from "prestige = within-panel sink (correctable)" to "**prestige clinical journals
are terminals even with a full source layer present; the source role belongs to specialty + basic-science venues.**"

---

## 3. Novelty & origins

Novelty chain (`data/v2/`): `novelty_semantic.parquet` (802,139 papers; mean same-field semantic novelty 0.69;
21,090 earliest-in-field), `cd_index.parquet` (CD-index, 5/10-yr windows), `kuzu_graph/` (property graph: 1.39M
Paper / 2.11M Author / 78 Journal / 107k Institution / 149 Topic; 7.14M CITES, 6.75M AUTHORED_BY, 5.54M
AFFILIATED_WITH, 517,712 ASSIGNED_TO), and the master `archetypes.parquet` (the dual-novelty 2×2; 27,220 papers
with complete semantic × structural novelty, spread evenly across disruptive-novel / incremental /
conventional-disruptive / novel-consolidating).

Origins (`V2/data_v2/origins/`): topic-origin attribution + sector/geo novelty + recombination built.
**Deferred (coverage, $0 session):** funding origin (OpenAlex grants parsed but not persisted — needs a schema
change + re-harvest) and citation intents (Semantic Scholar `citation_intents.parquet` empty — no S2 key).

---

## 4. Crosswalk (v1 ↔ v2)

`V2/data/crosswalk/topic_crosswalk.parquet` — **149/149 v1 topics map to v2** (top_k=3, centroid-cosine +
PMID-Jaccard). 447 mapping rows; best matches are strong (cosine **0.95–0.98**, e.g. v1-T0 knee → v2-T4 hip
cosine 0.95; v1-T1 → v2-T35 cosine 0.98). 149 v1 splits (>1 v2 candidate) and 49 v2 merges (>1 v1 source) — the
expected resolution shifts as the corpus broadens.

---

## 5. What changed from v1 → v2 (the prove-phase predictions, checked)

| Prove-phase expectation | v2 outcome |
|---|---|
| Expansion would *strengthen* the cascade and resolve S2/Null-1 | **Confirmed** — S2/Null-1 z +0.4 → **+27.7** |
| Generalists added as a source layer would surface as sources | **Split** — PNAS/AnnIntMed/BMJ source; NEJM/Lancet/JAMA/Nature/Science **terminal** |
| Roles would stay jackknife-stable | **Confirmed, tighter** — 0.945/0.962 → **0.9993** |
| Granularity robustness would hold | **Confirmed, tighter** — ρ 0.855 → **0.9879** |
| Held-out (Method A) becomes more informative with longer history | **Not yet** — still weak (leaf 0.12 / mid 0.38) |

---

## 6. Honest framing & caveats (carry with every claim)

1. **Topic noise = 35.5%** (284k papers unassigned). Intrinsic to a 78-journal corpus; the full-fit alternative
   was worse (38.8%). The cartography uses only the 517,712 assigned papers — rich, but the noise is a real
   coverage limit to state, and a `min_samples` sensitivity re-run is the documented way to characterize it.
2. **OpenAlex match-rate 88.1%** (165k unindexed papers) — ~2 pp under the v1 gate. Refs/institutions cover the
   matched 88%.
3. **Origins are still panel-conditional** (now a 78-journal panel, not the world) and **1995 left-censored** —
   though far less tie-dominated than v1 (jackknife forced count 3 vs 149).
4. **"Terminal" = net citation/flow sink within the 78-journal panel**, not a global dead-end. The continuous
   `source_z` / `net_outflow_share` are the nuanced output; the source/terminal label is a coarse argmax.
5. **Held-out years remain weak** — any forward-looking/trajectory claim must stay cautious.
6. **Funding + citation-intent layers deferred** (coverage gaps; $0 session).

---

## 7. Next

- Optional **`min_samples` sensitivity** re-run (e.g. 10→5) on Colab to characterize the 35.5% noise.
- Regenerate the interactive **`map_v0` for v2** (78-journal version) — V2-S09 map.
- Write-up / manuscript framing of the source/terminal split as the headline cartographic finding.
- Re-harvest with grant/intent persistence if the funding-origin layer is wanted.
