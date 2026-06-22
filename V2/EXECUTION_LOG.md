# V2 Cartography — Execution Log

Append-only log of cross-task hand-offs and batch status. Subagents: add a dated bullet under
your task heading if you learn something a downstream task must know. See `V2/CONTEXT.md`.

**Session scope:** Phases A–E (V2-S01 → V2-S07), $0, current corpus, stop at Gate G6.

---

## Batch 1 — Foundation (T1a charter/protocol, T1b flow+nulls, T2 infra spike)
_status: done — spec review PASS_

### T1a — Charter + validation protocol

- **2026-06-09** — Wrote `V2/docs/cartography/charter.md`, `V2/docs/cartography/validation_protocol.md`,
  `V2/README.md` (all LOCKED, dated 2026-06-09). The validation protocol RETIRES the V1 hypothesis
  gates and replaces them with four rigor methods (A held-out years, B null/permutation, C jackknife,
  D granularity). **V2-S03/S04/S05 must implement EXACTLY these locked constants** (validation_protocol §0):
  - **Held-out-year split:** learn on **1995–2018**, validate on **2019–2025** (temporal, never random).
    Method A statistic = Spearman `ρ_holdout` between train- and holdout-window per-journal seeding-score rankings.
  - **Permutation count = 1000; RNG seed = 20260609** (use `numpy.random.default_rng(20260609)`).
    Primary null = **Null-1 journal-label shuffle within topic**; secondary = **Null-2 first-appearance-year shuffle**.
    Test stats: **S1 = seeding-score spread (std across 10 journals)** [primary], **S2 = net directed lead-lag asymmetry**.
    p-value = add-one form `(1 + #{null ≥ obs}) / (1 + n_perm)`; effect size = z-score + percentile (BOTH required).
  - **Jackknife stability metric name = `jackknife_stability`**, two parts: **`attribution_flip_rate`**
    (origin-attribution change rate across 10 leave-one-out runs; excludes forced reattributions, reported
    separately as `forced_reattribution_count`) and **`role_rank_correlation`** (mean Spearman of role-score rankings).
  - **Granularity definition:** **leaf = 149 `topic_id`** vs **mid = 96 `mid_level_id`** (join
    `archetypes.topic_id → topic_hierarchy.topic_id → mid_level_id`, re-aggregate to mid). A conclusion is
    "robust" only if it holds at BOTH grains (leaf-vs-mid journal-ranking Spearman ρ ≥ 0.5 for PASS).
  - **min_papers = 1** to count a (topic, journal) appearance (matches `seeding.first_publication_year` default);
    secondary sensitivity at min_papers ∈ {1, 3}.
  - **Gate G6 PASS bar:** cascade PASS on Method B (S1/Null-1) AND Method C jackknife; roles PASS on
    `role_rank_correlation`; ≥1 novelty-origin finding PASS at both grains (Method D); map v0 coherent.
    Methods B+C are the make-or-break, gate-critical checks; Method A is supporting evidence only.
  - **Risk flagged for downstream:** the held-out 2019–2025 window is only 7 years; topics first appearing
    after 2019 are unscorable in the train window and must be excluded from the Method A comparison (report the
    excluded count). This is why Method A is supporting-only and B/C are gate-critical.

### T1b — Flow data model + null/jackknife primitives

- **2026-06-09** — Delivered `src/scifield/cartography/{__init__,flow,nulls}.py`, tests
  `tests/test_cartography_{flow,nulls}.py` (29 pass), builder `V2/scripts/build_flow.py`.
  Ran clean: ruff/black green, both flow tables materialized. **NO 11th journal** — the
  jama_surg collapse works (exactly 10 slugs in both grains). My `nulls.py` aligns with the
  T1a-locked protocol: same RNG seed 20260609, add-one p-value, `attribution_flip_rate` name,
  within-topic journal-label shuffle (Null-1).

- **flow.py public API** (pure, I/O-free; import as `scifield.cartography.flow`):
  - `build_flow_table(papers_topics, *, citation_edges=None, grain="leaf", min_papers=1) -> pd.DataFrame`
  - `first_appearance(papers_topics, *, topic_key="topic_id", min_papers=1) -> pd.DataFrame`
    (cols `[topic_key, journal_slug, first_year(int64)]`; delegates to
    `scifield.findings.seeding.first_publication_year`, NOT duplicated).
  - `add_citation_flows(flow, papers_topics, citation_edges, *, topic_key="topic_id") -> pd.DataFrame`
    (adds int64 `out_flow`/`in_flow`).
  - `assert_canonical_journals(journals) -> None` (raises if a non-slug / display name leaks in).
  - `topic_key_for_grain(grain) -> str` ("leaf"→"topic_id", "mid"→"mid_level_id"; else ValueError).
  - Constants `CANONICAL_JOURNAL_SLUGS` (the clean 10), `GRAIN_TOPIC_KEYS`.

- **Flow table schema (final column names + dtypes)** — same for both grains, topic key differs:
  `<topic_key>` int64 (`topic_id` leaf / `mid_level_id` mid), `journal_slug` str,
  `year` int64, `n_papers` int64 (volume), `is_first_appearance` bool (earliest year this
  (topic,journal) reached `min_papers` in a single year — the cascade anchor),
  `out_flow` int64, `in_flow` int64. Sorted by topic key, journal_slug, year.
  **Sentinel when `citation_edges=None`:** `out_flow=0` (int) AND `in_flow=NaN` (float64) =
  "citations NOT computed for this build" — a NaN `in_flow` means *unknown*, not a true zero.
  A genuine zero only appears when edges were supplied.

- **Citation-flow semantics:** `out_flow[cell]` = corpus-internal CITES edges whose **citing**
  paper falls in that (topic,journal,year) cell (citations *made by* the cell, anchored to the
  citing paper's pub year); `in_flow[cell]` = edges whose **cited** paper falls in the cell
  (citations *received*). Edges are corpus-internal (both endpoints in the 10-journal panel).
  Totals: out_flow=400,227 / in_flow=433,524 (≠ each other and < 574,478 edges, because
  endpoints in the dropped noise topic are unattributed — expected).

- **nulls.py public API** (pure, explicit `seed`, no global RNG; `scifield.cartography.nulls`):
  - `permute_labels(df, *, label_col, group_col=None, seed) -> pd.DataFrame` — shuffles
    `label_col` globally (group_col=None) OR **within** each `group_col` (preserves per-group
    label multiset; pass `label_col="journal_slug", group_col=<topic_key>` for protocol Null-1).
  - `permutation_test(observed_stat, null_stats, *, alternative="two-sided") -> dict`
    returns `{p_value, z, n_perm}`. p uses **add-one** estimator `(1+n_extreme)/(1+n_perm)`
    (never exactly 0). `alternative` ∈ {two-sided, greater, less}. NaN null replicates dropped.
  - `jackknife_drop_one(entities, compute_fn) -> pd.DataFrame` — calls `compute_fn(held_out)`
    per unique entity (dedups), stacks results with a prepended `held_out` column. `compute_fn`
    owns the actual exclusion+recompute; this only orchestrates+tags.
  - `attribution_flip_rate(per_run, *, key_col, value_col) -> float` in [0,1] — fraction of
    keys taking >1 distinct value across runs (NaN is its own value). Empty → 0.0.
    (NOTE: this does NOT itself separate the protocol's `forced_reattribution_count`; the
    S04/S05 caller must pre-filter forced reattributions and report that count separately,
    then pass the residual to this fn — keeps the primitive grain-agnostic.)
  - `rank_stability(per_run, *, key_col, score_col, run_col="held_out") -> float` — mean
    pairwise Spearman ρ of the ranking across runs (on shared entities); ≤1 run → 1.0.
    This is the primitive behind the protocol's `role_rank_correlation`.

- **Output parquets:** `V2/data/flow/flow_leaf.parquet` (13,507 rows, 149 topics × 10 journals,
  yr 1995–2026) and `V2/data/flow/flow_mid.parquet` (10,537 rows, 96 mid-level × 10 journals);
  each with a `.run.json` sidecar (record_run, task=V2-S01). **Consume `journal_slug` from these
  tables — do NOT re-derive it from archetypes' display `journal`.**

- **Canonical-journal join recipe (the resolver):** archetypes.pmid is BIGINT, papers_distinct.pmid
  is VARCHAR → `JOIN pdb.papers_distinct p ON CAST(a.pmid AS VARCHAR) = p.pmid`; take
  `p.journal_slug`. All 89,230 archetypes rows join (0 unmatched); jama_surg=4862 = the
  collapse of "JAMA surgery"(1817)+"Archives of surgery..."(3045). DuckDB opened READ_ONLY via ATTACH.

- **Gotchas for downstream:**
  - Builder drops the **noise topic `topic_id == -1`** (21,409 rows) → 67,821 papers remain
    (= 89,230 − 21,409 = the topic-assigned subset, matches Kuzu ASSIGNED_TO). If you reload
    archetypes yourself, filter `topic_id <> -1`.
  - **Stray `year == 2026`** (723 papers) is KEPT in the flow (real but partial harvest year).
    If you need a clean held-out cut, filter `year` yourself. (Aligns with T1a's 1995–2018 /
    2019–2025 split, where 2026 sits in the holdout window.)
  - pmid dtype: archetypes BIGINT, papers_distinct/Kuzu Paper.pmid STRING.
    `add_citation_flows` casts both endpoints + the attribution pmid to `string`, so int/str
    edge frames join fine — pass either.
  - `mid_level_id` requires joining `topic_hierarchy.parquet` (149 leaf → 96 mid); builder does
    this with a LEFT JOIN. Mid grain has fewer first-appearance rows (645 vs 909 leaf) because
    topics merge.

### T2 — Infrastructure spike

- 2026-06-09 — **DECISION: STAY** on Kùzu 0.11.3 + DuckDB 1.5.2 + Parquet for the prove
  phase (no migration). All 5–6 canonical map queries run **< 50 ms** on the current corpus
  (bench: `V2/scripts/infra_bench.py`; decision + thresholds:
  `V2/docs/cartography/infra_decision.md`). It is already the right hybrid: DuckDB owns
  aggregates/temporal/joins (Q1 first-appearance 41.6 ms, Q2 lag matrix 41.7 ms, Q4
  recombination 35.8 ms, Q5 sector/geo 23.4 ms); Kùzu owns traversal (Q3 seeded 1..3-hop
  cascade 8.7 ms — ~4× faster than the DuckDB recursive equivalent at 36.8 ms).
- **Slowest query = Q6 per-journal citational velocity (48.9 ms, Kùzu)** — a whole-graph CITES
  edge scan (574,478 edges), NOT seeded pathfinding. This is the canary for the Phase-F
  graph-traversal threshold. For S07: seeded cascades + all DuckDB aggregates are CHEAP;
  whole-graph edge aggregates (Q6) are the expensive pattern — pre-compute them offline.
- **GOTCHA (for S03/S04/S07):** `references_out.ref_pmid_if_known` is empty for **100%** of
  rows. To get corpus-internal citation edges in DuckDB/Parquet you MUST resolve
  `ref_openalex_id → archetypes.openalex_id → pmid` (recovers 541,493 of Kùzu's 574,478 edges;
  ~6% gap = ~33k corpus papers absent from archetypes). For EXACT corpus-internal traversal use
  the **Kùzu** graph (complete pre-resolved edge set). Q4 recombination already uses the
  `ref_openalex_id → topic_id` resolution path correctly.

## Batch 2 — Cascade engine (S03) + Novelty origins (S06)
_status: done — spec review PASS_

### S03 — Cascade engine

- **2026-06-09** — Delivered `src/scifield/cartography/cascade.py`, tests
  `tests/test_cartography_cascade.py` (13 pass; full cartography+seeding suite 56 pass, no
  regressions), builder `V2/scripts/build_cascade.py`, notebook `V2/notebooks/v2_01_cascades.ipynb`
  (executed headless, 0 errors). ruff/black green on all three code files. PURE + year-subset-safe:
  every function recomputes first-appearance from the rows it is handed (ignores any precomputed
  `is_first_appearance` flag), so S04 can call them on a `year <= 2018` slice for held-out validation.

- **cascade.py public API** (pure, I/O-free; import as `scifield.cartography.cascade`):
  - `origin_attribution(flow_or_first, *, topic_key="topic_id") -> pd.DataFrame`
    → `[<topic_key>, origin_journal_slug, origin_year(int64), n_journals(int64), tie(bool),
    n_co_earliest(int64)]`, one row per topic. **This is the cascade anchor S04 jackknifes.**
  - `lead_lag_matrix(flow_or_first, *, topic_key="topic_id") -> pd.DataFrame`
    → tidy long `[journal_i, journal_j, mean_lag(float), median_lag(float), n_shared(int64),
    n_i_leads(int64)]`, one row per ordered journal pair sharing ≥1 topic. **Sign convention:
    `mean_lag>0` ⇒ journal_i LEADS journal_j** (lag = first_year[j] − first_year[i]). Antisymmetric
    in means, symmetric in n_shared. **This is what S04 permutes (Null-1/Null-2).**
  - `diffusion_curve(flow_or_first, *, topic_key="topic_id", n_panel=10) -> pd.DataFrame`
    → `[<topic_key>, origin_year(int64), n_journals(int64), reach_fraction(float),
    span_years(int64), t50_empirical(float), t50_logistic(float), logistic_rate(float),
    curve_fitted(bool)]`, one row per topic.
  - `cascade_edges(papers_topics, *, topic_key="topic_id", min_papers=1) -> pd.DataFrame`
    → THIN WRAPPER over `seeding.directed_seeding_network` (not reimplemented); returns
    `[src, dst, n_precedes, n_shared, weight]` keyed on journal slugs.
  - Internal `_first_appearance_frame(flow_or_first, *, topic_key)` normalises EITHER a flow slice
    (uses `year`) OR a frame already carrying `first_year`; recomputes from the slice (year-subset-safe).
  - Constant `N_PANEL_JOURNALS = 10`.

- **Input contract:** every public fn accepts a flow-table slice (`[<topic_key>, journal_slug, year]`
  — exactly the shipped `flow_leaf/mid.parquet` columns) OR a `[<topic_key>, journal_slug, first_year]`
  frame. `topic_key` is `"topic_id"` (leaf) or `"mid_level_id"` (mid) — the output keys on whatever
  you pass. Consume `journal_slug` from the flow tables (never re-derive).

- **TIE-BREAK RULE (origin_attribution) — S04 must mirror this:** when ≥2 journals share the earliest
  first_year, `tie=True`, `n_co_earliest`=count, and `origin_journal_slug` = the **alphabetically-first
  co-earliest slug** (deterministic, jackknife-stable: a leave-one-out run yields the same winner
  unless THAT journal is dropped → a `forced_reattribution`, which T1a's protocol counts separately).
  For S04's `attribution_flip_rate`: a flip where the dropped journal WAS the prior origin is a forced
  reattribution (pre-filter + report `forced_reattribution_count`), not a genuine instability.

- **DIFFUSION-CURVE PARAMETRIZATION:** per topic, journals ordered by first_year; offset =
  first_year − origin_year (0 at origin). Cumulative fraction is over journals REACHED IN THE SLICE
  (always saturates at 1.0); `reach_fraction = n_journals/10` is the separate panel-saturation ceiling.
  `t50_empirical` = linear-interp offset where the step CDF hits 0.5 (=0.0 for single-journal topics).
  Logistic `f(t)=1/(1+exp(-rate*(t−t50)))` fit by scipy `curve_fit` when ≥3 distinct offsets;
  exponent clipped to ±50 + OptimizeWarning silenced (clean curves give singular covariance — params
  still valid). `curve_fitted=False` ⇒ `t50_logistic`/`logistic_rate` are NaN.

- **Output tables (both grains stacked, leading `grain` column "leaf"/"mid"); paths + counts:**
  - `V2/data/cascade/lag_matrix.parquet` — 180 rows (90 ordered pairs × 2 grains).
  - `V2/data/cascade/origin_attribution.parquet` — 245 rows (149 leaf + 96 mid topics).
  - `V2/data/cascade/diffusion_curves.parquet` — 245 rows.
  Each with a `.run.json` sidecar (`record_run`, config `task="V2-S03"`, records the tie-break +
  panel-conditional caveat strings). Builder: `.venv/bin/python V2/scripts/build_cascade.py`
  (`--grain leaf|mid|both`, `--spot-check N`).

- **SPOT-CHECK (5–10 leaf topics, richest cascades) reads sensibly:** topic 1 (PJI/periprosthetic
  infection) origin CORR→ortho journals at +0y then gen-surg later; topics 10 (resident training)
  & 18 (hospital mortality/care) origin Ann Surg→gen-surg fast, ortho slow; topics 35 (opioids),
  41 (VTE/aspirin), 54 (TXA/tranexamic) origin ortho journals (arthroplasty bleeding/thromboprophylaxis).
  Origin/follower ordering is domain-plausible.

- **CAVEATS for S04 (gate-critical) + S07 (map):**
  1. **1995 left-censoring is the dominant artifact.** Corpus starts 1995 → 139/149 leaf topics
     "originate" in 1995 (origin_year value_counts), and **tie rate = 79% leaf / 82% mid** because
     topics already present in multiple journals at panel start co-appear in year one. Read a 1995
     origin as "present at panel start", NOT "born 1995". S04's null/jackknife MUST account for this
     (e.g. exclude/flag 1995-origin ties, or the lead-lag permutation will be dominated by year-1 noise).
  2. **No single-journal topics at the current corpus** (0 of 149 leaf, 0 of 96 mid) — every topic
     reaches ≥2 journals, so every topic has a fittable diffusion curve (114/149 leaf logistic-fitted;
     the 35 unfitted have only 2 distinct offsets). At a year-subset (S04 held-out ≤2018) this may
     change → single-journal topics get `span_years=0, t50_empirical=0.0, curve_fitted=False`.
  3. **Mean reach ceiling ~0.61** — most topics never touch all 10 journals (only 18/149 reach all 10);
     reach_fraction is the cross-specialty-breadth signal.
  4. **Leader/follower ordering (supporting, not validation):** net-lead ranks CORR & J Bone Joint Surg
     top, J Arthroplasty & Spine bottom; Ann Surg is the most frequent origin (44 topics). Leaf-vs-mid
     net-lead **Spearman ρ = 0.96** (notebook §6, well above T1a's 0.5 robustness bar) — encouraging but
     S04 owns the formal Method-B/C verdict.
  5. lag_matrix is built from first-appearance years only (NOT citation flows); citations in the flow
     table (`out_flow`/`in_flow`) are untouched by S03 — available to S05 for velocity.

### S06 — Novelty origins

- **2026-06-09** — Delivered `src/scifield/cartography/origins.py` (pure, I/O-free),
  `tests/test_cartography_origins.py` (9 pass), builder `V2/scripts/build_origins.py`,
  notebook `V2/notebooks/v2_03_novelty_origins.ipynb` (executed headless, 0 errors, 4 figures).
  Additive `grants` parser support added to `src/scifield/corpus/openalex.py`. Ran clean:
  ruff/black green on all 4 touched files; `tests/test_corpus_*.py` 43 pass (parser edit did
  NOT regress the store/schema).

- **origins.py public API** (import as `scifield.cartography.origins`; all pure):
  - `novelty_by_institution_type(paper_novelty, paper_inst, institutions, *, novelty_cols, type_col="type", inst_key="institution_canonical_id") -> pd.DataFrame`
  - `novelty_by_country(..., novelty_cols, country_col="country_code", inst_key=..., low_n_threshold=30) -> pd.DataFrame`
  - `paper_sector_tags(paper_inst, institutions, *, type_col="type", inst_key=...) -> pd.DataFrame`
    (distinct `[pmid, type, is_company]`; the any-author rule lives here)
  - `recombination_events(references_topics, *, citing_col="citing_pmid", ref_topic_col="ref_topic_id") -> pd.DataFrame`
  - `recombination_by_topic(events, paper_topic, *, citing_col="citing_pmid", paper_key="pmid", topic_col="topic_id") -> pd.DataFrame`
  - Constant `COMPANY_TYPE = "company"`.

- **Paper → institution-type/country aggregation rule (any-author-affiliation):** join
  `paper_institutions.pmid → institutions.institution_canonical_id`, reduce to the **set** of
  distinct `type` (resp. `country_code`) per paper, then a paper contributes one observation to
  EACH sector/country it spans. A paper is "company-affiliated" iff ANY author institution has
  `type=='company'` (`is_company`). **Blank `type`/`country_code` dropped, never imputed.**
  CONSEQUENCE: per-row `n_papers` SUMS ABOVE the distinct paper total (multi-sector papers double-
  counted) — this is the "novelty *present in* sector X" framing, documented on each fn. Per-measure
  `<c>_n` is the non-null count for measure `c` (novelty NaNs skipped: sem_nov_mean 2,367 / cd5 5,271
  / cd10 4,712 nulls in archetypes).

- **Recombination metric:** per citing paper, from references resolved to a corpus topic
  (`references_out.ref_openalex_id → archetypes.openalex_id → topic_id`, T2's path; 433,519 resolved
  references): `n_distinct_topics` (count of distinct referenced topics; **≥2 ⇒ recombination event**)
  + `topic_entropy` (Shannon nats over the referenced-topic mix). Per-topic summary adds
  `mean_distinct_topics`, `mean_entropy`, `recombination_rate`. LOWER BOUND — out-of-corpus refs invisible.

- **Output schemas + paths** (all under `V2/data/origins/`, each with a `.run.json` sidecar, task=V2-S06):
  - `sector_novelty.parquet` (9 rows): `type`, `n_papers`(int64), then per novelty col in
    {sem_nov_mean, sem_nov_min, cd5, cd10}: `<c>_mean/_median/_std/_n`.
  - `geo_novelty.parquet` (63 well-sampled + low-n; 180-ish countries): same novelty cols +
    `country_code`, `n_papers`, `low_n`(bool, <30 papers).
  - `recombination.parquet` (81,581 papers): `citing_pmid`, `n_references_resolved`,
    `n_distinct_topics`, `topic_entropy`, `is_recombination`(bool).
  - `recombination_by_topic.parquet` (148 leaf topics): `topic_id`, `n_papers`, `mean_distinct_topics`,
    `mean_entropy`, `recombination_rate`, `top_words`(list, joined from topic_hierarchy).

- **Headline numbers (panel-conditional, 10-journal surgical/ortho corpus):**
  - SECTOR: company `sem_nov_mean=0.4107` (n=5,091) vs non-company paper-weighted `0.4103`
    (**delta +0.0004, essentially flat**); company `cd5=-0.3946` vs non-company `-0.4025` (delta +0.0079).
    Company sits MID-PACK on both axes → **novelty is NOT a tech/industry-sector phenomenon here. Weak signal.**
  - GEOGRAPHY: clear spread. Top semantic novelty = GH 0.447, HU 0.442, UG 0.432, MX 0.429, ET 0.426
    (LMIC/emerging research countries); bottom = KR 0.388, RS 0.391, TH 0.391, FI 0.392. **More
    differentiated axis** (likely topic-mix driven; descriptive only).
  - RECOMBINATION: 41.6% of papers (33,916/81,581) recombine ≥2 topics; mean 1.70 distinct ref-topics/paper.
    Most-recombinant focal topics = interdisciplinary bridges (mental-health/PROMIS, dialysis/transplant,
    biliary drainage, obesity/BMI, machine-learning, evidence-synthesis/trials).

- **GRANTS decision = DEFERRED store-wiring (parser-only).** Added `_parse_grants()` + a `grants` key
  to `parse_openalex_work`'s returned dict (funder/funder_display_name/award_id, prefix-stripped). Did
  NOT wire it through `enrich_store.OPENALEX_WORKS_SCHEMA` because: `pa.Table.from_pylist(rows, schema=...)`
  **silently drops** the extra key (verified), so the live 118k-row `openalex_works.parquet` is unchanged
  and grants are NOT persisted; adding the column to the schema would break the existing parquet +
  `test_corpus_enrich_store.py` round-trip UNLESS the store is re-pulled — and a re-harvest is out of scope
  ($0 session). **To enable funding origin later:** (a) add `grants` list<struct> to `OPENALEX_WORKS_SCHEMA`
  + a `_sample_row` entry, (b) targeted OpenAlex re-pull (polite pool) to backfill, (c) join `grants.funder
  → novelty`. **Citation-intent origin** likewise DEFERRED (`citation_intents.parquet` empty; needs
  `SEMANTIC_SCHOLAR_API_KEY` + a re-run). Both noted in the notebook's "Funding-origin: DEFERRED" cell.

- **Coverage caveats (carry into the map):** panel-conditional (10-journal surgical/ortho, not all science);
  type blanks ≈35% / country blanks ≈44% of institutions excluded (tables conditioned on resolved metadata);
  multi-affiliation double-counting (n_papers sums above paper total); recombination is a corpus-internal
  lower bound; funding+intent deferred.

- **FOR THE MAP (S07) + Gate G6:** ≥1 novelty-origin finding must be robust. **GEOGRAPHY is the most robust
  candidate** — it shows a wide, ordered spread across many well-sampled countries (vs the near-zero sector
  contrast). Surface the geo layer on the map; the SECTOR finding is essentially a NULL ("novelty is not
  sector-concentrated"), which is itself a clean, defensible result. The formal both-grains robustness check
  (leaf vs mid) is applied in the validation phase, not here — but geography is the axis to put through it.

## Batch 3 — Cascade validation (S04) + Roles & velocity (S05)
_status: done — spec review PASS_

### S04 — Cascade validation

- **2026-06-09** — Delivered `src/scifield/cartography/cascade_validation.py` (pure, I/O-free),
  `tests/test_cartography_cascade_validation.py` (17 pass; full cartography+seeding suite green,
  no regressions), builder `V2/scripts/run_cascade_validation.py` (runs in ~80 s, all cells at
  the LOCKED n_perm=1000 — no capping needed), doc `V2/docs/cartography/cascade_validation.md`,
  notebook `V2/notebooks/v2_01b_cascade_validation.ipynb` (executed headless, 0 errors, 5 figs).
  ruff/black green on all 3 code files. Locked constants used exactly (n_perm=1000, seed=20260609,
  add-one upper-tail p, Null-1 within-topic journal shuffle, Null-2 first-year shuffle, S1
  seeding-spread primary, S2 lead-lag-asymmetry secondary). Outputs:
  `V2/data/cascade/{null_results,jackknife_results,granularity_results}.parquet` + `.run.json`
  sidecars (task="V2-S04").

- **VERDICT = REAL (cascade is NOT a panel/temporal/1995-censoring artifact), with a documented
  PARTIAL qualification.** Gate-critical evidence is clean; secondary lines are weaker.
  - **Method B — null/permutation. Gate-critical primary S1/Null-1 (seeding-score spread,
    journal-label shuffle within topic) = PASS at BOTH grains AND on the resolvable subset:**
    leaf/all z=+3.27 p=0.0020; leaf/resolvable z=+3.22 p=0.0020 (138/149 topics);
    mid/all z=+4.04 p=0.0010; mid/resolvable z=+3.67 p=0.0010 (86/96 topics). Corroborated by
    S1/Null-2 (z 4.7–5.2). **The near-identical all-vs-resolvable z is the decisive 1995 test —
    the structure lives in genuinely-ordered topics, not the year-one ties.**
  - **S2 (secondary lead-lag asymmetry) does NOT corroborate under Null-1** (leaf z=+0.41 p=0.36;
    mid borderline z≈2.1 PARTIAL) because the within-topic label shuffle CONSERVES total asymmetry
    magnitude (it destroys who-leads-whom, not how-much); S2 PASSes under Null-2 (z 3.3–4.0). So
    the Method-B *block* verdict is PARTIAL at every grain×subset, driven entirely by S2/Null-1,
    while the designated primary S1/Null-1 is an unambiguous PASS.
  - **Method C — drop-one-journal jackknife of ORIGIN attributions = PASS everywhere:**
    `attribution_flip_rate = 0.0000` at leaf/all, leaf/resolvable, mid/all, mid/resolvable;
    `forced_reattribution_count` = topic count per grain (149/138/96/86), 0 genuine non-forced
    flips. CAVEAT: 0.0 is partly structural (origin=argmin(first_year) with deterministic
    alphabetical tie-break ⇒ dropping a non-origin journal can't change the earliest), so it
    confirms non-fragility rather than being a strong independent test.
  - **Method D — granularity = PASS:** leaf-vs-mid per-journal seeding-rank Spearman ρ = 0.855
    (bar 0.5).
  - **Method A — held-out years (SUPPORTING ONLY) = PARTIAL/FAIL:** leaf ρ=0.297 (PARTIAL,
    1 post-2018 topic excluded), mid ρ=0.091 (FAIL). Attributed to the 7-yr window + left-
    censoring; does not sink the gate; revisit with the expanded corpus.

- **Gate G6 cascade input = PROCEED (PASS-with-qualification):** the cascade earns PASS on the
  protocol's gate-critical primary (S1/Null-1, surviving the resolvable-subset 1995 sensitivity)
  + origin jackknife + granularity → systematic inter-journal seeding is REAL on the current
  corpus, not an artifact; qualifications = S2/Null-1 non-corroboration, PARTIAL held-out,
  structural jackknife. **SCOPE SPLIT for G6 synthesis: S04 owns the ORIGIN-attribution jackknife
  + the permutation nulls; S05 owns the ROLE-score jackknife (`role_rank_correlation`) — the
  OTHER half of Method C.** G6 must combine both halves of Method C with a robust V2-S06 novelty
  finding before reading the spend gate.

### S05 — Roles & velocity

- **2026-06-09** — Delivered `src/scifield/cartography/roles.py` (pure, networkx allowed in
  this layer; seeding.py stays networkx-free), `tests/test_cartography_roles.py` (12 pass; full
  cartography+seeding suite 68 pass, no regressions), builder `V2/scripts/build_roles.py`,
  notebook `V2/notebooks/v2_02_roles_velocity.ipynb` (executed headless, 0 errors, 3 figures),
  doc `V2/docs/cartography/roles_velocity.md`. ruff/black green on all 3 code files.

- **roles.py public API** (import as `scifield.cartography.roles`):
  - `role_scores(flow, directed_net, *, topic_key="topic_id", weight_threshold=0.5) -> pd.DataFrame`
    → per `journal_slug`: `seeding_score`, `net_outflow_share`, `betweenness`, the three
    components `source`/`bridge`/`terminal`, their z-scores `*_z`, and a discrete `role`.
  - `bridge_betweenness(directed_net, *, weight_threshold=0.5) -> pd.DataFrame` (networkx
    betweenness on the weight-filtered directed seeding network; distance = 1/weight).
  - `assign_roles(scores) -> pd.DataFrame` (role = **argmax of {source_z, bridge_z, terminal_z}**;
    ties break source > bridge > terminal).
  - `citational_velocity(citations, paper_meta, *, journal_col="journal_slug", ...) -> pd.DataFrame`
    → per journal: `median_lag`/`mean_lag`/`iqr_lag`/`q25`/`q75`/`frac_within_2y`/`velocity`.
  - `role_scores_jackknife(flow, *, topic_key, weight_threshold=0.5, score_col="source_z")` →
    drop-one-journal stacked frame; feed to `nulls.rank_stability(key_col="journal_slug",
    score_col="score", run_col="held_out")` for `role_rank_correlation`. **S05 owns the
    role-score jackknife; S04 owns origin-attribution.**
  - Constant `ROLE_COMPONENTS = ("source","bridge","terminal")`.

- **Score construction (documented rule):** SOURCE = mean of z(`seeding_score`) + z(net
  out-flow share, `Σ(out_flow−in_flow)` normalised by panel total |net|). BRIDGE = betweenness.
  TERMINAL = mean of z(−net out-flow share) + z(1−seeding) (near-mirror of source). The
  citation-flow + betweenness halves are **1995-censoring-free**; the seeding half is not, so
  it is given only half weight (S03 1995-tie caveat respected). The heatmap shows net-outflow +
  betweenness, not seeding, separate the labels.

- **ROLE ASSIGNMENT per journal (leaf grain):**
  - **source** = `br_j_surg`, `surgery`, `j_am_coll_surg` (the broad general-surgery generalists)
    + `j_arthroplasty` (biggest net citation *exporter*).
  - **bridge** = `arthroscopy`, `spine` (highest betweenness — topics route through them).
  - **terminal** = `ann_surg`, `j_bone_joint_surg_am`, `clin_orthop_relat_res`, `jama_surg`
    (the biggest net citation *importers* — heavily-cited prestige journals). NB: by the
    V2-S05 definition "terminal" = high in-flow/low out-flow = within-panel citation **sink**,
    NOT a global dead-end (panel-conditional). Mid grain: JAMA Surg→source, Spine→terminal flip
    (boundary), else identical.

- **`role_rank_correlation` (drop-one-journal jackknife, Method C) = 0.945 leaf / 0.962 mid —
  both PASS the protocol ≥0.7 bar.** Roles also robust across grain: leaf-vs-mid label
  agreement 0.80 (8/10), component Spearman ρ≈0.90 (source/terminal), 0.85 (bridge). **VERDICT:
  roles are STABLE and interpretable.**

- **VELOCITY headline:** median time-to-citation 4–7y. `j_arthroplasty` fastest (4y median,
  28% within 2y); the prestige terminals `ann_surg`/`j_bone_joint_surg_am`/`spine` slowest
  (6–7y) but accrue the MOST citations with the longest tails. Velocity ⟂ role-terminal status
  (a journal can be heavily cited *and* slow). Velocity join: `cited_by.parquet.focal_oa_id ==
  archetypes.openalex_id`, lag = `citing_year − pub_year`, journal via
  `archetypes.pmid → papers_distinct.journal_slug` (read-only DuckDB); lags <0 or >60 dropped.

- **Outputs:** `V2/data/roles/role_scores.parquet` (20 rows = 10 journals × 2 grains, carries
  `grain` + `specialty` cols) and `V2/data/roles/velocity.parquet` (10 journals, grain-
  independent paper-level), each with a `.run.json` sidecar (record_run, task=V2-S05; the
  role_scores sidecar records `role_rank_correlation_{leaf,mid}`). Builder:
  `.venv/bin/python V2/scripts/build_roles.py` (`--grain leaf|mid|both`).

- **Gate G6 roles input:** roles **PASS** the Method-C `role_rank_correlation` rubric (0.945/
  0.962 ≥ 0.7) and are interpretable to a domain reader → G6 roles requirement met. Carries
  forward to the G6 synthesis alongside S04's cascade null/jackknife verdict (the make-or-break).

- **FOR THE MAP (S07):** colour journals by `role` (`role_scores.parquet`, filter `grain=="leaf"`);
  the continuous `source`/`bridge`/`terminal` scores are the nuanced output (the label is a coarse
  argmax). Size nodes by `betweenness` for the bridge axis. Velocity (`velocity.parquet`) gives a
  per-journal speed badge. Reuse `ROLE_COLOR` {source #34a853, bridge #fbbc04, terminal #ea4335}
  from `v2_02_roles_velocity.ipynb`. Keep the panel-conditional "terminal = within-panel citation
  sink, not global dead-end" caveat on any role legend.

## Batch 4 — Map v0 (S07)
_status: done_

### S07 — Usable map v0

- **2026-06-09** — Delivered `src/scifield/cartography/mapio.py` (pure loader/prep layer,
  numpy-style docstrings, line-length 100), `tests/test_cartography_mapio.py` (16 tests pass;
  full cartography suite 96 passed + 1 skipped, no regressions), builder
  `V2/scripts/build_map.py`, and doc `V2/docs/cartography/map_v0_README.md` (dated 2026-06-09).
  ruff/black green on all three code files. **NO existing cartography module modified** — only
  `mapio.py` was added, as instructed.

- **OUTPUT:** `V2/map_v0/index.html` — self-contained static HTML, **126.8 KB**, **9 Plotly
  divs across 4 tabbed sections**. Opens by double-click, no server (streamlit/dash absent, not
  added). Plotly library loaded once from CDN via a single `<script src>` in the page head
  (`to_html(include_plotlyjs=False)` per div + one shared CDN ref); per-figure DATA is embedded,
  so the page is fully self-contained except the viewer's one-time library fetch. Sidecar
  `V2/map_v0/index.html.run.json` written (`record_run`, `task="V2-S07"`, 9 input parquet
  SHA-256s recorded).

- **SECTION LIST (the 4 tabs, each with an on-page caveat box):**
  - **a. Topic landscape** — 149 leaf topics as a sized, origin-coloured scatter (x = origin
    year, y = cross-journal reach, size = paper count, colour = origin journal; hover = top-words
    + recombination rate). Source: topic_hierarchy + origin_attribution + recombination_by_topic.
  - **b. Cascade flows** — (i) inter-journal lead-lag heatmap (signed mean lag), (ii) directed
    seeding network (10 nodes coloured by S05 role, edges lead→follow for pairs sharing ≥5 topics,
    node size ∝ |net-outflow share|), (iii) corpus adoption-by-breadth curve. Caveat carries
    panel-conditional + 1995-censoring + the **S04 REAL-with-qualification** verdict (S1/Null-1
    PASS both grains, jackknife flip-rate 0, granularity ρ=0.855).
  - **c. Journal roles** — source/bridge/terminal component bars + citational-velocity bars.
    Caveat carries the jackknife stability (role_rank_correlation 0.945/0.962) + the
    "terminal = within-panel citation sink, not global dead-end" caveat.
  - **d. Novelty origins** — sector novelty (company mid-pack, presented as a near-null),
    geography (top-vs-bottom well-sampled countries, the most-differentiated axis), top
    recombinant topics. Caveat flags the **deferred funding + citation-intent layers**.

- **mapio.py public API** (pure, read-only; import as `scifield.cartography.mapio`):
  `topic_labels()`, `load_lag_matrix(grain)`, `lag_matrix_wide(grain)`,
  `load_origin_attribution(grain)`, `load_diffusion_curves(grain)`, `adoption_curve(grain)`,
  `load_roles(grain)`, `load_velocity()`, `load_origins()` (dict of 4 frames),
  `geo_ranked()`, `topic_landscape_frame(grain="leaf")`, plus helpers `journal_display()`,
  constants `ROLE_COLOR` / `JOURNAL_DISPLAY` / `GRAIN_TOPIC_KEYS`. All grain-aware loaders
  default to `leaf` (the map's display grain); mid is the robustness grain, not the display.

- **Gate G6 map-coherence input = COHERENT (PASS-ready).** The assembled map renders all three
  required layers (cascades, roles, origins) on the current corpus, each with its caveat and its
  upstream validation verdict surfaced; the narrative lines up for a domain reader
  (generalist-surgery journals seed → subspecialty journals bridge → prestige journals are
  within-panel citation sinks), and the panel-conditional + 1995-censoring + deferred-layer
  limitations are foregrounded rather than buried. This is the last cartographic deliverable
  before the human Gate G6 synthesis.

---

## Post-batch fix — S06b novelty-origin robustness (closes final-review SHOULD-FIX #1)
_status: done_

### S06b — novelty-origin robustness numbers

- **2026-06-09** — The final reviewer found Gate-G6 ingredient (iii) ("≥1 robust novelty-origin
  finding") was only ASSERTED, with no quantified robustness. Topic-granularity Method-D is N/A by
  construction for per-institution `country_code`/`type` findings, so this fix supplied the
  appropriate grain-independent checks. Added two PURE functions to `src/scifield/cartography/origins.py`:
  `split_half_stability(...)` (random 50/50 + temporal early/late Spearman ρ of per-key novelty
  ranking, keys with ≥n_min papers in both halves) and `bootstrap_delta_ci(...)` (1000-rep stratified
  bootstrap of the company−non-company novelty delta). New builder `V2/scripts/build_origins_robustness.py`
  → `V2/data/origins/{geo_robustness,sector_robustness}.parquet` (+ sidecars, task=V2-S06b); new doc
  `V2/docs/cartography/novelty_origins.md`. +7 tests (origins test file now 16 pass). ruff/black green.
- **ACTUAL numbers (seed 20260609, 67,821 papers; 5,091 company vs 60,068 non-company):**
  - **Sector = robust, well-powered NEAR-NULL** (the cleanest ingredient-(iii) finding):
    `sem_nov_mean` delta=−0.0006 CI[−0.0031,+0.0019] spans_zero=True; `cd5` delta=+0.0034
    CI[−0.0106,+0.0173] spans_zero=True. Not a power problem (5,091 company papers).
  - **Geography = moderately robust** (secondary): split-half ρ `sem_nov_mean` random=0.669 /
    temporal=0.508; `cd5` random=0.490 / temporal=0.653.
- **Correction to S06 hand-off:** S06 asserted geography was "the most robust" origin finding — by the
  actual numbers, the **sector well-powered near-null is the cleaner robust finding**; geography is a
  reproducible-but-not-razor-sharp secondary. Ingredient (iii) is SATISFIED either way.

## Final review — whole-program acceptance
_status: ACCEPT_

- **2026-06-09** — Final reviewer over the full diff: **ACCEPT**. 117→103+ tests green, locked
  constants identical across all batches, canonical 10-slug rule holds in all 14 parquets, provenance
  chain verified (sidecar input-hashes match live files), no overclaiming, scope respected ($0, no
  migration, G6 left as a human decision). SHOULD-FIX #1 (novelty robustness) closed by S06b above.
  SHOULD-FIX #2 (`.gitignore` swallows `V2/data/`) resolved as INTENDED — tables are regenerable from
  builders at $0; documented in `V2/README.md` ("Reproducing the data tables"). Stale batch-status
  headers fixed. Gate-G6 readiness packet written to `V2/docs/cartography/G6_readiness.md`.

## 🚦 GATE G6 — SIGNED PASS
_status: SIGNED PASS by Samer Salman, 2026-06-09_

- **2026-06-09** — Samer signed **Gate G6 = PASS** in `V2/docs/cartography/G6_readiness.md`. All four
  ingredients cleared their pre-locked bars (cascade REAL-with-qualification + survives 1995-censoring
  control; roles stable 0.945/0.962; sector novelty robust well-powered near-null; map coherent).
  **Decision: fund Phases F–G.** Phases A–E prove-phase is CLOSED.
- **Next: V2-S08 (expansion harvest)** begins in a fresh Claude Code session. ⚠️ **DeepSeek spend-gating
  is now LIVE** — `feedback-deepseek-spend-gating`: dry-run cost estimate + explicit per-run OK from
  Samer before ANY DeepSeek extraction. Plan §"Phase F / V2-S08": hybrid corpus = source-layer
  generalists (Nature/Science/NEJM/Lancet/JAMA/PNAS) + OpenAlex-complete deep coverage in 2–3 existing
  specialties; re-harvest + re-enrich + **re-embed/re-cluster** (adding generalists moves the topic model
  — build an old↔new topic crosswalk, not "more rows through the same topics"); log spend in
  `docs/operations/api_costs.md`. Then V2-S09 re-runs cartography at scale (key test: do the source
  generalists actually show up as sources, and do they change the specialty-topic origin attributions?).

---

## V2-S10 — Trajectory / "future" layer
_status: COMMITTED & GREEN 2026-06-21_

- **What:** added the missing *future* dimension to the V2 cartography — an honest, **descriptive**
  per-topic trajectory projection with uncertainty bands. Headline metric = topic **share** of annual
  assigned output (removes the corpus-growth confound); raw **volume** kept as a secondary read. This
  is **explicitly NOT a predictive claim and NOT the dead F3 GNN** (F3 = emergence classifier, signed
  **NULL** at Gate G4: sealed test AUC **0.781** < no-graph **0.804**, H1 FAIL). No
  `src/scifield/forecasting/*` reuse — fresh descriptive layer.
- **Method:** per-topic **state-space local-linear-trend** (`statsmodels UnobservedComponents`), fit on
  `logit(share)` / `log(volume)` then back-transformed `conf_int(alpha=1−ci_level)` bands; deterministic
  **log/logit-linear fallback** for short or non-converging series. `ci_level=0.80`, horizon **+5y to 2030**,
  **leaf** grain. Fit window **1995–2025**; **2026 is a partial harvest year (~3,872 papers vs ~28k/yr) and
  is EXCLUDED from fitting** — it surfaces only as the first projected year. `topic_id == -1` excluded from
  both the numerator and the share denominator.
- **Code/artifacts:** `scifield.cartography.trajectory` (pure; public surface
  `build_topic_year_series` + `fit_trajectories`) · `V2/scripts/build_trajectory.py` ·
  `V2/notebooks/v2_05_trajectories.ipynb` (executed headless, 0 errors; 4 figures
  `figures/v2_05_{fanchart_grid,fan_rising,fan_falling,share_vs_volume}.png`) ·
  `V2/docs/cartography/trajectories.md` (methods doc). Outputs
  `V2/data_v2/trajectory/{trajectory_series,trajectory_summary}.parquet` (+ `.run.json` sidecars,
  `record_run` config `task="V2-S10"`, `ci_level=0.8`, `horizon=5`, `max_complete_year=2025`,
  `model="state_space_llt+loglinear_fallback"`; `input_hashes` verified against the live
  `archetypes.parquet` / `topic_hierarchy.parquet`).
- **ACTUAL numbers (data-version v2; 802,139 papers; 149 leaf topics):** **149/149 `fit_ok`**; model
  split **state_space_llt=146 / loglinear_fallback=3**; direction **flat=81 / falling=38 / rising=30**;
  series table **5,352 rows = 4,607 observed (1995–2025) + 745 projected (2026–2030, = 149 topics × 5-yr horizon)**.
  - **Rising (slope share/yr, illustrative):** topic 2 *health/information/data/care/digital*
    **+0.00331** (`last_obs` 0.139 → `proj2030` 0.155); topic 13 *covid19/influenza/sarscov2* **+0.00133**
    (0.020 → 0.027); topic 0 *knee/cartilage/acl/ligament* **+0.00132**; topic 72 *opioid/pain* **+0.00033**.
  - **Falling (slope share/yr, illustrative):** topic 10 *coronary/heart/myocardial/cardiac* **−0.00080**;
    topic 32 *sepsis/shock/septic* **−0.00070**; topic 9 *renal/kidney/glomerular* **−0.00063**;
    topic 7 *dna/rna/transcription* **−0.00054**.
- **Tests/gates:** `tests/test_cartography_trajectory.py` **17 passed**; full suite green;
  `nbconvert --execute` of `v2_05_trajectories.ipynb` runs clean (0 errors, 4 figures);
  ruff/black/mypy (pre-commit) green. **$0** — local CPU on existing tables, no DeepSeek/API.
- This is V2 **Phase F–G** work, continuing after the V2-S08 expansion harvest (1.49M papers / 78
  journals) and the V2-S09 scaled cartography re-run (logged via commits + findings docs, not in this file).
