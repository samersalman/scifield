# SciField V2 Cartography — Data Dictionary & Deposit README

This file documents the **derived cartography tables** that ship with the SciField
v0.2.0 release. It doubles as the README for the public data deposit.

The deposit is the small (~1.7 MB) set of *derived* tables under `V2/data_v2/`, plus a
topic-label dictionary. The raw 1.49M-paper corpus, embeddings, FAISS index, and topic
graph are **not** redistributed — they are reproducible from the pipeline in this repo
(see the repo `README.md` and `docs/`).

Every table has a machine-readable provenance sidecar next to it named
`<table>.parquet.run.json` (config, input hashes, git SHA, software versions, timestamp).
The sidecars are part of the deposit and are the authoritative record of how each table
was produced.

---

## Read this first — headline caveats

These caveats govern interpretation of *all* tables below. They are not optional footnotes.

1. **Origins are PANEL-CONDITIONAL.** Any "origin", "first appearance", "source", or
   "seeding" signal is *first/source within the 78-journal panel*, **not** first in the
   world. Several true generalist sources of science (e.g. *Nature*, *NEJM*, *Lancet*,
   *Science*, *PNAS*) **are** in this 78-journal panel, but are read only as they appear
   *inside the panel* — they cannot be credited as world-origins. (The panel's headline
   read is itself such a within-panel result: PNAS ranks #1 source while NEJM is the most
   terminal.) Read every origin claim as "earliest **observed in this panel**".

2. **The corpus is 1995 LEFT-CENSORED.** Coverage starts in 1995. Any topic, journal, or
   diffusion that began before 1995 has its onset clipped to the panel start. Signals that
   depend on *onset timing* (origin year, seeding, t50) inherit this bias; flow- and
   betweenness-based signals are comparatively robust to it (see `roles/`).

3. **`share` is panel-conditional.** Topic `share` is the fraction of *panel* papers in a
   topic-year, not a fraction of all of science.

4. **Trajectory is a DESCRIPTIVE projection, not a forecast.** The `trajectory/` tables
   extrapolate each topic's observed share/volume +5 years (to **2030**) with 80%
   uncertainty bands. This is a *descriptive* state-space/log-linear extrapolation. It is
   **not** causal, **not** a validated predictive model, and is explicitly **not** the V1
   F3 GNN forecasting result (which returned a null finding). Do not read the bands as a
   tested forecast.

5. **Grain.** Many tables exist at two clustering granularities: `leaf` (149 fine-grained
   topics, `topic_id` 0–148) and `mid` (coarser `mid_level_id` groupings). Where a `grain`
   column is present it tells you which rows are which; filter on it before aggregating.
   The headline cartography uses `grain == "leaf"`.

---

## Topic label dictionary

### `topic_hierarchy.parquet`
The `topic_id` → human-readable label lookup. **Join key for almost everything else.**
149 rows (one per leaf topic). Also carries the leaf→mid→top cluster hierarchy.

| column | dtype | meaning |
|---|---|---|
| `topic_id` | int64 | Leaf topic id, 0–148. Primary join key against the cascade/flow/trajectory tables. |
| `top_words` | list[str] | Ordered top c-TF-IDF keywords for the topic — use the first ~5 as a short label. |
| `size` | int64 | Number of corpus papers assigned to the topic. |
| `mid_level_id` | int64 | Mid-granularity cluster id this leaf rolls up into (join key for `grain == "mid"` rows). |
| `top_level_id` | int64 | Top-granularity cluster id (coarsest level of the hierarchy). |
| `representative_docs` | list[str] | A few representative document title/abstract snippets for eyeballing the topic. |

---

## `cascade/` — journal lead–lag diffusion (V2-S04/S09)

How topics move *between journals*: who leads, who follows, where a topic is first observed
in the panel, and how fast it spreads. **Origin signals here are panel-conditional and
left-censored (caveats 1–2).**

### `cascade/lag_matrix.parquet`
Pairwise journal lead–lag: for each ordered journal pair, the typical time offset at which
they enter shared topics. **Grain note:** has a `grain` column (`leaf` / `mid`).

| column | dtype | meaning |
|---|---|---|
| `grain` | str | `leaf` or `mid` — clustering granularity these lags were computed at. |
| `journal_i` | str | Journal slug *i* (the candidate leader). |
| `journal_j` | str | Journal slug *j* (the candidate follower). |
| `mean_lag` | float64 | Mean year offset across shared topics. Negative = *i* leads *j*; sign convention is i-relative. |
| `median_lag` | float64 | Median year offset across shared topics (robust version of `mean_lag`). |
| `n_shared` | int64 | Number of topics shared by *i* and *j* (support for the lag estimate). |
| `n_i_leads` | int64 | Of the shared topics, how many *i* entered first. |

### `cascade/origin_attribution.parquet`
For each topic, the journal in which it is **first observed in the panel**. Read strictly
as panel-conditional first appearance (caveats 1–2). **Grain note:** `grain` column.

| column | dtype | meaning |
|---|---|---|
| `grain` | str | `leaf` / `mid`. |
| `topic_id` | float64 | Topic id (leaf id when `grain == "leaf"`). |
| `origin_journal_slug` | str | Journal slug where the topic is earliest observed in the panel. |
| `origin_year` | int64 | Earliest observed year (≥ 1995; clipped onset if the topic predates the panel). |
| `n_journals` | int64 | Number of journals that ever carry the topic. |
| `tie` | bool | True if multiple journals share the earliest year (origin is ambiguous). |
| `n_co_earliest` | int64 | Number of journals tied at the earliest year. |
| `mid_level_id` | float64 | Mid cluster id (populated for `grain == "mid"` rows; NaN at leaf). |

### `cascade/diffusion_curves.parquet`
Per-topic spread curve across journals — reach over time and a fitted logistic. **Grain
note:** `grain` column.

| column | dtype | meaning |
|---|---|---|
| `grain` | str | `leaf` / `mid`. |
| `topic_id` | float64 | Topic id. |
| `origin_year` | int64 | Panel-conditional onset year (see caveats). |
| `n_journals` | int64 | Number of journals the topic reaches. |
| `reach_fraction` | float64 | Share of panel journals reached (reach normalized by panel size). |
| `span_years` | int64 | Years from onset to last observation. |
| `t50_empirical` | float64 | Empirical years-to-50%-reach. |
| `t50_logistic` | float64 | Years-to-50% from the fitted logistic. |
| `logistic_rate` | float64 | Fitted logistic growth rate (diffusion speed). |
| `curve_fitted` | bool | Whether the logistic fit converged (False ⇒ treat fitted columns as unreliable). |
| `mid_level_id` | float64 | Mid cluster id (NaN at leaf). |

### `cascade/null_results.parquet`
Permutation-null robustness for the cascade signal (within-topic and year-shuffle nulls).
Provenance/QA table — supports the "cascade is REAL" claim. **Grain note:** `grain` column.

| column | dtype | meaning |
|---|---|---|
| `grain` | str | `leaf` / `mid`. |
| `subset` | str | Which subset of cells the statistic was computed over. |
| `stat` | str | Statistic id (`S1`, `S2`). |
| `null` | str | Null model: `within_topic` or `year_shuffle`. |
| `observed` | float64 | Observed value of the statistic. |
| `null_mean` | float64 | Mean of the statistic under the null. |
| `null_std` | float64 | Std of the statistic under the null. |
| `z` | float64 | Standardized distance of observed from the null. |
| `p_value` | float64 | Permutation p-value. |
| `percentile` | float64 | Percentile of observed within the null distribution. |
| `n_perm` | int64 | Number of permutations. |
| `n_topics` | int64 | Number of topics entering the statistic. |
| `cell_verdict` | str | Per-cell PASS/FAIL verdict. |
| `method_b_verdict` | str | Verdict under the alternative (method-B) test. |

### `cascade/granularity_results.parquet`
Robustness of the cascade ordering across leaf-vs-mid granularity and held-out checks.
Small QA table.

| column | dtype | meaning |
|---|---|---|
| `method` | str | Robustness method (e.g. `D_granularity`, `A_heldout`). |
| `grain` | str | Comparison granularity (e.g. `leaf_vs_mid`, `leaf`). |
| `spearman_rho` | float64 | Rank correlation between the two orderings being compared. |
| `bar` | float64 | Pre-registered pass threshold for `spearman_rho`. |
| `n_compared` | int64 | Number of items compared. |
| `n_excluded` | int64 | Number excluded from the comparison. |
| `supporting_only` | bool | Whether the check is supporting-evidence only (not a gating test). |
| `verdict` | str | PASS / FAIL. |

### `cascade/jackknife_results.parquet`
Leave-one-out stability of origin attribution — how often the panel-origin flips when a
journal is dropped. Small QA table.

| column | dtype | meaning |
|---|---|---|
| `grain` | str | `leaf` / `mid`. |
| `subset` | str | Cells considered (`all`, `resolvable`). |
| `attribution_flip_rate` | float64 | Fraction of origins that change under jackknife. |
| `forced_reattribution_count` | int64 | Count of forced re-attributions. |
| `n_eligible_cells` | int64 | Eligible topic×journal cells. |
| `n_genuine_flips` | int64 | Genuine (non-tie) attribution flips. |
| `n_topics_flipped` | int64 | Topics whose origin flipped. |
| `n_topics` | int64 | Total topics. |
| `verdict` | str | PASS / FAIL. |

---

## `flow/` — topic-by-journal-by-year flow (V2-S04)

The substrate the cascade and roles are built on: how many papers each journal contributes
to each topic each year, and the directed in/out flow used for the seeding network. Two
files, one per grain.

### `flow/flow_leaf.parquet`
Leaf-grain flow. Grain is implicit in the filename (no `grain` column); join on `topic_id`.

| column | dtype | meaning |
|---|---|---|
| `topic_id` | int64 | Leaf topic id (join to `topic_hierarchy.topic_id`). |
| `journal_slug` | str | Journal slug. |
| `year` | int64 | Publication year (≥ 1995). |
| `n_papers` | int64 | Papers from this journal in this topic-year. |
| `is_first_appearance` | bool | True if this is the journal's first year in the topic (panel-conditional). |
| `out_flow` | int64 | Directed out-flow (this journal seeding others) for the topic-year. |
| `in_flow` | int64 | Directed in-flow (this journal following others). |

### `flow/flow_mid.parquet`
Mid-grain flow. Identical schema to `flow_leaf` except the topic key is `mid_level_id`
(int64) instead of `topic_id`. Join on `topic_hierarchy.mid_level_id`.

---

## `roles/` — journal roles & velocity (V2-S05/S09)

Where each journal sits in the diffusion network (source / bridge / terminal) and how fast
its papers get cited.

> **Role caveat (panel-conditional, caveats 1–2):** roles are *within-panel*. The seeding
> sub-signal is biased by the 1995 panel start; the flow and betweenness sub-signals are
> not. Generalist journals (e.g. *Nature*, *NEJM*, *PNAS*) **are** in this 78-journal
> panel, so a journal labelled `source` is a source *relative to this panel*, not a claim
> about world-first origination. The role rank is highly stable across grains
> (leaf↔mid Spearman ρ ≈ 0.999; see `role_scores.parquet.run.json`).

### `roles/role_scores.parquet`
Per-journal role classification from z-scored source/bridge/terminal sub-scores. **Grain
note:** `grain` column (`leaf` / `mid`).

| column | dtype | meaning |
|---|---|---|
| `grain` | str | `leaf` / `mid`. |
| `journal_slug` | str | Journal slug. |
| `seeding_score` | float64 | Tendency to introduce topics ahead of peers (left-censoring-sensitive). |
| `net_outflow_share` | float64 | Net directed out-flow share (seeds more than it follows). |
| `betweenness` | float64 | Betweenness centrality on the directed seeding network (robust to left-censoring). |
| `source` | float64 | Source sub-score = mean z(seeding_score) + z(net_outflow_share). |
| `bridge` | float64 | Bridge sub-score (= betweenness component). |
| `terminal` | float64 | Terminal sub-score = mean z(−net_outflow_share) + z(1 − seeding_score). |
| `source_z` | float64 | Z-scored source sub-score (used in the argmax). |
| `bridge_z` | float64 | Z-scored bridge sub-score. |
| `terminal_z` | float64 | Z-scored terminal sub-score. |
| `role` | str | Assigned role: argmax of the z-scores; ties broken source > bridge > terminal. One of `source` / `bridge` / `terminal`. |
| `specialty` | str | Journal specialty tag (may be null/NaN). |

### `roles/velocity.parquet`
Per-journal citation velocity — how quickly its papers accrue citations. 78 rows (one per
panel journal). No `grain` column.

| column | dtype | meaning |
|---|---|---|
| `journal_slug` | str | Journal slug. |
| `n_citations` | int64 | Citations observed for the journal's papers (in panel). |
| `n_papers_cited` | int64 | Distinct cited papers. |
| `median_lag` | float64 | Median years from publication to citation. |
| `mean_lag` | float64 | Mean years from publication to citation. |
| `iqr_lag` | float64 | Inter-quartile range of the lag. |
| `q25_lag` | float64 | 25th-percentile lag. |
| `q75_lag` | float64 | 75th-percentile lag. |
| `frac_within_2y` | float64 | Fraction of citations arriving within 2 years. |
| `velocity` | str | Discretized velocity class: `fast` or `slow`. |
| `specialty` | str | Journal specialty tag (may be null/NaN). |

---

## `origins/` — novelty origins (V2-S06/S09)

Where novelty comes from, by institution sector and by country, plus a recombination
(combinatorial novelty) signal. **All origin signals are panel-conditional and
left-censored (caveats 1–2).** Novelty metrics: `sem_nov_*` are embedding-based semantic
novelty; `cd5`/`cd10` are CD-index (disruption) at 5- and 10-year windows.

### `origins/sector_novelty.parquet`
Novelty aggregated by institution **sector type** (ROR institution type). 9 rows, one per
sector. No `grain` column; grain is the sector `type`.

| column | dtype | meaning |
|---|---|---|
| `type` | str | Institution sector (`archive`, `company`, `education`, `facility`, `funder`, `government`, `healthcare`, `nonprofit`, `other`). |
| `n_papers` | int64 | Papers attributed to the sector. |
| `sem_nov_mean_mean` | float64 | Mean over papers of the per-paper *mean* semantic novelty. |
| `sem_nov_mean_median` | float64 | Median of the per-paper mean semantic novelty. |
| `sem_nov_mean_std` | float64 | Std of the per-paper mean semantic novelty. |
| `sem_nov_mean_n` | int64 | N papers with a mean-semantic-novelty value. |
| `sem_nov_min_mean` | float64 | Mean of the per-paper *min* (most-novel-neighbor) semantic novelty. |
| `sem_nov_min_median` | float64 | Median of the per-paper min semantic novelty. |
| `sem_nov_min_std` | float64 | Std of the per-paper min semantic novelty. |
| `sem_nov_min_n` | int64 | N papers with a min-semantic-novelty value. |
| `cd5_mean` | float64 | Mean CD-index at the 5-year window (disruption; + = disruptive, − = consolidating). |
| `cd5_median` | float64 | Median CD-index (5y). |
| `cd5_std` | float64 | Std CD-index (5y). |
| `cd5_n` | int64 | N papers with a CD-5 value. |
| `cd10_mean` | float64 | Mean CD-index at the 10-year window. |
| `cd10_median` | float64 | Median CD-index (10y). |
| `cd10_std` | float64 | Std CD-index (10y). |
| `cd10_n` | int64 | N papers with a CD-10 value. |

### `origins/geo_novelty.parquet`
Same novelty metrics aggregated by **country**. 219 rows. Columns mirror
`sector_novelty.parquet` (the `sem_nov_*` and `cd5/cd10` families are identical) with the
grouping key and a low-sample flag swapped in:

| column | dtype | meaning |
|---|---|---|
| `country_code` | str | ISO country code of the contributing institution(s). |
| `n_papers` | int64 | Papers attributed to the country. |
| `sem_nov_mean_*`, `sem_nov_min_*`, `cd5_*`, `cd10_*` | float64 / int64 | Identical definitions to `sector_novelty.parquet` above (mean / median / std / n for each metric family). |
| `low_n` | bool | True if the country has too few papers for a stable estimate — **filter these out** for headline reads. |

### `origins/recombination.parquet`
Per-paper combinatorial-novelty signal: how many distinct topics a paper's resolved
references span. One row per citing paper. 53,431 rows.

| column | dtype | meaning |
|---|---|---|
| `citing_pmid` | str | PMID of the citing paper. |
| `n_references_resolved` | int64 | References resolved to a corpus topic. |
| `n_distinct_topics` | int64 | Distinct topics spanned by those references. |
| `topic_entropy` | float64 | Shannon entropy of the topic mix of the references (spread of recombination). |
| `is_recombination` | bool | True if the paper exceeds the recombination threshold (combinatorially novel). |

### `origins/recombination_by_topic.parquet`
Recombination rolled up **per topic**. 149 rows (one per leaf topic).

| column | dtype | meaning |
|---|---|---|
| `topic_id` | int64 | Leaf topic id (join to `topic_hierarchy`). |
| `n_papers` | int64 | Papers in the topic with a recombination value. |
| `mean_distinct_topics` | float64 | Mean distinct reference-topics per paper in the topic. |
| `mean_entropy` | float64 | Mean reference-topic entropy per paper. |
| `recombination_rate` | float64 | Fraction of the topic's papers flagged `is_recombination`. |
| `top_words` | list[str] | Convenience copy of the topic's top keywords. |

---

## `trajectory/` — descriptive topic trajectories (V2-S10)

Per-topic share/volume extrapolated +5 years to **2030** with 80% bands. **DESCRIPTIVE
PROJECTION, not a forecast — see caveat 4.** Not the F3 GNN.

### `trajectory/trajectory_summary.parquet`
One row per leaf topic (149): the fitted model, recent slope, direction, and the 2030
projection point with 80% bands. **Grain note:** `grain` column (`leaf`).

| column | dtype | meaning |
|---|---|---|
| `grain` | str | `leaf`. |
| `topic_id` | int64 | Leaf topic id (join to `topic_hierarchy`). |
| `label` | str | Short comma-joined topic label (first top-words). |
| `size` | Int64 | Topic size (papers). |
| `n_years_observed` | int64 | Number of observed years feeding the fit. |
| `last_obs_year` | int64 | Last observed year (the projection base; typically 2025). |
| `last_obs_share` | float64 | Observed panel share in the last observed year. |
| `model` | str | Fitted model: `state_space_llt` (local-linear-trend) or `loglinear_fallback`. |
| `slope_share_per_yr` | float64 | Recent share slope (per year); sign drives `direction`. |
| `direction` | str | `rising` / `flat` / `falling`. |
| `horizon_year` | int64 | Projection horizon (2030 = last_obs + 5). |
| `proj_share` | float64 | Projected panel share at the horizon (point estimate). |
| `proj_share_lo` | float64 | Lower 80% band on projected share. |
| `proj_share_hi` | float64 | Upper 80% band on projected share. |
| `proj_volume` | float64 | Projected paper volume at the horizon (point estimate). |
| `proj_volume_lo` | float64 | Lower 80% band on projected volume. |
| `proj_volume_hi` | float64 | Upper 80% band on projected volume. |
| `fit_ok` | bool | Whether the fit succeeded (all 149 are True in this release). |

### `trajectory/trajectory_series.parquet`
The long-form series behind the summary: observed (1995→last obs) and projected
(→2030) share/volume per topic-year. 5,352 rows. **Grain note:** `grain` column.

| column | dtype | meaning |
|---|---|---|
| `grain` | str | `leaf`. |
| `topic_id` | int64 | Leaf topic id. |
| `year` | int64 | Year (1995–2030). |
| `kind` | str | `observed` (≤ last obs year) or `projected` (after). |
| `share` | float64 | Panel share point value for the topic-year. |
| `share_lo` | float64 | Lower 80% band on share (NaN for `observed` rows). |
| `share_hi` | float64 | Upper 80% band on share (NaN for `observed` rows). |
| `volume` | float64 | Paper-volume point value for the topic-year. |
| `volume_lo` | float64 | Lower 80% band on volume (NaN for `observed` rows). |
| `volume_hi` | float64 | Upper 80% band on volume (NaN for `observed` rows). |

---

## Provenance sidecars (`*.parquet.run.json`)

Each table ships a JSON sidecar with the same basename. Fields:

- `config` — the parameters the table was generated with (grain, thresholds, model, etc.).
- `config_hash` — hash of that config.
- `input_hashes` — content hashes of the upstream inputs (lets you detect input drift).
- `git_sha` / `git_dirty` — the repo commit the table was built from.
- `software_versions` — platform / Python / package versions.
- `timestamp` — UTC build time.

Treat the sidecar as the source of truth for reproducibility; the prose above is a guide.

---

## Citation & license

**If you use these tables, please cite the Zenodo record and the software repository.**

- **Zenodo deposit (this release, v0.2.0):**
  `TODO: DOI minted on release — insert the Zenodo DOI here once the v0.2.0 record is published.`
  (Do not cite a DOI until it has been minted; the placeholder is intentional.)
- **Software repository:** SciField, <https://github.com/samersalman/scifield>
- See `CITATION.cff` at the repo root for the canonical citation metadata.

**License.** These derived cartography tables are released under the repository's
**Apache License 2.0** (see the `LICENSE` file at the repo root), consistent with the
`.zenodo.json` `license` field. No separate or additional data license is applied.

> Note on the underlying corpus: the tables are *derived aggregates* (lags, shares,
> novelty summaries, projections) computed over a panel of journal metadata harvested from
> PubMed/OpenAlex/Semantic Scholar. The raw corpus is **not** redistributed here; it is
> reproducible from the pipeline in this repository. Respect the terms of the upstream
> metadata providers when reconstructing it.
