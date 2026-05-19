# SciField: A Multi-Axis Framework for Monitoring Scientific Field Health

**Project plan v0.1**
**Author:** Samer Salman
**Date:** May 2026
**Target venue:** Nature Communications / Nature Human Behaviour / Nature Machine Intelligence
**Target timeline:** 12–18 months end-to-end

---

## 1. Project context

You are an MS1 at Baylor College of Medicine with an established research profile in orthopedic surgery, AI/ML clinical applications, and large-database analysis (NTDB, TriNetX, MIMIC-IV, T-MSIS). You recently completed a single-journal BERTopic study of *Arthroscopy* (1995–2025, 7,166 abstracts, 64 topics, hot/emerging/stable/declining classification). That manuscript identified meaningful trends but used a now-commoditized methodological framework — sentence-transformer embeddings + UMAP + HDBSCAN + linear-trend trend classification has been published in spine surgery, bone regeneration, LLM literature, and other subspecialties. The retrospective topic-modeling-of-one-journal genre is mature.

You want to do something genuinely new with the same starting tools, at a fundamentally higher scientific ambition, and release the entire pipeline as an open framework that other researchers can apply to any field. The goal is a Nature-family contribution to the science-of-science literature, not another subspecialty bibliometric study.

## 2. Scientific goals

The framework, provisionally **SciField**, characterizes any scientific corpus along four orthogonal axes simultaneously:

1. **Thematic structure** — what is being studied (transformer-based topic modeling, refined from the *Arthroscopy* pipeline).
2. **Epistemic quality** — how well it is being studied (NLP-extracted study design, sample size, effect direction, evidence strength).
3. **Novelty** — how new it is, on two complementary dimensions: semantic novelty (embedding distance from prior literature) and structural novelty (citation-graph disruption / CD index).
4. **Future trajectory** — where it is going (validated forecasting of topic emergence and decline using temporal graph neural networks).

The three target findings (each independently publishable if the others fall through):

- **F1 — Epistemic cascade.** Evidence quality declines precede thematic decline by a measurable lag (~3–5 years). When topics enter "hype" phases, low-quality papers propagate citation chains faster than high-quality papers in the same topic.
- **F2 — Dual-novelty divergence.** Semantic and structural novelty are partially decorrelated. The 2×2 of these signals identifies four paper archetypes (frontier work, methodological repackaging, isolated novelty, consolidation). Distribution across these archetypes has shifted measurably over 30 years.
- **F3 — Predictive validity.** A temporal-graph-network forecasting model predicts topic emergence 3 years ahead of conventional trend detection at materially higher accuracy than autoregressive, moving-average, and simple-MLP baselines.

The artifact: a permissively-licensed open-source Python package + CLI + documentation site that lets any researcher apply the full framework to any PubMed-indexed corpus.

## 3. Scope philosophy

This project is built around three principles that shape every design decision below.

**Start small and validate everything.** v1 uses 10 journals (5 ortho + 5 general surgery) over 30 years. This is large enough to be scientifically meaningful and small enough to iterate quickly. Each axis is validated independently before integration. Only after v1 demonstrates the findings hold does v2 expand the corpus to 25–40 journals spanning additional surgical and non-surgical specialties for the manuscript-grade analyses.

**Reproducibility from day one.** Every artifact — corpus, embeddings, topics, models, figures — is versioned, hashed, and reproducible from a single config file. The package structure, CLI, and documentation system exist from Phase 0, not bolted on at the end. The eventual GitHub release should require zero refactoring.

**De-risk via independently publishable outputs.** Each major phase produces an artifact that could anchor a smaller paper if the full vision doesn't land. Corpus + thematic structure → resource paper. Epistemic extraction validated against hand-labels → methods paper. Forecasting model → ML paper. The flagship paper integrates all three; if one underperforms, you still have publishable work.

## 4. v1 corpus: 10 journals

Selected for high volume, broad subspecialty coverage within surgery/orthopedics, strong PubMed indexing, and complete OpenAlex citation coverage post-2000.

**Orthopedics (5)**
- *Journal of Bone and Joint Surgery* (JBJS American)
- *Arthroscopy*
- *Journal of Arthroplasty*
- *Spine*
- *Clinical Orthopaedics and Related Research* (CORR)

*Note on journal selection.* AJSM was initially considered but excluded to avoid sports-medicine redundancy with *Arthroscopy*. *Journal of Arthroplasty* was selected in its place because (a) it is procedurally and philosophically complementary to *Arthroscopy* (joint replacement vs. joint preservation on overlapping anatomy), creating a natural contrast useful for the cross-journal seeding analysis; (b) it is volume-rich (~600–800 articles/year in recent years) and almost entirely original research; and (c) it adds methodological diversity (registry studies, RCTs, large cohorts) that strengthens the epistemic-quality (F1) analysis.

**General Surgery (5)**
- *Annals of Surgery*
- *JAMA Surgery*
- *Journal of the American College of Surgeons* (JACS)
- *British Journal of Surgery* (BJS)
- *Surgery*

Expected corpus size: 150,000–250,000 abstracts. Expected citation network (1-hop expansion): several million edges. This is the working scale for v1 and is well within what a single GPU + DuckDB + Kùzu can handle on a workstation.

**v2 expansion plan (post-validation):** add neurosurgery (*Neurosurgery*, *J Neurosurg*), CT surgery (*Ann Thorac Surg*, *J Thorac Cardiovasc Surg*), plastic (*PRS*), vascular (*J Vasc Surg*), and 3–5 non-surgical specialties (cardiology, hematology, oncology, neurology, internal medicine) for cross-specialty comparison. v2 targets 25–40 journals, 500k–1M papers.

## 5. Phase breakdown

### Phase 0 — Project scaffolding and infrastructure (2–3 weeks)

**Objectives.** Establish the package structure, config system, data layout, and reproducibility infrastructure that every subsequent phase will build into.

**Tools and rationale.**
- **uv** for Python environment and packaging. Faster than poetry, modern, well-supported.
- **pyproject.toml** with a single `scifield` package containing modules per axis. CLI entry points exposed via `[project.scripts]`.
- **Hydra** for hierarchical configs. Every analysis driven by a YAML config; reruns are deterministic by config hash.
- **DVC or plain Git-LFS + manifest files** for data versioning. DVC is heavier but tracks data lineage; the simpler manifest approach (record hashes in a YAML, store data on S3 or institutional storage) often wins for small teams.
- **Pre-commit hooks** with ruff (linting), black (formatting), mypy (typing). Catches drift early.
- **GitHub Actions** for CI from day 1. Run tests on every PR. Even with a private repo, this enforces discipline.
- **mkdocs-material** for the documentation site. Build the structure now, fill in as phases complete.
- **Sphinx autoapi** for API docs alternative — mkdocs is more user-friendly for non-developers, which matches the eventual audience.

**Deliverables.** A repo where `uv run scifield --help` works, the docs site builds, CI passes, and a placeholder config + dummy pipeline runs end-to-end on a 100-paper toy corpus.

**Success criteria.** A colleague with Python experience can clone the repo, run `uv sync && uv run scifield demo`, and see a result in under 10 minutes.

**Senior collaborator note.** This is the right moment to identify and informally engage a senior science-of-science collaborator (someone published on CD index, OpenAlex methodology, or temporal bibliometrics). Bring them in to react to the Phase 0–1 plan, *before* methodology is locked in. Candidates: Funk lab (Pittsburgh), Sinatra lab (Northeastern), Wang lab (Northwestern Kellogg), Fortunato lab (Indiana). Cold email with the abstract and Phase outline.

### Phase 1 — Corpus v1 construction (3–4 weeks)

**Objectives.** Build, validate, and document the v1 corpus. Every paper has: PMID, year, title, abstract, journal, authors (disambiguated), institution, MeSH terms, OpenAlex ID, full citation list, full reference list.

**Tools and rationale.**
- **Biopython Entrez** for PubMed harvesting. Same as your *Arthroscopy* pipeline; well-trodden.
- **OpenAlex API** for citations, authorship, institution disambiguation, concepts. Free, comprehensive, Crossref-fed, covers ~250M works, ~95% coverage post-2000. Critical: rate-limit politely (10 req/sec with email in header) and cache aggressively. Pre-2000 coverage drops; document this as a known limitation.
- **Semantic Scholar Graph API** as a secondary citation source and for citation-intent labels (background, method, result). Cross-validation against OpenAlex catches systematic gaps.
- **ROR API** for institution canonicalization.
- **DuckDB** as the primary tabular store. Columnar, fast, single-file, no server. Holds the paper-level features table.
- **Parquet** for raw and intermediate artifacts. Versioned by config hash.
- **httpx + tenacity** for async API harvesting with retry/backoff.

**Deliverables.** A DuckDB database with ~200k papers; one Parquet file per journal per year for raw artifacts; a corpus-statistics report (papers/year, abstract length distribution, citation coverage, MeSH coverage).

**Success criteria.** >95% of papers have abstract text, journal, year, MeSH; >90% have OpenAlex match; >80% have full citation list resolved. Documented gaps with rationale.

**Risk register for this phase.** OpenAlex coverage gaps for older papers; abstract quality variation across journals/eras; author disambiguation failures. All three need to be measured, reported, and considered in downstream analyses.

### Phase 2 — Thematic backbone (3–4 weeks)

**Objectives.** Replicate and refine the *Arthroscopy* pipeline at the v1 corpus scale. Produce a stable topic landscape across the entire 10-journal corpus, then per-journal sub-topic structures for finer-grained analyses.

**Tools and rationale.**
- **sentence-transformers** with `all-mpnet-base-v2` initially (you know it). Consider `BAAI/bge-large-en-v1.5` or `nomic-embed-text-v1` as alternatives — both materially stronger on biomedical text. Run a small embedding-quality bake-off on a labeled subset before committing.
- **FAISS** for nearest-neighbor search (needed for novelty in Phase 4). HNSW index. CPU is fine at this scale.
- **BERTopic** with the same UMAP + HDBSCAN + c-TF-IDF pipeline you used. Add hierarchical topic merging for the multi-journal corpus (otherwise you'll get hundreds of topics, many redundant).
- **OCTIS** or **Palmetto** for coherence scoring; cross-validate NPMI with C_v coherence to ensure your hyperparameter selection isn't artifact-dependent.
- **Plotly** for intertopic distance maps, temporal heatmaps, and the eventual dashboard.

**Deliverables.** A canonical topic hierarchy (probably 100–200 leaf topics, organized into ~20 mid-level and 5–7 top-level domains); per-paper topic assignment with probability; per-journal-per-year topic distributions; full reproducibility config.

**Success criteria.** Topic coherence ≥ your *Arthroscopy* baseline (NPMI ~0.18); topic hierarchy clinically interpretable on a 20-topic spot-check by a domain expert (you or a co-author); noise fraction <20%.

**Gate.** Before proceeding to Phase 3, the topic structure should look reasonable to a clinician reviewer. If it doesn't, the rest of the framework is built on sand.

### Phase 3 — Epistemic quality extraction (5–7 weeks; the hardest single phase)

**Objectives.** For every paper in the corpus, extract structured features describing the epistemic quality of the work: study design, sample size, presence/absence of controls, effect direction, statistical claims, conflict-of-interest disclosure (when in abstract). Validate against a hand-labeled subset.

**Tools and rationale.**
- **Claude / GPT-4-class LLM with structured output** (instructor library, JSON schema). LLMs are now strong enough to extract these fields from abstracts with high accuracy. This will be the primary extraction method. Use Claude via API with a carefully-engineered system prompt and a Pydantic schema.
- **Hand-labeled validation set.** 500 randomly-sampled abstracts, double-coded by you and Rohan (or another co-author), arbitrated on disagreements. This becomes the ground-truth set. Without this you have no defensible epistemic-quality claim.
- **Inter-rater reliability** computed with Cohen's kappa or Krippendorff's alpha. Target κ ≥ 0.7 on study design, ≥ 0.8 on presence of controls, ≥ 0.6 on effect direction (which is inherently noisier).
- **Cross-validation against existing tools:** Trialstreamer (Marshall et al.) for RCT detection; RobotReviewer for risk-of-bias signals where applicable. These won't cover everything but they let you triangulate.
- **Cost management.** 200k papers × LLM call is non-trivial. Batch processing via Claude Batch API or OpenAI Batch (50% discount). Estimate $500–$2000 in API costs for v1.
- **Validation gate:** if LLM extraction underperforms (e.g., κ < 0.6 on study design), pivot to a smaller hand-labeled corpus + fine-tuned BERT classifier. Slower but more defensible.

**Deliverables.** Per-paper feature table with epistemic quality fields; validation report (confusion matrices, κ, error analysis); the validated extraction prompt + schema as a reusable module.

**Success criteria.** Inter-rater κ targets met; LLM-vs-human agreement on the 500-paper test set within 10% of inter-rater agreement; full error analysis documented.

**This is the most novel methodological contribution.** Done well, it stands alone as a methods paper. Done poorly, it undermines the whole framework. Budget time accordingly. Pre-register the validation protocol on OSF before running it — this is a small ask that adds substantial credibility.

### Phase 4 — Novelty layer (semantic + structural) (3–4 weeks)

**Objectives.** Compute two novelty scores for every paper: semantic novelty (distance from prior literature in embedding space) and structural novelty (CD index from citation graph). Examine their joint distribution.

**Tools and rationale.**
- **Semantic novelty.** For each paper, compute the mean and minimum cosine distance from its embedding to the embeddings of all papers published before it in the same field. Several variants exist (Foster et al. 2015 used a similar idea); choose one principled variant and document. FAISS handles the lookups efficiently.
- **Structural novelty / CD index.** Funk-Owen-Smith CD_n index, computed on the citation graph. Multiple Python implementations exist (Funk lab's reference code in R; cdindex Python port). Validate your implementation against published values on the Park et al. 2023 replication corpus.
- **Kùzu** for the citation graph. Embedded, columnar, fast graph OLAP. Much lighter than Neo4j for distribution in an open-source tool — installs as a pip dependency, no server needed. Schema as described in the prior conversation: Paper, Author, Journal, Institution, Topic nodes; CITES, AUTHORED_BY, AFFILIATED_WITH, PUBLISHED_IN, ASSIGNED_TO edges.
- **NetworkX or igraph** as a fallback for analyses where Kùzu is awkward; convert subgraphs as needed.

**Deliverables.** Per-paper semantic and structural novelty scores; 2×2 archetype assignment (frontier / repackaging / isolated / consolidation); temporal trends in archetype distribution across journals and topics; the dual-novelty figure (this is potentially your headline figure).

**Success criteria.** Semantic and structural novelty correlate weakly enough (|ρ| < 0.4) to be meaningfully distinct; the 2×2 distribution is interpretable; at least one finding emerges that's surprising to a domain expert.

**Gate.** If the 2×2 reveals no interesting pattern (everything clusters in one quadrant or correlations are too tight), the dual-novelty finding (F2) is dead and the paper architecture pivots toward F1 and F3 only.

### Phase 5 — Forecasting layer (5–7 weeks)

**Objectives.** Build and validate a model that predicts topic emergence 3 years ahead of conventional detection. Pre-specify all evaluation metrics before training.

**Tools and rationale.**
- **PyTorch Geometric** for the heterogeneous temporal graph network. The standard library; well-documented.
- **TGN (Temporal Graph Network)** or **HGT (Heterogeneous Graph Transformer)** as the architecture. TGN handles continuous-time event streams; HGT handles type-heterogeneous nodes. The choice depends on whether you model citations as discrete time steps or continuous events. Start with HGT; it's better-documented and the heterogeneity (Paper/Author/Topic node types) maps cleanly.
- **Baselines (mandatory).**
  - Naive: last-3-years moving average of topic share.
  - Autoregressive: ARIMA per topic (statsmodels).
  - Simple MLP on topic-level time-series features.
  - Without-graph ablation: same features but no graph structure.
- **Temporal cross-validation.** Train on 1995–2017, validate on 2018–2020, test on 2021–2025. Never let test data leak into training. Pre-specify this split in writing before training.
- **Pre-registration.** OSF pre-registration of model architecture, evaluation metrics (AUC for emergence classification, MAPE for share prediction), baselines, and statistical comparison. This is non-negotiable for a credible forecasting claim at Nat Comm.
- **sktime** for forecasting baselines and evaluation utilities.

**Deliverables.** Trained model + checkpoints; full evaluation table comparing GNN vs. all baselines; calibration plots; topic-level forecast trajectories; ablation study isolating the contribution of graph structure.

**Success criteria.** GNN beats best baseline by a meaningful margin (target: >5 percentage points on emergence AUC; statistically significant on a Wilcoxon signed-rank test across topics) at 3-year horizon. If the GNN does not beat baselines, this is an important null finding — document it honestly, scale back F3, and pivot the paper toward F1 and F2.

### Phase 6 — Integration and cross-journal analyses (4–5 weeks)

**Objectives.** Combine the four axes into the three target findings. Generate the analyses and figures that drive the manuscript.

**Specific analyses.**
- **F1 (epistemic cascade).** For each topic, plot evidence-quality trajectory and topic-volume trajectory over time. Test the lead-lag relationship (does quality drop precede volume drop?) using cross-correlation or Granger causality. Replicate across all topics meeting size thresholds.
- **F2 (dual-novelty divergence).** Cross-tabulate semantic and structural novelty; trend archetype distribution over time; relate archetype to downstream citation impact.
- **F3 (predictive validity).** Already produced in Phase 5; here, frame it within the broader narrative.
- **Bonus analyses.** Cross-journal seeding networks (which journals lead vs. follow); per-specialty differences; institutional and geographic patterns.

**Tools and rationale.** Mostly pandas, statsmodels, scipy. The hard work is in the prior phases; this phase synthesizes.

**Deliverables.** A draft results section with all figures and tables; a preprint-ready figure set; analyst notebook with full reproducible computation.

**Success criteria.** At least two of the three target findings hold with appropriate statistical support; figures are publication-quality; co-authors agree the narrative is coherent.

### Phase 7 — Scaling validation and corpus v2 (4–6 weeks)

**Objectives.** Demonstrate the framework scales beyond surgery/orthopedics, and that the v1 findings generalize.

**Steps.**
- Expand corpus to v2 (25–40 journals, 500k–1M papers, additional surgical specialties + non-surgical specialties for breadth).
- Re-run the full pipeline. Validate that runtime and memory scale acceptably.
- Test whether v1 findings replicate in v2. If F1 holds in 8 of 10 surgical specialties but not in 2 non-surgical specialties, that itself is an interesting finding (specialty-dependence of epistemic cascade).

**Tools and rationale.** Same stack. This phase stress-tests the architecture. Any awkwardness in scaling (e.g., embedding memory blow-ups, graph query slowness) gets fixed here.

**Deliverables.** v2 corpus + results; scaling benchmark report (runtime / memory / cost vs. corpus size); updated figures incorporating v2 data.

**Success criteria.** Pipeline runs end-to-end on v2 with documented resource requirements; at least one v1 finding replicates; honest reporting of any v1-to-v2 discrepancies.

### Phase 8 — Open-source release (3–4 weeks)

**Objectives.** Ship the framework as a polished, installable, documented open-source tool that other researchers can apply to their own corpora.

**Tools and rationale.**
- **PyPI release** of the `scifield` package. `pip install scifield` should work.
- **mkdocs-material** documentation site with: quickstart, tutorial walkthrough of v1 results, API reference, contribution guidelines.
- **Streamlit or Plotly Dash** demo app deployed on Hugging Face Spaces or Render. Lets reviewers and readers explore the v1 results interactively without installing anything.
- **Pre-computed v1 and v2 result artifacts** released via Zenodo with DOI for citation.
- **CITATION.cff** for proper citation.
- **MIT or Apache 2.0 license.** Apache 2.0 is slightly preferred for academic tools because of explicit patent grant; MIT is simpler. Either is defensible.
- **Docker image** for users who don't want to manage Python environments.
- **Example notebooks** for 2–3 different fields (one outside surgery — pick something like radiology or oncology to demonstrate generality).

**Deliverables.** Public GitHub repo; PyPI package; documentation site; demo app; Zenodo deposit with DOI; example notebooks; tagged v1.0.0 release.

**Success criteria.** A researcher unfamiliar with the project can install the tool, run the tutorial, and apply it to their own corpus within an afternoon. Documented in a usability test (informal — have a non-author colleague try it and time them).

### Phase 9 — Manuscript and submission (4–6 weeks)

**Objectives.** Write, polish, submit.

**Steps.**
- Draft the manuscript with the structure: brief intro framing science-of-science gap → methods → three findings + open-source contribution → discussion of implications for science policy, journal editors, and field self-monitoring.
- Internal review by all co-authors.
- External review by 2–3 trusted senior colleagues, including the science-of-science collaborator engaged in Phase 0.
- Pre-print on bioRxiv simultaneously with submission.
- Target venue: Nature Communications first (broad audience, accepts methods + findings papers, science-of-science fits). Backup: Nature Human Behaviour (more behavioral framing) or Nature Machine Intelligence (more ML framing) or *Patterns* (Cell Press, open access, faster decisions).
- Authorship: co-first with Rohan probably; senior author the science-of-science collaborator if engaged seriously; you handle correspondence.

**Deliverables.** Submitted manuscript; preprint with DOI; cover letter; suggested reviewers list.

## 6. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| OpenAlex citation coverage too sparse pre-2000 | Medium | Document explicitly; restrict primary novelty/CD-index analyses to post-2000; report sensitivity |
| Epistemic quality LLM extraction unreliable | Medium-high | Hand-labeled validation set; pre-register protocol; fallback to fine-tuned BERT |
| Dual-novelty 2×2 shows no interesting pattern | Medium | F2 is dropped; paper pivots to F1+F3 |
| GNN forecasting fails to beat baselines | Medium-high | Null finding documented; F3 framed as "graph structure does not add forecasting value beyond temporal features" — itself publishable |
| All three findings underwhelming | Low-medium | Phase outputs are individually publishable; salvage as a resource paper + methods paper |
| Computational resources insufficient | Low | BCM HPC, Modal credits, or cloud GPU rental — budget $1–3k for v2 |
| Senior collaborator not recruited | Medium | Cold email early; even informal advisory role is valuable; backup is a 2–3 expert review pre-submission |
| Timeline slips with MS coursework / other projects | High | Phase structure allows pause/resume at any gate; phases independently valuable |
| Pre-registration concerns introduce delays | Low | OSF preregistration is fast (<1 day); benefits outweigh costs |

## 7. Estimated timeline

| Phase | Duration | Cumulative |
|---|---|---|
| 0. Scaffolding | 3 weeks | 3 weeks |
| 1. Corpus v1 | 4 weeks | 7 weeks |
| 2. Thematic backbone | 4 weeks | 11 weeks |
| 3. Epistemic quality | 7 weeks | 18 weeks |
| 4. Novelty layer | 4 weeks | 22 weeks |
| 5. Forecasting | 7 weeks | 29 weeks |
| 6. Integration | 5 weeks | 34 weeks |
| 7. Scaling v2 | 6 weeks | 40 weeks |
| 8. Open-source release | 4 weeks | 44 weeks |
| 9. Manuscript | 6 weeks | 50 weeks |

50 weeks of focused work ≈ 12–14 months of part-time work alongside MS coursework. 18 months is realistic with buffers; 12 months is achievable if you have stretches of dedicated time (summer, research blocks).



## 8. Compute infrastructure

### 8.1 Resources confirmed

- **NVIDIA Brev** organization account with ~$500 in credits (provisioned via personal NVIDIA contact)
- **Personal GPU** on local workstation for development and small experiments
- **Anthropic / OpenAI API access** for epistemic quality extraction (Phase 3) — separate budget line

### 8.2 Brev allocation strategy

Brev is a meta-broker; instance types are provisioned across underlying clouds (Lambda, GCP, etc.) and rates reflect those plus a small margin. The strategy is to match GPU class to phase workload, use spot instances for fault-tolerant work, and aggressively stop instances when not in use (the single largest source of wasted credits is forgotten running machines).

| Phase | Workload | Recommended instance | Rationale | Estimated GPU-hours | Estimated cost |
|---|---|---|---|---|---|
| 0 | Scaffolding | None / local | CPU-only | 0 | $0 |
| 1 | Corpus harvesting | None / local | API rate-limited, not compute-bound | 0 | $0 |
| 2 | Embedding 200k abstracts (v1) | L40S 48GB on-demand | Embedding is throughput-bound, memory-light, FP8 helps | 2–4 | $5–10 |
| 3 | Epistemic extraction | None (uses Anthropic/OpenAI API, not Brev) | LLM extraction via Batch API | 0 | $50–150 in API costs |
| 4 | Novelty computations | None / local + CPU instance if needed | FAISS + graph algorithms are CPU-bound | 0 | $0–5 |
| 5 | GNN training (HGT/TGN) | A100 80GB on-demand for final runs; A100 spot for hyperparameter sweep | NVLink + HBM matter for graph-batched training; spot OK for sweeps with checkpointing | 30–60 | $50–120 |
| 6 | Integration + figures | None / local | Pandas/statsmodels | 0 | $0 |
| 7 | v2 scale-up | A100 80GB on-demand | Full pipeline rerun at 5× scale | 20–40 | $40–80 |
| 8 | Open-source packaging | None | — | 0 | $0 |
| 9 | Manuscript | None | — | 0 | $0 |

**Total projected Brev spend across the project: $100–220.** This leaves substantial credit headroom for unexpected reruns, additional ablation studies, or v3 expansion. Even in a worst case (every phase requires a second pass), the project stays comfortably under $400.

### 8.3 Brev operational hygiene

These are the rules that keep $500 in credits from disappearing in two weeks of inattention:

- **Stop instances the moment a job finishes.** Brev keeps instances running and billing until explicitly stopped. Set a calendar reminder to check the console at end of every working session.
- **Use snapshots, not running instances, for state.** Snapshot the environment, stop the instance, resume from snapshot when next needed. Storage is cents per GB-month vs. dollars per GPU-hour.
- **Use spot instances for Phase 5 hyperparameter sweep.** A100 spot is roughly half on-demand cost. Implement checkpoint + resume in the training loop so spot reclaims are non-fatal.
- **Pin region.** Choose the region with the cheapest current pricing for your needed GPU and stick with it; persistent storage doesn't migrate between regions.
- **Pre-build container images.** Build the project Docker image once, push to a registry, pull on instance launch. Avoids burning GPU-time on pip installs.
- **Monitor the credit balance weekly.** The Brev console shows credit burn rate; if it's higher than the table above projects, investigate immediately.

### 8.4 NVIDIA insider connection as project asset

The contact who provisioned these credits is a strategic resource beyond the credits themselves. Consider engaging them at three points:

- **After Phase 2 results.** A polished topic map across 10 surgical journals is a tangible artifact. Share it casually as "thanks for the credits, here's what they're funding." This plants the seed.
- **If GNN forecasting works in Phase 5.** This is the result NVIDIA cares about (graph + temporal + interesting application domain). Ask about NVIDIA's Graph Analytics / cuGraph team — they would find this work directly relevant, may have technical advice, and could potentially co-author or amplify.
- **At Phase 8 release.** Ask about Inception program eligibility and the NVIDIA developer blog as a distribution channel. A feature on the NVIDIA blog would dwarf any other marketing for the open-source release.

Soft asks only. Don't burn the relationship on credit top-ups. The bigger value is access and distribution, not compute.

### 8.5 LLM API budget (separate)

Phase 3 epistemic extraction uses LLM API calls, not GPU compute, and draws from a separate budget. Estimate:

- 200k abstracts × ~1.5k input tokens × Claude Haiku 4.5 via Batch API (50% discount)
- Estimated cost: $50–150 for v1
- Estimated cost: $200–500 for v2

Total project API budget: ~$300–650 across both runs. Likely covered by Anthropic API credits if available; otherwise a small grant line.

### 8.6 Total project budget summary

| Category | v1 | v2 | Total |
|---|---|---|---|
| Brev GPU compute | $80–150 | $40–80 | $120–230 |
| LLM API (epistemic extraction) | $50–150 | $200–500 | $250–650 |
| Storage (Zenodo, optional cloud) | $0 | $0 | $0 |
| Domain / docs hosting | $0 | $0 | $0 (use GitHub Pages + Hugging Face Spaces) |
| **Total** | **$130–300** | **$240–580** | **$370–880** |

Everything fits comfortably within current resources. No additional grant funding required to complete the project, though departmental support would be welcome for the LLM API costs.

## 9. Decision points and gates

Explicit gates where the project should be re-evaluated before proceeding:

- **After Phase 2:** Is the topic structure clinically interpretable? If not, stop and fix the embedding/clustering before continuing.
- **After Phase 3:** Did epistemic extraction meet inter-rater κ targets? If not, the F1 finding cannot be defended — pivot.
- **After Phase 4:** Did the dual-novelty 2×2 reveal interesting structure? If not, drop F2 and rescope.
- **After Phase 5:** Did the GNN beat baselines meaningfully? If not, frame F3 as a null finding or drop it.
- **After Phase 6:** Do at least 2 of 3 findings hold? If yes, proceed to scaling. If no, downscope to a methods + resource paper.

## 10. Stretch ideas (not v1; consider for v2 or follow-up papers)

- **Gap detection.** Identify regions of semantic space where questions are raised but answers are absent.
- **Cross-domain technique migration.** Detect when methods (e.g., ML/AI) migrate from a source field into a target field; quantify the lag and adoption pattern.
- **Author trajectory analysis.** Topic-pivoter vs. deep-domain-expert classification; relationship to career outcomes.
- **Replication signal detection.** Identify topics where replication failures cluster.
- **Specialty-comparative cascades.** Does F1 hold differently in surgical vs. medical specialties? Hypothesis-generating cross-field comparison.

## 11. Decisions made (May 2026)

- **Findings prioritization.** All three findings (F1, F2, F3) developed in parallel as first-class citizens in the manuscript. Parallelization across phases is opportunistic, not forced.
- **Senior collaborator.** Project proceeds without a science-of-science senior author. Reviewer concerns will be anticipated and pre-empted in Methods and Supplementary Methods.
- **Compute.** NVIDIA Brev (~$500 in credits) for GPU workloads; local GPU for development; Anthropic API for Phase 3 epistemic extraction.

## 12. Remaining open questions

These don't block Phase 0 but should be settled before Phase 1 starts:

1. **Authorship plan.** Co-first with Rohan again? Other contributors anticipated? Who handles which axes?
2. **Timeline alignment.** Does 12–18 months fit with the MS2 trajectory, or should phases align with specific milestones (Step 1, research blocks, conference cycles)?
3. **Journal list confirmation.** Does the 10-journal v1 list look right, or do you want to swap any (e.g., add *Bone Joint J*, swap *Surgery* for something else)?
4. **Pre-registration.** Comfortable pre-registering the Phase 3 epistemic extraction validation protocol and the Phase 5 forecasting protocol on OSF? Strongly recommended; adds ~1 day of work and substantially strengthens the paper's defensibility, especially given the decision to proceed without a senior science-of-science co-author.
5. **Project name.** "SciField" is a placeholder. The eventual name matters for the open-source release and downstream citations. Alternatives: FieldAtlas, SciCartographer, ResearchPulse, FieldHealth, BiblioPulse, FieldVitals.

## 13. First sprint (next 2 weeks)

To convert this plan into momentum, the first sprint targets Phase 0 deliverables:

1. Create the `scifield` GitHub repo (private to start), initialize with the package skeleton, pyproject.toml, and CI.
2. Set up Hydra config structure with placeholder configs for each phase.
3. Get the mkdocs documentation site building locally and on GitHub Pages.
4. Run `scifield demo` on a 100-paper toy corpus end-to-end to validate the scaffolding.
5. Verify Brev access works: launch a smallest-tier instance, pull the repo, run `uv sync`, stop the instance. Document the workflow.
6. Identify which Anthropic API key / billing arrangement will fund Phase 3 LLM costs.

By end of week 2, you should have a working, reproducible scaffold and an empty pipeline that runs end-to-end. Every subsequent phase fills in one module.
