# SciField

SciField is a multi-axis framework for monitoring the health of a scientific
field. It builds a longitudinal corpus from a defined set of journals,
enriches each article with thematic, epistemic, and novelty signals, and uses
those signals to forecast where a field is headed. The project is structured
as a sequence of execution sessions across Phases 0–9; this repository is the
day-zero scaffolding.

See the `plan/` directory in the repository for the full project plan and
session-execution roadmap.

## Phases at a glance

- **[Phase 0 — Scaffolding](phases/0_scaffolding.md):** Establish the package
  structure, config system, data layout, and reproducibility infrastructure
  every subsequent phase builds on.
- **[Phase 1 — Corpus](phases/1_corpus.md):** Build, validate, and document
  the v1 corpus — PMID, year, title, abstract, journal, authors, institutions,
  MeSH terms, OpenAlex ID, and full citation/reference lists.
- **[Phase 2 — Thematic](phases/2_thematic.md):** Produce a stable topic
  landscape across the 10-journal corpus, with per-journal sub-topic
  structures for finer-grained analyses.
- **[Phase 3 — Epistemic](phases/3_epistemic.md):** Extract structured
  epistemic-quality features (study design, sample size, controls, effect
  direction) per paper and validate against hand-labels.
- **[Phase 4 — Novelty](phases/4_novelty.md):** Compute semantic novelty
  (embedding distance from prior work) and structural novelty (CD index from
  the citation graph), then examine their joint distribution.
- **[Phase 5 — Forecasting](phases/5_forecasting.md):** Train and validate a
  temporal graph neural network that predicts topic emergence 3 years ahead
  of conventional detection, against pre-specified baselines.
- **[Phase 6 — Integration](phases/6_integration.md):** Combine the four axes
  into the three target findings (epistemic cascade, dual-novelty divergence,
  predictive validity) and generate manuscript figures.
- **[Phase 7 — Validation](phases/7_validation.md):** Scale the framework to
  the v2 corpus (25–40 journals) and test whether the v1 findings generalize
  beyond surgery/orthopedics.
- **[Phase 8 — Publication](phases/8_publication.md):** Ship the framework as
  a polished, installable, documented open-source tool — PyPI, docs site,
  demo app, Zenodo deposit, tagged release.
- **[Phase 9 — Manuscript](phases/9_manuscript.md):** Draft, internally and
  externally review, preprint, and submit the flagship manuscript.
