"""Phase 6 integration findings (V1-S15).

This subpackage holds the *computed* findings that synthesize the four pipeline
axes into the manuscript's headline results:

* :mod:`scifield.findings.cascade` — **F1**, the pre-registered epistemic-cascade
  lead/lag analysis (per-topic evidence-quality vs. research-volume time series,
  cross-correlation + Granger causality, BH-FDR, and the pre-registered decision
  rule). Pre-registration: ``docs/preregistrations/PR3_epistemic_cascade.md``.
* :mod:`scifield.findings.seeding` — the exploratory, **non-gating** bonus layer
  (per-journal / institution / country lead-follow "seeding" scores, a directed
  seeding network, and the orthopedic-vs-general-surgery specialty grouping).
* :mod:`scifield.findings.impact` — the F2-enrichment archetype-by-impact
  statistic (Kruskal-Wallis across the four dual-novelty archetypes + a
  rank-based effect size).

Every module here is pure and synthetic-fixture tested; the notebooks
(``notebooks/11`` … ``notebooks/14``) do the I/O and import these helpers — they
never reimplement the logic (project packaging rule).
"""

from __future__ import annotations
