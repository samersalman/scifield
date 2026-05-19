# Phase 3 — Epistemic

## Phase objective

For every paper in the corpus, extract structured features describing the
epistemic quality of the work: study design, sample size, presence/absence
of controls, effect direction, statistical claims, and conflict-of-interest
disclosure (when present in the abstract). The primary extraction method is
a Claude/GPT-4-class LLM with structured Pydantic output via the Claude
Batch API. A 500-abstract hand-labeled validation set, double-coded and
arbitrated, provides ground truth, with inter-rater reliability targets of
κ ≥ 0.7 for study design, ≥ 0.8 for controls, and ≥ 0.6 for effect direction.
Cross-validation against Trialstreamer and RobotReviewer triangulates the
LLM extraction. The validation protocol is pre-registered on OSF before
running. This is the project's most novel methodological contribution and a
hard gate: if extraction underperforms the κ targets, the F1 (epistemic
cascade) finding cannot be defended and the design pivots to a fine-tuned
BERT classifier on a smaller hand-labeled corpus.
