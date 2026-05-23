# Epistemic-quality extraction — phase log

Operational log for V1-S07's Phase 3 work: prompt design, iteration
notes from the 50-abstract pilot, and any post-pilot edits to the
system prompt or few-shot mix before V1-S08 promotes the pipeline
to the full 200k-abstract batch.

The canonical source files are:

- Schema: `src/scifield/epistemic/schema.py` (frozen,
  `extra="forbid"`, 6 fields). Version constant
  `LABEL_SCHEMA_VERSION = "v0.1"`.
- Prompt: `src/scifield/epistemic/prompt.py`. Version constant
  `PROMPT_VERSION = "v0.1"` — must equal `prompt.version` in
  `conf/epistemic/v1.yaml`.
- Config: `conf/epistemic/v1.yaml`.

Every persisted row in `data/v1/epistemic_pilot.parquet` carries
the `prompt_version` it was generated under, so multiple iterations
can co-exist in the same parquet without ambiguity.

## Prompt iteration log

### v0.1.0 — baseline (2026-05-22)

Initial prompt shipped with V1-S07 Batch 2A. Design choices:

- **Six fields, no more, no less.** The system prompt enumerates
  `study_design`, `sample_size`, `has_control`, `effect_direction`,
  `statistical_claim_present`, and `coi_disclosed_in_abstract` with
  one operational paragraph each, mirroring master plan §5 Phase 3
  and the field table in `plans/2026-05-22-v1-s07-epistemic-prep.md`
  §A. Adding a seventh field is a breaking change and requires a
  major-version bump (`v0.2`), not an inline edit.
- **JSON-only output, no markdown fence, no prose.** The Claude Code
  CLI does not give us tool-use / structured-output enforcement, so
  the output discipline is purely prompt-driven. The closing
  paragraph of the system prompt restates this in plain language
  and the few-shot block reinforces it by always immediately
  following `Abstract:` with `JSON:` and a one-line object.
- **`"na"` vs JSON `null` convention.** This is the most common
  source of parse drift in early labeling-LLM work. The prompt
  draws an explicit line: `effect_direction` uses the literal string
  `"na"` when the concept does not apply (reviews, methods papers,
  descriptive series); `sample_size` and `has_control` use JSON
  `null`. The schema accepts JSON `null` for `effect_direction` as
  a soft fallback so a malformed model output does not blow up the
  whole pilot, but the prompt instructs the model never to use it
  there.
- **Few-shot mix (6 exemplars).** RCT (positive effect), cohort
  (observational with COI disclosed in abstract), case series
  (small N, no statistical claim), review (sample_size and
  has_control both `null`, effect_direction `"na"`), negative-result
  RCT (drives the `effect_direction = "null"` path that is easy to
  miss), case-control (covers the remaining design enum). Each
  exemplar's label dict is round-tripped through `EpistemicLabel`
  in `tests/test_epistemic_prompt.py` so a schema-drifting edit
  cannot land silently.
- **Single-stream stdin format.** `build_prompt` concatenates the
  system prompt, the few-shot block (`Abstract: ...\nJSON: ...\n\n`
  per exemplar), and a final `Abstract: <abstract>\nJSON:` cue. The
  output is one string suitable for `subprocess.run(["claude",
  "--print"], input=...)` — there is no role separation because the
  Claude Code CLI is not the Anthropic API.

### v0.1.0 — pilot run (2026-05-23, n=50)

Ran `scifield epistemic pilot --n 50` against the first 50 PMIDs of
`data/v1/handlabel_sample.parquet` (sorted ascending). Backend:
`claude --print` subprocess. Wall: 4m 12s (~5s/abstract). Outcome:

- **Parse / validation:** 50 / 50 valid JSON, 0 retries triggered,
  0 rows routed to `data/v1/epistemic_pilot_failed.parquet`. The
  ≥90% threshold from plan §"Acceptance tests" step 7 cleared
  unconditionally; no JSON-discipline drift observed.
- **Label distribution:**
  - `study_design`: case_series 25, other 15, RCT 4, cohort 3,
    review 3, case_control 0. The case_series-heavy skew reflects
    the corpus slice — the first 50 PMIDs in pmid-sorted order are
    all from 1995, an era dominated by technique reports and small
    surgical series.
  - `effect_direction`: na 23, positive 19, mixed 7, null 1. The
    `"na"` discipline held — no JSON `null` leaked into this field.
  - `has_control`: true 13, false 27 (10 nulls — reviews/other).
  - `statistical_claim_present`: true 20, false 30.
  - `coi_disclosed_in_abstract`: false 50. **Expected.** Per
    plan §E.3.6 operational definition, COI in the full text but
    absent from the abstract is `false`; for a 1995 cohort that is
    universally the case.
  - `sample_size`: populated for 36 / 50 rows, median 42. Nulls are
    almost entirely reviews and technique papers — the model is
    correctly emitting `null` rather than fabricating a number.
- **No prompt change required.** v0.1.0 clears every parse-quality
  bar set by the pre-registration; iteration is deferred until V1-S08
  produces inter-rater κ that can flag systematic LLM error patterns
  worth chasing.

### v0.1.1 — (deferred, pending V1-S08 κ signal)

Reserved. Will be populated only if V1-S08's first inter-rater κ
report reveals a systematic LLM error pattern that traces back to
prompt language (e.g. `effect_direction` confusion on diagnostic
accuracy studies). No speculative edits before that signal arrives.

### v0.1.2 — (deferred, edge-case audit)

Reserved. Manual review of pilot rows where the model's choice is
contestable (reviews vs case series, basic science as `other`,
ambiguous COI lines). Will run after V1-S08 handlabels exist so the
audit has a human reference, not as speculation against the LLM's
solo output.
