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

## V1-S08 closeout

### Infrastructure landed (2026-05-23)

- `src/scifield/epistemic/batch.py` — concurrent + resumable runner
  (`ProcessPoolExecutor`, chunk-append parquet, `record_run` per
  flush, `preregistration_url` threaded into every sidecar).
- `src/scifield/epistemic/arbitrate.py` — disagreement detection +
  arbitration workbook round-trip (`load_two_raters`,
  `find_disagreements`, `export_arbitration_xlsx`,
  `import_arbitration_xlsx`).
- `src/scifield/epistemic/extract.py` — added picklable
  `extract_one_subprocess` wrapper for `ProcessPoolExecutor` workers.
- `src/scifield/cli.py` — new commands `extract-batch`,
  `arbitrate-export`, `arbitrate-import`.
- `conf/epistemic/v1.yaml` — new `extract_batch:` block.
- `docs/operations/api_costs.md` — per-run usage ledger created and
  backfilled with the V1-S07 pilot row.
- Tests: `tests/test_epistemic_batch.py` (+4) and
  `tests/test_epistemic_arbitrate.py` (+6) green; CLI smoke tests
  appended to `tests/test_cli_epistemic.py` (+5). Full suite
  184 passed, 1 skipped.

### Hand-labeling workbooks shipped

- `data/v1/epistemic_handlabel_samer.xlsx` — 500 rows, 6 dropdown
  validations, Instructions + Labels sheets.
- `data/v1/epistemic_handlabel_partner.xlsx` — same shape.

Both produced via `scifield epistemic export-labels --rater {samer,
partner} --out ...` from the frozen `data/v1/handlabel_sample.parquet`
(Hare-rounded 500 from V1-S07). Human inter-rater work proceeds on
its own clock; arbitration plumbing is ready to consume both
workbooks via `arbitrate-export` / `arbitrate-import` whenever they
return.

### Smoke batch (2026-05-23, n=100, concurrency=4)

- Command: `scifield epistemic extract-batch --submit --limit 100 --concurrency 4`.
- Result: 100 / 100 ok, 0 failures.
- Wall time: 131.2 s (~1.31 s/call effective, ~4× speedup over the
  V1-S07 pilot's ~5 s/call sequential baseline — confirms the
  bounded process pool scales linearly at concurrency 4 with no
  observed Claude Code rate-limit pushback).
- Sidecar `data/v1/epistemic_extracted.parquet.run.json` carries
  `config.preregistration_url = "https://doi.org/10.17605/OSF.IO/8ZJHD"`
  (PR1 acceptance gate cleared).
- Output distribution looks plausible: `study_design` dominated by
  case_series (35), cohort (29), other (23); `effect_direction`
  weighted positive (45) + na (40); failed parquet empty but
  schema-pinned. Spot-checks against the smoke output are consistent
  with the pilot's drift profile (no new pathology).
- See `docs/operations/api_costs.md` row `v1-s08-smoke` for the
  ledger entry.

### Full-corpus extraction (status: launchable, not yet run)

- `papers_distinct WHERE abstract IS NOT NULL AND length > 50`
  currently contains 89,230 rows (plan's 99,938 reflected an earlier
  corpus snapshot — non-issue, the dedup + abstract filter is
  unchanged).
- Projected wall time at concurrency 4: ~8 h (89,230 × 1.31 s ÷ 4),
  well under the plan's 35–55 h budget.
- Launch with `scifield epistemic extract-batch --submit --concurrency 4`;
  runner is resumable, so stopping and restarting is safe. Use
  `scifield epistemic extract-batch --status` to query progress.
  After the run completes, append a `v1-s08-full` row to
  `docs/operations/api_costs.md` (transport `claude-code-cli`,
  model_id `claude-via-claude-code`, sidecar path same as smoke).
- `Session-Objectives-MAP.md` V1-S08 status: mark
  "✓ infrastructure + LLM extraction; awaiting full-corpus run +
  hand-label arbitration" rather than fully ✓, per the plan
  closeout rule (final ✓ waits on
  `epistemic_handlabel_final.parquet` arriving post-rater).

### Carryovers into V1-S09

- Inter-rater κ on the two hand-label workbooks once they land.
- LLM-vs-arbitrated agreement analysis using
  `epistemic_handlabel_final.parquet` as truth.
- Gate G2 report (V1-S09 deliverable).
