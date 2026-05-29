# API costs and run accounting

This file is the per-run operations log for every LLM/agent extraction call made across the SciField V1 pipeline. V1 pre-registration PR1 (DOI `10.17605/OSF.IO/8ZJHD`) commits us to recording every extraction run for reproducibility, and plan §8.5 requires explicit usage accounting. Marginal dollar cost on the Claude Code subscription transport is zero, but wall time and call counts are first-class artifacts and must be tracked here.

**Conventions.** Rows are appended chronologically. Failed runs still get a row (with `n_ok` zero) — we never edit a historical row, only append. The `transport` column is `claude-code-cli` for V1-S07/S08; future rows may use `batch-api` or `anthropic-sdk`.

**Cross-reference.** The matching `.run.json` sidecar at `sidecar_path` carries the full config, git SHA, input hashes, and software versions for the run.

| date | run_id | n_attempted | n_ok | n_failed | wall_seconds | mean_s_per_call | transport | model_id | sidecar_path |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-23 | v1-s07-pilot | 50 | 50 | 0 | 252 | 5.04 | claude-code-cli | claude-via-claude-code | data/v1/epistemic_pilot.parquet.run.json |
| 2026-05-23 | v1-s08-smoke | 100 | 100 | 0 | 131.2 | 1.31 | claude-code-cli | claude-via-claude-code | data/v1/epistemic_extracted.parquet.run.json |
| 2026-05-24 | v1-s08-claude-partial | 14000 | 1881 | 12119 | n/a | n/a | claude-code-cli | claude-via-claude-code | data/v1/epistemic_extracted.parquet.run.json |
| 2026-05-29 | v1-s08-deepseek-smoke-thinking-on | 100 | 31 | 69 | 164.2 | 1.64 | deepseek-http | deepseek-v4-flash | data/v1/epistemic_extracted.parquet.run.json |
| 2026-05-29 | v1-s08-deepseek-smoke-thinking-off | 100 | 100 | 0 | 42.6 | 0.43 | deepseek-http | deepseek-v4-flash | data/v1/epistemic_extracted.parquet.run.json |
| 2026-05-29 | v1-s08-deepseek-full | 87118 | 87001 | 117 | 7211.3 | 0.083 | deepseek-http | deepseek-v4-flash | data/v1/epistemic_extracted.parquet.run.json |
| 2026-05-29 | v1-s08-deepseek-retry | 136 | 136 | 0 | 23.7 | 0.174 | deepseek-http | deepseek-v4-flash | data/v1/epistemic_extracted.parquet.run.json |

## Realized cost (deepseek-v4-flash, summed across all 2026-05-29 runs)

Computed from the per-row `usage` envelopes in `raw_response`:

| bucket | tokens | unit price ($/Mtok) | cost |
|---|---|---|---|
| input cache hit | 156,203,264 | 0.0028 | $0.4374 |
| input cache miss | 32,721,488 | 0.14 | $4.5810 |
| output (completion) | 5,135,188 | 0.28 | $1.4379 |
| **TOTAL** | | | **$6.4666** |

**Pre-flight dry-run estimate**: $7.2683 (n_abstracts=87,249). **Realized**: $6.47 (87,268 deepseek extractions). 11% under projection — cache-hit ratio was lower than modeled (~83% of prefix tokens hit vs 100% assumed) but offset by shorter-than-modeled abstracts.

## Backlog

All `papers_distinct` PMIDs (89,230) are covered as of 2026-05-29; 87,268 carry a `deepseek-v4-flash` extraction and 1,981 carry the original `claude-via-claude-code` extraction from the V1-S08 partial run. 19 PMIDs have both — these can be used as inter-model agreement samples for V1-S09 validation or deduped at analysis time by preferring `deepseek-v4-flash` as the canonical V1-S08 row.

## Incidents

- **2026-05-24, v1-s08-claude-partial**: Claude Code subscription session limit tripped ~1,881 abstracts in; every subsequent subprocess call returned `"You've hit your session limit · resets 6:30pm (America/Chicago)\n"` instead of JSON, all 12,119 routed to failures parquet. Root cause: per-subscription session quota, not a code regression. Recovery: switch transport to `deepseek-http` via `DEEPSEEK_API_KEY` (see `src/scifield/epistemic/deepseek_extract.py`).
- **2026-05-29, v1-s08-deepseek-smoke-thinking-on**: First deepseek smoke had 69% failure rate due to `deepseek-v4-flash`'s default thinking mode consuming `max_tokens=256` before emitting JSON. Fix: explicit `thinking: {"type": "disabled"}` in the request body + raised `max_tokens` to 512. Confirmed clean on the next smoke (100/100).
