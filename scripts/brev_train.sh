#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# brev_train.sh
#
# Purpose: A100-spot GPU template for V1-S13 HGT Optuna sweep on Brev.
#          Launches the GPU instance, syncs the repo, runs
#          `uv run scifield forecasting gnn-sweep` (the 40-trial HGT sweep),
#          copies the resulting sweep artifacts + their .run.json sidecars back
#          locally, records credit-balance delta, and tears the instance down.
#
# IMPORTANT — committed but NOT executed:
#   V1-S13 ran locally on Mac CPU at $0. The problem is tiny (~1,298 train /
#   208 val rows; a 40-trial CPU sweep completes in ~30–90 min), so an A100
#   buys nothing here. This template exists so a future scaling session (v2
#   corpus, much larger graphs) has a vetted, ready-to-run GPU path with the
#   correct artifact names and invocation already proven at the CLI level.
#   Do NOT execute it against a real GPU for V1-S13 — use the local sweep
#   instead (uv run scifield forecasting gnn-sweep).
#
# Usage:   bash scripts/brev_train.sh
#
# Reference: docs/operations/brev.md (Brev CLI setup, auth, and troubleshooting).
#
# GPU spec: Default below uses `--gpu a100` which works on Brev CLI v0.6.x.
#           Operator should verify the canonical name with:
#               brev search gpu --sort price
#           and override BREV_GPU_TYPE if needed. If the installed brev CLI does
#           not accept `--gpu`, the script will fail fast and the operator can
#           switch to `--type <instance-spec>` after consulting the catalog.
#           NOTE: Some brev CLI versions support a spot-instance flag (e.g.
#           --spot) to reduce cost; check `brev create --help` for the current
#           CLI's flags. This script uses the conservative `--gpu` form (matching
#           brev_embed.sh) and leaves spot-flag opt-in to the operator.
#
# Env var forwarding: This script forwards any `SCIFIELD_*` env vars from the
#                     local environment into the remote `brev exec` shell. The
#                     sweep command reads from the local DuckDB mirror checked
#                     in via git+sync but forwarding keeps the door open for
#                     future flags (e.g. SCIFIELD_OPTUNA_TRIALS).
#
# After this script completes:
#   1. The sweep writes:
#        data/v1/forecasting_sweep.parquet  (trial results, best params)
#        data/v1/forecasting_sweep.parquet.run.json
#        models/v1/hgt_best.pt             (checkpoint of best trial)
#        models/v1/hgt_best.pt.run.json
#      All four files are copied back to the local repo by this script.
#   2. V1-S14 (test-set eval) reads hgt_best.pt and forecasting_sweep.parquet
#      from those local paths. Both .run.json sidecars reference the OSF
#      pre-registration DOI 10.17605/OSF.IO/XP94F.
#   3. Log the credit-balance delta (printed at end of this script) to
#        docs/operations/brev.md
#      under a new "V1-S13 HGT sweep run" subsection (match the V1-S05 format).
#
# Stop condition: if the `brev` CLI is not installed, this script exits 0
# (non-fatal) so it stays safe to run in CI/dev environments without Brev.
# ------------------------------------------------------------------------------

set -euo pipefail

INSTANCE_NAME="scifield-train-V1-S13"
REPO_URL="https://github.com/samersalman/scifield.git"
# A100 40/80 GB GPU spec. Operator: verify with `brev search gpu --sort price`
# before relying on this in CI. Override with BREV_GPU_TYPE=<value> if needed.
# For spot pricing, check whether your brev CLI version supports --spot and add
# that flag to the `brev create` call below.
BREV_GPU_TYPE="${BREV_GPU_TYPE:-a100}"
# GPU instances can take longer to provision than CPU; allow 30 min.
READY_TIMEOUT_SECONDS=1800
POLL_INTERVAL_SECONDS=30

LOCAL_DATA_DIR="./data/v1"
LOCAL_MODELS_DIR="./models/v1"

# 1. Bail out (non-fatally) if brev CLI is missing.
command -v brev >/dev/null || {
  echo "brev CLI not installed; see docs/operations/brev.md" >&2
  exit 0
}

# 2. Print brev version for the run log.
echo "[brev_train] brev --version:"
brev --version || true

# 3. Record credit balance BEFORE — best-effort, tolerate failures.
echo "[brev_train] Credit balance BEFORE (raw):"
BREV_BALANCE_BEFORE="$(brev org 2>&1 || true)"
if [[ -z "${BREV_BALANCE_BEFORE}" ]]; then
  BREV_BALANCE_BEFORE="$(brev profile 2>&1 || true)"
fi
echo "${BREV_BALANCE_BEFORE}"

# 4. Launch the A100 GPU instance, tagged for this sweep run.
# We don't use --startup-script for the clone because brev marks the instance
# "Ready" as soon as SSH is up, which races the async startup-script. Instead
# we do the clone inline inside brev exec (step 6) so timing is deterministic.
echo "[brev_train] Creating instance ${INSTANCE_NAME} (gpu=${BREV_GPU_TYPE})..."
brev create "${INSTANCE_NAME}" \
  --gpu "${BREV_GPU_TYPE}"

# 7. Guarantee teardown immediately after a successful create.
trap 'brev stop '"${INSTANCE_NAME}"' || true' EXIT

# 5. Poll until the instance is RUNNING/READY, with a ~30 minute timeout.
echo "[brev_train] Waiting up to ${READY_TIMEOUT_SECONDS}s for ${INSTANCE_NAME} to be ready..."
deadline=$(( $(date +%s) + READY_TIMEOUT_SECONDS ))
ready=0
while [[ $(date +%s) -lt ${deadline} ]]; do
  brev refresh >/dev/null 2>&1 || true
  ls_output="$(brev ls 2>&1 || true)"
  # Match the instance line and look for a ready-ish state.
  if echo "${ls_output}" | grep -E "^| ${INSTANCE_NAME}( |$)" | grep -Eiq "running|ready|deployed"; then
    ready=1
    break
  fi
  echo "[brev_train] not ready yet; sleeping ${POLL_INTERVAL_SECONDS}s..."
  sleep "${POLL_INTERVAL_SECONDS}"
done

if [[ "${ready}" -ne 1 ]]; then
  echo "[brev_train] Instance ${INSTANCE_NAME} never reached ready state within ${READY_TIMEOUT_SECONDS}s." >&2
  # Trap will still fire and stop the instance.
  exit 1
fi

echo "[brev_train] Instance is ready."

# 6. SSH-exec the HGT Optuna sweep command. Clone inline so we don't depend on
#    --startup-script timing (brev reports Ready when SSH is up, not when
#    the startup script finishes).
echo "[brev_train] Running HGT sweep on ${INSTANCE_NAME}..."
brev exec "${INSTANCE_NAME}" "\
set -euxo pipefail; \
cd \$HOME; \
[ -d scifield ] || git clone ${REPO_URL} scifield; \
cd scifield && git pull --ff-only origin main; \
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh); \
export PATH=\$HOME/.local/bin:\$PATH; \
uv sync && uv run scifield forecasting gnn-sweep"

# 8. Copy sweep artifacts + sidecars back to local dirs.
echo "[brev_train] Ensuring local ${LOCAL_DATA_DIR} and ${LOCAL_MODELS_DIR} exist..."
mkdir -p "${LOCAL_DATA_DIR}"
mkdir -p "${LOCAL_MODELS_DIR}"

echo "[brev_train] Copying forecasting_sweep.parquet back from ${INSTANCE_NAME}..."
brev cp "${INSTANCE_NAME}:scifield/data/v1/forecasting_sweep.parquet" "${LOCAL_DATA_DIR}/forecasting_sweep.parquet" || {
  echo "[brev_train] WARNING: brev cp failed for forecasting_sweep.parquet; try manual scp via 'brev ssh ${INSTANCE_NAME}'." >&2
}

echo "[brev_train] Copying forecasting_sweep.parquet.run.json back from ${INSTANCE_NAME}..."
brev cp "${INSTANCE_NAME}:scifield/data/v1/forecasting_sweep.parquet.run.json" "${LOCAL_DATA_DIR}/forecasting_sweep.parquet.run.json" || {
  echo "[brev_train] WARNING: brev cp failed for forecasting_sweep.parquet.run.json; try manual scp via 'brev ssh ${INSTANCE_NAME}'." >&2
}

echo "[brev_train] Copying hgt_best.pt back from ${INSTANCE_NAME}..."
brev cp "${INSTANCE_NAME}:scifield/models/v1/hgt_best.pt" "${LOCAL_MODELS_DIR}/hgt_best.pt" || {
  echo "[brev_train] WARNING: brev cp failed for hgt_best.pt; try manual scp via 'brev ssh ${INSTANCE_NAME}'." >&2
}

echo "[brev_train] Copying hgt_best.pt.run.json back from ${INSTANCE_NAME}..."
brev cp "${INSTANCE_NAME}:scifield/models/v1/hgt_best.pt.run.json" "${LOCAL_MODELS_DIR}/hgt_best.pt.run.json" || {
  echo "[brev_train] WARNING: brev cp failed for hgt_best.pt.run.json; try manual scp via 'brev ssh ${INSTANCE_NAME}'." >&2
}

# 9. Record credit balance AFTER and print delta (best-effort).
echo "[brev_train] Credit balance AFTER (raw):"
BREV_BALANCE_AFTER="$(brev org 2>&1 || true)"
if [[ -z "${BREV_BALANCE_AFTER}" ]]; then
  BREV_BALANCE_AFTER="$(brev profile 2>&1 || true)"
fi
echo "${BREV_BALANCE_AFTER}"

echo "[brev_train] --- Credit balance delta (raw before vs after) ---"
echo "[brev_train] BEFORE:"
echo "${BREV_BALANCE_BEFORE}"
echo "[brev_train] AFTER:"
echo "${BREV_BALANCE_AFTER}"
echo "[brev_train] (Parse credit values manually; format varies by brev CLI version.)"
echo "[brev_train] Log this delta into docs/operations/brev.md per the V1-S05 pattern."

# 10. Opt-in full deletion. Uncomment the line below to permanently delete the
#     instance instead of just stopping it (default trap behavior stops only).
# brev delete "${INSTANCE_NAME}"

echo "[brev_train] Done. Trap will stop ${INSTANCE_NAME} on exit."
echo "[brev_train] Artifacts written to ${LOCAL_DATA_DIR}/ and ${LOCAL_MODELS_DIR}/."
echo "[brev_train] .run.json sidecars reference OSF pre-registration DOI 10.17605/OSF.IO/XP94F."
echo "[brev_train] Next step: V1-S14 test-set eval — uv run scifield forecasting evaluate --checkpoint ${LOCAL_MODELS_DIR}/hgt_best.pt"
