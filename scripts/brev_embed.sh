#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# brev_embed.sh
#
# Purpose: Full-corpus embedding run for V1-S05 on a Brev L40S 48 GB GPU.
#          Launches the GPU instance, syncs the repo, runs `scifield embed`
#          on the abstract-bearing subset of the corpus, copies the resulting
#          `embeddings.parquet` (+ sidecar) back locally, records credit-balance
#          delta, and tears the instance down.
#
# Usage:   bash scripts/brev_embed.sh
#
# Reference: docs/operations/brev.md (Brev CLI setup, auth, and troubleshooting).
#
# GPU spec: Default below uses `--gpu l40s` which works on Brev CLI v0.6.x.
#           Operator should verify the canonical name with:
#               brev search gpu --sort price
#           and override BREV_GPU_TYPE if needed. If the installed brev CLI does
#           not accept `--gpu`, the script will fail fast and the operator can
#           switch to `--type <instance-spec>` after consulting the catalog.
#
# Env var forwarding: This script forwards any `SCIFIELD_*` env vars from the
#                     local environment into the remote `brev exec` shell. The
#                     `scifield embed` command itself does not need NCBI /
#                     OpenAlex / S2 keys (it only reads from the local DuckDB
#                     mirror checked into Brev via git+sync), but the forwarding
#                     keeps the door open for future flags.
#
# After this script completes:
#   1. Run locally:
#        uv run scifield faiss-build \
#          --embeddings data/v1/embeddings.parquet \
#          --out data/v1/faiss.index
#      to build the HNSW index + PMID map sidecar.
#   2. Log credit-balance delta (printed at end of this script) to
#        docs/operations/brev.md
#      under a new "V1-S05 embedding run" subsection (match the V1-S02 format).
#
# Stop condition: if the `brev` CLI is not installed, this script exits 0
# (non-fatal) so it stays safe to run in CI/dev environments without Brev.
# ------------------------------------------------------------------------------

set -euo pipefail

INSTANCE_NAME="scifield-embed-V1-S05"
REPO_URL="https://github.com/samersalman/scifield.git"
# L40S 48 GB GPU spec. Operator: verify with `brev search gpu --sort price`
# before relying on this in CI. Override with BREV_GPU_TYPE=<value> if needed.
BREV_GPU_TYPE="${BREV_GPU_TYPE:-l40s}"
# GPU instances can take longer to provision than CPU; allow 30 min.
READY_TIMEOUT_SECONDS=1800
POLL_INTERVAL_SECONDS=30

LOCAL_DATA_DIR="./data/v1"

# 1. Bail out (non-fatally) if brev CLI is missing.
command -v brev >/dev/null || {
  echo "brev CLI not installed; see docs/operations/brev.md" >&2
  exit 0
}

# 2. Print brev version for the run log.
echo "[brev_embed] brev --version:"
brev --version || true

# 3. Record credit balance BEFORE — best-effort, tolerate failures.
echo "[brev_embed] Credit balance BEFORE (raw):"
BREV_BALANCE_BEFORE="$(brev org 2>&1 || true)"
if [[ -z "${BREV_BALANCE_BEFORE}" ]]; then
  BREV_BALANCE_BEFORE="$(brev profile 2>&1 || true)"
fi
echo "${BREV_BALANCE_BEFORE}"

# 4. Launch the L40S GPU instance, tagged for this embedding run.
# We don't use --startup-script for the clone because brev marks the instance
# "Ready" as soon as SSH is up, which races the async startup-script. Instead
# we do the clone inline inside brev exec (step 6) so timing is deterministic.
echo "[brev_embed] Creating instance ${INSTANCE_NAME} (gpu=${BREV_GPU_TYPE})..."
brev create "${INSTANCE_NAME}" \
  --gpu "${BREV_GPU_TYPE}"

# 7. Guarantee teardown immediately after a successful create.
trap 'brev stop '"${INSTANCE_NAME}"' || true' EXIT

# 5. Poll until the instance is RUNNING/READY, with a ~30 minute timeout.
echo "[brev_embed] Waiting up to ${READY_TIMEOUT_SECONDS}s for ${INSTANCE_NAME} to be ready..."
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
  echo "[brev_embed] not ready yet; sleeping ${POLL_INTERVAL_SECONDS}s..."
  sleep "${POLL_INTERVAL_SECONDS}"
done

if [[ "${ready}" -ne 1 ]]; then
  echo "[brev_embed] Instance ${INSTANCE_NAME} never reached ready state within ${READY_TIMEOUT_SECONDS}s." >&2
  # Trap will still fire and stop the instance.
  exit 1
fi

echo "[brev_embed] Instance is ready."

# 6. SSH-exec the embedding command. Clone inline so we don't depend on
#    --startup-script timing (brev reports Ready when SSH is up, not when
#    the startup script finishes).
echo "[brev_embed] Running embedding command on ${INSTANCE_NAME}..."
brev exec "${INSTANCE_NAME}" "\
set -euxo pipefail; \
cd \$HOME; \
[ -d scifield ] || git clone ${REPO_URL} scifield; \
cd scifield && git pull --ff-only origin main; \
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh); \
export PATH=\$HOME/.local/bin:\$PATH; \
uv sync && uv run scifield embed --config conf/thematic/embed.yaml"

# 8. Copy embeddings + sidecar back to local data/v1/.
echo "[brev_embed] Ensuring local ${LOCAL_DATA_DIR} exists..."
mkdir -p "${LOCAL_DATA_DIR}"

echo "[brev_embed] Copying embeddings.parquet back from ${INSTANCE_NAME}..."
brev cp "${INSTANCE_NAME}:scifield/data/v1/embeddings.parquet" "${LOCAL_DATA_DIR}/embeddings.parquet" || {
  echo "[brev_embed] WARNING: brev cp failed for embeddings.parquet; try manual scp via 'brev ssh ${INSTANCE_NAME}'." >&2
}

echo "[brev_embed] Copying embeddings.parquet.run.json back from ${INSTANCE_NAME}..."
brev cp "${INSTANCE_NAME}:scifield/data/v1/embeddings.parquet.run.json" "${LOCAL_DATA_DIR}/embeddings.parquet.run.json" || {
  echo "[brev_embed] WARNING: brev cp failed for embeddings.parquet.run.json; try manual scp via 'brev ssh ${INSTANCE_NAME}'." >&2
}

# 9. Record credit balance AFTER and print delta (best-effort).
echo "[brev_embed] Credit balance AFTER (raw):"
BREV_BALANCE_AFTER="$(brev org 2>&1 || true)"
if [[ -z "${BREV_BALANCE_AFTER}" ]]; then
  BREV_BALANCE_AFTER="$(brev profile 2>&1 || true)"
fi
echo "${BREV_BALANCE_AFTER}"

echo "[brev_embed] --- Credit balance delta (raw before vs after) ---"
echo "[brev_embed] BEFORE:"
echo "${BREV_BALANCE_BEFORE}"
echo "[brev_embed] AFTER:"
echo "${BREV_BALANCE_AFTER}"
echo "[brev_embed] (Parse credit values manually; format varies by brev CLI version.)"
echo "[brev_embed] Log this delta into docs/operations/brev.md per the V1-S02 pattern."

# 10. Opt-in full deletion. Uncomment the line below to permanently delete the
#     instance instead of just stopping it (default trap behavior stops only).
# brev delete "${INSTANCE_NAME}"

echo "[brev_embed] Done. Trap will stop ${INSTANCE_NAME} on exit."
echo "[brev_embed] Next step: uv run scifield faiss-build --embeddings ${LOCAL_DATA_DIR}/embeddings.parquet --out ${LOCAL_DATA_DIR}/faiss.index"
