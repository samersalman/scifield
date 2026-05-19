#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# brev_smoke.sh
#
# Purpose: End-to-end smoke test of a SciField run on a Brev cloud instance.
#          Launches the smallest CPU instance, syncs the repo, runs `scifield
#          demo`, records credit-balance delta, and tears the instance down.
#
# Usage:   bash scripts/brev_smoke.sh
#
# Reference: docs/operations/brev.md (Brev CLI setup, auth, and troubleshooting).
#
# Stop condition: if the `brev` CLI is not installed, this script exits 0
# (non-fatal) so it stays safe to run in CI/dev environments without Brev.
# ------------------------------------------------------------------------------

set -euo pipefail

INSTANCE_NAME="scifield-smoke-v1s02"
REPO_URL="https://github.com/samersalman/scifield.git"
# Cheapest x86 CPU on Brev as of v0.6.x: n2d-highcpu-2 @ $0.05/hr (GCP).
# Verify with `brev search cpu --sort price` before relying on this in CI.
SMALLEST_CPU_INSTANCE="n2d-highcpu-2"
READY_TIMEOUT_SECONDS=600
POLL_INTERVAL_SECONDS=15

# 1. Bail out (non-fatally) if brev CLI is missing.
command -v brev >/dev/null || {
  echo "brev CLI not installed; see docs/operations/brev.md" >&2
  exit 0
}

# 2. Print brev version for the run log.
echo "[brev_smoke] brev --version:"
brev --version || true

# 3. Record credit balance BEFORE — best-effort, tolerate failures.
echo "[brev_smoke] Credit balance BEFORE (raw):"
BREV_BALANCE_BEFORE="$(brev org 2>&1 || true)"
if [[ -z "${BREV_BALANCE_BEFORE}" ]]; then
  BREV_BALANCE_BEFORE="$(brev profile 2>&1 || true)"
fi
echo "${BREV_BALANCE_BEFORE}"

# 4. Launch the smallest CPU instance, tagged for this smoke.
# We don't use --startup-script for the clone because brev marks the instance
# "Ready" as soon as SSH is up, which races the async startup-script. Instead
# we do the clone inline inside brev exec (step 6) so timing is deterministic.
echo "[brev_smoke] Creating instance ${INSTANCE_NAME} (type=${SMALLEST_CPU_INSTANCE})..."
brev create "${INSTANCE_NAME}" \
  --type "${SMALLEST_CPU_INSTANCE}"

# 7. Guarantee teardown immediately after a successful create.
trap 'brev stop '"${INSTANCE_NAME}"' || true' EXIT

# 5. Poll until the instance is RUNNING/READY, with a ~10 minute timeout.
echo "[brev_smoke] Waiting up to ${READY_TIMEOUT_SECONDS}s for ${INSTANCE_NAME} to be ready..."
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
  echo "[brev_smoke] not ready yet; sleeping ${POLL_INTERVAL_SECONDS}s..."
  sleep "${POLL_INTERVAL_SECONDS}"
done

if [[ "${ready}" -ne 1 ]]; then
  echo "[brev_smoke] Instance ${INSTANCE_NAME} never reached ready state within ${READY_TIMEOUT_SECONDS}s." >&2
  # Trap will still fire and stop the instance.
  exit 1
fi

echo "[brev_smoke] Instance is ready."

# 6. SSH-exec the smoke commands. Clone inline so we don't depend on
#    --startup-script timing (brev reports Ready when SSH is up, not when
#    the startup script finishes).
echo "[brev_smoke] Running smoke commands on ${INSTANCE_NAME}..."
brev exec "${INSTANCE_NAME}" "\
set -euxo pipefail; \
cd \$HOME; \
[ -d scifield ] || git clone ${REPO_URL} scifield; \
cd scifield && git pull --ff-only origin main; \
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh); \
export PATH=\$HOME/.local/bin:\$PATH; \
uv sync && uv run scifield demo"

# 8. Record credit balance AFTER and print delta (best-effort).
echo "[brev_smoke] Credit balance AFTER (raw):"
BREV_BALANCE_AFTER="$(brev org 2>&1 || true)"
if [[ -z "${BREV_BALANCE_AFTER}" ]]; then
  BREV_BALANCE_AFTER="$(brev profile 2>&1 || true)"
fi
echo "${BREV_BALANCE_AFTER}"

echo "[brev_smoke] --- Credit balance delta (raw before vs after) ---"
echo "[brev_smoke] BEFORE:"
echo "${BREV_BALANCE_BEFORE}"
echo "[brev_smoke] AFTER:"
echo "${BREV_BALANCE_AFTER}"
echo "[brev_smoke] (Parse credit values manually; format varies by brev CLI version.)"

# 9. Opt-in full deletion. Uncomment the line below to permanently delete the
#    instance instead of just stopping it (default trap behavior stops only).
# brev delete "${INSTANCE_NAME}"

echo "[brev_smoke] Done. Trap will stop ${INSTANCE_NAME} on exit."
