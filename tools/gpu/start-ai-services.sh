#!/usr/bin/env bash
set -euo pipefail

BASE="${MIRROR_SOUL_GPU_BASE:-/shareHost/C084003-ai}"
REPOSITORY="${MIRROR_SOUL_AI_REPOSITORY:-$BASE/Mirror_Soul_AI}"
LOG_DIR="$BASE/logs"
PREFLIGHT="$BASE/preflight-ai-services.sh"
FACE_RUNNER="$BASE/run-face-worker.sh"
FACE_LOG="$LOG_DIR/face-worker.log"

mkdir -p "$LOG_DIR"
bash "$PREFLIGHT"

bash "$REPOSITORY/tools/gpu/start-ditto-worker-pool.sh"

if tmux has-session -t face-worker 2>/dev/null; then
    printf '[OK] face-worker is already running\n'
else
    printf '\n===== face-worker started at %s =====\n' "$(date --iso-8601=seconds)" >> "$FACE_LOG"
    tmux new-session -d -s face-worker \
        "bash '$FACE_RUNNER' >> '$FACE_LOG' 2>&1"
    printf '[OK] face-worker started\n'
fi

sleep 3
if ! tmux has-session -t face-worker 2>/dev/null; then
    printf '[FAIL] face-worker exited during startup.\n' >&2
    tail -n 60 "$FACE_LOG"
    exit 1
fi

printf '[OK] face-worker is running\n'
printf '\nActive sessions\n'
tmux ls
printf '\nGPU\n'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
printf '\nLogs\n'
printf '  Ditto workers: %s/ditto-service-*.log\n' "$LOG_DIR"
printf '  Face worker  : %s\n' "$FACE_LOG"
