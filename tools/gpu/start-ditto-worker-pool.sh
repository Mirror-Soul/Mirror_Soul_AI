#!/usr/bin/env bash
set -euo pipefail

BASE="${MIRROR_SOUL_GPU_BASE:-/shareHost/C084003-ai}"
ENV_FILE="${DITTO_SERVICE_ENV_FILE:-$BASE/.env.ditto-service}"
REPOSITORY="${MIRROR_SOUL_AI_REPOSITORY:-$BASE/Mirror_Soul_AI}"
RUNNER="$REPOSITORY/tools/gpu/run-ditto-worker.sh"
LOG_DIR="$BASE/logs"

set -a
source "$ENV_FILE"
set +a

WORKERS="${DITTO_SERVICE_WORKERS:-2}"
BASE_PORT="${DITTO_SERVICE_BASE_PORT:-${DITTO_SERVICE_PORT:-8080}}"

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || (( WORKERS > 4 )); then
    printf '[FAIL] DITTO_SERVICE_WORKERS must be between 1 and 4.\n' >&2
    exit 1
fi
if ! [[ "$BASE_PORT" =~ ^[0-9]+$ ]] || (( BASE_PORT < 1 || BASE_PORT + WORKERS - 1 > 65535 )); then
    printf '[FAIL] DITTO_SERVICE_BASE_PORT is invalid.\n' >&2
    exit 1
fi
if tmux has-session -t ditto-service 2>/dev/null; then
    printf '[FAIL] Legacy tmux session ditto-service is running. Stop it before starting the pool.\n' >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
for index in $(seq 1 "$WORKERS"); do
    port=$((BASE_PORT + index - 1))
    session="ditto-service-$index"
    log_file="$LOG_DIR/$session.log"

    if ! tmux has-session -t "$session" 2>/dev/null; then
        printf '\n===== %s started at %s =====\n' "$session" "$(date --iso-8601=seconds)" >> "$log_file"
        tmux new-session -d -s "$session" \
            "bash '$RUNNER' '$port' >> '$log_file' 2>&1"
        printf '[OK] %s started on port %s\n' "$session" "$port"
    else
        printf '[OK] %s is already running on port %s\n' "$session" "$port"
    fi

    printf 'Waiting for %s /ready' "$session"
    ready=0
    for _ in $(seq 1 90); do
        if wget -qO- "http://127.0.0.1:$port/ready" >/dev/null 2>&1; then
            ready=1
            break
        fi
        printf '.'
        sleep 2
    done
    printf '\n'
    if [[ "$ready" -ne 1 ]]; then
        printf '[FAIL] %s did not become ready.\n' "$session" >&2
        tail -n 60 "$log_file"
        exit 1
    fi
    printf '[OK] %s is ready\n' "$session"
done

printf '\nDitto worker pool\n'
for index in $(seq 1 "$WORKERS"); do
    port=$((BASE_PORT + index - 1))
    wget -qO- "http://127.0.0.1:$port/ready"
    printf '\n'
done
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
