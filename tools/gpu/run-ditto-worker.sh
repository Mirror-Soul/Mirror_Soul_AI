#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    printf 'Usage: %s <port>\n' "$0" >&2
    exit 2
fi

PORT="$1"
BASE="${MIRROR_SOUL_GPU_BASE:-/shareHost/C084003-ai}"
ENV_FILE="${DITTO_SERVICE_ENV_FILE:-$BASE/.env.ditto-service}"
PYTHON="${DITTO_SERVICE_PYTHON:-/shareHost/C084003-ditto/conda-env/bin/python}"
REPOSITORY="${MIRROR_SOUL_AI_REPOSITORY:-$BASE/Mirror_Soul_AI}"

cd "$REPOSITORY"
set -a
source "$ENV_FILE"
set +a
export DITTO_SERVICE_PORT="$PORT"

exec "$PYTHON" -m ditto_server.main
