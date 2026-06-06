#!/usr/bin/env bash
set -euo pipefail

RESTART=0
if [[ "${1:-}" == "--restart" ]]; then
    RESTART=1
    shift
fi

PORT="${1:-8501}"
REPO_ROOT="${VLM_UQ_REPO_ROOT:-/home/tgroot/vlm_uncertainty}"
APP_PATH="${PATCHING_QUAL_VIEWER_APP:-$REPO_ROOT/qualitative_viewer/app.py}"
PYTHON="${PYTHON:-/gpfs/home4/tgroot/.conda/envs/llava-experiments/bin/python}"
FILE_WATCHER_TYPE="${STREAMLIT_FILE_WATCHER_TYPE:-poll}"
RUN_ON_SAVE="${STREAMLIT_RUN_ON_SAVE:-true}"

cd "$REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

if [[ "${RESTART_STREAMLIT:-$RESTART}" == "1" ]]; then
    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
        if [[ -n "$pids" ]]; then
            kill $pids
            sleep 1
        fi
    else
        echo "lsof is not available; cannot automatically clear port $PORT." >&2
    fi
fi

"$PYTHON" -m streamlit run "$APP_PATH" \
    --server.address 127.0.0.1 \
    --server.port "$PORT" \
    --server.headless true \
    --server.runOnSave "$RUN_ON_SAVE" \
    --server.fileWatcherType "$FILE_WATCHER_TYPE"
