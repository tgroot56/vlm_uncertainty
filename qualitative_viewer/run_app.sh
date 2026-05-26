#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8501}"
REPO_ROOT="${VLM_UQ_REPO_ROOT:-/home/tgroot/vlm_uncertainty}"
APP_PATH="${PATCHING_QUAL_VIEWER_APP:-$REPO_ROOT/qualitative_viewer/app.py}"
PYTHON="${PYTHON:-/gpfs/home4/tgroot/.conda/envs/llava-experiments/bin/python}"

cd "$REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m streamlit run "$APP_PATH" --server.address 127.0.0.1 --server.port "$PORT"
