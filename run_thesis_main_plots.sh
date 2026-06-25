#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-/gpfs/home4/tgroot/.conda/envs/llava-experiments/bin/python}"

"${PYTHON_BIN}" scripts/run_reproduce_thesis_main_plots.py
