#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PREDICTIONS="${1:-multi_dataset_results_fixed_labels/asl_large_dataset/predictions_full_model_seed42.csv}"
OUTPUT_DIR="${2:-results/evaluation/asl_large_dataset_full_model_seed42}"

python src/evaluate.py \
  --predictions "$PREDICTIONS" \
  --output-dir "$OUTPUT_DIR" \
  --overwrite
