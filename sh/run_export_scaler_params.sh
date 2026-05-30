#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python src/export_scaler_params.py \
  --scaler "${1:-models/scaler_full_model_seed42.pkl}" \
  --output-dir "${2:-deployment/jetson}"
