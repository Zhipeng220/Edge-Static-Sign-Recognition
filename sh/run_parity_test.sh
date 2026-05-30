#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ROOT_DIR="${ROOT_DIR:-./multi_dataset_results_fixed_labels}"
DATASET="${DATASET:-asl_large_dataset}"
VARIANT="${VARIANT:-full_model}"
SEED="${SEED:-42}"

PARITY_DIR="${PARITY_DIR:-./parity_inputs}"
WORKER="${WORKER:-./trt_infer_worker}"
OUT_DIR="${OUT_DIR:-./parity_results/${DATASET}_${VARIANT}_seed${SEED}}"

FP32_ENGINE="${FP32_ENGINE:-${ROOT_DIR}/${DATASET}/deployment/${VARIANT}_seed${SEED}_fp32.engine}"
FP16_ENGINE="${FP16_ENGINE:-${ROOT_DIR}/${DATASET}/deployment/${VARIANT}_seed${SEED}_fp16.engine}"

python3 "${REPO_ROOT}/parity_test_trt.py" \
  --parity-dir "${PARITY_DIR}" \
  --worker "${WORKER}" \
  --fp32-engine "${FP32_ENGINE}" \
  --fp16-engine "${FP16_ENGINE}" \
  --output-dir "${OUT_DIR}"
