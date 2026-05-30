#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_DIR="${SOURCE_DIR:-./parity_source_small_asl}"
INPUT_DIR="${INPUT_DIR:-./parity_inputs_small_asl}"
OUT_DIR="${OUT_DIR:-./parity_results/asl_large_dataset_full_model_seed42_small_asl}"

SCALER_STATS="${SCALER_STATS:-./multi_dataset_results_fixed_labels/asl_large_dataset/deployment/scaler_stats_full_model_seed42.npz}"
WORKER="${WORKER:-./trt_infer_worker}"
FP32_ENGINE="${FP32_ENGINE:-./multi_dataset_results_fixed_labels/asl_large_dataset/deployment/full_model_seed42_fp32.engine}"
FP16_ENGINE="${FP16_ENGINE:-./multi_dataset_results_fixed_labels/asl_large_dataset/deployment/full_model_seed42_fp16.engine}"

python3 "${REPO_ROOT}/prepare_small_asl_parity_inputs.py" \
  --source-dir "${SOURCE_DIR}" \
  --scaler-stats "${SCALER_STATS}" \
  --output-dir "${INPUT_DIR}"

python3 "${REPO_ROOT}/parity_fp16_vs_fp32.py" \
  --inputs "${INPUT_DIR}/parity_inputs.npy" \
  --worker "${WORKER}" \
  --fp32-engine "${FP32_ENGINE}" \
  --fp16-engine "${FP16_ENGINE}" \
  --output-dir "${OUT_DIR}"

echo ""
echo "Finished."
echo "Use:"
echo "  ${OUT_DIR}/parity_summary.csv"
