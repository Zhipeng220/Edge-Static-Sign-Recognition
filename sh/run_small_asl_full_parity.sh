#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PARITY_DIR="${PARITY_DIR:-./parity_inputs_small_asl}"
OUT_DIR="${OUT_DIR:-./parity_results/asl_large_dataset_full_model_seed42_small_asl_full}"
WORKER="${WORKER:-./trt_infer_worker}"

DEPLOYMENT_DIR="${DEPLOYMENT_DIR:-./multi_dataset_results_fixed_labels/asl_large_dataset/deployment}"
FP32_ENGINE="${FP32_ENGINE:-${DEPLOYMENT_DIR}/full_model_seed42_fp32.engine}"
FP16_ENGINE="${FP16_ENGINE:-${DEPLOYMENT_DIR}/full_model_seed42_fp16.engine}"
DEPLOYED_LABEL_CLASSES="${DEPLOYED_LABEL_CLASSES:-${DEPLOYMENT_DIR}/label_classes_full_model_seed42.npy}"

python3 "${REPO_ROOT}/evaluate_small_asl_full_parity.py" \
  --parity-dir "${PARITY_DIR}" \
  --deployed-label-classes "${DEPLOYED_LABEL_CLASSES}" \
  --worker "${WORKER}" \
  --fp32-engine "${FP32_ENGINE}" \
  --fp16-engine "${FP16_ENGINE}" \
  --output-dir "${OUT_DIR}"

echo ""
echo "Finished."
echo "Use:"
echo "  ${OUT_DIR}/full_parity_summary.csv"
