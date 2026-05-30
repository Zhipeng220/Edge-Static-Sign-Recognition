#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RESULT_ROOT="./multi_dataset_results_fixed_labels"
WORKSPACE_MB=1024

echo "=============================================="
echo "TensorRT batch engine build"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "WORKSPACE_MB=${WORKSPACE_MB}"
echo "=============================================="

# Jetson performance setting
sudo nvpmodel -m 5 || true
sudo jetson_clocks || true

# Check trtexec workspace flag
if trtexec --help | grep -q "memPoolSize"; then
    WORKSPACE_FLAG="--memPoolSize=workspace:${WORKSPACE_MB}"
else
    WORKSPACE_FLAG="--workspace=${WORKSPACE_MB}"
fi

echo "Using workspace flag: ${WORKSPACE_FLAG}"

SUMMARY_CSV="${RESULT_ROOT}/ALL_DATASETS_tensorrt_engine_sizes.csv"
echo "dataset,variant,seed,onnx_path,onnx_size_mb,fp32_engine_path,fp32_engine_size_mb,fp16_engine_path,fp16_engine_size_mb,fp32_status,fp16_status,fp32_command,fp16_command" > "${SUMMARY_CSV}"

find "${RESULT_ROOT}" -path "*/deployment/*.onnx" | sort | while read -r ONNX_PATH; do
    DEPLOY_DIR="$(dirname "${ONNX_PATH}")"
    DATASET_DIR="$(dirname "${DEPLOY_DIR}")"
    DATASET="$(basename "${DATASET_DIR}")"

    ONNX_FILE="$(basename "${ONNX_PATH}")"
    BASE="${ONNX_FILE%.onnx}"

    # Parse variant and seed from names like full_model_seed42.onnx
    SEED="$(echo "${BASE}" | sed -n 's/.*_seed\([0-9]*\)$/\1/p')"
    VARIANT="$(echo "${BASE}" | sed 's/_seed[0-9]*$//')"

    FP32_ENGINE="${DEPLOY_DIR}/${BASE}_fp32.engine"
    FP16_ENGINE="${DEPLOY_DIR}/${BASE}_fp16.engine"

    FP32_LOG="${DEPLOY_DIR}/trtexec_${BASE}_fp32.log"
    FP16_LOG="${DEPLOY_DIR}/trtexec_${BASE}_fp16.log"

    ONNX_SIZE_MB="$(du -m "${ONNX_PATH}" | awk '{print $1}')"

    echo ""
    echo "----------------------------------------------"
    echo "Dataset: ${DATASET}"
    echo "Variant: ${VARIANT}"
    echo "Seed: ${SEED}"
    echo "ONNX: ${ONNX_PATH}"
    echo "----------------------------------------------"

    FP32_CMD="trtexec --onnx=${ONNX_PATH} --saveEngine=${FP32_ENGINE} --minShapes=landmarks:1x21x3 --optShapes=landmarks:1x21x3 --maxShapes=landmarks:1x21x3 ${WORKSPACE_FLAG}"
    FP16_CMD="trtexec --onnx=${ONNX_PATH} --saveEngine=${FP16_ENGINE} --minShapes=landmarks:1x21x3 --optShapes=landmarks:1x21x3 --maxShapes=landmarks:1x21x3 ${WORKSPACE_FLAG} --fp16"

    echo "[FP32] ${FP32_CMD}"
    if eval "${FP32_CMD}" > "${FP32_LOG}" 2>&1; then
        FP32_STATUS="ok"
    else
        FP32_STATUS="failed"
    fi

    echo "[FP16] ${FP16_CMD}"
    if eval "${FP16_CMD}" > "${FP16_LOG}" 2>&1; then
        FP16_STATUS="ok"
    else
        FP16_STATUS="failed"
    fi

    if [ -f "${FP32_ENGINE}" ]; then
        FP32_SIZE_MB="$(du -m "${FP32_ENGINE}" | awk '{print $1}')"
    else
        FP32_SIZE_MB=""
    fi

    if [ -f "${FP16_ENGINE}" ]; then
        FP16_SIZE_MB="$(du -m "${FP16_ENGINE}" | awk '{print $1}')"
    else
        FP16_SIZE_MB=""
    fi

    echo "${DATASET},${VARIANT},${SEED},${ONNX_PATH},${ONNX_SIZE_MB},${FP32_ENGINE},${FP32_SIZE_MB},${FP16_ENGINE},${FP16_SIZE_MB},${FP32_STATUS},${FP16_STATUS},\"${FP32_CMD}\",\"${FP16_CMD}\"" >> "${SUMMARY_CSV}"

    echo "FP32 status: ${FP32_STATUS}, size: ${FP32_SIZE_MB} MB"
    echo "FP16 status: ${FP16_STATUS}, size: ${FP16_SIZE_MB} MB"
done

echo ""
echo "=============================================="
echo "Finished TensorRT batch conversion."
echo "Summary saved to:"
echo "${SUMMARY_CSV}"
echo "=============================================="
