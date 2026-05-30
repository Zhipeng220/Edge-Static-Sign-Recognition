#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ============================================================
# run_peak_ram_trtexec.sh
#
# Purpose:
#   Measure classifier-only TensorRT latency / throughput and
#   peak Jetson system RAM for FP16 and FP32 engines.
#
# Scope:
#   TensorRT classifier-only benchmark.
#   This script does NOT measure image loading, MediaPipe,
#   camera capture, display, speech, STM32, or actuator latency.
#
# Usage:
#   bash sh/run_peak_ram_trtexec.sh
#
# Optional environment-variable overrides:
#   DATASET=asl_large_dataset VARIANT=full_model SEED=42 \
#   INTERVAL_MS=100 IDLE_SECONDS=10 DURATION_SEC=10 \
#   bash sh/run_peak_ram_trtexec.sh
# ============================================================

ROOT_DIR="${ROOT_DIR:-./multi_dataset_results_fixed_labels}"
DATASET="${DATASET:-asl_large_dataset}"
VARIANT="${VARIANT:-full_model}"
SEED="${SEED:-42}"

INTERVAL_MS="${INTERVAL_MS:-100}"
IDLE_SECONDS="${IDLE_SECONDS:-10}"
WARMUP_MS="${WARMUP_MS:-1000}"
DURATION_SEC="${DURATION_SEC:-10}"
ITERATIONS="${ITERATIONS:-1000}"
INPUT_SHAPE="${INPUT_SHAPE:-landmarks:1x21x3}"
NVP_MODEL="${NVP_MODEL:-5}"
USE_SPIN_WAIT="${USE_SPIN_WAIT:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRTEXEC_BIN="${TRTEXEC_BIN:-trtexec}"
TEGRSTATS_BIN="${TEGRSTATS_BIN:-tegrastats}"

SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-${REPO_ROOT}/summarize_peak_ram.py}"

ENGINE_DIR="${ROOT_DIR}/${DATASET}/deployment"
FP16_ENGINE="${FP16_ENGINE:-${ENGINE_DIR}/${VARIANT}_seed${SEED}_fp16.engine}"
FP32_ENGINE="${FP32_ENGINE:-${ENGINE_DIR}/${VARIANT}_seed${SEED}_fp32.engine}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-./peak_ram_results/${DATASET}_${VARIANT}_seed${SEED}_${STAMP}}"
mkdir -p "${OUT_DIR}"

ENV_LOG="${OUT_DIR}/environment.txt"
COMMAND_LOG="${OUT_DIR}/commands_used.txt"
IDLE_LOG="${OUT_DIR}/tegrastats_idle.log"
FP16_TEGRA_LOG="${OUT_DIR}/tegrastats_trtexec_fp16.log"
FP32_TEGRA_LOG="${OUT_DIR}/tegrastats_trtexec_fp32.log"
FP16_TRT_LOG="${OUT_DIR}/trtexec_fp16.log"
FP32_TRT_LOG="${OUT_DIR}/trtexec_fp32.log"
SUMMARY_PREFIX="${OUT_DIR}/classifier_only_peak_ram"

stop_tegrastats() {
    "${TEGRSTATS_BIN}" --stop >/dev/null 2>&1 || true
    sleep 0.4
}

cleanup() {
    stop_tegrastats
}
trap cleanup EXIT INT TERM

require_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: ${cmd}" >&2
        exit 1
    fi
}

require_file() {
    local path="$1"
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Required file not found: ${path}" >&2
        exit 1
    fi
}

start_tegrastats() {
    local logfile="$1"
    stop_tegrastats
    rm -f "${logfile}"
    "${TEGRSTATS_BIN}" --interval "${INTERVAL_MS}" --logfile "${logfile}" >/dev/null 2>&1 &
    sleep 0.6
}

run_trtexec_one() {
    local precision="$1"
    local engine="$2"
    local tegra_log="$3"
    local trt_log="$4"

    local -a cmd=(
        "${TRTEXEC_BIN}"
        "--loadEngine=${engine}"
        "--shapes=${INPUT_SHAPE}"
        "--warmUp=${WARMUP_MS}"
        "--duration=${DURATION_SEC}"
        "--iterations=${ITERATIONS}"
    )

    if [ "${USE_SPIN_WAIT}" = "1" ]; then
        cmd+=("--useSpinWait")
    fi

    echo "" | tee -a "${COMMAND_LOG}"
    echo "[${precision}] ${cmd[*]}" | tee -a "${COMMAND_LOG}"

    start_tegrastats "${tegra_log}"
    set +e
    "${cmd[@]}" 2>&1 | tee "${trt_log}"
    local trt_status=${PIPESTATUS[0]}
    set -e
    stop_tegrastats

    if [ "${trt_status}" -ne 0 ]; then
        echo "[ERROR] ${precision} trtexec failed with exit code ${trt_status}" >&2
        exit "${trt_status}"
    fi
}

require_command "${TRTEXEC_BIN}"
require_command "${TEGRSTATS_BIN}"
require_command "${PYTHON_BIN}"
require_file "${FP16_ENGINE}"
require_file "${FP32_ENGINE}"
require_file "${SUMMARY_SCRIPT}"

echo "============================================================"
echo "Jetson classifier-only TensorRT peak RAM benchmark"
echo "============================================================"
echo "Dataset            : ${DATASET}"
echo "Variant            : ${VARIANT}"
echo "Seed               : ${SEED}"
echo "Sampling tool      : NVIDIA tegrastats"
echo "Sampling interval  : ${INTERVAL_MS} ms"
echo "Idle baseline      : ${IDLE_SECONDS} s"
echo "Input shape        : ${INPUT_SHAPE}"
echo "Warm-up            : ${WARMUP_MS} ms"
echo "Timed duration     : ${DURATION_SEC} s minimum"
echo "Iterations         : ${ITERATIONS} minimum"
echo "Output directory   : ${OUT_DIR}"
echo "============================================================"

{
    echo "Collected at: $(date --iso-8601=seconds 2>/dev/null || date)"
    echo ""
    echo "=== Device model ==="
    tr -d '\0' < /proc/device-tree/model 2>/dev/null || true
    echo ""
    echo ""
    echo "=== L4T ==="
    cat /etc/nv_tegra_release 2>/dev/null || true
    echo ""
    echo "=== Kernel ==="
    uname -a || true
    echo ""
    echo "=== TensorRT ==="
    "${PYTHON_BIN}" -c "import tensorrt as trt; print(trt.__version__)" 2>/dev/null || true
    echo ""
    echo "=== CUDA ==="
    nvcc --version 2>/dev/null || true
    echo ""
    echo "=== Power mode ==="
    sudo nvpmodel -q 2>/dev/null || true
    echo ""
    echo "=== jetson_clocks --show ==="
    sudo jetson_clocks --show 2>/dev/null || true
} > "${ENV_LOG}"

echo "[INFO] Applying requested Jetson performance settings..."
sudo nvpmodel -m "${NVP_MODEL}" || true
sudo jetson_clocks || true

echo "[INFO] Collecting idle RAM baseline for ${IDLE_SECONDS} seconds..."
start_tegrastats "${IDLE_LOG}"
sleep "${IDLE_SECONDS}"
stop_tegrastats

echo "[INFO] Running FP16 classifier-only benchmark..."
run_trtexec_one "FP16" "${FP16_ENGINE}" "${FP16_TEGRA_LOG}" "${FP16_TRT_LOG}"

echo "[INFO] Running FP32 classifier-only benchmark..."
run_trtexec_one "FP32" "${FP32_ENGINE}" "${FP32_TEGRA_LOG}" "${FP32_TRT_LOG}"

echo "[INFO] Summarizing RAM and trtexec logs..."
"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
    --idle-log "${IDLE_LOG}" \
    --benchmark "FP16::${FP16_TEGRA_LOG}::${FP16_TRT_LOG}::${FP16_ENGINE}" \
    --benchmark "FP32::${FP32_TEGRA_LOG}::${FP32_TRT_LOG}::${FP32_ENGINE}" \
    --sampling-tool "NVIDIA tegrastats" \
    --sampling-interval-ms "${INTERVAL_MS}" \
    --scope "TensorRT classifier only" \
    --output-prefix "${SUMMARY_PREFIX}"

echo ""
echo "============================================================"
echo "Finished."
echo "Use this CSV for the deployment table:"
echo "  ${SUMMARY_PREFIX}_summary.csv"
echo ""
echo "Keep these files as raw evidence:"
echo "  ${ENV_LOG}"
echo "  ${COMMAND_LOG}"
echo "  ${IDLE_LOG}"
echo "  ${FP16_TEGRA_LOG}"
echo "  ${FP32_TEGRA_LOG}"
echo "  ${FP16_TRT_LOG}"
echo "  ${FP32_TRT_LOG}"
echo "============================================================"
