#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ============================================================
# run_trtexec_5rounds.sh
#
# Final reproducible TensorRT classifier-only benchmark protocol.
#
# Default protocol:
#   - device: Jetson Xavier NX
#   - batch size: 1 (explicit)
#   - fixed input shape: landmarks:1x21x3
#   - one stream
#   - data transfers enabled
#   - warm-up: at least 1000 ms
#   - timed benchmark: at least 10 seconds and at least 1000 iterations
#   - --useSpinWait enabled
#   - 5 independent rounds for FP16 and FP32
#   - NVIDIA tegrastats sampled every 100 ms
#
# Important:
#   With both --duration=10 and --iterations=1000, trtexec runs for
#   whichever condition is longer. Therefore, do NOT describe this
#   protocol as "exactly 1000 timed inferences".
#
# Usage:
#   bash sh/run_trtexec_5rounds.sh
#
# Optional override example:
#   DATASET=asl_large_dataset VARIANT=full_model SEED=42 ROUNDS=5 \
#   bash sh/run_trtexec_5rounds.sh
# ============================================================

ROOT_DIR="${ROOT_DIR:-./multi_dataset_results_fixed_labels}"
DATASET="${DATASET:-asl_large_dataset}"
VARIANT="${VARIANT:-full_model}"
SEED="${SEED:-42}"

ROUNDS="${ROUNDS:-5}"
INTERVAL_MS="${INTERVAL_MS:-100}"
IDLE_SECONDS="${IDLE_SECONDS:-10}"
WARMUP_MS="${WARMUP_MS:-1000}"
DURATION_SEC="${DURATION_SEC:-10}"
ITERATIONS="${ITERATIONS:-1000}"
INPUT_SHAPE="${INPUT_SHAPE:-landmarks:1x21x3}"
NVP_MODEL="${NVP_MODEL:-5}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRTEXEC_BIN="${TRTEXEC_BIN:-trtexec}"
TEGRASTATS_BIN="${TEGRASTATS_BIN:-tegrastats}"

SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-${REPO_ROOT}/summarize_trtexec_5rounds.py}"

ENGINE_DIR="${ROOT_DIR}/${DATASET}/deployment"
FP16_ENGINE="${FP16_ENGINE:-${ENGINE_DIR}/${VARIANT}_seed${SEED}_fp16.engine}"
FP32_ENGINE="${FP32_ENGINE:-${ENGINE_DIR}/${VARIANT}_seed${SEED}_fp32.engine}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-./trtexec_final_results/${DATASET}_${VARIANT}_seed${SEED}_${STAMP}}"
mkdir -p "${OUT_DIR}"

ENV_LOG="${OUT_DIR}/environment.txt"
COMMAND_LOG="${OUT_DIR}/commands_used.txt"
IDLE_LOG="${OUT_DIR}/tegrastats_idle.log"

stop_tegrastats() {
    "${TEGRASTATS_BIN}" --stop >/dev/null 2>&1 || true
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
    "${TEGRASTATS_BIN}" --interval "${INTERVAL_MS}" --logfile "${logfile}" >/dev/null 2>&1 &
    sleep 0.6
}

run_one_round() {
    local precision="$1"
    local engine="$2"
    local round="$3"

    local precision_lc
    precision_lc="$(echo "${precision}" | tr '[:upper:]' '[:lower:]')"

    local trt_log="${OUT_DIR}/trtexec_${precision_lc}_round${round}.log"
    local tegra_log="${OUT_DIR}/tegrastats_${precision_lc}_round${round}.log"

    local -a cmd=(
        "${TRTEXEC_BIN}"
        "--loadEngine=${engine}"
        "--shapes=${INPUT_SHAPE}"
        "--warmUp=${WARMUP_MS}"
        "--duration=${DURATION_SEC}"
        "--iterations=${ITERATIONS}"
        "--useSpinWait"
    )

    echo "" | tee -a "${COMMAND_LOG}"
    echo "[${precision} round ${round}] ${cmd[*]}" | tee -a "${COMMAND_LOG}"

    start_tegrastats "${tegra_log}"

    set +e
    "${cmd[@]}" 2>&1 | tee "${trt_log}"
    local status=${PIPESTATUS[0]}
    set -e

    stop_tegrastats

    if [ "${status}" -ne 0 ]; then
        echo "[ERROR] ${precision} round ${round} failed with exit code ${status}" >&2
        exit "${status}"
    fi
}

require_command "${TRTEXEC_BIN}"
require_command "${TEGRASTATS_BIN}"
require_command "${PYTHON_BIN}"
require_file "${FP16_ENGINE}"
require_file "${FP32_ENGINE}"
require_file "${SUMMARY_SCRIPT}"

echo "============================================================"
echo "Final TensorRT classifier-only benchmark protocol"
echo "============================================================"
echo "Dataset              : ${DATASET}"
echo "Variant              : ${VARIANT}"
echo "Seed                 : ${SEED}"
echo "Rounds               : ${ROUNDS}"
echo "Input shape          : ${INPUT_SHAPE}"
echo "Warm-up              : at least ${WARMUP_MS} ms"
echo "Timed run            : at least ${DURATION_SEC} s and ${ITERATIONS} iterations"
echo "Spin wait            : enabled"
echo "Sampling tool        : NVIDIA tegrastats"
echo "Sampling interval    : ${INTERVAL_MS} ms"
echo "Output directory     : ${OUT_DIR}"
echo "============================================================"

echo "[INFO] Applying Jetson performance settings..."
sudo nvpmodel -m "${NVP_MODEL}" || true
sudo jetson_clocks || true

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

echo "[INFO] Collecting idle RAM baseline for ${IDLE_SECONDS} seconds..."
start_tegrastats "${IDLE_LOG}"
sleep "${IDLE_SECONDS}"
stop_tegrastats

for round in $(seq 1 "${ROUNDS}"); do
    echo "[INFO] Running FP16 round ${round}/${ROUNDS}..."
    run_one_round "FP16" "${FP16_ENGINE}" "${round}"
done

for round in $(seq 1 "${ROUNDS}"); do
    echo "[INFO] Running FP32 round ${round}/${ROUNDS}..."
    run_one_round "FP32" "${FP32_ENGINE}" "${round}"
done

echo "[INFO] Summarizing final protocol logs..."
"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
    --input-dir "${OUT_DIR}" \
    --idle-log "${IDLE_LOG}" \
    --rounds "${ROUNDS}" \
    --sampling-tool "NVIDIA tegrastats" \
    --sampling-interval-ms "${INTERVAL_MS}" \
    --scope "TensorRT classifier only" \
    --output-prefix "${OUT_DIR}/trtexec_final"

echo ""
echo "============================================================"
echo "Finished."
echo "Use these files for the paper:"
echo "  ${OUT_DIR}/trtexec_final_round_summary.csv"
echo "  ${OUT_DIR}/trtexec_final_aggregate_summary.csv"
echo "  ${OUT_DIR}/trtexec_final_report.md"
echo ""
echo "Keep raw logs in:"
echo "  ${OUT_DIR}"
echo "============================================================"
