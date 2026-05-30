#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Formal offline image-to-label deployment benchmark without sklearn on Jetson.
#
# Preliminary smoke test:
#   IMAGE_DIR=./test_data/sample_image EXPECTED_IMAGES=28 ROUNDS=1 \
#   bash sh/run_pipeline_5rounds.sh
#
# Formal run:
#   IMAGE_DIR=./test_data/benchmark_1000_images EXPECTED_IMAGES=1000 ROUNDS=5 \
#   bash sh/run_pipeline_5rounds.sh

ROOT_DIR="${ROOT_DIR:-./multi_dataset_results_fixed_labels}"
DATASET="${DATASET:-asl_large_dataset}"
VARIANT="${VARIANT:-full_model}"
SEED="${SEED:-42}"

IMAGE_DIR="${IMAGE_DIR:-./test_data/benchmark_1000_images}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-1000}"
ROUNDS="${ROUNDS:-5}"
WARMUP_IMAGES="${WARMUP_IMAGES:-30}"
INTERVAL_MS="${INTERVAL_MS:-100}"
IDLE_SECONDS="${IDLE_SECONDS:-10}"
NVP_MODEL="${NVP_MODEL:-5}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TEGRSTATS_BIN="${TEGRSTATS_BIN:-tegrastats}"
WORKER="${WORKER:-./trt_infer_worker}"

BENCHMARK_PY="${BENCHMARK_PY:-${REPO_ROOT}/benchmark_image_to_label.py}"
SUMMARY_PY="${SUMMARY_PY:-${REPO_ROOT}/summarize_pipeline_results.py}"

ENGINE_DIR="${ROOT_DIR}/${DATASET}/deployment"
FP16_ENGINE="${FP16_ENGINE:-${ENGINE_DIR}/${VARIANT}_seed${SEED}_fp16.engine}"
FP32_ENGINE="${FP32_ENGINE:-${ENGINE_DIR}/${VARIANT}_seed${SEED}_fp32.engine}"

# Lightweight NumPy metadata exported on the training PC.
SCALER_STATS="${SCALER_STATS:-${ENGINE_DIR}/scaler_stats_${VARIANT}_seed${SEED}.npz}"
LABEL_CLASSES="${LABEL_CLASSES:-${ENGINE_DIR}/label_classes_${VARIANT}_seed${SEED}.npy}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-./pipeline_final_results/${DATASET}_${VARIANT}_seed${SEED}_${STAMP}}"
mkdir -p "${OUT_DIR}"

IDLE_LOG="${OUT_DIR}/tegrastats_idle.log"
COMMAND_LOG="${OUT_DIR}/commands_used.txt"
ENV_LOG="${OUT_DIR}/environment.txt"

stop_tegrastats() {
    "${TEGRSTATS_BIN}" --stop >/dev/null 2>&1 || true
    sleep 0.4
}

cleanup() {
    stop_tegrastats
}
trap cleanup EXIT INT TERM

require_file() {
    if [ ! -f "$1" ]; then
        echo "[ERROR] Missing file: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [ ! -d "$1" ]; then
        echo "[ERROR] Missing directory: $1" >&2
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

run_round() {
    local precision="$1"
    local engine="$2"
    local round="$3"
    local lc
    lc="$(echo "${precision}" | tr '[:upper:]' '[:lower:]')"

    local csv="${OUT_DIR}/pipeline_${lc}_round${round}.csv"
    local summary="${OUT_DIR}/pipeline_${lc}_round${round}_summary.json"
    local worker_log="${OUT_DIR}/worker_${lc}_round${round}.stderr.log"
    local tegra_log="${OUT_DIR}/tegrastats_pipeline_${lc}_round${round}.log"
    local stdout_log="${OUT_DIR}/pipeline_${lc}_round${round}.stdout.log"

    local -a cmd=(
        "${PYTHON_BIN}" "${BENCHMARK_PY}"
        "--image-dir" "${IMAGE_DIR}"
        "--engine" "${engine}"
        "--worker" "${WORKER}"
        "--scaler-stats" "${SCALER_STATS}"
        "--label-classes" "${LABEL_CLASSES}"
        "--output-csv" "${csv}"
        "--summary-json" "${summary}"
        "--worker-stderr-log" "${worker_log}"
        "--precision-label" "${precision}"
        "--round-id" "${round}"
        "--expected-images" "${EXPECTED_IMAGES}"
        "--warmup-images" "${WARMUP_IMAGES}"
    )

    echo "[${precision} round ${round}] ${cmd[*]}" | tee -a "${COMMAND_LOG}"

    start_tegrastats "${tegra_log}"
    set +e
    "${cmd[@]}" 2>&1 | tee "${stdout_log}"
    local status=${PIPESTATUS[0]}
    set -e
    stop_tegrastats

    if [ "${status}" -ne 0 ]; then
        echo "[ERROR] ${precision} pipeline round ${round} failed with ${status}" >&2
        exit "${status}"
    fi
}

require_dir "${IMAGE_DIR}"
require_file "${WORKER}"
require_file "${FP16_ENGINE}"
require_file "${FP32_ENGINE}"
require_file "${SCALER_STATS}"
require_file "${LABEL_CLASSES}"
require_file "${BENCHMARK_PY}"
require_file "${SUMMARY_PY}"

echo "============================================================"
echo "Offline image-to-label pipeline benchmark (NumPy metadata)"
echo "============================================================"
echo "Dataset            : ${DATASET}"
echo "Variant            : ${VARIANT}"
echo "Seed               : ${SEED}"
echo "Images             : ${IMAGE_DIR}"
echo "Expected images    : ${EXPECTED_IMAGES}"
echo "Rounds             : ${ROUNDS}"
echo "Warm-up images     : ${WARMUP_IMAGES}"
echo "Sampling tool      : NVIDIA tegrastats"
echo "Sampling interval  : ${INTERVAL_MS} ms"
echo "Scaler stats       : ${SCALER_STATS}"
echo "Label classes      : ${LABEL_CLASSES}"
echo "Output directory   : ${OUT_DIR}"
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
    echo "=== MediaPipe ==="
    "${PYTHON_BIN}" -c "import mediapipe as mp; print(mp.__version__)" 2>/dev/null || true
    echo ""
    echo "=== OpenCV ==="
    "${PYTHON_BIN}" -c "import cv2; print(cv2.__version__)" 2>/dev/null || true
    echo ""
    echo "=== Power mode ==="
    sudo nvpmodel -q 2>/dev/null || true
    echo ""
    echo "=== jetson_clocks --show ==="
    sudo jetson_clocks --show 2>/dev/null || true
} > "${ENV_LOG}"

echo "[INFO] Collecting idle telemetry for ${IDLE_SECONDS} seconds..."
start_tegrastats "${IDLE_LOG}"
sleep "${IDLE_SECONDS}"
stop_tegrastats

for round in $(seq 1 "${ROUNDS}"); do
    echo "[INFO] Running FP16 round ${round}/${ROUNDS} ..."
    run_round "FP16" "${FP16_ENGINE}" "${round}"
done

for round in $(seq 1 "${ROUNDS}"); do
    echo "[INFO] Running FP32 round ${round}/${ROUNDS} ..."
    run_round "FP32" "${FP32_ENGINE}" "${round}"
done

echo "[INFO] Summarizing pipeline results ..."
"${PYTHON_BIN}" "${SUMMARY_PY}" \
    --input-dir "${OUT_DIR}" \
    --idle-log "${IDLE_LOG}" \
    --rounds "${ROUNDS}" \
    --sampling-tool "NVIDIA tegrastats" \
    --sampling-interval-ms "${INTERVAL_MS}" \
    --output-prefix "${OUT_DIR}/pipeline_final"

echo ""
echo "============================================================"
echo "Finished."
echo "Use these files for the paper:"
echo "  ${OUT_DIR}/pipeline_final_round_summary.csv"
echo "  ${OUT_DIR}/pipeline_final_aggregate_summary.csv"
echo "  ${OUT_DIR}/pipeline_final_report.md"
echo "============================================================"
