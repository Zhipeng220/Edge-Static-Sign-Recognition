#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="./required_table_logs_${STAMP}"
ARCHIVE="./required_table_logs_${STAMP}.tar.gz"

mkdir -p "$OUT_DIR"

latest_subdir() {
    local root="$1"

    find "$root" \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      2>/dev/null \
      | sort \
      | tail -n 1
}

copy_if_exists() {
    local src="$1"

    if [ -f "$src" ]; then
        cp -a "$src" "$OUT_DIR/"
        echo "[COPIED] $src"
    else
        echo "[MISSING] $src"
    fi
}

TRTEXEC_OUT="$(latest_subdir ./trtexec_final_results)"
PEAK_RAM_OUT="$(latest_subdir ./peak_ram_results)"
PIPELINE_OUT="$(latest_subdir ./pipeline_final_results)"

echo "============================================================"
echo "Detected directories"
echo "============================================================"
echo "TRTEXEC_OUT  = ${TRTEXEC_OUT:-NOT FOUND}"
echo "PEAK_RAM_OUT = ${PEAK_RAM_OUT:-NOT FOUND}"
echo "PIPELINE_OUT = ${PIPELINE_OUT:-NOT FOUND}"
echo

echo "============================================================"
echo "Copy classifier-only formal results"
echo "============================================================"

if [ -n "${TRTEXEC_OUT:-}" ]; then
    copy_if_exists "${TRTEXEC_OUT}/trtexec_final_aggregate_summary.csv"
    copy_if_exists "${TRTEXEC_OUT}/trtexec_final_round_summary.csv"
    copy_if_exists "${TRTEXEC_OUT}/environment.txt"
    copy_if_exists "${TRTEXEC_OUT}/commands_used.txt"
fi

if [ -n "${PEAK_RAM_OUT:-}" ]; then
    copy_if_exists "${PEAK_RAM_OUT}/classifier_only_peak_ram_summary.csv"
fi

echo
echo "============================================================"
echo "Copy 28-image pipeline smoke-test results"
echo "============================================================"

if [ -n "${PIPELINE_OUT:-}" ]; then
    copy_if_exists "${PIPELINE_OUT}/pipeline_final_aggregate_summary.csv"
    copy_if_exists "${PIPELINE_OUT}/pipeline_final_round_summary.csv"

    copy_if_exists "${PIPELINE_OUT}/pipeline_fp16_round1.csv"
    copy_if_exists "${PIPELINE_OUT}/pipeline_fp32_round1.csv"

    copy_if_exists "${PIPELINE_OUT}/pipeline_fp16_round1_summary.json"
    copy_if_exists "${PIPELINE_OUT}/pipeline_fp32_round1_summary.json"

    copy_if_exists "${PIPELINE_OUT}/tegrastats_pipeline_fp16_round1.log"
    copy_if_exists "${PIPELINE_OUT}/tegrastats_pipeline_fp32_round1.log"

    copy_if_exists "${PIPELINE_OUT}/environment.txt"
    copy_if_exists "${PIPELINE_OUT}/commands_used.txt"
fi

echo
echo "============================================================"
echo "Generate file manifest"
echo "============================================================"

find "$OUT_DIR" \
  -type f \
  -printf '%10s bytes\t%f\n' \
  | sort \
  > "${OUT_DIR}/MANIFEST.txt"

cat > "${OUT_DIR}/README.txt" <<'EOF_README'
This package contains:

1. Formal five-round classifier-only TensorRT results.
2. Classifier-only peak RAM summary.
3. Current 28-image image-to-label pipeline smoke-test results.
4. FP16 and FP32 smoke-test tegrastats logs for power-rail checking.

Important:
The pipeline results contain only one round per precision mode and 28 images.
They are smoke-test evidence only and must not be described as final five-round
image-to-label benchmark results.
EOF_README

tar -czf "$ARCHIVE" "$OUT_DIR"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

echo
echo "============================================================"
echo "Completed"
echo "============================================================"
echo "Archive:"
echo "  $ARCHIVE"
echo
echo "Checksum:"
echo "  ${ARCHIVE}.sha256"
echo
echo "Contents:"
tar -tzf "$ARCHIVE"
