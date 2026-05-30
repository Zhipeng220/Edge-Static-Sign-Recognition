#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Build the TensorRT 8.x C++ worker on Jetson Xavier NX.
# Usage:
#   bash sh/build_trt_worker.sh

CXX="${CXX:-g++}"
SRC="${SRC:-trt_infer_worker.cpp}"
OUT="${OUT:-trt_infer_worker}"

CUDA_INC="${CUDA_INC:-/usr/local/cuda/include}"
CUDA_LIB="${CUDA_LIB:-/usr/local/cuda/lib64}"

TRT_INC="${TRT_INC:-/usr/include/aarch64-linux-gnu}"
TRT_LIB="${TRT_LIB:-/usr/lib/aarch64-linux-gnu}"

echo "[INFO] Building ${OUT} ..."
"${CXX}" \
  -O3 \
  -std=c++14 \
  -Wall \
  -Wextra \
  "${SRC}" \
  -I"${CUDA_INC}" \
  -I"${TRT_INC}" \
  -L"${CUDA_LIB}" \
  -L"${TRT_LIB}" \
  -Wl,-rpath,"${CUDA_LIB}" \
  -Wl,-rpath,"${TRT_LIB}" \
  -lnvinfer \
  -lcudart \
  -o "${OUT}"

echo "[OK] Built: ${OUT}"
echo "[INFO] Verify with:"
echo "  ./${OUT} --engine ./multi_dataset_results_fixed_labels/asl_large_dataset/deployment/full_model_seed42_fp16.engine"
