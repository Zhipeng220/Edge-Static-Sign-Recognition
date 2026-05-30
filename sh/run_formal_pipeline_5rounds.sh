#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Run repeated offline image-to-label benchmarks with the existing
# run_pipeline_5rounds.sh script.
#
# Default: 28 held-out smoke-test images, 5 rounds per precision.
#
# Formal paper example with 1000 genuine held-out images:
#   IMAGE_DIR=./test_data/benchmark_1000_images EXPECTED_IMAGES=1000 \
#   ROUNDS=5 bash sh/run_formal_pipeline_5rounds.sh
#
# Do not duplicate images merely to reach 1000. Report the real held-out count.

IMAGE_DIR="${IMAGE_DIR:-./test_data/sample_image}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-28}"
ROUNDS="${ROUNDS:-5}"
WARMUP_IMAGES="${WARMUP_IMAGES:-30}"
INTERVAL_MS="${INTERVAL_MS:-100}"

exec env \
  IMAGE_DIR="${IMAGE_DIR}" \
  EXPECTED_IMAGES="${EXPECTED_IMAGES}" \
  ROUNDS="${ROUNDS}" \
  WARMUP_IMAGES="${WARMUP_IMAGES}" \
  INTERVAL_MS="${INTERVAL_MS}" \
  bash "${SCRIPT_DIR}/run_pipeline_5rounds.sh"
