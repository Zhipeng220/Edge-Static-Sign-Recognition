#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PACK_ROOT="./paper_hand_packages/${STAMP}"
MANIFEST_DIR="${PACK_ROOT}/manifests"
WARN_FILE="${MANIFEST_DIR}/missing_files_warning.txt"

mkdir -p "$MANIFEST_DIR"
: > "$WARN_FILE"

READY_LIST="${MANIFEST_DIR}/paper_ready_files.txt"
PIPELINE_LIST="${MANIFEST_DIR}/pipeline_smoke_test_files.txt"
REPRO_LIST="${MANIFEST_DIR}/reproducibility_files.txt"

: > "$READY_LIST"
: > "$PIPELINE_LIST"
: > "$REPRO_LIST"

add_if_exists() {
    local path="$1"
    local list_file="$2"

    if [ -e "$path" ]; then
        printf '%s\n' "$path" >> "$list_file"
    else
        printf 'WARN: missing file or directory: %s\n' "$path" \
          | tee -a "$WARN_FILE"
    fi
}

echo "============================================================"
echo "1. Generate a fresh folder inventory"
echo "============================================================"

find . \
  -path './venv' -prune -o \
  -path './paper_hand_packages' -prune -o \
  -printf '%y\t%TY-%Tm-%Td %TH:%TM\t%10s bytes\t%p\n' \
  | sort \
  > "${MANIFEST_DIR}/folder_inventory_no_venv.txt"

echo "============================================================"
echo "2. Select paper-ready CSV files"
echo "============================================================"

for f in \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_final_holdout_summary.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_baseline_comparison_summary.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_baseline_comparison_all_seeds.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_ablation_summary.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_ablation_all_seeds.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_clean_anatomical_ablation_summary.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_clean_anatomical_ablation.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_mcnemar.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_dataset_accounting_summary.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_split_accounting.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_split_class_support.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_onnx_export_stats_full_model.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_model_file_sizes.csv \
  ./multi_dataset_results_fixed_labels/ALL_DATASETS_tensorrt_engine_sizes.csv
do
    add_if_exists "$f" "$READY_LIST"
done

TRTEXEC_DIR="$(
    find ./trtexec_final_results \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      | sort \
      | tail -n 1
)"

PEAK_RAM_DIR="$(
    find ./peak_ram_results \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      | sort \
      | tail -n 1
)"

if [ -n "${TRTEXEC_DIR:-}" ]; then
    add_if_exists "$TRTEXEC_DIR" "$READY_LIST"
else
    echo "WARN: no trtexec_final_results directory found." \
      | tee -a "$WARN_FILE"
fi

if [ -n "${PEAK_RAM_DIR:-}" ]; then
    add_if_exists "$PEAK_RAM_DIR" "$READY_LIST"
else
    echo "WARN: no peak_ram_results directory found." \
      | tee -a "$WARN_FILE"
fi

echo "============================================================"
echo "3. Select current pipeline smoke-test evidence"
echo "============================================================"

add_if_exists ./pipeline_final_results "$PIPELINE_LIST"

echo "============================================================"
echo "4. Select reproducibility scripts and deployment files"
echo "============================================================"

for f in \
  ./README.md \
  ./README_PATCH.md \
  ./benchmark_image_to_label.py \
  ./benchmark_offline_trt_pipeline.py \
  ./sh/build_all_trt_engines.sh \
  ./sh/build_trt_worker.sh \
  ./check_benchmark_images.py \
  ./collect_model_sizes.py \
  ./parity_test_trt.py \
  ./prepare_parity_inputs.py \
  ./sh/run_parity_test.sh \
  ./sh/run_peak_ram_trtexec.sh \
  ./sh/run_pipeline_5rounds.sh \
  ./sh/run_trtexec_5rounds.sh \
  ./summarize_peak_ram.py \
  ./summarize_pipeline_results.py \
  ./summarize_trtexec_5rounds.py \
  ./trt_infer_worker.cpp \
  ./trt_infer_worker \
  ./engines \
  ./models \
  ./scripts \
  ./multi_dataset_results_fixed_labels/asl_large_dataset/deployment/full_model_seed42.onnx \
  ./multi_dataset_results_fixed_labels/asl_large_dataset/scaler_full_model_seed42.pkl \
  ./multi_dataset_results_fixed_labels/asl_large_dataset/label_encoder_full_model_seed42.pkl
do
    add_if_exists "$f" "$REPRO_LIST"
done

# TensorRT parity results may not exist yet. Include them automatically later
# when the parity experiment has been completed.
if [ -d ./parity_results ]; then
    add_if_exists ./parity_results "$REPRO_LIST"
fi

sort -u -o "$READY_LIST" "$READY_LIST"
sort -u -o "$PIPELINE_LIST" "$PIPELINE_LIST"
sort -u -o "$REPRO_LIST" "$REPRO_LIST"

echo "============================================================"
echo "5. Create archives"
echo "============================================================"

READY_TAR="${PACK_ROOT}/paper_ready_data_${STAMP}.tar.gz"
PIPELINE_TAR="${PACK_ROOT}/pipeline_smoke_test_evidence_${STAMP}.tar.gz"
REPRO_TAR="${PACK_ROOT}/reproducibility_scripts_${STAMP}.tar.gz"
UPLOAD_TAR="${PACK_ROOT}/paper_hand_upload_bundle_${STAMP}.tar.gz"

tar -czf "$READY_TAR" -T "$READY_LIST"
tar -czf "$PIPELINE_TAR" -T "$PIPELINE_LIST"
tar -czf "$REPRO_TAR" -T "$REPRO_LIST"

(
    cd "$PACK_ROOT"

    sha256sum \
      "$(basename "$READY_TAR")" \
      "$(basename "$PIPELINE_TAR")" \
      "$(basename "$REPRO_TAR")" \
      > SHA256SUMS.txt
)

tar -czf "$UPLOAD_TAR" \
  -C "$PACK_ROOT" \
  "$(basename "$READY_TAR")" \
  "$(basename "$PIPELINE_TAR")" \
  "$(basename "$REPRO_TAR")" \
  SHA256SUMS.txt \
  manifests

(
    cd "$PACK_ROOT"
    sha256sum "$(basename "$UPLOAD_TAR")" \
      > UPLOAD_BUNDLE_SHA256.txt
)

echo
echo "============================================================"
echo "6. Completed"
echo "============================================================"
echo "Package directory:"
echo "  ${PACK_ROOT}"
echo
echo "Recommended upload file:"
echo "  ${UPLOAD_TAR}"
echo
echo "Generated files:"
ls -lh "$PACK_ROOT"
echo
echo "Missing-file warnings:"
if [ -s "$WARN_FILE" ]; then
    cat "$WARN_FILE"
else
    echo "  None"
fi
