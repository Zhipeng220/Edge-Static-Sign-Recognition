"""
Unified MediaPipe landmark extraction script for ASL / Indian Sign Language / NUS Hand Posture.

This script combines three previous workflows:
1) Move one image per class into final_test for folder-structured datasets.
2) Extract MediaPipe hand landmarks and save X_data.npy / y_labels.npy / class_mapping.npy.
3) Handle NUS_Hand_Posture where images are stored flat in Color/ or BW/ and labels are parsed from filenames.

Default output protocol:
    max_num_hands = 1  -> X_data.npy shape [N, 63]

This matches the current single-hand 21 x 3 landmark training protocol. If you need the old
126-dimensional two-hand protocol, set max_num_hands=2 for that dataset config.
"""

from __future__ import annotations

import os
import re
import csv
import json
import shutil
import random
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# 0. Global settings
# ============================================================

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

SKIP_DIRS = {
    "final_test",
    "npy_for_training",
    "processed_split",
    "processed_spatial_tokens",
    "__pycache__",
}


@dataclass
class DatasetExtractSpec:
    name: str
    dataset_type: str  # "folder" or "flat_nus"
    input_dir: str
    output_dir: str
    final_test_dir: str = ""
    extract_final_test: bool = True
    seed: int = 42
    max_num_hands: int = 1  # 1 -> 63 dims, 2 -> 126 dims
    min_detection_confidence: float = 0.5
    enabled: bool = True


# ============================================================
# 1. Dataset configuration
# ============================================================
# 修改这里即可。建议当前论文主实验统一用 max_num_hands=1，即 [N, 63]。

DATASETS: List[DatasetExtractSpec] = [
    DatasetExtractSpec(
        name="asl_dataset",
        dataset_type="folder",
        input_dir=r".\data\raw_datasets\asl_dataset",
        output_dir=r".\data\asl_dataset\processed_landmarks",
        final_test_dir=r".\data\asl_dataset\final_holdout",
        extract_final_test=True,
        max_num_hands=1,
        enabled=True,
    ),
    DatasetExtractSpec(
        name="indian_sign_language",
        dataset_type="folder",
        input_dir=r".\data\raw_datasets\indian_sign_language",
        output_dir=r".\data\indian_sign_language\processed_landmarks",
        final_test_dir=r".\data\indian_sign_language\final_holdout",
        extract_final_test=True,
        max_num_hands=1,
        enabled=True,
    ),
    DatasetExtractSpec(
        name="nus_hand_posture",
        dataset_type="flat_nus",
        input_dir=r".\data\raw_datasets\nus_hand_posture\Color",
        output_dir=r".\data\nus_hand_posture\processed_landmarks",
        final_test_dir=r".\data\nus_hand_posture\final_holdout",
        extract_final_test=True,
        max_num_hands=1,
        enabled=True,
    ),
    DatasetExtractSpec(
        name="asl_large_dataset",
        dataset_type="folder",
        input_dir=r".\data\raw_datasets\asl_large_dataset",
        output_dir=r".\data\asl_large_dataset\processed_landmarks",
        # 大 ASL 如果你已经有独立测试集 asl_alphabet_test，则不建议再从 train 里 move 图片。
        # 如果你确实想从大 ASL train 中每类再抽 1 张 final_test，可把 extract_final_test 改成 True，
        # 并把 final_test_dir 改成你想保存的位置。
        final_test_dir=r".\data\asl_large_dataset\final_holdout",
        extract_final_test=False,
        max_num_hands=1,
        enabled=True,
    ),
]


# ============================================================
# 2. Common utilities
# ============================================================

def is_valid_image_file(filename: str) -> bool:
    """Return True for normal image files, excluding hidden/macOS resource files."""
    name = os.path.basename(filename)
    if name.startswith("._"):
        return False
    if name.startswith("."):
        return False
    return name.lower().endswith(IMAGE_EXTS)


def get_class_dirs(input_dataset_dir: str) -> List[str]:
    """Read category subfolders for folder-structured datasets."""
    classes = []
    for d in os.listdir(input_dataset_dir):
        full_path = os.path.join(input_dataset_dir, d)
        if not os.path.isdir(full_path):
            continue
        if d.startswith("."):
            continue
        if d in SKIP_DIRS:
            continue
        classes.append(d)
    classes.sort()
    return classes


def parse_nus_label_from_filename(filename: str) -> str:
    """
    Parse NUS class label from filenames such as:
        _g1.jpg        -> g1
        _g1 (1).jpg    -> g1
        _g10.jpg       -> g10
        g2.jpg         -> g2
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = stem.strip().lstrip("_")
    stem = re.sub(r"\s*\(\d+\)$", "", stem)

    match = re.search(r"([A-Za-z]+\d+)", stem)
    if match:
        return match.group(1)

    match = re.search(r"([A-Za-z]+)", stem)
    if match:
        return match.group(1)

    raise ValueError(f"Cannot parse NUS label from filename: {filename}")


def collect_folder_samples(input_dir: str) -> List[Dict[str, str]]:
    """Collect samples from folder layout: input_dir/class_name/image.jpg."""
    samples: List[Dict[str, str]] = []
    classes = get_class_dirs(input_dir)
    if not classes:
        raise RuntimeError(f"No class folders found in: {input_dir}")

    for cls_name in classes:
        cls_dir = os.path.join(input_dir, cls_name)
        for filename in os.listdir(cls_dir):
            if not is_valid_image_file(filename):
                continue
            samples.append({
                "label": cls_name,
                "filename": filename,
                "path": os.path.join(cls_dir, filename),
            })
    return samples


def collect_flat_nus_samples(input_dir: str) -> List[Dict[str, str]]:
    """Collect NUS samples from flat layout: Color/g1.jpg, Color/g1 (1).jpg, ..."""
    samples: List[Dict[str, str]] = []
    for filename in os.listdir(input_dir):
        if not is_valid_image_file(filename):
            continue
        label = parse_nus_label_from_filename(filename)
        samples.append({
            "label": label,
            "filename": filename,
            "path": os.path.join(input_dir, filename),
        })
    if not samples:
        raise RuntimeError(f"No valid images found in: {input_dir}")
    return samples


def collect_samples(spec: DatasetExtractSpec) -> List[Dict[str, str]]:
    if spec.dataset_type == "folder":
        return collect_folder_samples(spec.input_dir)
    if spec.dataset_type == "flat_nus":
        return collect_flat_nus_samples(spec.input_dir)
    raise ValueError(f"Unknown dataset_type: {spec.dataset_type}")


def count_by_label(samples: List[Dict[str, str]]) -> Dict[str, int]:
    return dict(Counter(str(s["label"]) for s in samples))


def count_final_test_images(final_test_dir: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not final_test_dir or not os.path.exists(final_test_dir):
        return counts

    for label in os.listdir(final_test_dir):
        label_dir = os.path.join(final_test_dir, label)
        if not os.path.isdir(label_dir):
            continue
        counts[label] = sum(
            1 for f in os.listdir(label_dir)
            if is_valid_image_file(f)
        )
    return counts


# ============================================================
# 3. Final-test extraction
# ============================================================

def extract_final_test_samples(spec: DatasetExtractSpec) -> None:
    """
    Move one image per class to final_test/class_name/.

    This function is intentionally idempotent at class level:
    if final_test/class_name already contains at least one image, that class is skipped.
    This prevents repeated runs from continuously removing training images.
    """
    if not spec.final_test_dir:
        print(f"[{spec.name}] final_test_dir is empty; skipping final_test extraction.")
        return

    random.seed(spec.seed)
    os.makedirs(spec.final_test_dir, exist_ok=True)

    samples = collect_samples(spec)
    class_to_samples: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for sample in samples:
        class_to_samples[str(sample["label"])].append(sample)

    print("\n================= final_test extraction =================")
    print(f"Dataset: {spec.name}")
    print(f"Input dir: {spec.input_dir}")
    print(f"final_test dir: {spec.final_test_dir}")
    print(f"Classes: {len(class_to_samples)}")
    print("=========================================================\n")

    for label, label_samples in sorted(class_to_samples.items()):
        cls_test_dir = os.path.join(spec.final_test_dir, label)
        os.makedirs(cls_test_dir, exist_ok=True)

        existing = [f for f in os.listdir(cls_test_dir) if is_valid_image_file(f)]
        if existing:
            print(f"{label}: final_test already exists; skip moving")
            continue

        if not label_samples:
            print(f"{label}: no images; skip")
            continue

        chosen = random.choice(label_samples)
        src = chosen["path"]
        dst = os.path.join(cls_test_dir, chosen["filename"])

        shutil.move(src, dst)
        print(
            f"{label}: {chosen['filename']} -> final_test, "
            f"{len(label_samples) - 1} images remain for training/extraction"
        )

    print(f"\n[{spec.name}] final_test extraction finished: {spec.final_test_dir}\n")


# ============================================================
# 4. MediaPipe extraction
# ============================================================

def hand_bbox_area(hand_landmarks) -> float:
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return max(max(xs) - min(xs), 0.0) * max(max(ys) - min(ys), 0.0)


def landmarks_to_feature_row(results, max_num_hands: int) -> Optional[List[float]]:
    """
    Convert MediaPipe results to either 63-dim single-hand or 126-dim two-hand vector.

    max_num_hands=1:
      choose the hand with the largest 2D bounding box -> 63 dims

    max_num_hands=2:
      use up to two detected hands and zero-pad missing second hand -> 126 dims
    """
    if not results.multi_hand_landmarks:
        return None

    detected_hands = list(results.multi_hand_landmarks)

    if max_num_hands == 1:
        best_hand = max(detected_hands, key=hand_bbox_area)
        row: List[float] = []
        for lm in best_hand.landmark:
            row.extend([lm.x, lm.y, lm.z])
        return row if len(row) == 63 else None

    if max_num_hands == 2:
        # Stable order by larger hand first. This avoids arbitrary order changes when two hands appear.
        detected_hands = sorted(detected_hands, key=hand_bbox_area, reverse=True)[:2]
        row = []
        for hand in detected_hands:
            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])

        if len(detected_hands) == 1:
            row.extend([0.0] * 63)

        return row if len(row) == 126 else None

    raise ValueError("max_num_hands must be 1 or 2")


def extract_mediapipe_landmarks(spec: DatasetExtractSpec) -> Dict[str, object]:
    """
    Extract MediaPipe landmarks and save:
        X_data.npy
        y_labels.npy
        class_mapping.npy
        metadata.csv
        failed_images.csv
        per_class_accounting.csv
        dataset_accounting_summary.json/csv
    """
    os.makedirs(spec.output_dir, exist_ok=True)

    samples = collect_samples(spec)
    labels = sorted(set(str(s["label"]) for s in samples))
    class_to_idx = {label: idx for idx, label in enumerate(labels)}

    np.save(os.path.join(spec.output_dir, "class_mapping.npy"), class_to_idx)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=spec.max_num_hands,
        min_detection_confidence=spec.min_detection_confidence,
    )

    X_data: List[List[float]] = []
    y_labels: List[int] = []
    metadata_rows: List[Dict[str, str]] = []
    failed_rows: List[Dict[str, str]] = []

    expected_dim = 63 if spec.max_num_hands == 1 else 126

    print("\n================= MediaPipe landmark extraction =================")
    print(f"Dataset: {spec.name}")
    print(f"Input dir: {spec.input_dir}")
    print(f"Output dir: {spec.output_dir}")
    print(f"Candidate images: {len(samples)}")
    print(f"Classes: {len(labels)}")
    print(f"max_num_hands: {spec.max_num_hands}")
    print(f"Expected feature dimension: {expected_dim}")
    print(f"Class mapping: {class_to_idx}")
    print("=================================================================\n")

    total_images = 0
    valid_images = 0
    unreadable = 0
    failed_detect = 0
    invalid_feature = 0

    class_progress = Counter()
    class_success = Counter()
    class_failed = Counter()

    for sample in samples:
        label = str(sample["label"])
        img_path = sample["path"]
        filename = sample["filename"]
        total_images += 1
        class_progress[label] += 1

        image = cv2.imread(img_path)
        if image is None:
            unreadable += 1
            class_failed[label] += 1
            metadata_rows.append({
                "label": label,
                "filename": filename,
                "path": img_path,
                "status": "unreadable",
            })
            failed_rows.append({
                "label": label,
                "filename": filename,
                "path": img_path,
                "reason": "cv2_imread_failed",
            })
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            failed_detect += 1
            class_failed[label] += 1
            metadata_rows.append({
                "label": label,
                "filename": filename,
                "path": img_path,
                "status": "no_hand",
            })
            failed_rows.append({
                "label": label,
                "filename": filename,
                "path": img_path,
                "reason": "mediapipe_no_hand",
            })
            continue

        row_features = landmarks_to_feature_row(results, spec.max_num_hands)
        if row_features is None or len(row_features) != expected_dim:
            invalid_feature += 1
            class_failed[label] += 1
            actual_len = 0 if row_features is None else len(row_features)
            metadata_rows.append({
                "label": label,
                "filename": filename,
                "path": img_path,
                "status": "invalid_feature_length",
            })
            failed_rows.append({
                "label": label,
                "filename": filename,
                "path": img_path,
                "reason": f"invalid_feature_length_{actual_len}",
            })
            continue

        X_data.append(row_features)
        y_labels.append(class_to_idx[label])
        valid_images += 1
        class_success[label] += 1
        metadata_rows.append({
            "label": label,
            "filename": filename,
            "path": img_path,
            "status": "success",
        })

        if total_images % 500 == 0:
            print(f"[{spec.name}] processed {total_images}/{len(samples)}, success {valid_images}")

    hands.close()

    X_data_np = np.asarray(X_data, dtype=np.float32)
    y_labels_np = np.asarray(y_labels, dtype=np.int64)

    if len(X_data_np) == 0:
        raise RuntimeError(f"[{spec.name}] No samples were successfully extracted.")

    np.save(os.path.join(spec.output_dir, "X_data.npy"), X_data_np)
    np.save(os.path.join(spec.output_dir, "y_labels.npy"), y_labels_np)

    metadata_path = os.path.join(spec.output_dir, "metadata.csv")
    failed_path = os.path.join(spec.output_dir, "failed_images.csv")

    with open(metadata_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "filename", "path", "status"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    with open(failed_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "filename", "path", "reason"])
        writer.writeheader()
        writer.writerows(failed_rows)

    summary = {
        "dataset": spec.name,
        "input_dir": spec.input_dir,
        "output_dir": spec.output_dir,
        "final_test_dir": spec.final_test_dir,
        "extract_final_test": bool(spec.extract_final_test),
        "max_num_hands": int(spec.max_num_hands),
        "feature_dim": int(expected_dim),
        "num_classes": int(len(labels)),
        "candidate_images_after_final_test": int(total_images),
        "mediapipe_success": int(valid_images),
        "unreadable": int(unreadable),
        "no_hand_detected": int(failed_detect),
        "invalid_feature_length": int(invalid_feature),
        "x_shape": list(X_data_np.shape),
        "y_shape": list(y_labels_np.shape),
        "metadata_csv": metadata_path,
        "failed_images_csv": failed_path,
    }

    summary_json_path = os.path.join(spec.output_dir, "dataset_accounting_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_csv_path = os.path.join(spec.output_dir, "dataset_accounting_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n================= Extraction finished =================")
    print(f"Dataset: {spec.name}")
    print(f"Total candidate images: {total_images}")
    print(f"MediaPipe success: {valid_images}")
    print(f"Unreadable: {unreadable}")
    print(f"No hand detected: {failed_detect}")
    print(f"Invalid feature length: {invalid_feature}")
    print(f"X_data shape: {X_data_np.shape}")
    print(f"y_labels shape: {y_labels_np.shape}")
    print(f"metadata.csv: {metadata_path}")
    print(f"failed_images.csv: {failed_path}")
    print(f"dataset_accounting_summary.csv: {summary_csv_path}")
    print("=======================================================\n")

    return {
        "summary": summary,
        "metadata_rows": metadata_rows,
        "class_progress": class_progress,
        "class_success": class_success,
        "class_failed": class_failed,
    }


# ============================================================
# 5. Accounting table
# ============================================================

def write_per_class_accounting(
    spec: DatasetExtractSpec,
    raw_before_counts: Dict[str, int],
    final_test_counts: Dict[str, int],
    extraction_info: Dict[str, object],
) -> str:
    metadata_rows = extraction_info["metadata_rows"]

    status_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in metadata_rows:
        status_counts[str(row["label"])][str(row["status"])] += 1

    labels = sorted(set(raw_before_counts) | set(final_test_counts) | set(status_counts))

    out_path = os.path.join(spec.output_dir, "per_class_accounting.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "dataset",
            "class_label",
            "raw_images_estimated",
            "final_test_images",
            "candidate_images_after_final_test",
            "mediapipe_success",
            "no_hand_detected",
            "unreadable",
            "invalid_feature_length",
            "retained_samples",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        totals = Counter()
        for label in labels:
            success = status_counts[label].get("success", 0)
            no_hand = status_counts[label].get("no_hand", 0)
            unreadable = status_counts[label].get("unreadable", 0)
            invalid = status_counts[label].get("invalid_feature_length", 0)
            candidate_after = success + no_hand + unreadable + invalid
            final_count = int(final_test_counts.get(label, 0))
            # If final_test was created before this run, raw_before_counts may not include moved images.
            # Therefore raw_images_estimated is candidate_after + current final_test images.
            raw_estimated = candidate_after + final_count

            row = {
                "dataset": spec.name,
                "class_label": label,
                "raw_images_estimated": int(raw_estimated),
                "final_test_images": int(final_count),
                "candidate_images_after_final_test": int(candidate_after),
                "mediapipe_success": int(success),
                "no_hand_detected": int(no_hand),
                "unreadable": int(unreadable),
                "invalid_feature_length": int(invalid),
                "retained_samples": int(success),
            }
            writer.writerow(row)
            for k, v in row.items():
                if isinstance(v, int):
                    totals[k] += v

        total_row = {
            "dataset": spec.name,
            "class_label": "TOTAL",
            "raw_images_estimated": int(totals["raw_images_estimated"]),
            "final_test_images": int(totals["final_test_images"]),
            "candidate_images_after_final_test": int(totals["candidate_images_after_final_test"]),
            "mediapipe_success": int(totals["mediapipe_success"]),
            "no_hand_detected": int(totals["no_hand_detected"]),
            "unreadable": int(totals["unreadable"]),
            "invalid_feature_length": int(totals["invalid_feature_length"]),
            "retained_samples": int(totals["retained_samples"]),
        }
        writer.writerow(total_row)

    print(f"[{spec.name}] Saved per-class accounting table: {out_path}")
    return out_path


# ============================================================
# 6. Main orchestration
# ============================================================

def prepare_one_dataset(spec: DatasetExtractSpec) -> Optional[Dict[str, object]]:
    if not spec.enabled:
        print(f"[{spec.name}] disabled; skipping.")
        return None

    if not os.path.exists(spec.input_dir):
        raise FileNotFoundError(f"[{spec.name}] input_dir does not exist: {spec.input_dir}")

    print("\n" + "#" * 90)
    print(f"Preparing dataset: {spec.name}")
    print("#" * 90)

    # Count current samples before any final_test move.
    # If final_test already exists from a previous run, this is not the original raw count;
    # the accounting table will estimate raw as current candidates + final_test images.
    raw_before_samples = collect_samples(spec)
    raw_before_counts = count_by_label(raw_before_samples)

    if spec.extract_final_test:
        extract_final_test_samples(spec)
    else:
        print(f"[{spec.name}] extract_final_test=False; no images will be moved.")

    final_test_counts = count_final_test_images(spec.final_test_dir)

    extraction_info = extract_mediapipe_landmarks(spec)
    per_class_path = write_per_class_accounting(
        spec=spec,
        raw_before_counts=raw_before_counts,
        final_test_counts=final_test_counts,
        extraction_info=extraction_info,
    )

    extraction_info["per_class_accounting_csv"] = per_class_path
    return extraction_info


def main() -> None:
    all_summaries: List[Dict[str, object]] = []

    for spec in DATASETS:
        info = prepare_one_dataset(spec)
        if info is None:
            continue
        all_summaries.append(info["summary"])

    # Save a combined summary near the script working directory.
    if all_summaries:
        out_path = "ALL_DATASETS_extraction_summary.csv"
        keys = sorted(set().union(*(row.keys() for row in all_summaries)))
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in all_summaries:
                writer.writerow(row)
        print(f"\nSaved combined extraction summary: {out_path}")


if __name__ == "__main__":
    main()
