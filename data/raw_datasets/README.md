# Raw Dataset Staging

Place downloaded raw image datasets here before running landmark extraction. Raw images are not committed to Git.

Expected layout:

```text
data/raw_datasets/
  asl_dataset/
    <class_name>/
      image.jpg
  indian_sign_language/
    <class_name>/
      image.jpg
  nus_hand_posture/
    Color/
      g1.jpg
      g1 (1).jpg
  asl_large_dataset/
    <class_name>/
      image.jpg
```

Run extraction from the repository root:

```bash
python src/extract_landmarks.py
```

The generated landmark arrays are written to:

```text
data/<dataset_name>/processed_landmarks/
```

Training reads those processed landmark folders through `configs/experiment_config.json`.
