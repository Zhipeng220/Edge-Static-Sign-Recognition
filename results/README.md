# Results

Selected summary tables, manifests, and figure output folders are stored here. Large raw experiment outputs are intentionally not duplicated.

Current layout:

- `tables/`: combined `ALL_DATASETS_*.csv` summary tables copied from the experiment outputs.
- `manifests/`: repository and large-artifact file manifests.
- `figures/confusion_matrices/`: placeholder for exported confusion-matrix figures.
- `figures/latency/`: placeholder for Jetson or pipeline latency figures.
- `figures/tsne/`: placeholder for prediction t-SNE figures.
- `raw_csv/`: placeholder for optional per-run raw CSV exports.

Jetson-side deployment result folders may be generated in the deployment package rather than committed to this GitHub repository:

| Directory | Purpose |
|---|---|
| `parity_results/` | ONNX/TensorRT and FP32/FP16 agreement checks |
| `trtexec_final_results/` | Classifier-only TensorRT raw logs, `commands_used.txt`, and summary CSV files |
| `pipeline_final_results/` | Offline stored-image-to-label pipeline raw logs, `commands_used.txt`, and summary CSV files |
| `peak_ram_results/` | Peak-RAM telemetry logs and summary CSV files |
| `required_table_logs_20260530_170954/` | Packaged evidence used for deployment-related tables, when present |

Raw logs, summary CSV files, command records, and `.sha256` checksum files should be kept with the archived deployment evidence. Do not fabricate missing result filenames.
