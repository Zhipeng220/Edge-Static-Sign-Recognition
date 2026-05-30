# Jetson Xavier NX TensorRT Deployment

The Jetson-side workflow is command-oriented. Users run shell scripts from the Jetson deployment package root:

```text
~/paper_hand_deploy
```

Auxiliary Python tools are invoked automatically by the shell wrappers when needed. They are not required manual entry points for normal reproduction.

## Shell Commands

| Command | Purpose | Main outputs |
|---|---|---|
| `bash build_all_trt_engines.sh` | Build TensorRT engines on the target Jetson device | `engines/` |
| `bash build_trt_worker.sh` | Build the persistent TensorRT inference worker | worker executable |
| `bash run_parity_test.sh` | Run ONNX / TensorRT parity checks | `parity_results/` |
| `bash run_small_asl_trt_parity.sh` | Run the retained-sample TensorRT parity test | `parity_results/` |
| `bash run_small_asl_full_parity.sh` | Run ONNX, TensorRT FP32, and TensorRT FP16 consistency checks | `parity_results/` |
| `bash run_trtexec_5rounds.sh` | Run five-round classifier-only TensorRT benchmarks | `trtexec_final_results/` |
| `bash run_peak_ram_trtexec.sh` | Measure peak RAM and collect telemetry | `peak_ram_results/` |
| `bash run_pipeline_5rounds.sh` | Run the offline stored-image-to-label pipeline benchmark | pipeline result directory |
| `bash run_formal_pipeline_5rounds.sh` | Run the formal five-round FP16 and FP32 pipeline benchmarks | `pipeline_final_results/` |
| `bash pack_required_table_logs.sh` | Package the logs used for paper tables | timestamped archive |
| `bash pack_scanned_paper_data.sh` | Package reproducibility evidence and checksums | `paper_hand_packages/` |

## Benchmark Scope

Classifier-only TensorRT benchmark:

- entry point: `bash run_trtexec_5rounds.sh`
- scope: TensorRT engine execution only
- excludes: image loading, colour conversion, MediaPipe, camera capture, display rendering, speech synthesis, STM32 communication, and actuator response

Offline stored-image-to-label pipeline benchmark:

- entry point: `bash run_formal_pipeline_5rounds.sh`
- scope: stored image loading, colour conversion, MediaPipe landmark extraction, normalization/scaler, persistent TensorRT worker inference, and prediction decoding
- formal setup: 1789 held-out images per round, 26 letter classes A-Z, FP16 five rounds, FP32 five rounds
- excludes: camera capture, display rendering, speech synthesis, STM32 communication, and actuator response

The offline pipeline benchmark is not a live-camera FPS benchmark.

## Verified Environment

| Item | Verified setting |
|---|---|
| Device | NVIDIA Jetson Xavier NX Developer Kit |
| L4T | R35.4.1 |
| Kernel | Linux 5.10.120-tegra, aarch64 |
| TensorRT | 8.5.2.2 |
| MediaPipe | 0.10.9 |
| OpenCV | 4.8.1 |
| Power mode | `MODE_10W_DESKTOP`, mode 5 |
| Online CPU cores | CPU 0-5 |
| CPU frequency | 1.9072 GHz |
| GPU frequency | 510 MHz |
| EMC frequency | 1600 MHz |
| Telemetry tool | NVIDIA `tegrastats` |
| Telemetry interval | 100 ms |
| Idle system RAM mean | 1294.03 MB |

The exact warm-up value used for the formal benchmark remains to be verified from the archived formal commands. The earlier 30-image smoke-test setting is not reported as a formal benchmark parameter unless it matches the archived commands.
