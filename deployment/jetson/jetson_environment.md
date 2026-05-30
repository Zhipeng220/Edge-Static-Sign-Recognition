# Jetson Environment

Fill this file with the exact hardware and software environment used for TensorRT and pipeline measurements.

## Hardware

```text
Jetson model:
Module RAM:
Storage:
Power supply:
Cooling mode:
```

## NVIDIA Software Stack

```text
JetPack version:
L4T version:
CUDA version:
cuDNN version:
TensorRT version:
ONNX opset:
```

## Python Environment

```text
Python version:
PyTorch version:
NumPy version:
OpenCV version:
MediaPipe version:
```

## Benchmark Settings

```text
Power mode:
jetson_clocks enabled:
Batch size:
Input shape:
Precision modes:
Warm-up:
Timed runs:
tegrastats interval:
```

## Reporting Notes

- Report classifier-only TensorRT results separately from end-to-end image-to-label pipeline results.
- Record whether results came from `sh/run_trtexec_5rounds.sh`, `sh/run_peak_ram_trtexec.sh`, or `sh/run_pipeline_5rounds.sh`.
- Include raw logs and command records when packaging reproducibility evidence.
