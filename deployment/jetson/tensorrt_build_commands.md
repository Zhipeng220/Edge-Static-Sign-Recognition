# TensorRT Build Commands

These are low-level command examples. The recommended repository-level wrapper is:

```bash
bash sh/build_all_trt_engines.sh
```

```bash
trtexec --onnx=models/full_model_single_hand_classifier.onnx \
        --saveEngine=models/full_model_seed42_fp32.engine \
        --explicitBatch

trtexec --onnx=models/full_model_single_hand_classifier.onnx \
        --saveEngine=models/full_model_seed42_fp16.engine \
        --explicitBatch \
        --fp16
```
