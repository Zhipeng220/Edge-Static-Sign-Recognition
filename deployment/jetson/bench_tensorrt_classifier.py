import os
import time
import argparse
import numpy as np
import pandas as pd

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine file not found: {engine_path}")

    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

    return engine


def get_engine_io_info(engine):
    """
    TensorRT 8.x compatible binding API.
    """
    inputs = []
    outputs = []

    for binding_idx in range(engine.num_bindings):
        name = engine.get_binding_name(binding_idx)
        dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
        shape = tuple(engine.get_binding_shape(binding_idx))
        is_input = engine.binding_is_input(binding_idx)

        item = {
            "binding_idx": binding_idx,
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "is_input": is_input,
        }

        if is_input:
            inputs.append(item)
        else:
            outputs.append(item)

    return inputs, outputs


def allocate_buffers(engine, context, input_shape):
    """
    Allocate host/device buffers.

    input_shape should be fixed, e.g. (1, 21, 3).
    """
    inputs_info, outputs_info = get_engine_io_info(engine)

    if len(inputs_info) != 1:
        raise RuntimeError(f"Expected one input, got {len(inputs_info)} inputs: {inputs_info}")

    input_info = inputs_info[0]
    input_idx = input_info["binding_idx"]

    # If dynamic shape exists, set it.
    engine_shape = tuple(engine.get_binding_shape(input_idx))
    if any(dim < 0 for dim in engine_shape):
        context.set_binding_shape(input_idx, input_shape)

    bindings = [None] * engine.num_bindings

    host_inputs = []
    host_outputs = []
    device_inputs = []
    device_outputs = []

    # Allocate input
    input_dtype = trt.nptype(engine.get_binding_dtype(input_idx))
    input_size = int(np.prod(input_shape))

    h_input = cuda.pagelocked_empty(input_size, input_dtype)
    d_input = cuda.mem_alloc(h_input.nbytes)

    bindings[input_idx] = int(d_input)
    host_inputs.append(h_input)
    device_inputs.append(d_input)

    # Allocate outputs
    for output_info in outputs_info:
        output_idx = output_info["binding_idx"]
        output_dtype = trt.nptype(engine.get_binding_dtype(output_idx))

        output_shape = tuple(context.get_binding_shape(output_idx))
        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(f"Output shape is still dynamic: {output_shape}")

        output_size = int(np.prod(output_shape))

        h_output = cuda.pagelocked_empty(output_size, output_dtype)
        d_output = cuda.mem_alloc(h_output.nbytes)

        bindings[output_idx] = int(d_output)
        host_outputs.append(h_output)
        device_outputs.append(d_output)

    stream = cuda.Stream()

    return {
        "bindings": bindings,
        "stream": stream,
        "input_shape": input_shape,
        "input_dtype": input_dtype,
        "host_inputs": host_inputs,
        "host_outputs": host_outputs,
        "device_inputs": device_inputs,
        "device_outputs": device_outputs,
        "inputs_info": inputs_info,
        "outputs_info": outputs_info,
    }


def infer_once(context, buffers, input_array):
    """
    One inference pass.

    Returns:
        output numpy array
        host_end_to_end_ms
        gpu_inference_ms
    """
    stream = buffers["stream"]

    h_input = buffers["host_inputs"][0]
    d_input = buffers["device_inputs"][0]

    h_output = buffers["host_outputs"][0]
    d_output = buffers["device_outputs"][0]

    bindings = buffers["bindings"]

    input_array = np.asarray(input_array, dtype=buffers["input_dtype"]).ravel()
    np.copyto(h_input, input_array)

    start_event = cuda.Event()
    end_event = cuda.Event()

    host_start = time.perf_counter()

    cuda.memcpy_htod_async(d_input, h_input, stream)

    start_event.record(stream)
    context.execute_async_v2(
        bindings=bindings,
        stream_handle=stream.handle
    )
    end_event.record(stream)

    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    host_end = time.perf_counter()

    gpu_inference_ms = start_event.time_till(end_event)
    host_end_to_end_ms = (host_end - host_start) * 1000.0

    return h_output.copy(), host_end_to_end_ms, gpu_inference_ms


def load_samples(samples_path, input_shape):
    """
    Expected sample file:
        sample_landmarks_scaled.npy

    Shape can be:
        [N, 63]
        [N, 21, 3]
        [63]
        [1, 21, 3]
    """
    if samples_path is None:
        print("No sample file provided. Using random input.")
        return None

    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Sample file not found: {samples_path}")

    X = np.load(samples_path).astype(np.float32)

    if X.ndim == 1:
        X = X.reshape(1, *input_shape[1:])
    elif X.ndim == 2 and X.shape[1] == 63:
        X = X.reshape(-1, 21, 3)
    elif X.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported sample shape: {X.shape}")

    if X.shape[1:] != tuple(input_shape[1:]):
        raise ValueError(
            f"Sample shape mismatch. Expected [N, {input_shape[1]}, {input_shape[2]}], "
            f"got {X.shape}"
        )

    print(f"Loaded samples: {X.shape}")
    return X


def load_labels(labels_path):
    if labels_path is None:
        return None

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    y = np.load(labels_path)
    print(f"Loaded labels: {y.shape}")
    return y


def benchmark(
    engine_path,
    samples_path=None,
    labels_path=None,
    input_shape=(1, 21, 3),
    warmup=100,
    runs=1000,
    repeat=5,
):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    inputs_info, outputs_info = get_engine_io_info(engine)

    print("\n================ TensorRT Engine IO ================")
    for info in inputs_info + outputs_info:
        role = "INPUT" if info["is_input"] else "OUTPUT"
        print(
            f"{role} | idx={info['binding_idx']} | "
            f"name={info['name']} | shape={info['shape']} | dtype={info['dtype']}"
        )

    buffers = allocate_buffers(
        engine=engine,
        context=context,
        input_shape=input_shape,
    )

    X = load_samples(samples_path, input_shape)
    y = load_labels(labels_path)

    if X is None:
        X = np.random.randn(1, *input_shape[1:]).astype(np.float32)

    n_samples = len(X)

    print("\n================ Benchmark Config ================")
    print(f"Engine: {engine_path}")
    print(f"Input shape: {input_shape}")
    print(f"Warm-up: {warmup}")
    print(f"Runs per repeat: {runs}")
    print(f"Repeat: {repeat}")
    print(f"Sample count: {n_samples}")
    print(f"Engine size MB: {os.path.getsize(engine_path) / 1024 / 1024:.4f}")

    # Warm-up
    for i in range(warmup):
        sample = X[i % n_samples].reshape(input_shape)
        infer_once(context, buffers, sample)

    repeat_rows = []
    prediction_rows = []

    for r in range(repeat):
        host_latencies = []
        gpu_latencies = []

        correct = 0
        total_with_label = 0

        for i in range(runs):
            sample_idx = i % n_samples
            sample = X[sample_idx].reshape(input_shape)

            output, host_ms, gpu_ms = infer_once(context, buffers, sample)

            host_latencies.append(host_ms)
            gpu_latencies.append(gpu_ms)

            pred_id = int(np.argmax(output))

            if y is not None and sample_idx < len(y):
                true_id = int(y[sample_idx])
                total_with_label += 1
                if pred_id == true_id:
                    correct += 1

                if r == 0:
                    prediction_rows.append({
                        "sample_idx": sample_idx,
                        "true_id": true_id,
                        "pred_id": pred_id,
                    })

        host_latencies = np.asarray(host_latencies)
        gpu_latencies = np.asarray(gpu_latencies)

        accuracy = None
        if total_with_label > 0:
            accuracy = correct / total_with_label

        row = {
            "repeat": r,
            "engine": os.path.basename(engine_path),
            "warmup": warmup,
            "runs": runs,

            "host_latency_mean_ms": float(host_latencies.mean()),
            "host_latency_std_ms": float(host_latencies.std(ddof=1)),
            "host_latency_median_ms": float(np.median(host_latencies)),
            "host_latency_min_ms": float(host_latencies.min()),
            "host_latency_max_ms": float(host_latencies.max()),
            "host_fps_classifier_only": float(1000.0 / host_latencies.mean()),

            "gpu_latency_mean_ms": float(gpu_latencies.mean()),
            "gpu_latency_std_ms": float(gpu_latencies.std(ddof=1)),
            "gpu_latency_median_ms": float(np.median(gpu_latencies)),
            "gpu_latency_min_ms": float(gpu_latencies.min()),
            "gpu_latency_max_ms": float(gpu_latencies.max()),
            "gpu_fps_classifier_only": float(1000.0 / gpu_latencies.mean()),

            "accuracy": accuracy,
            "model_size_mb": float(os.path.getsize(engine_path) / 1024 / 1024),
        }

        repeat_rows.append(row)

        print(
            f"Repeat {r} | "
            f"Host latency: {row['host_latency_mean_ms']:.6f} ± {row['host_latency_std_ms']:.6f} ms | "
            f"GPU latency: {row['gpu_latency_mean_ms']:.6f} ± {row['gpu_latency_std_ms']:.6f} ms | "
            f"Host FPS: {row['host_fps_classifier_only']:.2f} | "
            f"GPU FPS: {row['gpu_fps_classifier_only']:.2f} | "
            f"Acc: {accuracy}"
        )

    raw_df = pd.DataFrame(repeat_rows)

    summary = {
        "engine": os.path.basename(engine_path),
        "engine_path": engine_path,
        "warmup": warmup,
        "runs_per_repeat": runs,
        "repeat": repeat,
        "batch_size": input_shape[0],
        "input_shape": str(input_shape),
        "model_size_mb": float(os.path.getsize(engine_path) / 1024 / 1024),

        "host_latency_mean_ms": raw_df["host_latency_mean_ms"].mean(),
        "host_latency_std_across_repeats_ms": raw_df["host_latency_mean_ms"].std(ddof=1),
        "host_fps_classifier_only_mean": raw_df["host_fps_classifier_only"].mean(),
        "host_fps_classifier_only_std": raw_df["host_fps_classifier_only"].std(ddof=1),

        "gpu_latency_mean_ms": raw_df["gpu_latency_mean_ms"].mean(),
        "gpu_latency_std_across_repeats_ms": raw_df["gpu_latency_mean_ms"].std(ddof=1),
        "gpu_fps_classifier_only_mean": raw_df["gpu_fps_classifier_only"].mean(),
        "gpu_fps_classifier_only_std": raw_df["gpu_fps_classifier_only"].std(ddof=1),
    }

    if raw_df["accuracy"].notna().any():
        summary["accuracy_mean"] = raw_df["accuracy"].mean()
        summary["accuracy_std"] = raw_df["accuracy"].std(ddof=1)

    summary_df = pd.DataFrame([summary])
    pred_df = pd.DataFrame(prediction_rows)

    return raw_df, summary_df, pred_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine",
        required=True,
        help="Path to TensorRT engine file."
    )

    parser.add_argument(
        "--samples",
        default=None,
        help="Optional .npy file. Recommended: sample_landmarks_scaled.npy"
    )

    parser.add_argument(
        "--labels",
        default=None,
        help="Optional .npy labels file for accuracy checking."
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size. Default: 1"
    )

    parser.add_argument(
        "--landmarks",
        type=int,
        default=21,
        help="Number of landmarks. Default: 21"
    )

    parser.add_argument(
        "--coord_dim",
        type=int,
        default=3,
        help="Coordinate dimension. Default: 3"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warm-up iterations. Default: 100"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1000,
        help="Timed runs per repeat. Default: 1000"
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of repeated benchmark rounds. Default: 5"
    )

    parser.add_argument(
        "--out_prefix",
        default="trt_classifier",
        help="Output prefix for CSV files."
    )

    args = parser.parse_args()

    input_shape = (args.batch, args.landmarks, args.coord_dim)

    raw_df, summary_df, pred_df = benchmark(
        engine_path=args.engine,
        samples_path=args.samples,
        labels_path=args.labels,
        input_shape=input_shape,
        warmup=args.warmup,
        runs=args.runs,
        repeat=args.repeat,
    )

    raw_path = f"{args.out_prefix}_raw.csv"
    summary_path = f"{args.out_prefix}_summary.csv"
    pred_path = f"{args.out_prefix}_predictions.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if len(pred_df) > 0:
        pred_df.to_csv(pred_path, index=False)

    print("\n================ Saved Results ================")
    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")

    if len(pred_df) > 0:
        print(f"Predictions: {pred_path}")

    print("\n================ Summary ================")
    print(summary_df.T)


if __name__ == "__main__":
    main()