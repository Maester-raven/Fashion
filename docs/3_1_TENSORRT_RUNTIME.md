# 3.1 TensorRT Runtime and GPU-Specific Engine Policy

## Requirement Versus Validated Runtime

- Specified project requirement: TensorRT 8.6.1
- Validated runtime in this compatibility pass: TensorRT 10.13.3.9
- TensorRT 8.6.1 was not recovered or validated here
- TensorRT 10.13.3.9 is operational for the validated non-sealed smoke path, but it is not equivalent to 8.6.1

## Project-Local TensorRT

TensorRT must be resolved from the project-local runtime before any environment-level TensorRT package:

```bash
export PROJECT_ROOT=/path/to/Fashion
export TRT_ROOT=/path/to/project-local/trt_runtime
export PYTHONPATH="$TRT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TRT_ROOT/tensorrt_libs:${LD_LIBRARY_PATH:-}"
```

The project-local TensorRT package, shared libraries, ONNX files, and engine files are deployment assets. They are not committed to GitHub.

## Engine Compatibility

TensorRT engines are GPU-specific deployment artifacts. An engine that deserializes is not automatically safe to execute. The old 3.1.1 engine may deserialize under TensorRT 10.13.3.9 but was execution-blocked on a 76-SM RTX 4080 because the engine expected an 80-SM GPU.

Use the checker before execution:

```bash
python scripts/deployment/check_tensorrt_engine_compatibility.py \
  --engine models/tensorrt/shared_dual_cls_1024_fp16.engine \
  --trt-root "$TRT_ROOT" \
  --engine-built-sm-count 80 \
  --json-output outputs/tensorrt_engine_compatibility.json
```

By default the checker only deserializes. It does not create an execution context and does not enqueue inference unless explicitly requested. When an engine build manifest or historical record provides the engine build SM count, pass `--engine-built-sm-count`; the checker will block execution if that SM count differs from the current GPU.

## Building for the Current GPU

Build a fresh engine for each deployment GPU from the validated ONNX file:

```bash
python scripts/deployment/build_rtmdet_engine_for_current_gpu.py \
  --onnx models/onnx/shared_dual_cls_1024.onnx \
  --output-dir models/tensorrt \
  --trt-root "$TRT_ROOT" \
  --workspace-gb 8 \
  --fp16 \
  --manifest outputs/rtmdet_engine_build_manifest.json
```

The builder refuses to overwrite an existing engine unless `--overwrite` is provided. Engine names should include TensorRT, GPU, and SM-count context. Do not commit generated engines.
