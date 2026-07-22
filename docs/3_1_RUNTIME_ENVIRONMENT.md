# 3.1 Unified Runtime Environment

This document records the validated shared runtime for module 3.1.1 instance segmentation and the 3.1.2 Generic Single-hit BBox-Mask MVP.

## Validation Status

- Result: `single_shared_environment_validated_with_project_local_tensorrt`
- Validation scope: non-sealed smoke and static/runtime compatibility only
- Formal 3.1.1 metrics were not rerun or changed
- Formal 3.1.2 metrics were not rerun or changed
- The earlier temporary dual-environment recommendation is superseded by this validation

## Validated Core

- Python: 3.10.20
- PyTorch: 2.1.2+cu121
- torchvision: 0.16.2+cu121
- MMCV: 2.1.0
- MMEngine: 0.10.7
- MMDetection: 3.3.0
- Project-local TensorRT runtime: 10.13.3.9

The project requirement table previously listed TensorRT 8.6.1. TensorRT 8.6.1 was not recovered or validated in this pass. The validated project-local runtime is TensorRT 10.13.3.9, which is operational for the non-sealed smoke path but is not equivalent to TensorRT 8.6.1.

## Environment Files

- `environments/environment_3_1.yml`
- `environments/requirements_3_1.txt`
- `environments/requirements_3_1_test.txt`
- `environments/environment_3_1_lock.txt`

These files intentionally do not include host-specific prefixes, data paths, checkpoint binaries, ONNX files, TensorRT engines, SSH hosts, or tokens.

## Required Runtime Variables

```bash
export PROJECT_ROOT=/path/to/Fashion
export TRT_ROOT=/path/to/project-local/trt_runtime
export PYTHONPATH="$TRT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TRT_ROOT/tensorrt_libs:${LD_LIBRARY_PATH:-}"
```

The TensorRT Python package and shared libraries are project-local deployment assets and are not committed to GitHub.

## 3.1.1 Boundary

Historical formal 3.1.1 metrics remain:

- Recall: 84.21%
- Precision-like: 84.08%
- Mean TP IoU: 85.28%
- Average latency: 47.84 ms

The original 3.1.1 TensorRT engine may deserialize under TensorRT 10.13.3.9, but it can be execution-blocked when the deployment GPU does not match the engine build GPU. In the validation run, the old engine required 80 SM while the RTX 4080 had 76 SM, so execution was blocked.

## 3.1.2 Boundary

Formal 3.1.2 metrics remain:

- Positive BBox@0.5: 0.5173
- Positive BBox-Mask E2E: 0.2733
- Natural MVP Mixed: 0.5322
- Balanced v2 MVP Mixed: 0.4953

Latency and quality boundaries remain:

- Official localization latency target: 30 ms
- SAM-HQ stage average: about 112-123 ms
- SAM-HQ stage P95: about 220 ms
- `full_end_to_end_latency_measured=false`
- `latency_target_met=false`
- `production_quality_target_met=false`
- `mask_quality_limited=true`

The 3.1.2 frozen policy, thresholds, checkpoints, and formal metrics are unchanged by this environment patch.
