#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Deserialize a TensorRT engine and report GPU/runtime compatibility without executing inference by default.")
    ap.add_argument("--engine", required=True, help="Path to a TensorRT engine file.")
    ap.add_argument("--trt-root", help="Project-local TensorRT Python package root. LD_LIBRARY_PATH must include its tensorrt_libs before process start.")
    ap.add_argument("--json-output")
    ap.add_argument("--create-context", action="store_true", help="Optionally create an execution context. Does not enqueue inference.")
    ap.add_argument("--engine-built-sm-count", type=int, help="Optional SM count recorded when this engine was built. If provided, it is compared with the current GPU SM count.")
    args = ap.parse_args()
    if args.trt_root:
        sys.path.insert(0, args.trt_root)
    report = {"engine_path": args.engine, "engine_exists": Path(args.engine).exists(), "deserialize_attempted": False, "deserialize_passed": False, "context_created": False, "execution_compatibility_verified": False, "compatible_for_execution": False, "execution_blocked": True, "warnings": [], "errors": []}
    try:
        import torch
        report.update({"torch_version": getattr(torch, "__version__", "unknown"), "cuda_available": bool(torch.cuda.is_available())})
        if torch.cuda.is_available():
            report.update({"gpu_name": torch.cuda.get_device_name(0), "gpu_sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "gpu_compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0)))})
    except Exception as exc:
        report["warnings"].append(f"torch_probe_failed:{exc!r}")
    if report["engine_exists"]:
        report["engine_sha256"] = sha256_file(args.engine)
    try:
        import tensorrt as trt
        report.update({"tensorrt_version": getattr(trt, "__version__", "unknown"), "tensorrt_module": str(Path(trt.__file__).resolve())})
        logger = trt.Logger(trt.Logger.INFO)
        stdout = io.StringIO(); stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            runtime = trt.Runtime(logger)
            report["deserialize_attempted"] = True
            with open(args.engine, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                report["errors"].append("deserialize_returned_none")
            else:
                report["deserialize_passed"] = True
                try:
                    report["num_io_tensors"] = int(engine.num_io_tensors)
                    report["io_tensors"] = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
                except Exception:
                    pass
                if args.create_context:
                    ctx = engine.create_execution_context()
                    report["context_created"] = ctx is not None
        captured = "\n".join([stdout.getvalue(), stderr.getvalue()]).strip()
        if captured:
            report["warnings"].extend([line for line in captured.splitlines() if line.strip()])
    except Exception as exc:
        report["errors"].append(repr(exc))
    if args.engine_built_sm_count is not None:
        report["engine_built_sm_count"] = int(args.engine_built_sm_count)
        current_sm = report.get("gpu_sm_count")
        if current_sm is not None and int(current_sm) != int(args.engine_built_sm_count):
            report["warnings"].append(f"engine_built_sm_count_mismatch:current={current_sm}:engine={args.engine_built_sm_count}")
            report["engine_required_sm_count"] = int(args.engine_built_sm_count)
    warning_text = "\n".join(report.get("warnings", []) + report.get("errors", []))
    patterns = [r"different device model", r"requires\s+(\d+)\s+multiprocessors", r"deadlocking is likely", r"SM\s*count", r"compute capability", r"engine_built_sm_count_mismatch"]
    report["gpu_mismatch_detected"] = any(re.search(p, warning_text, flags=re.I) for p in patterns)
    m = re.search(r"requires\s+(\d+)\s+multiprocessors", warning_text, flags=re.I)
    if m:
        report["engine_required_sm_count"] = int(m.group(1))
    report["execution_compatibility_verified"] = bool(args.create_context and report.get("context_created"))
    if not report["execution_compatibility_verified"]:
        report["compatible_for_execution"] = False
        report["execution_blocked"] = True
        report["block_reason"] = "execution_context_not_created"
    else:
        report["compatible_for_execution"] = bool(not report["gpu_mismatch_detected"] and not report["errors"])
        report["execution_blocked"] = not report["compatible_for_execution"]
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_output:
        Path(args.json_output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 2 if report["execution_blocked"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
