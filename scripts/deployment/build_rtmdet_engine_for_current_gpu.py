#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def gpu_report():
    rep = {}
    try:
        import torch
        rep["torch_version"] = getattr(torch, "__version__", "unknown")
        rep["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            rep["gpu_name"] = torch.cuda.get_device_name(0)
            rep["gpu_sm_count"] = torch.cuda.get_device_properties(0).multi_processor_count
            rep["gpu_compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as exc:
        rep["torch_probe_error"] = repr(exc)
    try:
        import tensorrt as trt
        rep["tensorrt_version"] = getattr(trt, "__version__", "unknown")
        rep["tensorrt_module"] = str(Path(trt.__file__).resolve())
    except Exception as exc:
        rep["tensorrt_probe_error"] = repr(exc)
    return rep


def main():
    ap = argparse.ArgumentParser(description="Build a GPU-specific RTMDet TensorRT engine from an existing ONNX file.")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", help="Explicit output engine path. Required unless --output-dir is used.")
    ap.add_argument("--output-dir", help="Directory for a generated GPU/TRT-specific engine name.")
    ap.add_argument("--trt-root", help="Project-local TensorRT runtime root to prepend to PYTHONPATH.")
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--input-shape", default="1x3x1024x1024")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest", help="Where to write build manifest JSON.")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    onnx = Path(args.onnx)
    gpu = gpu_report()
    trt_ver = safe_name(str(gpu.get("tensorrt_version", "unknown_trt")))
    gpu_name = safe_name(str(gpu.get("gpu_name", "unknown_gpu")))
    sm = gpu.get("gpu_sm_count", "unknown_sm")
    engine = Path(args.engine) if args.engine else Path(args.output_dir or "models/tensorrt") / f"rtmdet_shared_dual_cls_1024_fp16_{trt_ver}_{gpu_name}_{sm}sm.engine"
    manifest = {"onnx": str(onnx), "onnx_exists": onnx.exists(), "engine": str(engine), "engine_exists_before": engine.exists(), "workspace_gb": args.workspace_gb, "fp16": args.fp16, "input_shape": args.input_shape, "dry_run": args.dry_run, **gpu}
    if onnx.exists():
        manifest["onnx_sha256"] = sha256_file(onnx)
    if engine.exists() and not args.overwrite:
        manifest["blocked"] = True
        manifest["block_reason"] = "engine_exists_and_overwrite_false"
    else:
        manifest["blocked"] = False
    cmd = [sys.executable, str(repo / "scripts/export/build_rtmdet_tensorrt_engine.py"), "--onnx", str(onnx), "--engine", str(engine), "--workspace-gb", str(args.workspace_gb)]
    if args.fp16:
        cmd.append("--fp16")
    manifest["command"] = cmd
    if not args.dry_run and not manifest["blocked"]:
        env = os.environ.copy()
        if args.trt_root:
            env["PYTHONPATH"] = f"{args.trt_root}:{repo / 'src'}:{repo}:{env.get('PYTHONPATH','')}"
            env["LD_LIBRARY_PATH"] = f"{Path(args.trt_root) / 'tensorrt_libs'}:{env.get('LD_LIBRARY_PATH','')}"
        proc = subprocess.run(cmd, cwd=str(repo), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        manifest["returncode"] = proc.returncode
        manifest["build_output_tail"] = proc.stdout[-8000:]
        if engine.exists():
            manifest["engine_sha256"] = sha256_file(engine)
            manifest["engine_size"] = engine.stat().st_size
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 2 if manifest.get("blocked") else int(manifest.get("returncode", 0))

if __name__ == "__main__":
    raise SystemExit(main())
