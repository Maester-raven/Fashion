"""Single-image TensorRT inference wrapper for PRD 3.1.1 instance segmentation.

This module intentionally keeps the actual TensorRT/MMDetection inference logic
inside ``scripts/eval/run_rtmdet_tensorrt_pipeline.py``.  It only adapts a
single image into the COCO-style input expected by that pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


CLASSES = (
    "top",
    "pants",
    "skirt",
    "outerwear",
    "dress",
    "shoes",
    "bag",
    "accessory",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-image RTMDet-Ins TensorRT inference by wrapping the "
            "existing COCO-style TensorRT pipeline."
        )
    )
    parser.add_argument("--config", required=True, help="Path to RTMDet config.")
    parser.add_argument(
        "--engine",
        required=True,
        help=(
            "Path to TensorRT engine. If missing, build it first following "
            "README.md."
        ),
    )
    parser.add_argument("--image", required=True, help="Path to one input image.")
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to final single-image JSON output.",
    )
    parser.add_argument("--score-thr", type=float, default=0.15)
    parser.add_argument("--nms-pre", type=int, default=300)
    parser.add_argument("--max-per-img", type=int, default=80)
    parser.add_argument("--disable-d1-fusion", action="store_true")
    parser.add_argument(
        "--benchmark-json",
        help=(
            "Optional benchmark JSON path. Defaults to a sidecar file next to "
            "--output-json."
        ),
    )
    parser.add_argument(
        "--raw-output-json",
        help=(
            "Optional raw COCO prediction JSON path. Defaults to a sidecar file "
            "next to --output-json."
        ),
    )
    return parser.parse_args()


def resolve_existing_file(path_text: str, root: Path, label: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        if label == "engine":
            raise FileNotFoundError(
                f"TensorRT engine not found: {path}. "
                "Please build models/tensorrt/shared_dual_cls_1024_fp16.engine "
                "following README.md before running inference."
            )
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def make_single_image_coco(image_path: Path) -> dict:
    return {
        "images": [
            {
                "id": 1,
                "file_name": str(image_path),
            }
        ],
        "annotations": [],
        "categories": [
            {
                "id": index + 1,
                "name": name,
            }
            for index, name in enumerate(CLASSES)
        ],
    }


def prepend_pythonpath(env: dict[str, str], paths: list[Path]) -> dict[str, str]:
    current = env.get("PYTHONPATH", "")
    prefix = os.pathsep.join(str(path) for path in paths)
    env["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return env


def main() -> int:
    args = parse_args()
    root = repo_root()

    config_path = resolve_existing_file(args.config, root, "config")
    engine_path = resolve_existing_file(args.engine, root, "engine")
    image_path = resolve_existing_file(args.image, root, "image")

    output_path = Path(args.output_json).expanduser()
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_output_path = Path(
        args.raw_output_json
        or output_path.with_name(output_path.stem + ".raw_predictions.json")
    ).expanduser()
    if not raw_output_path.is_absolute():
        raw_output_path = root / raw_output_path
    raw_output_path = raw_output_path.resolve()
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_path = Path(
        args.benchmark_json
        or output_path.with_name(output_path.stem + ".benchmark.json")
    ).expanduser()
    if not benchmark_path.is_absolute():
        benchmark_path = root / benchmark_path
    benchmark_path = benchmark_path.resolve()
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_path = root / "scripts" / "eval" / "run_rtmdet_tensorrt_pipeline.py"
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"TensorRT pipeline script not found: {pipeline_path}")

    with tempfile.TemporaryDirectory(prefix="fashion_single_image_") as tmp_dir:
        val_json_path = Path(tmp_dir) / "single_image_coco.json"
        val_json_path.write_text(
            json.dumps(make_single_image_coco(image_path), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        command = [
            sys.executable,
            str(pipeline_path),
            "--config",
            str(config_path),
            "--engine",
            str(engine_path),
            "--val-json",
            str(val_json_path),
            "--data-root",
            "/",
            "--output-json",
            str(raw_output_path),
            "--benchmark-json",
            str(benchmark_path),
            "--warmup",
            "0",
            "--benchmark-images",
            "1",
            "--max-images",
            "1",
            "--score-thr",
            str(args.score_thr),
            "--nms-pre",
            str(args.nms_pre),
            "--max-per-img",
            str(args.max_per_img),
        ]
        if args.disable_d1_fusion:
            command.append("--disable-d1-fusion")

        env = prepend_pythonpath(
            os.environ.copy(),
            [root / "src", root, root / "scripts" / "eval"],
        )
        result = subprocess.run(command, cwd=str(root), env=env, check=False)
        if result.returncode != 0:
            return result.returncode

    try:
        instances = json.loads(raw_output_path.read_text(encoding="utf-8"))
        parse_error = None
    except Exception as exc:  # pragma: no cover - defensive output path
        instances = []
        parse_error = f"Failed to parse raw prediction JSON: {exc}"

    payload = {
        "image": args.image,
        "raw_prediction_json": str(raw_output_path),
        "benchmark_json": str(benchmark_path),
        "instances": instances if isinstance(instances, list) else [],
    }
    if parse_error:
        payload["error"] = parse_error

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if parse_error:
        print(parse_error, file=sys.stderr)
        return 1

    print(f"Wrote single-image prediction JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
