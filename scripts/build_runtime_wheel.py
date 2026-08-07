#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Fashion runtime wheel outside the source checkout.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tmp-root", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output_dir).expanduser().resolve()
    tmp_root = Path(args.tmp_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    build_src = tmp_root / "fashion_runtime_wheel_source"
    shutil.rmtree(build_src, ignore_errors=True)
    ignore = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.egg-info", "build", "dist", ".pytest_cache")
    shutil.copytree(root, build_src, ignore=ignore)
    env = os.environ.copy()
    env.setdefault("TMPDIR", str(tmp_root))
    subprocess.check_call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(out), str(build_src)], env=env)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
