#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODULES = {
    "3.1.1": Path("envs/3_1_1"),
    "3.1.2": Path("envs/3_1_2"),
    "3.1.3": Path("envs/3_1_3"),
}
ALIASES = {"311": "3.1.1", "312": "3.1.2", "313": "3.1.3"}
DEFAULT_ENV_NAMES = {"3.1.1": "fashion311", "3.1.2": "fashion312", "3.1.3": "fashion313"}


def run(cmd: list[str], *, env: dict[str, str], cwd: Path | None = None) -> None:
    print(json.dumps({"run": cmd, "cwd": str(cwd) if cwd else None}, ensure_ascii=False), flush=True)
    subprocess.check_call(cmd, env=env, cwd=str(cwd) if cwd else None)


def disk_report(path: Path) -> dict[str, int | str]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return {"path": str(path), "total": usage.total, "used": usage.used, "free": usage.free}


def copy_source_for_build(root: Path, build_root: Path) -> Path:
    src = build_root / "source"
    ignore = shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", "*.pyo", "*.egg-info", "build", "dist", ".pytest_cache"
    )
    shutil.copytree(root, src, ignore=ignore)
    return src


def build_runtime_wheel(root: Path, pip: Path, cache_root: Path, tmp_root: Path, env: dict[str, str]) -> Path:
    build_parent = cache_root / "wheel_build"
    wheel_dir = cache_root / "wheels"
    shutil.rmtree(build_parent, ignore_errors=True)
    build_parent.mkdir(parents=True, exist_ok=True)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    build_src = copy_source_for_build(root, build_parent)
    before = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    run([str(pip), "wheel", "--no-deps", "--wheel-dir", str(wheel_dir), str(build_src)], env=env)
    after = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    dirtied = sorted(after - before)
    if dirtied:
        raise RuntimeError(f"source tree dirtied while building wheel: {dirtied[:20]}")
    wheels = sorted(wheel_dir.glob("fashion313_runtime-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise RuntimeError(f"no runtime wheel produced in {wheel_dir}")
    return wheels[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a module-level Fashion 3.1 runtime environment.")
    parser.add_argument("--module", required=True, choices=["3.1.1", "3.1.2", "3.1.3", "311", "312", "313"])
    parser.add_argument("--backend", default="conda", choices=["conda", "venv"])
    parser.add_argument("--prefix", help="Exact environment prefix. Prefer --env-root for deployment docs.")
    parser.add_argument("--env-root", help="Directory where the module environment will be created.")
    parser.add_argument("--cache-root", default=os.environ.get("FASHION_CACHE_ROOT"))
    parser.add_argument("--tmp-root", default=os.environ.get("FASHION_TMP_ROOT"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-package-install", action="store_true", help="Only create dependency env and run verify script.")
    args = parser.parse_args(argv)

    module = ALIASES.get(args.module, args.module)
    root = Path(__file__).resolve().parents[1]
    env_dir = root / MODULES[module]
    if args.prefix:
        prefix = Path(args.prefix).expanduser().resolve()
    elif args.env_root:
        prefix = Path(args.env_root).expanduser().resolve() / DEFAULT_ENV_NAMES[module]
    else:
        parser.error("provide --prefix or --env-root")

    cache_root = Path(args.cache_root or (prefix.parent / "fashion_cache")).expanduser().resolve()
    tmp_root = Path(args.tmp_root or (prefix.parent / "fashion_tmp")).expanduser().resolve()
    pip_cache = cache_root / "pip"
    xdg_cache = cache_root / "xdg"
    torch_home = cache_root / "torch"
    conda_pkgs = cache_root / "conda_pkgs"
    for p in [prefix.parent, cache_root, tmp_root, pip_cache, xdg_cache, torch_home, conda_pkgs]:
        p.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "TMPDIR": str(tmp_root),
        "PIP_CACHE_DIR": str(pip_cache),
        "XDG_CACHE_HOME": str(xdg_cache),
        "TORCH_HOME": str(torch_home),
        "CONDA_PKGS_DIRS": str(conda_pkgs),
        "PYTHONNOUSERSITE": "1",
    })

    plan = {
        "module": module,
        "backend": args.backend,
        "prefix": str(prefix),
        "environment_dir": str(env_dir),
        "cache_root": str(cache_root),
        "tmp_root": str(tmp_root),
        "disk": {"env_parent": disk_report(prefix.parent), "cache_root": disk_report(cache_root), "tmp_root": disk_report(tmp_root)},
    }
    if prefix.exists():
        raise SystemExit(f"environment prefix already exists: {prefix}")

    if args.backend == "conda":
        create_cmd = ["conda", "env", "create", "--solver", "libmamba", "--prefix", str(prefix), "-f", str(env_dir / "environment-minimal.yml")]
    else:
        create_cmd = [sys.executable, "-m", "venv", str(prefix)]
    plan["create_command"] = create_cmd
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    run(create_cmd, env=env)
    pip = prefix / ("Scripts/pip.exe" if sys.platform == "win32" else "bin/pip")
    py = prefix / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    run([str(pip), "install", "--upgrade", "pip", "setuptools", "wheel"], env=env)
    req = env_dir / "requirements-runtime.txt"
    if req.exists():
        run([str(pip), "install", "-r", str(req)], env=env)
    if not args.skip_package_install:
        wheel = build_runtime_wheel(root, pip, cache_root, tmp_root, env)
        run([str(pip), "install", "--no-deps", str(wheel)], env=env)
    verify = env_dir / "verify_environment.py"
    if verify.exists():
        run([str(py), str(verify)], env=env, cwd=root)
    print(json.dumps({"created": True, "module": module, "prefix": str(prefix), "activate": f"conda activate {prefix}" if args.backend == "conda" else f"source {prefix}/bin/activate"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
