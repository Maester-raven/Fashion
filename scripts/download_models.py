#!/usr/bin/env python3
import argparse, hashlib, os, sys, urllib.request
from pathlib import Path

MODELS = {
    "attribute": {
        "filename": "fashion313_attribute_model_v1.pth",
        "sha256": "2842eeea66c79cf03ae3b5958859dc150669d8e76914edf6089b64a011853920",
        "url": "",
    },
    "region_family": {
        "filename": "fashion313_region_family_model_v1.pth",
        "sha256": "06c3711e88721eaa135f1ece750c2911fb55a76b8e2b90b4d489bcefdec12bfb",
        "url": "",
    },
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify(path: Path, expected: str) -> bool:
    if not path.exists():
        print(f"missing: {path}", file=sys.stderr)
        return False
    actual = sha256_file(path)
    if actual != expected:
        print(f"sha256 mismatch for {path}: {actual} != {expected}", file=sys.stderr)
        return False
    print(f"verified: {path.name} {actual}")
    return True

def main(argv=None):
    p = argparse.ArgumentParser(description="Download or verify Fashion 3.1.3 model assets.")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args(argv)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for meta in MODELS.values():
        target = model_dir / meta["filename"]
        if target.exists():
            ok = verify(target, meta["sha256"]) and ok
            continue
        if args.verify_only:
            print(f"missing: {target}", file=sys.stderr)
            ok = False
            continue
        if not meta["url"]:
            print("No public model URL is configured yet. Place " + meta["filename"] + " manually or attach it as a GitHub Release asset in a later release step.", file=sys.stderr)
            ok = False
            continue
        tmp = target.with_suffix(target.suffix + ".partial")
        urllib.request.urlretrieve(meta["url"], tmp)
        if not verify(tmp, meta["sha256"]):
            tmp.unlink(missing_ok=True)
            ok = False
            continue
        tmp.replace(target)
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
