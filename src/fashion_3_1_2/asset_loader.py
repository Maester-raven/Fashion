import hashlib
import json
import os
from pathlib import Path

def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)

def expand_path(value, config=None, model_root=None):
    value = str(value)
    for key, env in {
        "${FASHION_MODEL_ROOT}": os.environ.get("FASHION_MODEL_ROOT", ""),
        "${FASHION_PROJECT_ROOT}": os.environ.get("FASHION_PROJECT_ROOT", ""),
        "${SAM_HQ_REPO_ROOT}": os.environ.get("SAM_HQ_REPO_ROOT", ""),
    }.items():
        value = value.replace(key, env)
    p = Path(value)
    if p.is_absolute():
        return p
    root = model_root or os.environ.get("FASHION_MODEL_ROOT") or os.environ.get("FASHION_PROJECT_ROOT") or "."
    return Path(root) / p

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_file_hash(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"checkpoint_hash_mismatch:{path}:{actual}!={expected}")
    return actual
