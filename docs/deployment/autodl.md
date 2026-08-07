# AutoDL deployment notes for Fashion 3.1

AutoDL containers often have a small `/root` overlay and a much larger data disk at `/root/autodl-tmp`. Build environments, pip caches, torch caches, and temporary files on the data disk.

Example:

```bash
export ENV_ROOT=/root/autodl-tmp/fashion_envs
export CACHE_ROOT=/root/autodl-tmp/fashion_cache
export TMP_ROOT=/root/autodl-tmp/fashion_tmp
mkdir -p "$ENV_ROOT" "$CACHE_ROOT" "$TMP_ROOT"

python scripts/setup_module_environment.py --module 3.1.1 --backend conda --env-root "$ENV_ROOT" --cache-root "$CACHE_ROOT" --tmp-root "$TMP_ROOT"
python scripts/setup_module_environment.py --module 3.1.2 --backend conda --env-root "$ENV_ROOT" --cache-root "$CACHE_ROOT" --tmp-root "$TMP_ROOT"
python scripts/setup_module_environment.py --module 3.1.3 --backend conda --env-root "$ENV_ROOT" --cache-root "$CACHE_ROOT" --tmp-root "$TMP_ROOT"
```

Do not place large venvs, conda package caches, PyTorch wheels, or temporary build directories on `/root` when the overlay is small.

The setup script builds the runtime wheel outside the source checkout and installs that wheel with `--no-deps`, so a clean clone should remain free of `*.egg-info`, `build/`, and `dist/` after deployment installation.
