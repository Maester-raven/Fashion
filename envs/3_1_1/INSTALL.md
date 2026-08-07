# Fashion 3.1.1 environment

3.1.1 uses a dedicated OpenMMLab runtime with compiled MMCV ops. Do not use mmcv-lite.

Recommended install:

```bash
python scripts/setup_module_environment.py \
  --module 3.1.1 \
  --backend conda \
  --env-root $HOME/fashion_envs \
  --cache-root $HOME/fashion_cache \
  --tmp-root $HOME/fashion_tmp
```

The lock uses torch 2.1.2+cu121, torchvision 0.16.2+cu121, mmcv 2.1.0, mmengine 0.10.7, mmdet 3.3.0, and the torch runtime dependency fsspec 2026.4.0. The setup script installs fsspec from the 3.1.1 runtime requirements; users should not need an extra manual `pip install fsspec` step.
