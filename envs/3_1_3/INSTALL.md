# Fashion 3.1.3 environment

3.1.3 uses the frozen PyTorch CUDA runtime verified for the native design attribute prototype. It intentionally does not depend on the conda `pytorch-cuda=12.8` meta-package because that package is not consistently available from public conda channels.

```bash
python scripts/setup_module_environment.py \
  --module 3.1.3 \
  --backend conda \
  --env-root $HOME/fashion_envs \
  --cache-root $HOME/fashion_cache \
  --tmp-root $HOME/fashion_tmp
```

The runtime lock installs torch 2.11.0+cu128 and torchvision 0.26.0+cu128 from the PyTorch CUDA 12.8 wheel index.

Runtime note: OpenCV is pinned to 4.8.1.78 to preserve NumPy 1.26.4 compatibility in clean pip resolution.
