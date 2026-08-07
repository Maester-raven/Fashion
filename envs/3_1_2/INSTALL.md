# Fashion 3.1.2 environment

3.1.2 uses its own multimodal/local-grounding runtime. Install it as a module-level environment; do not merge it with 3.1.1 OpenMMLab unless explicitly validating a custom deployment.

```bash
python scripts/setup_module_environment.py \
  --module 3.1.2 \
  --backend conda \
  --env-root $HOME/fashion_envs \
  --cache-root $HOME/fashion_cache \
  --tmp-root $HOME/fashion_tmp
```

PyTorch is installed by conda using pytorch=2.1.2, torchvision=0.16.2, and pytorch-cuda=12.1; pip installs only the remaining runtime packages.
