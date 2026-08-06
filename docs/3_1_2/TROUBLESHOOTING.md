# Troubleshooting

- Put `CONDA_PKGS_DIRS`, `TMPDIR`, the pip cache and environment prefix on a data volume with at least 20 GB free.
- Do not replace the pinned Torch 2.1.2+cu121 or compiled MMCV wheel with a CPU or incompatible ABI build.
- TensorRT must report 10.13.3.9 and its Python module and shared libraries must originate inside the fresh environment.
- Run all three verifiers before inference. Confirm model assets are not Git LFS pointer files.
- Positive 3.1.2 calls are seconds-scale and do not meet the 30 ms PRD target.
