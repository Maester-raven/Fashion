# 3.1.2 Dependency Delta

3.1.2 reuses the repository's core Python stack where possible: Python, PyTorch, torchvision, NumPy, Pillow, PyYAML/JSON-compatible config parsing, and pycocotools or the internal uncompressed RLE utilities.

Additional runtime requirement: a local SAM-HQ repository must be available and configured through `SAM_HQ_REPO_ROOT` or the config path. The package must import `segment_anything` from the SAM-HQ repo root, not from `seginw/segment_anything`.

Validated clone environment:

- torch = 2.11.0+cu128
- torchvision = 0.26.0+cu128
- CUDA runtime = 12.8
- GPU = NVIDIA GeForce RTX 4080

This is a validated environment, not the only supported environment. Do not upgrade Python, PyTorch, CUDA, MMDetection, MMCV, or MMEngine globally just for 3.1.2.
