# 3.1.2 zero/one/N functional v1
Default profile is `zero_one_n_functional_v1`; legacy `single_hit_v1` remains available. The chain is Presence G2, Route A cap500, live FashionCLIP/smoke_r1, fixed threshold/NMS/max10, and per-bbox SAM-HQ.

## Environment compatibility update

The release environment now pins the validated 3.1.1/3.1.2 shared stack, including CUDA 12.1 Torch, compiled MMCV, MMDetection and TensorRT 10.13.3.9. Large caches and the environment prefix should be placed on a data volume.
