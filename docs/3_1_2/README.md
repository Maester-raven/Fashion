# 3.1.2 Generic Single-hit BBox-Mask MVP v1

This package exposes a frozen 3.1.2 MVP interface:

`generic query + parent garment crop -> zero or one bbox + COCO uncompressed RLE mask`.

Official sealed metrics:

- Positive BBox@0.5 = 0.5173
- Positive BBox-Mask E2E = 0.2733
- Natural MVP Mixed = 0.5322
- Balanced v2 MVP Mixed = 0.4953

It does not return all instances, does not support constrained-single, does not support spatial/relation queries, is weak on micro/tiny parts, has limited mask quality, has P95 latency around 220 ms, and does not meet production quality.
Confidence is not fully calibrated.

Path resolution order: explicit CLI args / `--model-root`, then `FASHION_MODEL_ROOT`, `FASHION_PROJECT_ROOT`, and `SAM_HQ_REPO_ROOT`, then config-relative paths.
