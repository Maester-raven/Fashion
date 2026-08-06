# Fashion 3.1.3 Model Assets

Model checkpoint binaries are intentionally not committed to Git. Place the files below in this directory before running real inference.

| File | SHA256 | Notes |
| --- | --- | --- |
| `fashion313_attribute_model_v1.pth` | `2842eeea66c79cf03ae3b5958859dc150669d8e76914edf6089b64a011853920` | Native Design attribute checkpoint |
| `fashion313_region_family_model_v1.pth` | `06c3711e88721eaa135f1ece750c2911fb55a76b8e2b90b4d489bcefdec12bfb` | Region-family routing checkpoint |

No public download URL is configured yet. Use `python scripts/download_models.py --model-dir models --verify-only` after manually placing the files.
