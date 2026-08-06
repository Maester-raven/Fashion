# Fashion 3.1.3 Model Assets

Model weights are not committed to Git and are not yet hosted publicly.

Expected files under `models/`:

| Filename | SHA256 |
| --- | --- |
| `fashion313_attribute_model_v1.pth` | `2842eeea66c79cf03ae3b5958859dc150669d8e76914edf6089b64a011853920` |
| `fashion313_region_family_model_v1.pth` | `06c3711e88721eaa135f1ece750c2911fb55a76b8e2b90b4d489bcefdec12bfb` |

Manual placement:

```bash
mkdir -p models
# copy the two .pth files into models/
python scripts/verify_assets.py --attribute-checkpoint models/fashion313_attribute_model_v1.pth --region-family-checkpoint models/fashion313_region_family_model_v1.pth
```

Future GitHub Release contract: tag `v0.1.0-rc1`, release asset filenames exactly as listed above.

To use custom paths, pass them to `Fashion313Runtime(...)` or the CLI flags.
