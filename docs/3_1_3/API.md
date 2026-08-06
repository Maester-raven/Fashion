# Fashion 3.1.3 API

```python
from fashion313_runtime import Fashion313Runtime
runtime = Fashion313Runtime("models/fashion313_attribute_model_v1.pth", "models/fashion313_region_family_model_v1.pth")
result = runtime.predict("image.jpg", "target_mask.png", parent_mask="parent_mask.png")
```

`image` may be a path, PIL image, or NumPy RGB array. Masks may be PNG, NumPy binary mask, polygon, or uncompressed COCO RLE.
