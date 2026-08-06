from pathlib import Path
from fashion313_runtime import Fashion313Runtime
root = Path(__file__).resolve().parents[1]
rt = Fashion313Runtime(root/'models/attribute_model.pth', root/'models/region_family_model.pth')
out = rt.predict(root/'examples/example.jpg', root/'examples/target_mask.png', root/'examples/parent_mask.png')
print(out['status'], out['region_family']['id'])
