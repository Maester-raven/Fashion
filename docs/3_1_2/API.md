# 3.1.2 API
```python
from fashion_3_1_2 import Fashion312Runtime
r=Fashion312Runtime.from_config('configs/3_1_2/zero_one_n_functional_v1.yaml')
x=r.predict(image_path='example.jpg',parent_bbox=[0,0,512,512],query_text='find all sleeves')
```
`parent_crop_path` is also supported.
