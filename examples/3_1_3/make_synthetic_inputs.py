from pathlib import Path
import numpy as np
from PIL import Image
root = Path(__file__).resolve().parent
img = np.zeros((96, 96, 3), dtype=np.uint8)
img[..., 0] = 180; img[..., 1] = 160; img[..., 2] = 140
img[32:64, 32:64] = [220, 40, 80]
target = np.zeros((96, 96), dtype=np.uint8); target[36:58, 38:60] = 255
parent = np.zeros((96, 96), dtype=np.uint8); parent[20:76, 20:76] = 255
Image.fromarray(img).save(root/'example.jpg')
Image.fromarray(target).save(root/'target_mask.png')
Image.fromarray(parent).save(root/'parent_mask.png')
