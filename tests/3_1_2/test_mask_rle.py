import numpy as np
from fashion_3_1_2.mask_utils import encode_binary_mask, decode_rle, mask_area, mask_tight_bbox
m = np.zeros((8, 9), dtype=bool); m[2:6, 3:7] = True
r = encode_binary_mask(m)
assert np.array_equal(m, decode_rle(r))
assert mask_area(r) == 16
assert mask_tight_bbox(m) == [3.0, 2.0, 7.0, 6.0]
