import numpy as np

def encode_binary_mask(mask):
    mask = np.asarray(mask).astype(np.uint8)
    pixels = mask.T.flatten()
    counts = []
    last = 0
    run = 0
    for p in pixels:
        p = int(p)
        if p == last:
            run += 1
        else:
            counts.append(run)
            run = 1
            last = p
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": [int(c) for c in counts]}

def decode_rle(rle):
    h, w = [int(v) for v in rle["size"]]
    vals = []
    val = 0
    for c in rle["counts"]:
        vals.extend([val] * int(c))
        val = 1 - val
    arr = np.asarray(vals[: h * w], dtype=np.uint8)
    if arr.size < h * w:
        arr = np.pad(arr, (0, h * w - arr.size))
    return arr.reshape((w, h)).T.astype(bool)

def mask_area(mask_or_rle):
    if isinstance(mask_or_rle, dict):
        return int(decode_rle(mask_or_rle).sum())
    return int(np.asarray(mask_or_rle).astype(bool).sum())

def mask_tight_bbox(mask):
    mask = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

def validate_canvas(mask, height, width):
    return tuple(np.asarray(mask).shape[:2]) == (int(height), int(width))

def ensure_json_safe_rle(rle):
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": [int(c) for c in rle["counts"]]}
