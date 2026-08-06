import math
import numpy as np
import torch
import torch.nn.functional as F
from .constants import DINO_MEAN, DINO_STD, DINO_MEAN255, INPUT_SIZE
from .mask_utils import mask_bbox

DINO_MEAN_NP = np.asarray(DINO_MEAN, dtype=np.float32)
DINO_STD_NP = np.asarray(DINO_STD, dtype=np.float32)
DINO_MEAN255_NP = np.asarray(DINO_MEAN255, dtype=np.uint8)

def _floorceil_box(b, w, h):
    x1, y1, x2, y2 = b
    x1 = max(0, min(w - 1, math.floor(x1)))
    y1 = max(0, min(h - 1, math.floor(y1)))
    x2 = max(x1 + 1, min(w, math.ceil(x2)))
    y2 = max(y1 + 1, min(h, math.ceil(y2)))
    return [int(x1), int(y1), int(x2), int(y2)]

def _resize_rgb(rgb):
    # G1 uses CUDA bicubic interpolation rather than PIL. This helper is retained for API compatibility.
    x = torch.from_numpy(np.asarray(rgb)).to("cuda", non_blocking=False)
    x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bicubic", align_corners=False, antialias=True)
    x = (x.clamp(0, 1)[0].permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
    return x

def _norm_tensor(rgb):
    mean = torch.tensor(DINO_MEAN, device="cuda", dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(DINO_STD, device="cuda", dtype=torch.float32).view(3, 1, 1)
    x = torch.from_numpy(np.asarray(rgb)).to("cuda", non_blocking=False)
    x = x.permute(2, 0, 1).float() / 255.0
    return (x - mean) / std

def make_tensors(image, target_mask, parent_mask=None):
    h, w = image.shape[:2]
    tx1, ty1, tx2, ty2 = mask_bbox(target_mask)
    if parent_mask is not None:
        px1, py1, px2, py2 = mask_bbox(parent_mask)
        pad = 0.15 * max(tx2 - tx1, ty2 - ty1)
        lbox = _floorceil_box([max(px1, tx1 - pad), max(py1, ty1 - pad), min(px2, tx2 + pad), min(py2, ty2 + pad)], w, h)
        ppad = 0.075 * max(px2 - px1, py2 - py1)
        pbox = _floorceil_box([px1 - ppad, py1 - ppad, px2 + ppad, py2 + ppad], w, h)
    else:
        pad = 0.075 * max(tx2 - tx1, ty2 - ty1)
        lbox = _floorceil_box([tx1 - pad, ty1 - pad, tx2 + pad, ty2 + pad], w, h)
        pbox = [0, 0, w, h]

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t2 = torch.cuda.Event(enable_timing=True)
    t3 = torch.cuda.Event(enable_timing=True)
    t0.record()
    img = torch.from_numpy(np.asarray(image)).to("cuda", non_blocking=False)
    mask = torch.from_numpy(np.asarray(target_mask)).to("cuda", non_blocking=False)
    t1.record()
    local = img[lbox[1]:lbox[3], lbox[0]:lbox[2], :].clone()
    lm = mask[lbox[1]:lbox[3], lbox[0]:lbox[2]]
    local[~lm] = torch.tensor(DINO_MEAN255_NP, device="cuda", dtype=torch.uint8)
    context = img[pbox[1]:pbox[3], pbox[0]:pbox[2], :].clone()
    t2.record()
    mean = torch.tensor(DINO_MEAN, device="cuda", dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(DINO_STD, device="cuda", dtype=torch.float32).view(1, 3, 1, 1)
    def prep(x):
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bicubic", align_corners=False, antialias=True)
        return ((x - mean) / std)[0]
    local_t = prep(local)
    context_t = prep(context)
    t3.record()
    torch.cuda.synchronize()
    meta = {
        "target_bbox_xyxy": [tx1, ty1, tx2, ty2],
        "parent_mask_provided": parent_mask is not None,
        "target_area": int(target_mask.sum()),
        "image_width": int(w),
        "image_height": int(h),
        "g1_timing_ms": {
            "h2d": float(t0.elapsed_time(t1)),
            "gpu_crop": float(t1.elapsed_time(t2)),
            "gpu_resize_normalize": float(t2.elapsed_time(t3)),
            "gpu_preprocessing_total": float(t0.elapsed_time(t3)),
        },
    }
    return local_t, context_t, meta

def geometry_row(meta):
    x1, y1, x2, y2 = meta["target_bbox_xyxy"]
    return {"image_width": meta["image_width"], "image_height": meta["image_height"], "bbox_xyxy": [x1, y1, x2, y2], "area": meta["target_area"]}
