import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .exceptions import InputValidationError

def load_image(image):
    if isinstance(image, (str, Path)):
        arr = np.asarray(Image.open(image).convert('RGB'))
    elif isinstance(image, Image.Image):
        arr = np.asarray(image.convert('RGB'))
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise InputValidationError('numpy image must have shape HxWx3')
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise InputValidationError('unsupported image input')
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise InputValidationError('image must be RGB')
    return arr

def _decode_uncompressed_rle(rle, shape):
    h,w = shape
    counts = rle.get('counts')
    if not isinstance(counts, list):
        raise InputValidationError('compressed COCO RLE is not supported in this baseline; provide PNG, numpy mask, polygon, or uncompressed RLE counts list')
    flat = []
    val = 0
    for c in counts:
        flat.extend([val] * int(c)); val = 1 - val
    arr = np.asarray(flat[:h*w], dtype=np.uint8)
    if arr.size < h*w:
        arr = np.pad(arr, (0, h*w-arr.size))
    return arr.reshape((w, h)).T.astype(bool)

def load_mask(mask, image_shape, name='mask'):
    h,w = image_shape[:2]
    if isinstance(mask, (str, Path)):
        arr = np.asarray(Image.open(mask).convert('L')) > 0
    elif isinstance(mask, np.ndarray):
        arr = mask.astype(bool)
    elif isinstance(mask, dict):
        if 'counts' in mask and 'size' in mask:
            arr = _decode_uncompressed_rle(mask, (h,w))
        else:
            raise InputValidationError(f'{name}: unsupported dict mask')
    elif isinstance(mask, (list, tuple)):
        canvas = Image.new('L', (w,h), 0)
        draw = ImageDraw.Draw(canvas)
        polys = mask
        if polys and isinstance(polys[0], (int,float)):
            polys = [polys]
        for poly in polys:
            if len(poly) < 6:
                raise InputValidationError(f'{name}: invalid polygon')
            pts = [(float(poly[i]), float(poly[i+1])) for i in range(0, len(poly), 2)]
            draw.polygon(pts, outline=1, fill=1)
        arr = np.asarray(canvas) > 0
    else:
        raise InputValidationError(f'{name}: unsupported mask input')
    if arr.shape != (h,w):
        raise InputValidationError(f'{name}: mask shape {arr.shape} does not match image shape {(h,w)}')
    if not arr.any():
        raise InputValidationError(f'{name}: mask is empty')
    return arr

def mask_bbox(mask):
    ys,xs = np.where(mask)
    if len(xs) == 0:
        raise InputValidationError('mask is empty')
    return [float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)]
