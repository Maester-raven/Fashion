import math
import numpy as np

def is_valid_xyxy(box):
    return bool(box and len(box) == 4 and float(box[2]) > float(box[0]) and float(box[3]) > float(box[1]))

def clip_xyxy(box, width, height):
    if not is_valid_xyxy(box):
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    b = [max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))]
    return b if is_valid_xyxy(b) else None

def parent_origin(parent_bbox):
    return int(math.floor(float(parent_bbox[0]))), int(math.floor(float(parent_bbox[1])))

def parent_size(parent_bbox):
    return max(1, int(math.ceil(float(parent_bbox[2]) - float(parent_bbox[0])))), max(1, int(math.ceil(float(parent_bbox[3]) - float(parent_bbox[1]))))

def full_image_bbox_to_parent_crop(box, parent_bbox):
    ox, oy = parent_origin(parent_bbox)
    w, h = parent_size(parent_bbox)
    x1, y1, x2, y2 = [float(v) for v in box]
    return clip_xyxy([x1 - ox, y1 - oy, x2 - ox, y2 - oy], w, h)

def sam_resize_identity_for_predictor(box_parent):
    # SamPredictor handles ResizeLongestSide internally. The public API keeps parent-crop xyxy.
    return [float(v) for v in box_parent]

def validate_mask_canvas(mask, parent_bbox):
    h, w = np.asarray(mask).shape[:2]
    pw, ph = parent_size(parent_bbox)
    return (w, h) == (pw, ph)
