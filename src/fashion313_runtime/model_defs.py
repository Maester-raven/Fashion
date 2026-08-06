
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import json
import sys
import numpy as np
import torch
import torch.nn as nn

GEOM_NAMES = (
    ["bbox_width_ratio", "bbox_height_ratio", "mask_area_image_ratio", "mask_area_bbox_ratio", "bbox_aspect_ratio", "centroid_x_ratio", "centroid_y_ratio",
     "mask_touch_top", "mask_touch_bottom", "mask_touch_left", "mask_touch_right", "horizontal_span_ratio", "vertical_span_ratio"]
    + [f"vertical_width_profile_{i}" for i in range(8)]
    + [f"horizontal_height_profile_{i}" for i in range(8)]
)

def _package_root():
    return Path(__file__).resolve().parents[2]

def _read_json(path):
    return json.loads(Path(path).read_text())

def load_c2_mapping(config_dir=None):
    config_dir = Path(config_dir or (_package_root() / "config"))
    mapping = _read_json(config_dir / "c2_class_mapping.json")["groups"]
    groups = OrderedDict((g, spec["class_value_names"]) for g, spec in mapping.items())
    label_to_index = {g: {k: int(v) for k, v in spec["label_to_index"].items()} for g, spec in mapping.items()}
    return mapping, groups, label_to_index

class GroupHeads(nn.Module):
    def __init__(self, group_to_classes, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleDict()
        for group, labels in group_to_classes.items():
            self.heads[group] = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, len(labels)))
    def forward_group(self, feat, group):
        return self.heads[group](feat)

class TwoViewModel(nn.Module):
    def __init__(self, group_to_classes, dropout=0.1):
        super().__init__()
        repo = _package_root() / "third_party" / "dinov2-official"
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from dinov2.models.vision_transformer import vit_small
        self.backbone = vit_small(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
        self.heads = GroupHeads(group_to_classes, dropout=dropout)
    def encode(self, local, parent):
        return torch.cat([self.backbone(local), self.backbone(parent)], dim=1)

class TaskHeads(nn.Module):
    def __init__(self, task_to_classes, input_dim=768, hidden=384, dropout=0.1):
        super().__init__()
        self.task_to_classes = task_to_classes
        self.heads = nn.ModuleDict({t: nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, len(classes))) for t, classes in task_to_classes.items()})
    def forward_task(self, x, task):
        return self.heads[task](x)

class P1Model(nn.Module):
    def __init__(self, old_group_to_classes, new_mapping, dropout=0.1):
        super().__init__()
        self.c2 = TwoViewModel(old_group_to_classes, dropout=dropout)
        self.new_heads = TaskHeads({t: spec["class_names"] for t, spec in new_mapping.items()}, dropout=dropout)
    def encode(self, local, context):
        return self.c2.encode(local, context)
    def forward_new(self, feat, task):
        return self.new_heads.forward_task(feat, task)
    def forward_c2(self, feat, group):
        return self.c2.heads.forward_group(feat, group)

class RegionFamilyHead(nn.Module):
    def __init__(self, input_dim, local_classes):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 256), nn.GELU(), nn.Dropout(0.1))
        self.scope_head = nn.Linear(256, 2)
        self.local_head = nn.Linear(256, local_classes)
    def forward(self, x):
        h = self.proj(x)
        return self.scope_head(h), self.local_head(h)

def box_xyxy(row):
    if row.get("bbox_xyxy"):
        return [float(x) for x in row["bbox_xyxy"]]
    b = row.get("target_bbox") or row.get("bbox")
    return [float(b[0]), float(b[1]), float(b[0]) + float(b[2]), float(b[1]) + float(b[3])]

def geometry_for_row_fast(row):
    w, h = float(row["image_width"]), float(row["image_height"])
    x1, y1, x2, y2 = box_xyxy(row)
    bw, bh = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    area = float(row.get("area") or bw * bh)
    vals = []
    def add(v):
        vals.append(float(v) if np.isfinite(v) else 0.0)
    add(bw / max(1.0, w)); add(bh / max(1.0, h))
    add(area / max(1.0, w * h)); add(area / max(1e-6, bw * bh))
    add(bw / max(1e-6, bh))
    add(((x1 + x2) * 0.5) / max(1.0, w)); add(((y1 + y2) * 0.5) / max(1.0, h))
    add(float(y1 <= 1)); add(float(y2 >= h - 1)); add(float(x1 <= 1)); add(float(x2 >= w - 1))
    add(bw / max(1.0, w)); add(bh / max(1.0, h))
    fill = min(1.0, max(0.0, area / max(1e-6, bw * bh)))
    vals.extend([fill] * 8)
    vals.extend([fill] * 8)
    valid = np.ones(len(vals), dtype=np.float32)
    return np.array(vals, dtype=np.float32), valid

def normalized_geometry(row, mean, std):
    g, v = geometry_for_row_fast(row)
    z = (g - mean) / std
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0).astype(np.float32)
    return z, v
