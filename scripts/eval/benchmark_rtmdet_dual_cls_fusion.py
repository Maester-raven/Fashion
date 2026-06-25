#!/usr/bin/env python3
"""Benchmark the shared-backbone A/D1 RTMDet selective fusion pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.runner import load_state_dict
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


CLASSES = ("top", "pants", "skirt", "outerwear", "dress", "shoes", "bag", "accessory")
THRESHOLDS = torch.tensor([0.325, 0.4125, 0.4875, 0.425, 0.35, 0.30, 0.3875, 0.4125])
SORT_MULTIPLIERS = torch.tensor([1.0, 1.18, 0.65, 1.0, 0.95, 1.0, 1.0, 1.0])
LOWER_CONFLICTS = {frozenset((1, 2)), frozenset((1, 4)), frozenset((2, 4))}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--a-checkpoint", required=True)
    parser.add_argument("--d1-checkpoint", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def ema_model_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    # EMAHook swaps fields before saving: state_dict contains the averaged
    # deployment weights, while ema_state_dict keeps the original model.
    return checkpoint["state_dict"]


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def bbox_pair_stats(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    iou = inter / max(1e-9, area_a + area_b - inter)
    contain = inter / max(1e-9, min(area_a, area_b))
    return iou, contain


def mask_matrix_iou(a_masks, b_masks):
    if len(a_masks) == 0 or len(b_masks) == 0:
        return torch.zeros((len(a_masks), len(b_masks)), device=a_masks.device)
    a = a_masks.flatten(1).to(torch.float16)
    b = b_masks.flatten(1).to(torch.float16)
    inter = a @ b.transpose(0, 1)
    area_a = a.sum(dim=1, dtype=torch.float32).unsqueeze(1)
    area_b = b.sum(dim=1, dtype=torch.float32).unsqueeze(0)
    return inter.float() / (area_a + area_b - inter.float()).clamp_min(1.0)


def selective_relabel(a_result, d1_result):
    labels = a_result.labels.clone()
    scores = a_result.scores
    a_indices = torch.nonzero(scores >= 0.20, as_tuple=False).flatten()
    d_indices = torch.nonzero(d1_result.scores >= 0.20, as_tuple=False).flatten()
    if len(a_indices) == 0 or len(d_indices) == 0:
        return labels, 0
    ious = mask_matrix_iou(a_result.masks[a_indices], d1_result.masks[d_indices])
    pairs = []
    positions = torch.nonzero(ious >= 0.50, as_tuple=False)
    for a_pos, d_pos in positions.tolist():
        pairs.append((float(ious[a_pos, d_pos]), a_pos, d_pos))
    used_a = set()
    used_d = set()
    relabeled = 0
    for _, a_pos, d_pos in sorted(pairs, reverse=True):
        if a_pos in used_a or d_pos in used_d:
            continue
        used_a.add(a_pos)
        used_d.add(d_pos)
        ai = int(a_indices[a_pos])
        di = int(d_indices[d_pos])
        a_label = int(labels[ai])
        d_label = int(d1_result.labels[di])
        if frozenset((a_label, d_label)) != frozenset((0, 4)):
            continue
        if float(d1_result.scores[di]) < float(scores[ai]):
            continue
        labels[ai] = d_label
        relabeled += 1
    return labels, relabeled


def final_resolver(a_result, labels):
    device = a_result.scores.device
    thresholds = THRESHOLDS.to(device)
    multipliers = SORT_MULTIPLIERS.to(device)
    selected = torch.nonzero(a_result.scores >= thresholds[labels], as_tuple=False).flatten()
    if len(selected) == 0:
        return selected
    sort_scores = a_result.scores[selected] * multipliers[labels[selected]]
    order = selected[torch.argsort(sort_scores, descending=True)]
    ordered_boxes = a_result.bboxes[order]
    ordered_labels = labels[order]
    ordered_masks = a_result.masks[order]
    left_top = torch.maximum(
        ordered_boxes[:, None, :2], ordered_boxes[None, :, :2]
    )
    right_bottom = torch.minimum(
        ordered_boxes[:, None, 2:], ordered_boxes[None, :, 2:]
    )
    sizes = (right_bottom - left_top).clamp_min(0)
    box_intersection = sizes[..., 0] * sizes[..., 1]
    box_areas = (
        (ordered_boxes[:, 2] - ordered_boxes[:, 0]).clamp_min(0)
        * (ordered_boxes[:, 3] - ordered_boxes[:, 1]).clamp_min(0)
    )
    box_iou = box_intersection / (
        box_areas[:, None] + box_areas[None, :] - box_intersection
    ).clamp_min(1e-9)
    box_contain = box_intersection / torch.minimum(
        box_areas[:, None], box_areas[None, :]
    ).clamp_min(1e-9)

    same_class = ordered_labels[:, None] == ordered_labels[None, :]
    lower_group = (
        ((ordered_labels[:, None] == 1) | (ordered_labels[:, None] == 2) | (ordered_labels[:, None] == 4))
        & ((ordered_labels[None, :] == 1) | (ordered_labels[None, :] == 2) | (ordered_labels[None, :] == 4))
    )
    category_conflict = same_class | lower_group
    bbox_gate = (box_iou >= 0.10) | (box_contain >= 0.55)

    flat_masks = ordered_masks.flatten(1).to(torch.float16)
    mask_intersection = (flat_masks @ flat_masks.transpose(0, 1)).float()
    mask_areas = flat_masks.sum(dim=1, dtype=torch.float32)
    mask_iou = mask_intersection / (
        mask_areas[:, None] + mask_areas[None, :] - mask_intersection
    ).clamp_min(1.0)
    mask_contain = mask_intersection / torch.minimum(
        mask_areas[:, None], mask_areas[None, :]
    ).clamp_min(1.0)
    contain_threshold = torch.where(
        same_class,
        torch.full_like(mask_contain, 0.75),
        torch.full_like(mask_contain, 0.775),
    )
    suppress = (
        category_conflict
        & bbox_gate
        & ((mask_iou >= 0.275) | (mask_contain >= contain_threshold))
    )
    suppress.fill_diagonal_(False)
    suppress_cpu = suppress.detach().cpu().numpy()
    dropped = [False] * len(order)
    kept_positions = []
    for position in range(len(order)):
        if dropped[position]:
            continue
        kept_positions.append(position)
        for lower_position in range(position + 1, len(order)):
            if suppress_cpu[lower_position, position]:
                dropped[lower_position] = True
    kept_positions = torch.tensor(kept_positions, dtype=torch.long, device=device)
    return order[kept_positions]


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, checkpoint=None, device="cuda:0")
    load_state_dict(model, ema_model_state(args.a_checkpoint), strict=True)
    model.eval()

    d1_cls = copy.deepcopy(model.bbox_head.rtm_cls).cuda().eval()
    d1_state = ema_model_state(args.d1_checkpoint)
    d1_cls.load_state_dict({
        key.removeprefix("bbox_head.rtm_cls."): value
        for key, value in d1_state.items()
        if key.startswith("bbox_head.rtm_cls.")
    }, strict=True)

    test_pipeline_cfg = get_test_pipeline_cfg(cfg)
    test_pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
    test_pipeline = Compose(test_pipeline_cfg)
    with open(args.val_json, "r", encoding="utf-8") as file:
        coco = json.load(file)
    image_paths = [str(Path(args.data_root) / item["file_name"]) for item in coco["images"]]
    requested = args.warmup + args.max_images
    if len(image_paths) < requested:
        repeats = (requested + len(image_paths) - 1) // len(image_paths)
        image_paths = (image_paths * repeats)[:requested]
    else:
        image_paths = image_paths[:requested]

    latencies = []
    decode_latencies = []
    compute_latencies = []
    output_counts = []
    category_counts = {name: 0 for name in CLASSES}
    completeness_failures = 0
    total_relabels = 0
    amp_dtype = torch.float16

    def infer_array(image):
        data = test_pipeline(dict(img=image, img_id=0))
        data["inputs"] = [data["inputs"]]
        data["data_samples"] = [data["data_samples"]]
        data = model.data_preprocessor(data, training=False)
        batch_inputs = data["inputs"]
        samples = data["data_samples"]
        batch_img_metas = [sample.metainfo for sample in samples]
        with torch.inference_mode(), torch.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
            feats = model.extract_feat(batch_inputs)
            outs = model.bbox_head(feats)
            predict_outs = tuple(
                tuple(item.float() for item in value)
                if isinstance(value, (tuple, list))
                else value.float()
                for value in outs
            ) if args.amp else outs
            a_results = model.bbox_head.predict_by_feat(
                *predict_outs, batch_img_metas=batch_img_metas, rescale=True
            )
            d1_scores = []
            for index, feat in enumerate(feats):
                cls_feat = feat
                for layer in model.bbox_head.cls_convs[index]:
                    cls_feat = layer(cls_feat)
                d1_score = d1_cls[index](cls_feat)
                d1_scores.append(d1_score.float() if args.amp else d1_score)
            d1_results = model.bbox_head.predict_by_feat(
                tuple(d1_scores),
                predict_outs[1],
                predict_outs[2],
                predict_outs[3],
                batch_img_metas=batch_img_metas,
                rescale=True,
            )
            labels, relabeled = selective_relabel(a_results[0], d1_results[0])
            kept = final_resolver(a_results[0], labels)
            output = {
                "scores": a_results[0].scores[kept].detach().cpu().numpy(),
                "labels": labels[kept].detach().cpu().numpy(),
                "bboxes": a_results[0].bboxes[kept].detach().cpu().numpy(),
                "masks": a_results[0].masks[kept].detach().cpu().numpy(),
            }
        return output, relabeled

    for index, image_path in enumerate(image_paths):
        start = time.perf_counter()
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        decode_done = time.perf_counter()
        if image is None:
            raise FileNotFoundError(image_path)
        torch.cuda.synchronize()
        compute_start = time.perf_counter()
        output, relabeled = infer_array(image)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if index < args.warmup:
            continue
        decode_ms = (decode_done - start) * 1000
        compute_ms = (end - compute_start) * 1000
        decode_latencies.append(decode_ms)
        compute_latencies.append(compute_ms)
        latencies.append((end - start) * 1000)
        total_relabels += relabeled
        count = len(output["labels"])
        output_counts.append(count)
        if not (
            len(output["scores"]) == count
            and len(output["bboxes"]) == count
            and len(output["masks"]) == count
            and output["bboxes"].shape == (count, 4)
            and output["masks"].ndim == 3
        ):
            completeness_failures += 1
        for label in output["labels"].tolist():
            category_counts[CLASSES[int(label)]] += 1

    result = {
        "device": torch.cuda.get_device_name(0),
        "precision_mode": "amp_fp16" if args.amp else "fp32",
        "warmup_images": args.warmup,
        "measured_images": len(latencies),
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p95": percentile(latencies, 95),
            "max": max(latencies),
            "pass_rate_le_50ms": sum(value <= 50 for value in latencies) / len(latencies),
        },
        "decode_ms": {
            "mean": statistics.mean(decode_latencies),
            "p50": percentile(decode_latencies, 50),
            "p95": percentile(decode_latencies, 95),
        },
        "compute_and_output_ms": {
            "mean": statistics.mean(compute_latencies),
            "p50": percentile(compute_latencies, 50),
            "p95": percentile(compute_latencies, 95),
        },
        "sequential_qps": 1000.0 / statistics.mean(latencies),
        "output_completeness": {
            "failures": completeness_failures,
            "passed": completeness_failures == 0,
            "mean_instances_per_image": statistics.mean(output_counts),
            "category_counts": category_counts,
            "observed_all_8_categories": all(value > 0 for value in category_counts.values()),
            "selective_relabels": total_relabels,
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
