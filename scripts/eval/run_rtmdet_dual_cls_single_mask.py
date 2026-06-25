#!/usr/bin/env python3
"""Run RTMDet A/D1 fusion with one NMS and one mask generation pass."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.ops import batched_nms
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.runner import load_state_dict
from mmengine.structures import InstanceData
from mmdet.apis import init_detector
from mmdet.models.utils import filter_scores_and_topk
from mmdet.structures.bbox import cat_boxes, get_box_tensor, get_box_wh, scale_boxes
from mmdet.utils import get_test_pipeline_cfg
from pycocotools import mask as mask_utils

from benchmark_rtmdet_dual_cls_fusion import THRESHOLDS, ema_model_state, final_resolver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--a-checkpoint", required=True)
    parser.add_argument("--d1-checkpoint", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--benchmark-json", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--benchmark-images", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--nms-pre", type=int, default=1000)
    parser.add_argument("--max-per-img", type=int, default=150)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--disable-d1-fusion", action="store_true")
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def dual_single_predict(head, outs, d1_scores, img_meta):
    a_cls_scores, bbox_preds, kernel_preds, mask_feat = outs
    cfg = copy.deepcopy(head.test_cfg)
    cfg.score_thr = float(dual_single_predict.score_thr)
    cfg.nms_pre = int(dual_single_predict.nms_pre)
    cfg.max_per_img = int(dual_single_predict.max_per_img)
    featmap_sizes = [score.shape[-2:] for score in a_cls_scores]
    priors_by_level = head.prior_generator.grid_priors(
        featmap_sizes,
        dtype=a_cls_scores[0].dtype,
        device=a_cls_scores[0].device,
        with_stride=True,
    )

    selected_bboxes = []
    selected_priors = []
    selected_scores = []
    selected_labels = []
    selected_kernels = []
    selected_source_ids = []
    expert_bboxes = []
    expert_priors = []
    expert_scores_out = []
    expert_labels_out = []
    expert_source_ids = []
    source_offset = 0
    for a_logits, d1_logits, bbox_pred, kernel_pred, priors in zip(
        a_cls_scores, d1_scores, bbox_preds, kernel_preds, priors_by_level
    ):
        a_logits = a_logits[0]
        d1_logits = d1_logits[0]
        bbox_pred = bbox_pred[0].permute(1, 2, 0).reshape(-1, head.bbox_coder.encode_size)
        kernel_pred = kernel_pred[0].permute(1, 2, 0).reshape(-1, head.num_gen_params)
        a_scores = a_logits.permute(1, 2, 0).reshape(-1, head.cls_out_channels).sigmoid()
        expert_scores = d1_logits.permute(1, 2, 0).reshape(-1, head.cls_out_channels).sigmoid()
        source_ids = torch.arange(
            a_scores.shape[0], dtype=torch.long, device=a_scores.device
        ) + source_offset
        source_offset += a_scores.shape[0]
        scores, labels, _, filtered = filter_scores_and_topk(
            a_scores,
            cfg.score_thr,
            cfg.nms_pre,
            dict(
                bbox_pred=bbox_pred,
                priors=priors,
                kernel_pred=kernel_pred,
                d1_scores=expert_scores,
                source_ids=source_ids,
            ),
        )
        selected_bboxes.append(filtered["bbox_pred"])
        selected_priors.append(filtered["priors"])
        selected_scores.append(scores)
        selected_labels.append(labels)
        selected_kernels.append(filtered["kernel_pred"])
        selected_source_ids.append(filtered["source_ids"])
        if not getattr(dual_single_predict, "disable_d1_fusion", False):
            expert_selected_scores, expert_selected_labels, _, expert_filtered = filter_scores_and_topk(
                expert_scores,
                cfg.score_thr,
                cfg.nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors, source_ids=source_ids),
            )
            expert_bboxes.append(expert_filtered["bbox_pred"])
            expert_priors.append(expert_filtered["priors"])
            expert_scores_out.append(expert_selected_scores)
            expert_labels_out.append(expert_selected_labels)
            expert_source_ids.append(expert_filtered["source_ids"])

    bbox_pred = torch.cat(selected_bboxes)
    priors = cat_boxes(selected_priors)
    bboxes = head.bbox_coder.decode(priors[..., :2], bbox_pred, max_shape=img_meta["img_shape"])
    results = InstanceData(
        bboxes=bboxes,
        priors=priors,
        scores=torch.cat(selected_scores),
        labels=torch.cat(selected_labels),
        kernels=torch.cat(selected_kernels),
        source_ids=torch.cat(selected_source_ids),
    )

    scale_factor = [1 / value for value in img_meta["scale_factor"]]
    results.bboxes = scale_boxes(results.bboxes, scale_factor)
    if cfg.get("min_bbox_size", -1) >= 0:
        widths, heights = get_box_wh(results.bboxes)
        valid = (widths > cfg.min_bbox_size) & (heights > cfg.min_bbox_size)
        if not valid.all():
            results = results[valid]

    if results.bboxes.numel() == 0:
        height, width = img_meta["ori_shape"][:2]
        results.masks = torch.zeros((0, height, width), dtype=torch.bool, device=mask_feat.device)
        return results, 0

    det_bboxes, keep_indices = batched_nms(
        get_box_tensor(results.bboxes), results.scores, results.labels, cfg.nms
    )
    results = results[keep_indices]
    results.scores = det_bboxes[:, -1]
    results = results[:cfg.max_per_img]

    labels = results.labels.clone()
    pairs = []
    if not getattr(dual_single_predict, "disable_d1_fusion", False):
        expert_bbox_pred = torch.cat(expert_bboxes)
        expert_prior_tensor = cat_boxes(expert_priors)
        expert_decoded = head.bbox_coder.decode(
            expert_prior_tensor[..., :2], expert_bbox_pred, max_shape=img_meta["img_shape"]
        )
        expert_results = InstanceData(
            bboxes=scale_boxes(expert_decoded, scale_factor),
            scores=torch.cat(expert_scores_out),
            labels=torch.cat(expert_labels_out),
            source_ids=torch.cat(expert_source_ids),
        )
        expert_det_bboxes, expert_keep = batched_nms(
            get_box_tensor(expert_results.bboxes),
            expert_results.scores,
            expert_results.labels,
            cfg.nms,
        )
        expert_results = expert_results[expert_keep]
        expert_results.scores = expert_det_bboxes[:, -1]
        expert_results = expert_results[:cfg.max_per_img]
        a_positions = torch.nonzero(results.scores >= 0.20, as_tuple=False).flatten()
        d_positions = torch.nonzero(expert_results.scores >= 0.20, as_tuple=False).flatten()
    else:
        a_positions = torch.empty(0, dtype=torch.long, device=results.scores.device)
        d_positions = torch.empty(0, dtype=torch.long, device=results.scores.device)
    if len(a_positions) and len(d_positions):
        a_boxes = results.bboxes[a_positions]
        d_boxes = expert_results.bboxes[d_positions]
        left_top = torch.maximum(a_boxes[:, None, :2], d_boxes[None, :, :2])
        right_bottom = torch.minimum(a_boxes[:, None, 2:], d_boxes[None, :, 2:])
        sizes = (right_bottom - left_top).clamp_min(0)
        intersections = sizes[..., 0] * sizes[..., 1]
        a_areas = (
            (a_boxes[:, 2] - a_boxes[:, 0]).clamp_min(0)
            * (a_boxes[:, 3] - a_boxes[:, 1]).clamp_min(0)
        )[:, None]
        d_areas = (
            (d_boxes[:, 2] - d_boxes[:, 0]).clamp_min(0)
            * (d_boxes[:, 3] - d_boxes[:, 1]).clamp_min(0)
        )[None, :]
        pair_ious = intersections / (a_areas + d_areas - intersections).clamp_min(1e-9)
        same_source = (
            results.source_ids[a_positions, None]
            == expert_results.source_ids[d_positions][None, :]
        )
        pair_ious = torch.where(same_source, torch.ones_like(pair_ious), pair_ious)
        pair_positions = torch.nonzero(pair_ious >= 0.50, as_tuple=False)
        pair_values = pair_ious[pair_positions[:, 0], pair_positions[:, 1]]
        a_position_values = a_positions.detach().cpu().tolist()
        d_position_values = d_positions.detach().cpu().tolist()
        for value, (a_index, d_index) in zip(
            pair_values.detach().cpu().tolist(), pair_positions.detach().cpu().tolist()
        ):
            pairs.append((value, a_position_values[a_index], d_position_values[d_index]))
    used_a = set()
    used_d = set()
    relabeled = 0
    for _, a_position, d_position in sorted(pairs, reverse=True):
        if a_position in used_a or d_position in used_d:
            continue
        used_a.add(a_position)
        used_d.add(d_position)
        a_label = int(labels[a_position])
        d_label = int(expert_results.labels[d_position])
        if frozenset((a_label, d_label)) != frozenset((0, 4)):
            continue
        if float(expert_results.scores[d_position]) < float(results.scores[a_position]):
            continue
        labels[a_position] = d_label
        relabeled += 1
    results.labels = labels

    # The validated PRD pipeline applies class thresholds before conflict
    # resolution. Move the same gate ahead of dynamic mask generation so only
    # final-output candidates receive full-resolution masks.
    pre_mask_keep = results.scores >= THRESHOLDS.to(results.scores.device)[results.labels]
    results = results[pre_mask_keep]
    if len(results) == 0:
        ori_h, ori_w = img_meta["ori_shape"][:2]
        results.masks = torch.zeros(
            (0, ori_h, ori_w), dtype=torch.bool, device=mask_feat.device
        )
        return results, relabeled

    stride = head.prior_generator.strides[0][0]
    mask_logits = head._mask_predict_by_feat_single(mask_feat[0], results.kernels, results.priors)
    mask_logits = F.interpolate(mask_logits.unsqueeze(0), scale_factor=stride, mode="bilinear")
    ori_h, ori_w = img_meta["ori_shape"][:2]
    mask_logits = F.interpolate(
        mask_logits,
        size=[
            math.ceil(mask_logits.shape[-2] * scale_factor[0]),
            math.ceil(mask_logits.shape[-1] * scale_factor[1]),
        ],
        mode="bilinear",
        align_corners=False,
    )[..., :ori_h, :ori_w]
    results.masks = mask_logits.sigmoid().squeeze(0) > cfg.mask_thr_binary
    return results, relabeled


def encode_prediction(image_id, category_ids, result, kept):
    rows = []
    scores = result.scores[kept].detach().cpu().numpy()
    labels = result.labels[kept].detach().cpu().numpy()
    boxes = result.bboxes[kept].detach().cpu().numpy()
    masks = result.masks[kept].detach().cpu().numpy()
    for score, label, box, mask in zip(scores, labels, boxes, masks):
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("ascii")
        x1, y1, x2, y2 = [float(value) for value in box]
        rows.append({
            "image_id": int(image_id),
            "category_id": int(category_ids[int(label)]),
            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
            "score": float(score),
            "segmentation": rle,
        })
    return rows


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dual_single_predict.score_thr = args.score_thr
    dual_single_predict.nms_pre = args.nms_pre
    dual_single_predict.max_per_img = args.max_per_img
    dual_single_predict.disable_d1_fusion = args.disable_d1_fusion
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

    def shared_forward(batch_inputs):
        features = model.extract_feat(batch_inputs)
        model_outputs = model.bbox_head(features)
        expert_outputs = []
        for level, feature in enumerate(features):
            cls_feature = feature
            for layer in model.bbox_head.cls_convs[level]:
                cls_feature = layer(cls_feature)
            expert_outputs.append(d1_cls[level](cls_feature))
        return model_outputs, tuple(expert_outputs)

    if args.compile:
        shared_forward = torch.compile(
            shared_forward, mode="reduce-overhead", fullgraph=False
        )
    pipeline_cfg = get_test_pipeline_cfg(cfg)
    pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
    pipeline = Compose(pipeline_cfg)

    with open(args.val_json, "r", encoding="utf-8") as file:
        coco = json.load(file)
    name_to_id = {category["name"]: category["id"] for category in coco["categories"]}
    category_ids = [name_to_id[name] for name in ("top", "pants", "skirt", "outerwear", "dress", "shoes", "bag", "accessory")]
    images = coco["images"][: args.max_images or None]
    benchmark_limit = min(len(images), args.warmup + args.benchmark_images)
    latencies = []
    inference_latencies = []
    rle_output_latencies = []
    relabeled_total = 0
    predictions = []

    for index, image_info in enumerate(images):
        start = time.perf_counter()
        path = str(Path(args.data_root) / image_info["file_name"])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        transformed = pipeline(dict(img=image, img_id=image_info["id"]))
        data = {
            "inputs": [transformed["inputs"]],
            "data_samples": [transformed["data_samples"]],
        }
        data = model.data_preprocessor(data, training=False)
        img_meta = data["data_samples"][0].metainfo
        with torch.inference_mode():
            outs, expert_scores = shared_forward(data["inputs"])
            result, relabeled = dual_single_predict(model.bbox_head, outs, expert_scores, img_meta)
            kept = final_resolver(result, result.labels)
        torch.cuda.synchronize()
        inference_done = time.perf_counter()
        rows = encode_prediction(image_info["id"], category_ids, result, kept)
        output_done = time.perf_counter()
        predictions.extend(rows)
        relabeled_total += relabeled
        torch.cuda.synchronize()
        end = time.perf_counter()
        if index < benchmark_limit and index >= args.warmup:
            latencies.append((end - start) * 1000)
            inference_latencies.append((inference_done - start) * 1000)
            rle_output_latencies.append((output_done - inference_done) * 1000)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(predictions, file)
    report = {
        "device": torch.cuda.get_device_name(0),
        "images": len(images),
        "predictions": len(predictions),
        "relabeled": relabeled_total,
        "test_cfg": {
            "score_thr": args.score_thr,
            "nms_pre": args.nms_pre,
            "max_per_img": args.max_per_img,
        },
        "torch_compile": args.compile,
        "d1_fusion_enabled": not args.disable_d1_fusion,
        "latency_ms": {
            "samples": len(latencies),
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": percentile(latencies, 50) if latencies else None,
            "p90": percentile(latencies, 90) if latencies else None,
            "p95": percentile(latencies, 95) if latencies else None,
            "pass_rate_le_50ms": (
                sum(value <= 50 for value in latencies) / len(latencies) if latencies else None
            ),
        },
        "inference_before_rle_ms": {
            "mean": statistics.mean(inference_latencies) if inference_latencies else None,
            "p50": percentile(inference_latencies, 50) if inference_latencies else None,
            "p95": percentile(inference_latencies, 95) if inference_latencies else None,
        },
        "rle_output_ms": {
            "mean": statistics.mean(rle_output_latencies) if rle_output_latencies else None,
            "p50": percentile(rle_output_latencies, 50) if rle_output_latencies else None,
            "p95": percentile(rle_output_latencies, 95) if rle_output_latencies else None,
        },
        "sequential_qps": 1000.0 / statistics.mean(latencies) if latencies else None,
    }
    benchmark_path = Path(args.benchmark_json)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
