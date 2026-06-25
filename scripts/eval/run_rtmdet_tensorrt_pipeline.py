#!/usr/bin/env python3
"""Run and benchmark the TensorRT RTMDet dual-classification pipeline."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
from mmcv.transforms import Compose
from mmengine.config import Config
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from pycocotools import mask as mask_utils

from benchmark_rtmdet_dual_cls_fusion import final_resolver
from run_rtmdet_dual_cls_single_mask import dual_single_predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--benchmark-json", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--benchmark-images", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--score-thr", type=float, default=0.15)
    parser.add_argument("--nms-pre", type=int, default=300)
    parser.add_argument("--max-per-img", type=int, default=80)
    parser.add_argument("--disable-d1-fusion", action="store_true")
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class TensorRTRunner:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.outputs = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            if mode == trt.TensorIOMode.OUTPUT:
                if dtype != trt.float32:
                    raise TypeError(f"Unsupported output dtype {name}: {dtype}")
                tensor = torch.empty(shape, dtype=torch.float32, device="cuda")
                self.outputs[name] = tensor
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"Failed to bind output {name}")

    def __call__(self, inputs):
        if inputs.shape != (1, 3, 1024, 1024):
            raise ValueError(f"Unexpected input shape: {tuple(inputs.shape)}")
        inputs = inputs.contiguous()
        if not self.context.set_tensor_address("input", inputs.data_ptr()):
            raise RuntimeError("Failed to bind TensorRT input")
        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            if not self.context.execute_async_v3(
                stream_handle=self.stream.cuda_stream
            ):
                raise RuntimeError("TensorRT execute_async_v3 failed")
        current_stream.wait_stream(self.stream)
        return self.outputs


def encode_rows(image_id, category_ids, result, kept):
    scores = result.scores[kept].detach().cpu().numpy()
    labels = result.labels[kept].detach().cpu().numpy()
    boxes = result.bboxes[kept].detach().cpu().numpy()
    masks = result.masks[kept].detach().cpu().numpy()
    rows = []
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
    cfg = Config.fromfile(args.config)
    # Learned layers run inside TensorRT. This model supplies the identical
    # preprocessor and parameter-free RTMDet mask/bbox decoding helpers.
    helper_model = init_detector(cfg, checkpoint=None, device="cuda:0")
    helper_model.eval()
    pipeline_cfg = get_test_pipeline_cfg(cfg)
    pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
    pipeline = Compose(pipeline_cfg)
    runner = TensorRTRunner(args.engine)
    dual_single_predict.score_thr = args.score_thr
    dual_single_predict.nms_pre = args.nms_pre
    dual_single_predict.max_per_img = args.max_per_img
    dual_single_predict.disable_d1_fusion = args.disable_d1_fusion

    with open(args.val_json, "r", encoding="utf-8") as file:
        coco = json.load(file)
    name_to_id = {category["name"]: category["id"] for category in coco["categories"]}
    classes = ("top", "pants", "skirt", "outerwear", "dress", "shoes", "bag", "accessory")
    category_ids = [name_to_id[name] for name in classes]
    images = coco["images"][: args.max_images or None]
    benchmark_limit = min(len(images), args.warmup + args.benchmark_images)

    predictions = []
    total_latencies = []
    before_rle_latencies = []
    rle_latencies = []
    core_latencies = []
    relabeled_total = 0

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
        data = helper_model.data_preprocessor(data, training=False)
        img_meta = data["data_samples"][0].metainfo

        core_start = torch.cuda.Event(enable_timing=True)
        core_end = torch.cuda.Event(enable_timing=True)
        core_start.record()
        outputs = runner(data["inputs"])
        core_end.record()
        a_outputs = (
            (outputs["a_cls_0"], outputs["a_cls_1"], outputs["a_cls_2"]),
            (outputs["bbox_0"], outputs["bbox_1"], outputs["bbox_2"]),
            (outputs["kernel_0"], outputs["kernel_1"], outputs["kernel_2"]),
            outputs["mask_feat"],
        )
        expert_outputs = (
            outputs.get("d1_cls_0"), outputs.get("d1_cls_1"), outputs.get("d1_cls_2")
        )
        with torch.inference_mode():
            result, relabeled = dual_single_predict(
                helper_model.bbox_head, a_outputs, expert_outputs, img_meta
            )
            kept = final_resolver(result, result.labels)
        torch.cuda.synchronize()
        inference_done = time.perf_counter()
        rows = encode_rows(image_info["id"], category_ids, result, kept)
        output_done = time.perf_counter()
        predictions.extend(rows)
        relabeled_total += relabeled
        if index < benchmark_limit and index >= args.warmup:
            total_latencies.append((output_done - start) * 1000)
            before_rle_latencies.append((inference_done - start) * 1000)
            rle_latencies.append((output_done - inference_done) * 1000)
            core_latencies.append(core_start.elapsed_time(core_end))

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as file:
        json.dump(predictions, file)
    report = {
        "device": torch.cuda.get_device_name(0),
        "tensorrt": trt.__version__,
        "images": len(images),
        "predictions": len(predictions),
        "relabeled": relabeled_total,
        "test_cfg": {
            "score_thr": args.score_thr,
            "nms_pre": args.nms_pre,
            "max_per_img": args.max_per_img,
        },
        "d1_fusion_enabled": not args.disable_d1_fusion,
        "latency_ms": {
            "mean": statistics.mean(total_latencies),
            "p50": percentile(total_latencies, 50),
            "p90": percentile(total_latencies, 90),
            "p95": percentile(total_latencies, 95),
            "pass_rate_le_50ms": sum(value <= 50 for value in total_latencies) / len(total_latencies),
        },
        "before_rle_ms": {
            "mean": statistics.mean(before_rle_latencies),
            "p50": percentile(before_rle_latencies, 50),
            "p95": percentile(before_rle_latencies, 95),
        },
        "rle_ms": {
            "mean": statistics.mean(rle_latencies),
            "p50": percentile(rle_latencies, 50),
            "p95": percentile(rle_latencies, 95),
        },
        "tensorrt_core_ms": {
            "mean": statistics.mean(core_latencies),
            "p50": percentile(core_latencies, 50),
            "p95": percentile(core_latencies, 95),
        },
        "sequential_qps": 1000.0 / statistics.mean(total_latencies),
    }
    Path(args.benchmark_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.benchmark_json, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
