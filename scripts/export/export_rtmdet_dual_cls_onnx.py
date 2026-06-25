#!/usr/bin/env python3
"""Export the shared RTMDet core with A and D1 classification logits."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import load_state_dict
from mmdet.apis import init_detector

from benchmark_rtmdet_dual_cls_fusion import ema_model_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--a-checkpoint", required=True)
    parser.add_argument("--d1-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--a-only", action="store_true")
    return parser.parse_args()


class SharedDualClassificationCore(torch.nn.Module):
    def __init__(self, model, d1_cls=None):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.bbox_head = model.bbox_head
        self.d1_cls = d1_cls

    def forward(self, inputs):
        features = self.backbone(inputs)
        features = self.neck(features)
        a_cls, bbox, kernels, mask_feat = self.bbox_head(features)
        base_outputs = (
            a_cls[0], a_cls[1], a_cls[2],
            bbox[0], bbox[1], bbox[2],
            kernels[0], kernels[1], kernels[2],
            mask_feat,
        )
        if self.d1_cls is None:
            return base_outputs
        d1_cls = []
        for level, feature in enumerate(features):
            cls_feature = feature
            for layer in self.bbox_head.cls_convs[level]:
                cls_feature = layer(cls_feature)
            d1_cls.append(self.d1_cls[level](cls_feature))
        return base_outputs + (d1_cls[0], d1_cls[1], d1_cls[2])


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, checkpoint=None, device="cpu")
    load_state_dict(model, ema_model_state(args.a_checkpoint), strict=True)
    model.eval()
    d1_cls = None
    if not args.a_only:
        if not args.d1_checkpoint:
            raise ValueError("--d1-checkpoint is required unless --a-only is set")
        d1_cls = copy.deepcopy(model.bbox_head.rtm_cls).eval()
        d1_state = ema_model_state(args.d1_checkpoint)
        d1_cls.load_state_dict({
            key.removeprefix("bbox_head.rtm_cls."): value
            for key, value in d1_state.items()
            if key.startswith("bbox_head.rtm_cls.")
        }, strict=True)
    core = SharedDualClassificationCore(model, d1_cls).eval()
    dummy = torch.zeros((args.batch_size, 3, 1024, 1024), dtype=torch.float32)
    output_names = [
        "a_cls_0", "a_cls_1", "a_cls_2",
        "bbox_0", "bbox_1", "bbox_2",
        "kernel_0", "kernel_1", "kernel_2",
        "mask_feat",
    ]
    if not args.a_only:
        output_names += ["d1_cls_0", "d1_cls_1", "d1_cls_2"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            core,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=None,
        )
    import onnx

    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    print(f"ONNX_OK path={output_path} size_bytes={output_path.stat().st_size}")
    for output in graph.graph.output:
        shape = [
            dimension.dim_value if dimension.dim_value else dimension.dim_param
            for dimension in output.type.tensor_type.shape.dim
        ]
        print(output.name, shape)


if __name__ == "__main__":
    main()
