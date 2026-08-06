#!/usr/bin/env python3
import argparse
from fashion_3_1_2 import SingleHitBBoxMaskPipeline
from fashion_3_1_2.pipeline import save_json, save_overlay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='single_hit_v1', choices=['single_hit_v1','zero_one_n_functional_v1'])
    ap.add_argument("--image", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-overlay")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-root")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--adapter-mode", default="mock", choices=["mock", "sam_hq_smoke"])
    ap.add_argument("--mock-bbox", default="")
    args = ap.parse_args()
    bbox = [float(x) for x in args.mock_bbox.split(",")] if args.mock_bbox else None
    pipe = SingleHitBBoxMaskPipeline.from_config(args.config, device=args.device, model_root=args.model_root, adapter_mode=args.adapter_mode, mock_bbox=bbox)
    result = pipe.predict(args.image, args.query)
    save_json(result, args.output_json)
    if args.output_overlay:
        save_overlay(args.image, result, args.output_overlay)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
