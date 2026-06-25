#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts/eval:${PYTHONPATH:-}"

python -m fashion_system.instance_segmentation.infer \
  --config configs/instance_segmentation/rtmdet_ins_l_fashionpedia8_copypaste13000_1024_e24_v1.py \
  --engine models/tensorrt/shared_dual_cls_1024_fp16.engine \
  --image examples/images/demo.jpg \
  --output-json outputs/demo_prediction.json \
  --score-thr 0.15 \
  --nms-pre 300 \
  --max-per-img 80 \
  --disable-d1-fusion
