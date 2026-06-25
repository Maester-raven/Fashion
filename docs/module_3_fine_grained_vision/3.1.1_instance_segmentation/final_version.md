# 3.1.1 服饰实例分割最终版本

## 最终交付版本

- 最终模型：**RTMDet-Ins-L A/epoch5**
- 推理后端：**TensorRT FP16**
- TensorRT engine 路径：`work_dirs/rtmdet_tensorrt/shared_dual_cls_1024_fp16.engine`
- checkpoint 路径：`work_dirs/rtmdet_ins_l_e24_hardft_A_repeat2_headlr_v1/epoch_5.pth`

上述路径仅用于版本追溯；engine 与 checkpoint 均属于大体积运行产物，不纳入 Git 仓库。

## 正式项目环境

AutoDL 正式项目环境名称为 `vibe`。复现前统一执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vibe
```

## 验证集

- 数据集：Fashionpedia 统一 8 类 `val600`
- 图像数：600
- GT 实例数：2572

## 最终指标

| 指标 | 结果 |
| --- | ---: |
| TP | 2166 |
| FP | 410 |
| FN | 406 |
| Recall | 84.21% |
| Precision-like | 84.08% |
| Mean TP IoU | 85.28% |
| 平均端到端单图时间 | 47.84 ms |

## 验收结论

PRD 3.1.1 原始要求为：

- mask IoU ≥ 0.85
- 平均单图时间 ≤ 50 ms

当前版本 Mean TP IoU 为 85.28%，平均端到端单图时间为 47.84 ms，因此**通过 3.1.1 原始验收**。

补充质量目标要求 Recall、Precision-like、IoU 均 ≥ 85%。当前 Recall 84.21%、Precision-like 84.08%，尚未完全达到该补充目标；后续应优先降低 FN 与 FP，同时保持 IoU 和端到端时延不退化。

## 8. 当前 GitHub 整理版说明

当前 GitHub 整理版已保留 3.1.1 最终模型说明、配置文件、评估脚本和数据地基相关脚本，并已补齐 3 个部署脚本：

* scripts/export/export_rtmdet_dual_cls_onnx.py
* scripts/export/build_rtmdet_tensorrt_engine.py
* scripts/eval/run_rtmdet_tensorrt_pipeline.py

但仓库仍不包含 checkpoint、ONNX、TensorRT engine、原始数据集和训练输出。完整运行前需要在本地准备模型制品、数据集和兼容的 CUDA/TensorRT/MMDetection 环境。
