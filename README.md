# 多模态驱动的电商服饰细粒度语义增强与智能解析系统

## 1. 项目简介

本项目面向电商服饰场景，目标是构建从商品图片与自然语言查询出发，到服饰实例分割、局部区域定位、属性提取，再到多模态问答、商品标签生成与卖点生成的一体化智能解析系统。

整体链路可概括为：

```text
商品图片 + 自然语言查询
  → 服饰实例分割
  → 局部区域定位
  → 属性提取
  → 多模态问答 / 商品标签 / 卖点生成
```

当前 GitHub 仓库只正式整理并上传 **3.1.1 服饰实例分割** 与 **数据地基** 相关内容。3.1.2 局部区域定位、3.1.3 属性提取、3.2、3.3 与 serving 等模块目前仅保留目录结构，尚未完成。

本仓库已包含 3.1.1 的 ONNX / TensorRT 部署脚本，但不包含 checkpoint、ONNX、TensorRT engine、原始数据集、预测 JSON 和训练输出。运行部署命令前需要在本地准备模型制品、数据和兼容的 CUDA / TensorRT / MMDetection / MMCV / MMEngine 环境。

## 2. 当前进度

- 已完成：3.1.1 服饰实例分割
- 最终版本：RTMDet-Ins-L A/epoch5 + 8 类 class-specific threshold + mask overlap conflict resolver + TensorRT FP16
- 支持 8 类：`top`、`pants`、`skirt`、`outerwear`、`dress`、`shoes`、`bag`、`accessory`
- Recall：84.21%
- Precision-like：84.08%
- Mean TP IoU：85.28%
- 平均端到端单图时间：47.84 ms
- 3.1.2 / 3.1.3 / 3.2 / 3.3 / serving 尚未完成

3.1.1 的详细验收结论见 [`docs/module_3_fine_grained_vision/3.1.1_instance_segmentation/final_version.md`](docs/module_3_fine_grained_vision/3.1.1_instance_segmentation/final_version.md)。

## 3. 仓库结构

- `configs/instance_segmentation/`：3.1.1 服饰实例分割相关 RTMDet-Ins 配置。
- `configs/data_foundation/`：统一数据 schema、属性 schema 与外部数据集映射。
- `scripts/data/`：数据来源审计、Fashionpedia 类别/属性审计、FashionAI 切分与 review 辅助脚本。
- `scripts/eval/`：实例分割评估、误差分析与 TensorRT pipeline 脚本。
  - `scripts/eval/benchmark_rtmdet_dual_cls_fusion.py`：TensorRT pipeline 后处理 / 冲突消解依赖脚本。
  - `scripts/eval/run_rtmdet_dual_cls_single_mask.py`：TensorRT output 转单图实例预测依赖脚本。
- `scripts/export/`：ONNX 导出与 TensorRT engine 构建脚本。
- `docs/`：PRD、模块文档、系统设计、周报与 3.1.1 最终版本说明。
- `src/fashion_system/`：系统工程化代码骨架，包含 common、data_foundation、instance_segmentation、local_grounding、attribute_extraction、multimodal、rag_agent、serving 等预留模块；当前尚未补齐标准单图推理入口 `src/fashion_system/instance_segmentation/infer.py`。
- `data/`：本地数据目录，已被 `.gitignore` 排除；数据放置说明见 `data/README.md`。
- `models/`：本地模型制品目录，已被 `.gitignore` 排除；模型制品放置说明见 `models/README.md`。
- `outputs/`：本地输出目录，已被 `.gitignore` 排除。

## 4. 环境准备

AutoDL 正式项目环境名称为 `vibe`。进入环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vibe
```

安装精简依赖：

```bash
pip install -r requirements.txt
```

说明：

- `requirements.txt` 是从当前环境中提取的精简核心依赖，便于理解项目主要技术栈。
- `requirements.lock.txt` 是当前环境快照。
- RTMDet、MMDetection、MMCV、MMEngine、TensorRT、CUDA、PyTorch 与显卡驱动强相关，完整训练、导出和部署环境需要根据目标机器单独确认。

## 5. 数据准备

GitHub 仓库不包含原始数据集。用户需要在本地或服务器上自行准备数据，例如：

```text
data/
├── raw/
│   ├── FashionAI/
│   ├── DeepFashion2/
│   └── Fashionpedia/
├── processed/
├── manifests/
└── README.md 或 .gitkeep
```

`data/` 目录已被 `.gitignore` 排除，不会上传到 GitHub。

## 6. 模型制品准备

GitHub 仓库不包含 checkpoint、ONNX 和 TensorRT engine。建议在本地按如下结构放置模型制品：

```text
models/
├── rtmdet/
│   └── epoch_5.pth
├── rtmdet_d1/
│   └── epoch_1.pth
├── onnx/
│   └── shared_dual_cls_1024.onnx
└── tensorrt/
    └── shared_dual_cls_1024_fp16.engine
```

说明：

- `models/rtmdet/epoch_5.pth` 是本地放置路径示例，GitHub 不包含该文件。
- `models/tensorrt/shared_dual_cls_1024_fp16.engine` 是本地生成路径示例，GitHub 不包含该文件。
- 如果不使用 D1 分支，需要检查导出脚本是否支持跳过 `--d1-checkpoint`。

3.1.1 最终版本的历史服务器路径记录如下：

```text
checkpoint:
work_dirs/rtmdet_ins_l_e24_hardft_A_repeat2_headlr_v1/epoch_5.pth

TensorRT engine:
work_dirs/rtmdet_tensorrt/shared_dual_cls_1024_fp16.engine
```

这些路径仅用于记录历史最终版本，不表示相关二进制文件已包含在当前仓库中。

## 7. 当前可运行内容

当前整理版可按模板运行或检查以下内容：

- ONNX 导出
- TensorRT engine 构建
- TensorRT pipeline 推理/测速
- 单图推理 demo 入口
- PRD 指标评估
- 配置文件与数据地基脚本检查

运行前必须准备本地模型权重、Fashionpedia val600 或其他 COCO 格式数据，以及 CUDA / TensorRT / MMDetection / MMCV / MMEngine 兼容环境。

### 7.1 查看配置

```bash
ls configs/instance_segmentation
ls configs/data_foundation
```

### 7.2 ONNX / TensorRT 部署命令模板

当前仓库已包含 3 个部署脚本：

- `scripts/export/export_rtmdet_dual_cls_onnx.py`
- `scripts/export/build_rtmdet_tensorrt_engine.py`
- `scripts/eval/run_rtmdet_tensorrt_pipeline.py`

TensorRT pipeline 还依赖以下两个辅助脚本，当前仓库已补齐：

- `scripts/eval/benchmark_rtmdet_dual_cls_fusion.py`：后处理 / 冲突消解逻辑。
- `scripts/eval/run_rtmdet_dual_cls_single_mask.py`：TensorRT output 到单图实例预测的转换逻辑。

Clean-server 部署说明：

- `configs/instance_segmentation/rtmdet_ins_l_fashionpedia8_copypaste13000_1024_e24_v1.py` 已整理为 GitHub clean 版自包含配置，不再依赖旧服务器 `/root/autodl-tmp/fashion_prd/...` 绝对 `_base_` 路径。
- `scripts/export/export_rtmdet_dual_cls_onnx.py` 会自动把仓库根目录、`src/` 和 `scripts/eval/` 加入 import path，因此直接运行导出命令时不需要手动设置 `PYTHONPATH`。
- 如果用户手动运行其他脚本，仍建议先执行：`export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts/eval:${PYTHONPATH:-}"`。

```bash
# 1. 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vibe

# 2. 导出 ONNX
python scripts/export/export_rtmdet_dual_cls_onnx.py \
  --config configs/instance_segmentation/rtmdet_ins_l_fashionpedia8_copypaste13000_1024_e24_v1.py \
  --a-checkpoint models/rtmdet/epoch_5.pth \
  --d1-checkpoint models/rtmdet_d1/epoch_1.pth \
  --output models/onnx/shared_dual_cls_1024.onnx \
  --opset 17 \
  --batch-size 1

# 3. 构建 TensorRT FP16 engine
python scripts/export/build_rtmdet_tensorrt_engine.py \
  --onnx models/onnx/shared_dual_cls_1024.onnx \
  --engine models/tensorrt/shared_dual_cls_1024_fp16.engine \
  --workspace-gb 8 \
  --fp16

# 4. TensorRT pipeline 推理与测速
python scripts/eval/run_rtmdet_tensorrt_pipeline.py \
  --config configs/instance_segmentation/rtmdet_ins_l_fashionpedia8_copypaste13000_1024_e24_v1.py \
  --engine models/tensorrt/shared_dual_cls_1024_fp16.engine \
  --val-json data/fashionpedia8/annotations/val600.json \
  --data-root data/fashionpedia8 \
  --output-json outputs/val600_final_predictions.json \
  --benchmark-json outputs/final_single120_benchmark.json \
  --warmup 20 \
  --benchmark-images 120 \
  --score-thr 0.15 \
  --nms-pre 300 \
  --max-per-img 80 \
  --disable-d1-fusion

# 5. 计算 PRD 指标
python scripts/eval/evaluate_coco_instance_prd_metrics.py \
  --gt-json data/fashionpedia8/annotations/val600.json \
  --pred-segm-json outputs/val600_final_predictions.json \
  --output-json outputs/val600_prd_metrics.json \
  --min-score 0.15 \
  --match-iou 0.05 \
  --thresholds top=0.325,pants=0.4125,skirt=0.4875,outerwear=0.425,dress=0.35,shoes=0.30,bag=0.3875,accessory=0.4125 \
  --mask-iou-thr 0.275 \
  --same-contain-thr 0.75 \
  --cross-contain-thr 0.775 \
  --skirt-score-penalty 0.65 \
  --pants-score-bonus 1.18 \
  --dress-score-bonus 0.95 \
  --accessory-cloth-contain-thr 0
```

### 7.3 单图推理 demo 入口

当前已有单图入口：

```bash
python -m fashion_system.instance_segmentation.infer
```

运行前必须准备：

- `models/tensorrt/shared_dual_cls_1024_fp16.engine`
- `examples/images/demo.jpg` 或用户自己的图片
- CUDA / TensorRT / MMDetection / MMCV / MMEngine 兼容环境

`examples/images/demo.jpg` 不随仓库提供，用户需要自行放置图片。

命令模板：

```bash
PYTHONPATH=src:. python -m fashion_system.instance_segmentation.infer \
  --config configs/instance_segmentation/rtmdet_ins_l_fashionpedia8_copypaste13000_1024_e24_v1.py \
  --engine models/tensorrt/shared_dual_cls_1024_fp16.engine \
  --image examples/images/demo.jpg \
  --output-json outputs/demo_prediction.json \
  --score-thr 0.15 \
  --nms-pre 300 \
  --max-per-img 80 \
  --disable-d1-fusion
```

也可以使用 demo 脚本：

```bash
bash scripts/demo/run_single_image.sh
```

该入口会将单图临时包装为 COCO 格式，并调用现有 `scripts/eval/run_rtmdet_tensorrt_pipeline.py`。当前实现是 v0.1.1-deployable-311 的最小工程入口，不复制或重构 TensorRT pipeline 逻辑。

### 7.4 运行数据地基相关脚本

可先查看脚本帮助或脚本顶部参数定义：

```bash
python scripts/data/audit_data_sources.py --help
python scripts/data/audit_fashionpedia_categories.py --help
python scripts/data/audit_fashionpedia_attributes.py --help
```

如果某些脚本不支持 `--help`，请查看脚本顶部参数定义后再运行。

## 8. 当前仓库不包含的内容

当前 GitHub 整理版仍不包含：

- checkpoint
- ONNX
- TensorRT engine
- 原始数据集
- 预测 JSON
- 真实 demo 图片，例如 `examples/images/demo.jpg`

`src/fashion_system/instance_segmentation/infer.py` 已作为最小单图推理入口补齐；后续仍需要继续完善返回格式、可视化输出与错误处理。

## 9. 3.1.1 历史复现流程

以下内容是历史复现流程 / 服务器原路径记录。当前仓库已经补齐 ONNX / TensorRT 部署脚本，但完整运行仍需要准备本地模型制品、数据集和兼容环境。

1. 激活正式 AutoDL 环境 `vibe`。
2. 准备 checkpoint：`work_dirs/rtmdet_ins_l_e24_hardft_A_repeat2_headlr_v1/epoch_5.pth`。
3. 准备 Fashionpedia 统一 8 类 val600 验证集。
4. 使用 `scripts/export/export_rtmdet_dual_cls_onnx.py` 导出 ONNX。
5. 使用 `scripts/export/build_rtmdet_tensorrt_engine.py` 构建 TensorRT engine：`work_dirs/rtmdet_tensorrt/shared_dual_cls_1024_fp16.engine`。
6. 使用 `scripts/eval/run_rtmdet_tensorrt_pipeline.py` 运行 TensorRT pipeline。
7. 使用 `scripts/eval/evaluate_coco_instance_prd_metrics.py` 计算 PRD metrics。

## 10. GitHub 上传策略

本仓库采用轻量化 GitHub 上传策略：

- 不上传 `data/`
- 不上传 `models/`
- 不上传 `outputs/`
- 不上传 checkpoint、engine、ONNX
- 不上传 logs、`work_dirs/`、`repos/`

这些内容应保留在本地机器、AutoDL 工作目录或外部对象存储中。

## 11. 后续计划

1. 完善 v0.1.1 单图 CLI 的输出格式与可视化能力
2. 补齐最小 demo 图片与端到端示例记录
3. 增加单图入口的轻量单元测试
4. 完成 Fashionpedia 294 属性盘点与映射
5. 完成 unified schema v2
6. 完成 FashionAI 实例补全
7. 开始 3.1.2 局部区域定位
8. 开始 3.1.3 属性提取

### A-only TensorRT deployment note

If only the final A/epoch5 checkpoint is available and the D1 checkpoint is not provided, export ONNX with `--a-only`:

```bash
python scripts/export/export_rtmdet_dual_cls_onnx.py \
  --config configs/instance_segmentation/rtmdet_ins_l_fashionpedia8_copypaste13000_1024_e24_v1.py \
  --a-checkpoint models/rtmdet/epoch_5.pth \
  --output models/onnx/shared_dual_cls_1024.onnx \
  --opset 17 \
  --batch-size 1 \
  --a-only
```

For A-only TensorRT engines, run inference with `--disable-d1-fusion`. The single-image demo script already uses this mode.
