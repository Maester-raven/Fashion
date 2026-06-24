# 多模态驱动的电商服饰细粒度语义增强与智能解析系统

本项目面向电商服饰场景，建设从数据地基、细粒度视觉理解到多模态检索、智能解析与服务部署的一体化系统。当前仓库仅正式整理并提交 **3.1.1 服饰实例分割**及其数据地基相关内容，其余模块为后续迭代预留。

## 当前进度

- 已完成：3.1.1 服饰实例分割。
- 最终版本：**RTMDet-Ins-L A/epoch5 + TensorRT FP16**。
- 支持 8 类：`top`、`pants`、`skirt`、`outerwear`、`dress`、`shoes`、`bag`、`accessory`。
- 最终指标：Recall **84.21%**，Precision-like **84.08%**，Mean TP IoU **85.28%**，平均端到端单图时间 **47.84 ms**。
- 预留但尚未完成：3.1.2 局部定位、3.1.3 属性抽取、3.2、3.3 及 `deploy/serving` 相关能力。

3.1.1 的详细验收结论见 [`docs/module_3_fine_grained_vision/3.1.1_instance_segmentation/final_version.md`](docs/module_3_fine_grained_vision/3.1.1_instance_segmentation/final_version.md)。

## 仓库结构

- `configs/instance_segmentation/`：RTMDet-Ins 配置。
- `configs/data_foundation/`：统一数据与属性 schema、外部数据集映射。
- `src/fashion_system/`：系统模块代码骨架；当前仅为后续工程化预留。
- `scripts/data/`：数据审计、规范化与 review 辅助脚本。
- `scripts/eval/`：实例分割评测与误差分析脚本。
- `scripts/export/`：模型导出与部署构建脚本目录。
- `docs/`：PRD、模块文档、系统设计与周报目录。
- `data/`、`models/`、`outputs/`：本地运行目录，内容不会提交到 Git。

## 环境说明

`requirements.txt` 仅列出从当前锁定环境中筛选出的核心运行与评测依赖，便于快速理解技术栈；完整、可追溯的环境快照以 `requirements.lock.txt` 为准。CUDA、PyTorch 和 TensorRT 需要结合目标机器的驱动与运行时兼容性安装，不能仅依赖通用 CPU 环境直接复现。

AutoDL 正式项目环境名称为 `vibe`。复现前统一执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vibe
```

## 数据与模型安全

原始/中间/统一数据、模型权重、TensorRT engine、训练输出、日志和第三方仓库均由 `.gitignore` 排除，不应提交到远端仓库。文档中出现的 checkpoint 和 engine 路径仅用于记录最终版本，不代表这些二进制文件包含在本仓库中。
