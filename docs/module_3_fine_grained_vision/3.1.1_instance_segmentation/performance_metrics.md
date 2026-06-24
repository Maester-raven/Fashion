# Fine-Grained Vision Performance Metrics

本文档定义 `3.1 细粒度视觉基础模块` 的性能测试口径，并对应
`scripts/perf_eval_finegrained.py` 的输出字段。

## 1. 服饰实例分割

- 核心指标：`segmentation.mean_iou`
- 计算口径：以标注实例为基准，在同图预测实例中寻找最佳匹配；当 GT 和预测都带有
  `category` 时优先匹配同类实例。优先使用 mask polygon IoU，缺少 mask 时退化为
  bbox IoU。
- 达标线：`mean_iou >= 0.85`
- 响应时间：`latency.stages.segmentation.p95 <= 50 ms`
- 辅助字段：
  - `pass_rate_at_threshold`：IoU 不低于 `0.85` 的 GT 实例占比
  - `matched_instances`：成功找到几何匹配的 GT 实例数

## 2. 语言引导的局部区域定位

- 核心指标：`region_localization.accuracy`
- 计算口径：以 GT 局部区域为基准，在同图预测区域中寻找相同 `attribute_id`
  或相同 `target_part` 的最佳匹配；IoU 不低于 `region_iou_threshold`
  时记为定位正确。
- 默认正确阈值：`region_iou_threshold = 0.50`
- 达标线：`accuracy >= 0.92`
- 响应时间：`latency.stages.localization.p95 <= 30 ms`
- 支持区域：领口、袖口、下摆、口袋、肩部、腰部、图案、装饰等服饰局部。

## 3. 细粒度属性提取

- 核心指标：`attribute_recognition.accuracy`
- 计算口径：按同图、同 `attribute_id` 比较 top-1 属性值，默认忽略
  `Invisible` 标签。
- 达标线：`accuracy >= 0.88`
- 响应时间：`latency.stages.attribute.p95 <= 20 ms`
- 属性范围：当前 taxonomy 已覆盖 FashionAI 8 类属性，可继续扩展到
  面料、工艺、设计等 14 大类 200+ 细粒度属性。

## 4. 响应时间

- 核心指标：`latency.stages.<stage>.p95`
- 统计口径：按样本记录耗时，输出 `mean / p50 / p95 / p99 / min / max`
  和阈值通过率。
- 支持阶段：
  - `segmentation`：实例分割耗时，阈值 50 ms
  - `localization`：局部区域定位耗时，阈值 30 ms
  - `attribute`：属性提取耗时，阈值 20 ms
  - `total`：端到端总耗时，可通过 `--total-time-ms` 设置阈值

## 5. 远端运行示例

远端项目路径：

```bash
cd /root/autodl-tmp/fashion_prd
```

打印指标定义：

```bash
/root/miniconda3/envs/vibe/bin/python scripts/perf_eval_finegrained.py --print-metric-defs
```

用 DeepFashion2 标注文件做分割指标 smoke：

```bash
/root/miniconda3/envs/vibe/bin/python scripts/perf_eval_finegrained.py \
  --gt-jsonl data_interim/deepfashion2_instances.jsonl \
  --pred-jsonl data_interim/deepfashion2_instances.jsonl \
  --max-records 5 \
  --bbox-iou-only \
  --out-report logs/perf_smoke_report.json
```

评估已有预测结果：

```bash
/root/miniconda3/envs/vibe/bin/python scripts/perf_eval_finegrained.py \
  --gt-jsonl data_interim/your_ground_truth.jsonl \
  --pred-jsonl data_interim/your_predictions.jsonl \
  --latency-jsonl logs/your_latency.jsonl \
  --out-report logs/perf_report.json
```

按样本运行真实推理命令并测总耗时：

```bash
/root/miniconda3/envs/vibe/bin/python scripts/perf_eval_finegrained.py \
  --cases-jsonl data_interim/perf_cases.jsonl \
  --command-template 'python scripts/your_infer.py --image "{image_path}" --query "{query}" --out "{out_path}"' \
  --out-latency-jsonl logs/perf_latency.jsonl \
  --total-time-ms 100 \
  --out-report logs/perf_command_report.json
```

当某类 GT 或耗时日志暂未提供时，报告会显示 `not_available`，不会把缺失项当成 0 分。
