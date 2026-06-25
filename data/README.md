# Data Directory

本目录用于本地放置数据集与中间数据，不提交到 GitHub。

推荐结构：

```text
data/
├── fashionpedia8/
│   ├── annotations/
│   │   └── val600.json
│   └── images/
├── raw/
│   ├── FashionAI/
│   ├── DeepFashion2/
│   └── Fashionpedia/
├── processed/
└── manifests/
```

说明：

* GitHub 不包含原始数据集。
* `data/fashionpedia8/annotations/val600.json` 应为 COCO 格式标注。
* `val600.json` 中 `images[].file_name` 需要能与 `--data-root data/fashionpedia8` 拼接成真实图片路径。
* 大规模数据、中间产物和缓存均由 `.gitignore` 排除。
* 若只复现 PRD 指标评估，需要准备 GT COCO JSON 和预测 JSON。
* 若运行 TensorRT pipeline，需要准备图片目录和对应 COCO 标注。
