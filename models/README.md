# Models Directory

本目录用于本地放置模型权重、ONNX 和 TensorRT engine，不提交到 GitHub。

推荐结构：

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

* GitHub 不包含 `.pth`、`.onnx`、`.engine`。
* `models/rtmdet/epoch_5.pth` 对应 3.1.1 最终 A/epoch5 checkpoint 的本地示例路径。
* `models/rtmdet_d1/epoch_1.pth` 是历史 D1 分支示例路径；最终部署默认 `--disable-d1-fusion`，是否需要 D1 checkpoint 取决于导出脚本参数。
* TensorRT engine 与 GPU 架构、CUDA、TensorRT 版本强相关，建议在目标机器本地由 ONNX 重新构建。
* 原始历史路径：

  * checkpoint: `work_dirs/rtmdet_ins_l_e24_hardft_A_repeat2_headlr_v1/epoch_5.pth`
  * TensorRT engine: `work_dirs/rtmdet_tensorrt/shared_dual_cls_1024_fp16.engine`
