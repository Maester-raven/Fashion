# Model Asset Manifest: 3.1.1 RTMDet-Ins-L A/epoch5

- asset: `epoch_5.pth`
- type: PyTorch checkpoint
- model: RTMDet-Ins-L A/epoch5
- task: 8-class fashion instance segmentation
- release tag: `v0.1.2-real-deploy-311`
- sha256: `cb5e2ae8916568954882bfc5f5f741e9753c3bfdd969e8441c1f0e6ce3d882da`
- upload method: GitHub Release asset
- not stored in git repository: true
- ONNX/engine: not uploaded; regenerated locally from checkpoint

## Classes

The checkpoint is trained/adapted for the following 8 e-commerce fashion classes:

1. top
2. pants
3. skirt
4. outerwear
5. dress
6. shoes
7. bag
8. accessory

## Validation summary

Fashionpedia unified 8-class val600:

- GT instances: 2572
- Predictions: 2576
- TP: 2166
- FP: 410
- FN: 406
- Recall: 84.21%
- Precision-like: 84.08%
- Mean TP IoU: 85.28%

## Deployment validation

Validated with GitHub tag `v0.1.2-real-deploy-311` on a clean server:

- Clean-server clone: passed
- Clean-server config load: passed
- A-only ONNX export: passed
- ONNX checker: passed
- TensorRT FP16 engine build: passed
- Single-image demo inference: passed
- `outputs/demo_prediction.json` generated: passed

Do not commit `epoch_5.pth`, `.onnx`, or `.engine` files to the Git repository.
