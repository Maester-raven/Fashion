import importlib,json,sys
expected={'torch':'2.1.2+cu121','torchvision':'0.16.2+cu121','mmcv':'2.1.0','mmengine':'0.10.7','mmdet':'3.3.0','tensorrt':'10.13.3.9','onnx':'1.16.2','numpy':'1.26.4','cv2':'4.11.0','transformers':'4.32.0','timm':'1.0.27'}
versions={};origins={};fail=[]
for n,v in expected.items():
 try:
  m=importlib.import_module(n);versions[n]=getattr(m,'__version__',None);origins[n]=getattr(m,'__file__',None)
  if versions[n]!=v:fail.append([n,'version',versions[n],v])
 except Exception as e:fail.append([n,'import',repr(e)])
try:
 import torch
 from mmcv.ops import nms
 import tensorrt as trt
 compiled={'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,'mmcv_nms_callable':callable(nms),'tensorrt_logger':bool(trt.Logger(trt.Logger.WARNING))}
 if not all([compiled['cuda_available'],compiled['torch_cuda']=='12.1',compiled['mmcv_nms_callable'],compiled['tensorrt_logger']]):fail.append(['compiled_ops',compiled])
except Exception as e:compiled={'error':repr(e)};fail.append(['compiled_ops',repr(e)])
print(json.dumps({'passed':not fail,'python':sys.version,'versions':versions,'origins':origins,'compiled':compiled,'failures':fail}));raise SystemExit(bool(fail))
