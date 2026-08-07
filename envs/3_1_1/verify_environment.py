from __future__ import annotations
import importlib, json, sys
rows=[]
for m in ['torch','torchvision','numpy','PIL','cv2','mmcv','mmengine','mmdet','onnx']:
    try:
        x=importlib.import_module(m); rows.append({'module':m,'ok':True,'version':getattr(x,'__version__',None),'file':getattr(x,'__file__',None)})
    except Exception as e:
        rows.append({'module':m,'ok':False,'error':repr(e)})
ops={'nms':False,'roi_align':False,'smoke':False}
try:
    import torch
    from mmcv.ops import nms, RoIAlign
    boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device='cuda')
    scores=torch.tensor([0.9,0.7],device='cuda')
    dets, inds = nms(boxes, scores, 0.5)
    roi=RoIAlign(output_size=(2,2), spatial_scale=1.0, sampling_ratio=0).cuda()
    feat=torch.randn(1,1,16,16,device='cuda')
    rois=torch.tensor([[0.,0.,0.,8.,8.]],device='cuda')
    out=roi(feat, rois)
    ops={'nms':True,'roi_align':True,'smoke':bool(out.numel()>0 and dets.numel()>0)}
except Exception as e:
    ops['error']=repr(e)
try:
    import torch
    cuda={'available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
except Exception as e:
    cuda={'error':repr(e)}
passed=all(r['ok'] for r in rows) and ops.get('nms') and ops.get('roi_align') and ops.get('smoke') and cuda.get('available')
print(json.dumps({'passed':passed,'python':sys.version,'cuda':cuda,'modules':rows,'mmcv_ops':ops},indent=2))
raise SystemExit(0 if passed else 1)
