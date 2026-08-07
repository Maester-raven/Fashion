from __future__ import annotations
import importlib, json, sys
rows=[]
mods=['torch','torchvision','numpy','PIL','cv2','yaml','timm','fashion313_runtime','fashion313_runtime.preprocessing','fashion313_runtime.runtime']
for m in mods:
    try:
        x=importlib.import_module(m); rows.append({'module':m,'ok':True,'version':getattr(x,'__version__',None),'file':getattr(x,'__file__',None)})
    except Exception as e: rows.append({'module':m,'ok':False,'error':repr(e)})
smoke={'cuda_tensor':False,'g1_preprocessing':False}
try:
    import numpy as np, torch
    from fashion313_runtime.preprocessing import make_tensors
    image=np.zeros((64,64,3),dtype=np.uint8); image[16:48,16:48]=180
    mask=np.zeros((64,64),dtype=bool); mask[20:44,22:46]=True
    parent=np.zeros((64,64),dtype=bool); parent[8:56,8:56]=True
    x,y,meta=make_tensors(image,mask,parent)
    smoke={'cuda_tensor':torch.cuda.is_available() and torch.ones(1,device='cuda').item()==1.0,'g1_preprocessing':bool(x.shape[-1]==518 and y.shape[-1]==518 and meta['target_area']>0),'local_shape':list(x.shape),'context_shape':list(y.shape),'meta':meta}
except Exception as e: smoke['error']=repr(e)
try:
    import torch
    cuda={'available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
except Exception as e: cuda={'error':repr(e)}
passed=all(r['ok'] for r in rows) and smoke.get('cuda_tensor') and smoke.get('g1_preprocessing') and cuda.get('available')
print(json.dumps({'passed':passed,'python':sys.version,'cuda':cuda,'modules':rows,'gpu_preprocessing_smoke':smoke},indent=2))
raise SystemExit(0 if passed else 1)
