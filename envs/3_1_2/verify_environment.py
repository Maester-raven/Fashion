from __future__ import annotations
import importlib, json, sys
mods=['torch','torchvision','numpy','PIL','cv2','yaml','transformers','tokenizers','safetensors','timm','scipy','pycocotools','fashion_3_1_2','fashion_3_1_2.api','fashion_3_1_2.asset_loader','fashion_3_1_2.runtime.constraints','fashion_3_1_2.components.mask_codec']
rows=[]
for m in mods:
    try:
        x=importlib.import_module(m); rows.append({'module':m,'ok':True,'version':getattr(x,'__version__',None),'file':getattr(x,'__file__',None)})
    except Exception as e:
        rows.append({'module':m,'ok':False,'error':repr(e)})
try:
    import torch
    cuda={'available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
except Exception as e: cuda={'error':repr(e)}
passed=all(r['ok'] for r in rows) and cuda.get('available')
print(json.dumps({'passed':passed,'python':sys.version,'cuda':cuda,'modules':rows},indent=2))
raise SystemExit(0 if passed else 1)
