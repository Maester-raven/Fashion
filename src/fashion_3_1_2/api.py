from pathlib import Path
class Fashion312Runtime:
 def __init__(self,profile='zero_one_n_functional_v1',device='cuda',asset_root=None):
  self.profile=profile
  if asset_root:
   import os;os.environ['FASHION_3_1_2_ASSET_ROOT']=str(asset_root)
  if profile=='single_hit_v1':
   from .profiles.single_hit_v1 import SingleHitBBoxMaskPipeline;self.runtime=SingleHitBBoxMaskPipeline
  else:
   from .runtime import Fashion312ZeroOneNFunctionalRuntime;self.runtime=Fashion312ZeroOneNFunctionalRuntime(device=device)
 @classmethod
 def from_config(cls,path,device='cuda',asset_root=None,profile=None):
  import yaml;c=yaml.safe_load(open(path));return cls(profile or c.get('profile','zero_one_n_functional_v1'),device,asset_root)
 def predict(self,**kw):
  if self.profile=='single_hit_v1': raise RuntimeError('single_hit_v1 uses legacy SingleHitBBoxMaskPipeline.from_config API')
  x=self.runtime.predict(**kw);x['profile']=self.profile;return x
