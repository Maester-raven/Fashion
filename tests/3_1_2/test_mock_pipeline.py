import tempfile
from PIL import Image
from fashion_3_1_2 import SingleHitBBoxMaskPipeline
p=tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
Image.new("RGB",(80,80),"white").save(p)
cfg="configs/3_1_2/single_hit_bbox_mask_mvp_v1.yaml"
pipe=SingleHitBBoxMaskPipeline.from_config(cfg, device="cpu", adapter_mode="mock", mock_bbox=[10,10,40,50])
r=pipe.predict(p,"find pocket")
assert r["status"]=="success" and len(r["instances"])==1
pipe=SingleHitBBoxMaskPipeline.from_config(cfg, device="cpu", adapter_mode="mock", mock_present=False)
assert pipe.predict(p,"find pocket")["status"]=="empty"
pipe=SingleHitBBoxMaskPipeline.from_config(cfg, device="cpu", adapter_mode="mock", mock_selector_empty=True)
assert pipe.predict(p,"find pocket")["status"]=="present_but_no_candidate"
pipe=SingleHitBBoxMaskPipeline.from_config(cfg, device="cpu", adapter_mode="mock", mock_sam_fail=True)
assert pipe.predict(p,"find pocket")["instances"][0]["mask_source"]=="coarse_bbox_runtime_fallback"
