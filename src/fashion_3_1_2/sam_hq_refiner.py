import sys
from pathlib import Path
import numpy as np
from PIL import Image
from .asset_loader import validate_file_hash
from .coordinate_utils import is_valid_xyxy

class MockSamRefiner:
    def __init__(self, fail=False, empty=False):
        self.fail = fail
        self.empty = empty
    def refine(self, image_path, bbox, device="cpu"):
        if self.fail:
            raise RuntimeError("mock_sam_runtime_failure")
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        mask = np.zeros((h, w), dtype=bool)
        if not self.empty and is_valid_xyxy(bbox):
            x1,y1,x2,y2=[int(round(v)) for v in bbox]
            mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = True
        return {"mask": mask, "quality_score": 1.0 if mask.sum() else 0.0, "source": "mock_sam"}

class SamHQRefiner:
    def __init__(self, repo_root, checkpoint_path, checkpoint_sha256, model_type="vit_l", device="cuda", multimask_output=False):
        self.repo_root = Path(repo_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_sha256 = checkpoint_sha256
        self.model_type = model_type
        self.device = device
        self.multimask_output = multimask_output
        self.predictor = None
        self.audit = {}

    def load(self):
        validate_file_hash(self.checkpoint_path, self.checkpoint_sha256)
        sys.path = [str(self.repo_root)] + [p for p in sys.path if "sam-hq-official/seginw" not in p]
        import torch
        import segment_anything
        actual = str(Path(segment_anything.__file__).resolve())
        expected_suffix = "segment_anything/__init__.py"
        if not actual.endswith(expected_suffix) or "seginw/segment_anything" in actual:
            raise RuntimeError(f"wrong_segment_anything_import:{actual}")
        from segment_anything import SamPredictor, sam_model_registry
        model = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        hf = [k for k in model.state_dict().keys() if "hf_token" in k]
        if not hf:
            raise RuntimeError("sam_hq_hf_token_missing")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda_requested_but_unavailable")
        model = model.to(device=self.device)
        model_device = str(next(model.parameters()).device)
        if not model_device.startswith(self.device):
            raise RuntimeError(f"sam_hq_wrong_device:{model_device}")
        self.predictor = SamPredictor(model)
        self.audit = {"segment_anything_file": actual, "hf_token_keys": hf, "model_device": model_device}
        return self

    def refine(self, image_path, bbox, device=None):
        if self.predictor is None:
            self.load()
        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img)
        self.predictor.set_image(arr)
        masks, scores, logits = self.predictor.predict(box=np.asarray(bbox, dtype=np.float32), multimask_output=self.multimask_output)
        mask = masks[0].astype(bool)
        return {"mask": mask, "quality_score": float(scores[0]) if len(scores) else None, "source": "sam_hq_bbox_prompt"}
