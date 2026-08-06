import os
from pathlib import Path
from ..asset_loader import load_config, expand_path
from ..presence_gate import MockPresenceGate
from ..smoke_r1_selector import MockSmokeR1Selector
from ..sam_hq_refiner import MockSamRefiner, SamHQRefiner
from ..pipeline import Pipeline, save_json, save_overlay

class SingleHitBBoxMaskPipeline(Pipeline):
    @classmethod
    def from_config(cls, config_path, device="cuda", model_root=None, adapter_mode="mock", mock_bbox=None, mock_present=True, mock_selector_empty=False, mock_sam_fail=False):
        cfg = load_config(config_path)
        if adapter_mode == "mock":
            presence = MockPresenceGate(present=mock_present)
            selector = MockSmokeR1Selector(bbox=None if mock_selector_empty else (mock_bbox or [10, 10, 60, 60]))
            sam = MockSamRefiner(fail=mock_sam_fail)
            return cls(presence, selector, sam, cfg, device=device)
        if adapter_mode == "sam_hq_smoke":
            presence = MockPresenceGate(present=mock_present)
            selector = MockSmokeR1Selector(bbox=None if mock_selector_empty else (mock_bbox or [10, 10, 60, 60]), source="demo_adapter")
            repo = os.environ.get("SAM_HQ_REPO_ROOT") or str(expand_path(cfg["sam_hq"]["repo"], model_root=model_root))
            ckpt = expand_path(cfg["sam_hq"]["checkpoint"], model_root=model_root)
            sam = SamHQRefiner(repo, ckpt, cfg["sam_hq"]["checkpoint_sha256"], cfg["sam_hq"]["model_type"], device=device, multimask_output=cfg["sam_hq"]["multimask_output"])
            return cls(presence, selector, sam, cfg, device=device)
        raise ValueError(f"unsupported adapter_mode: {adapter_mode}")
