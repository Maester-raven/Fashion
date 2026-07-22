import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .coordinate_utils import clip_xyxy
from .mask_utils import encode_binary_mask
from .schemas import LIMITATIONS, validate_result
from .status import Status

def coarse_mask(image_path, bbox):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mask = np.zeros((h, w), dtype=bool)
    b = clip_xyxy(bbox, w, h)
    if b:
        x1,y1,x2,y2=[int(round(v)) for v in b]
        mask[y1:y2, x1:x2] = True
    return mask

class Pipeline:
    def __init__(self, presence_gate, selector, sam_refiner, config=None, device="cuda"):
        self.presence_gate = presence_gate
        self.selector = selector
        self.sam_refiner = sam_refiner
        self.config = config or {}
        self.device = device

    def predict(self, parent_image_path, query_text):
        p = Path(parent_image_path)
        if not p.exists():
            return self._error(query_text, Status.PARENT_IMAGE_MISSING, "parent_image_missing", f"missing image: {p}")
        if not str(query_text).strip():
            return self._error(query_text, Status.EMPTY_QUERY, "empty_query", "query_text is empty")
        try:
            pres = self.presence_gate.predict(str(p), query_text)
        except Exception as e:
            return self._error(query_text, Status.PRESENCE_RUNTIME_FAILURE, "presence_runtime_failure", repr(e))
        if not pres.get("present"):
            return {"query_text": query_text, "presence": False, "status": Status.EMPTY.value, "instances": [], "limitations": LIMITATIONS}
        try:
            cand = self.selector.select(str(p), query_text)
        except Exception as e:
            return self._error(query_text, Status.SELECTOR_RUNTIME_FAILURE, "selector_runtime_failure", repr(e), presence=True)
        if not cand:
            return {"query_text": query_text, "presence": True, "status": Status.PRESENT_BUT_NO_CANDIDATE.value, "instances": [], "presence_score": pres.get("score"), "limitations": LIMITATIONS}
        img = Image.open(p).convert("RGB")
        bbox = clip_xyxy(cand["bbox"], img.width, img.height)
        if not bbox:
            return self._error(query_text, Status.CANDIDATE_BBOX_INVALID, "candidate_bbox_invalid", "invalid candidate bbox", presence=True)
        status = Status.SUCCESS.value
        mask_source = "sam_hq_bbox_prompt"
        precise = False
        try:
            refined = self.sam_refiner.refine(str(p), bbox, device=self.device)
            mask = refined["mask"]
            if mask is None or tuple(mask.shape[:2]) != (img.height, img.width) or int(mask.sum()) == 0:
                status = Status.SUCCESS_WITH_COARSE_MASK_FALLBACK.value
                mask_source = "coarse_bbox_runtime_fallback"
                refined = {"mask": coarse_mask(str(p), bbox), "quality_score": None}
        except Exception:
            status = Status.SUCCESS_WITH_COARSE_MASK_FALLBACK.value
            mask_source = "coarse_bbox_runtime_fallback"
            refined = {"mask": coarse_mask(str(p), bbox), "quality_score": None}
        inst = {
            "bbox": [float(v) for v in bbox],
            "mask_rle": encode_binary_mask(refined["mask"]),
            "presence_score": pres.get("score"),
            "selector_score": cand.get("score"),
            "mask_quality_score": refined.get("quality_score"),
            "candidate_id": cand.get("candidate_id", ""),
            "candidate_source": cand.get("source", ""),
            "selector_name": cand.get("selector_name", "smoke_r1_top1"),
            "mask_source": mask_source,
            "precise_mask_guaranteed": precise,
        }
        result = {"query_text": query_text, "presence": True, "status": status, "instances": [inst], "limitations": LIMITATIONS}
        validate_result(result)
        return result

    def _error(self, query_text, status, code, message, presence=False):
        return {"query_text": query_text, "presence": presence, "status": status.value, "instances": [], "error_code": code, "error_message": message, "limitations": LIMITATIONS}

def save_json(result, path):
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def save_overlay(image_path, result, path):
    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    for inst in result.get("instances", []):
        b = inst["bbox"]
        d.rectangle(b, outline=(255, 220, 0, 255), width=4)
    img.save(path)
