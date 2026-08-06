import torch
from .constants import LEGACY_TASK_TO_C2_GROUP
from .score_adapter import softmax_candidates

LEGACY_C2_ATTRIBUTE_IDS = {
    'neckline': ['182', '183', '185', '186', '187', '189', '190', '200'],
    'pocket': ['218', '219', '222', '223', '224'],
}

class AttributeRuntime:
    def __init__(self, loaded):
        self.loaded = loaded
    def encode(self, local_t, context_t):
        with torch.inference_mode():
            return self.loaded.attr_model.encode(local_t.to(self.loaded.device), context_t.to(self.loaded.device))
    def predict_task(self, feat, task_id, max_candidates=3):
        with torch.inference_mode():
            if task_id in self.loaded.mapping:
                logits = self.loaded.attr_model.forward_new(feat, task_id)
                m = self.loaded.mapping[task_id]
                return softmax_candidates(logits, m['class_ids'], m['class_names'], max_candidates=max_candidates), logits.detach().float().cpu().numpy()[0].tolist()
            if task_id in LEGACY_TASK_TO_C2_GROUP:
                group = LEGACY_TASK_TO_C2_GROUP[task_id]
                logits = self.loaded.attr_model.forward_c2(feat, group)
                classes = list(self.loaded.old_groups[group])
                class_names = [str(c.get('attribute_name', c.get('name', c))) if isinstance(c, dict) else str(c) for c in classes]
                class_ids = LEGACY_C2_ATTRIBUTE_IDS.get(task_id)
                if class_ids is None:
                    class_ids = [str(c.get('attribute_id', c.get('id', i))) if isinstance(c, dict) else str(i) for i,c in enumerate(classes)]
                if len(class_ids) != len(class_names):
                    raise ValueError(f'legacy class id/name length mismatch for {task_id}')
                return softmax_candidates(logits, class_ids, class_names, max_candidates=max_candidates), logits.detach().float().cpu().numpy()[0].tolist()
            return [], []
