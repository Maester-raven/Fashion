import time
from .mask_utils import load_image, load_mask
from .preprocessing import make_tensors
from .model_loader import LoadedModels
from .region_family_runtime import RegionFamilyRuntime
from .attribute_runtime import AttributeRuntime
from .candidate_policy import CandidatePolicy
from .schemas import validate_prediction
from .exceptions import ParentMaskRequiredError

class Fashion313Runtime:
    def __init__(self, attribute_checkpoint, region_family_checkpoint, project_root=None, device='cuda', expected_attribute_sha=None, expected_region_family_sha=None, max_candidates_per_task=3):
        self.loaded = LoadedModels(project_root, attribute_checkpoint, region_family_checkpoint, device, expected_attribute_sha, expected_region_family_sha)
        self.region_family_runtime = RegionFamilyRuntime(self.loaded)
        self.attribute_runtime = AttributeRuntime(self.loaded)
        self.candidate_policy = CandidatePolicy(max_candidates_per_task=max_candidates_per_task)
        self.device = str(self.loaded.device)
    def predict(self, image, target_mask, parent_mask=None):
        t0=time.perf_counter(); warnings=[]
        arr = load_image(image)
        tm = load_mask(target_mask, arr.shape, 'target_mask')
        pm = load_mask(parent_mask, arr.shape, 'parent_mask') if parent_mask is not None else None
        local_t, context_t, meta = make_tensors(arr, tm, pm)
        t1=time.perf_counter()
        feat = self.attribute_runtime.encode(local_t.unsqueeze(0), context_t.unsqueeze(0))
        t2=time.perf_counter()
        rf = self.region_family_runtime.predict(feat, meta)
        t3=time.perf_counter()
        active = self.candidate_policy.active_tasks(rf['id'])
        if rf['id'] != 'garment_instance' and active and pm is None:
            warnings.append('parent_mask_missing_for_predicted_local_family; returning no candidates')
            active = []
        predictions=[]
        for task in active:
            cands, logits = self.attribute_runtime.predict_task(feat, task, self.candidate_policy.max_candidates_per_task)
            predictions.append({'task_id': task, 'candidate_count': len(cands), 'candidates': cands})
        t4=time.perf_counter()
        out = {'status':'ok', 'region_family': {'id': rf['id'], 'confidence': rf['confidence']}, 'active_tasks': active, 'predictions': predictions, 'runtime_metadata': {'candidate_policy': self.candidate_policy.name, 'candidate_policy_version': self.candidate_policy.version, 'runtime_version': '0.1.0rc1', 'preprocessing_backend': 'g1_gpu', 'baseline_policy_is_final': self.candidate_policy.is_final, 'device': self.device}, 'timing_ms': {'preprocessing': (t1-t0)*1000, 'attribute_model': (t2-t1)*1000, 'region_family': (t3-t2)*1000, 'score_adapter_and_candidate_policy': (t4-t3)*1000, 'full_runtime': (t4-t0)*1000}, 'warnings': warnings}
        validate_prediction(out)
        return out
