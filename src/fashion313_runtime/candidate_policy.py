from .constants import FAMILY_TO_TASKS, FORMAL_SUPPORTED_FAMILIES

class CandidatePolicy:
    name = 'transparent_softmax_top3_baseline_v1'
    version = 'v1'
    is_final = False
    def __init__(self, max_candidates_per_task=3):
        self.max_candidates_per_task = int(max_candidates_per_task)
    def active_tasks(self, family):
        return list(FAMILY_TO_TASKS.get(family, [])) if family in FORMAL_SUPPORTED_FAMILIES else []
