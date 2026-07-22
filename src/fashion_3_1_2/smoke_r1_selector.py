class MockSmokeR1Selector:
    def __init__(self, bbox=None, score=0.8, candidate_id="mock_candidate", source="mock", fail=False):
        self.bbox = bbox
        self.score = score
        self.candidate_id = candidate_id
        self.source = source
        self.fail = fail
    def select(self, image_path, query_text):
        if self.fail:
            raise RuntimeError("mock_selector_failure")
        if self.bbox is None:
            return None
        return {"bbox": [float(v) for v in self.bbox], "score": float(self.score), "candidate_id": self.candidate_id, "source": self.source, "selector_name": "smoke_r1_top1"}
