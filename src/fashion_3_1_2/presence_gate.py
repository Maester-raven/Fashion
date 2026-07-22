class MockPresenceGate:
    def __init__(self, present=True, score=0.9, fail=False):
        self.present = present
        self.score = score
        self.fail = fail
    def predict(self, image_path, query_text):
        if self.fail:
            raise RuntimeError("mock_presence_failure")
        return {"present": bool(self.present), "score": float(self.score)}
