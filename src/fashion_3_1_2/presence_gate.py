import math
from pathlib import Path

VIEW_COUNT = 14

class MockPresenceGate:
    def __init__(self, present=True, score=0.9, fail=False):
        self.present = present
        self.score = score
        self.fail = fail
    def predict(self, image_path, query_text):
        if self.fail:
            raise RuntimeError("mock_presence_failure")
        return {"present": bool(self.present), "score": float(self.score)}


def _torch_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def _sim_summary(views, q):
    torch, _, F = _torch_modules()
    vn = F.normalize(views, dim=-1)
    qn = F.normalize(q, dim=-1)
    sims = (vn * qn[:, None, :]).sum(-1)
    sorted_s, _ = torch.sort(sims, dim=1, descending=True)
    entropy = -(torch.softmax(sims, dim=1) * torch.log_softmax(sims, dim=1)).sum(1, keepdim=True)
    return torch.cat([
        sims[:, 0:1],
        sims.max(1, keepdim=True).values,
        sims.min(1, keepdim=True).values,
        sims.mean(1, keepdim=True),
        sims.var(1, keepdim=True),
        sorted_s[:, 0:1] - sorted_s[:, 1:2],
        entropy,
    ], dim=1), sims


def _presence_g2_class():
    torch, nn, F = _torch_modules()
    class PresenceG2Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.qp = nn.Linear(dim, 256)
            self.vp = nn.Linear(dim, 256)
            self.net = nn.Sequential(
                nn.Linear(dim * 3 + 7, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        def forward(self, views, q):
            summary, _ = _sim_summary(views, q)
            qh = self.qp(q)
            vh = self.vp(views)
            att = (vh * qh[:, None, :]).sum(-1) / math.sqrt(vh.shape[-1])
            w = F.softmax(att, dim=1)
            attended = (views * w[:, :, None]).sum(1)
            x = torch.cat([q, views[:, 0, :], attended, summary], dim=1)
            return self.net(x).squeeze(1)
    return PresenceG2Model


def load_presence_g2_checkpoint_strict(checkpoint_path, device="cpu"):
    torch, _, _ = _torch_modules()
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(str(checkpoint_path), map_location=device)
    model_state = state.get("model_state", state) if isinstance(state, dict) else state
    dim = int(model_state["qp.weight"].shape[1])
    model_cls = _presence_g2_class()
    model = model_cls(dim)
    result = model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"dim": dim, "strict": True, "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys), "checkpoint_keys": len(model_state)}


class StrictPresenceGate:
    def __init__(self, checkpoint_path, threshold=0.35, temperature=1.5, device="cpu"):
        self.checkpoint_path = checkpoint_path
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.device = device
        self.model = None
        self.audit = None
    def load(self):
        self.model, self.audit = load_presence_g2_checkpoint_strict(self.checkpoint_path, self.device)
        return self
    def predict_from_features(self, views, query):
        torch, _, _ = _torch_modules()
        if self.model is None:
            self.load()
        with torch.no_grad():
            views_t = torch.as_tensor(views, dtype=torch.float32, device=self.device)
            query_t = torch.as_tensor(query, dtype=torch.float32, device=self.device)
            if views_t.ndim == 2:
                views_t = views_t.unsqueeze(0)
            if query_t.ndim == 1:
                query_t = query_t.unsqueeze(0)
            logit = self.model(views_t, query_t) / self.temperature
            score = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)[0]
        return {"present": bool(score >= self.threshold), "score": float(score)}
