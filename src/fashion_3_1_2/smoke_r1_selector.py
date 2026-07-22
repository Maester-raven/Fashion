from pathlib import Path

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


def _torch_modules():
    import torch
    import torch.nn as nn
    return torch, nn


def _set_ranker_class():
    torch, nn = _torch_modules()
    class SetRanker(nn.Module):
        def __init__(self, input_dim, hidden=256, heads=4, layers=2, dropout=0.1):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
            enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=512, dropout=dropout, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.rank_head = nn.Linear(hidden, 1)
            self.keep_head = nn.Linear(hidden, 1)
            self.count_head = nn.Linear(hidden, 5)
        def forward(self, x):
            z = self.proj(x.unsqueeze(0))
            z = self.encoder(z).squeeze(0)
            rank = self.rank_head(z).squeeze(-1)
            keep = self.keep_head(z).squeeze(-1)
            pooled = z.mean(dim=0)
            count = self.count_head(pooled)
            return {"rank": rank, "keep": keep, "count": count}
    return SetRanker


def infer_set_ranker_config(state_dict):
    input_dim = int(state_dict["proj.0.weight"].shape[1])
    hidden = int(state_dict["proj.0.weight"].shape[0])
    layer_ids = sorted({int(k.split(".")[2]) for k in state_dict if k.startswith("encoder.layers.") and k.split(".")[2].isdigit()})
    layers = max(layer_ids) + 1 if layer_ids else 2
    return {"input_dim": input_dim, "hidden_dim": hidden, "heads": 4, "layers": layers, "dropout": 0.1}


def load_smoke_r1_checkpoint_strict(checkpoint_path, device="cpu", model_config=None):
    torch, _ = _torch_modules()
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(str(checkpoint_path), map_location=device)
    model_state = state.get("model_state", state) if isinstance(state, dict) else state
    cfg = dict(infer_set_ranker_config(model_state))
    if model_config:
        cfg.update({k: model_config[k] for k in ["input_dim", "hidden_dim", "heads", "layers", "dropout"] if k in model_config})
    cls = _set_ranker_class()
    model = cls(int(cfg["input_dim"]), hidden=int(cfg["hidden_dim"]), heads=int(cfg["heads"]), layers=int(cfg["layers"]), dropout=float(cfg["dropout"]))
    result = model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"strict": True, "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys), "checkpoint_keys": len(model_state), **cfg}


class StrictSmokeR1Selector:
    def __init__(self, checkpoint_path, device="cpu", model_config=None):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model_config = model_config or {}
        self.model = None
        self.audit = None
    def load(self):
        self.model, self.audit = load_smoke_r1_checkpoint_strict(self.checkpoint_path, self.device, self.model_config)
        return self
    def score_feature_matrix(self, features):
        torch, _ = _torch_modules()
        if self.model is None:
            self.load()
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            out = self.model(x)
            return out["rank"].detach().cpu().numpy()
