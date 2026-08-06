import numpy as np
import torch
import torch.nn.functional as F
from .preprocessing import geometry_row

class RegionFamilyRuntime:
    def __init__(self, loaded):
        self.loaded = loaded
    def predict(self, feat, meta):
        row = geometry_row(meta)
        z, v = self.loaded.rf.normalized_geometry(row, self.loaded.geom_mean, self.loaded.geom_std)
        x = np.concatenate([feat.detach().float().cpu().numpy()[0], z, v]).astype(np.float32)
        with torch.inference_mode():
            scope_logits, local_logits = self.loaded.rf_model(torch.from_numpy(x).unsqueeze(0).to(self.loaded.device))
            scope_prob = F.softmax(scope_logits.float(), dim=1)[0]
            local_prob = F.softmax(local_logits.float(), dim=1)[0]
        scope_idx = int(torch.argmax(scope_prob).item())
        if scope_idx == 0:
            fam = 'garment_instance'; conf = float(scope_prob[0].detach().cpu())
        else:
            li = int(torch.argmax(local_prob).item())
            fam = self.loaded.rf_mapping['index_to_local_family'][str(li)]
            conf = float((scope_prob[1] * local_prob[li]).detach().cpu())
        return {'id': fam, 'name': fam, 'confidence': conf, 'scope_logits': scope_logits.detach().float().cpu().numpy()[0].tolist(), 'local_logits': local_logits.detach().float().cpu().numpy()[0].tolist()}
