import math
import torch
import torch.nn.functional as F

def softmax_candidates(logits, class_ids, class_names, max_candidates=3):
    probs = F.softmax(logits.float(), dim=1)[0]
    order = torch.argsort(probs, descending=True).detach().cpu().tolist()
    out=[]
    for rank, idx in enumerate(order[:max_candidates], 1):
        conf = float(probs[idx].detach().cpu())
        if not math.isfinite(conf):
            raise ValueError('non-finite confidence')
        out.append({'attribute_id': str(class_ids[idx]), 'attribute_name': str(class_names[idx]), 'class_index': int(idx), 'rank': rank, 'confidence': conf})
    return out
