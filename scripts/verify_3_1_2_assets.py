import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--asset-root',default='checkpoints/3_1_2');a=p.parse_args();r=Path(a.asset_root);req=['presence_g2/best_model.pth','route_a/detector_v2.pth','route_a/detector_v3.pth','fashionclip/model.safetensors','smoke_r1/best_model.pth','sam_hq/sam_hq_vit_l.pth'];missing=[x for x in req if not (r/x).is_file()];print(json.dumps({'passed':not missing,'missing':missing}));raise SystemExit(bool(missing))
