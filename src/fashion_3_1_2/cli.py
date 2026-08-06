import argparse,json
from .api import Fashion312Runtime
def main():
 p=argparse.ArgumentParser();p.add_argument('--image');p.add_argument('--parent-crop');p.add_argument('--parent-bbox',nargs=4,type=float);p.add_argument('--query',required=True);p.add_argument('--output',required=True);p.add_argument('--profile',default='zero_one_n_functional_v1',choices=['zero_one_n_functional_v1','single_hit_v1']);p.add_argument('--device',default='cuda');p.add_argument('--overlay-output');p.add_argument('--asset-root');a=p.parse_args()
 if a.profile=='single_hit_v1': raise SystemExit('Use scripts/inference/infer_3_1_2_single_hit_bbox_mask.py for rollback profile')
 r=Fashion312Runtime(profile=a.profile,device=a.device,asset_root=a.asset_root);x=r.predict(image_path=a.image,parent_bbox=a.parent_bbox,parent_crop_path=a.parent_crop,query_text=a.query);open(a.output,'w').write(json.dumps(x,indent=2,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
