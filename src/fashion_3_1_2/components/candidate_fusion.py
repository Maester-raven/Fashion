#!/usr/bin/env python3
import argparse, gzip, hashlib, json, math
from collections import defaultdict
from pathlib import Path

POLICY_VERSION = "route_a_live_trisource_policy_v2"
PRIORS = {"detector_v2": 1.00, "detector_v3": 1.03, "samhq": 0.92}

def read_jsonl(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    mode='rt' if str(path).endswith('.gz') else 'r'
    with opener(path, mode, encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def open_out(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, 'wt', encoding='utf-8') if str(path).endswith('.gz') else open(path, 'w', encoding='utf-8')

def area(b): return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def iou(a,b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); u=area(a)+area(b)-inter
    return inter/u if u>0 else 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--detector-v2-jsonl', required=True)
    ap.add_argument('--detector-v3-jsonl', required=True)
    ap.add_argument('--samhq-jsonl', required=True)
    ap.add_argument('--output-jsonl', required=True)
    ap.add_argument('--status-csv', required=True)
    ap.add_argument('--policy-config', required=True)
    args=ap.parse_args()
    by=defaultdict(list)
    for path in [args.detector_v2_jsonl,args.detector_v3_jsonl,args.samhq_jsonl]:
        for r in read_jsonl(path):
            by[r['parent_instance_id']].append(r)
    out=open_out(args.output_jsonl)
    status=[]
    for pid,cands in sorted(by.items()):
        scored=[]
        for r in cands:
            b=r['candidate_bbox_full_image_xyxy']; pb=r['parent_bbox_full_image_xyxy']
            rel=area(b)/max(1.0, area(pb))
            small_boost=0.03 if rel < 0.002 else (0.015 if rel < 0.01 else 0.0)
            score=float(r.get('candidate_score_normalized') or r.get('candidate_score_raw') or 0.0)*PRIORS.get(r['source_name'],1.0)+small_boost
            scored.append((score,r))
        scored.sort(key=lambda x:(x[0], -area(x[1]['candidate_bbox_full_image_xyxy']), x[1]['source_name']), reverse=True)
        kept=[]
        for score,r in scored:
            b=r['candidate_bbox_full_image_xyxy']
            if any(iou(b,k['candidate_bbox_full_image_xyxy'])>=0.985 for k in kept):
                continue
            kept.append(r | {
                'candidate_final_id': f'route_a_live_v2:{pid}:fused:{len(kept)+1:04d}',
                'candidate_score_fused': score,
                'candidate_final_rank': len(kept)+1,
                'candidate_primary_source': r['source_name'],
                'candidate_supporting_sources': [r['source_name']],
                'source_candidate_ids': [r['candidate_source_id']],
                'fusion_trigger_trace': {'source_prior': PRIORS.get(r['source_name'],1.0), 'small_box_boost_applied': area(b)/max(1.0, area(r['parent_bbox_full_image_xyxy'])) < 0.01, 'nms_iou': 0.985},
            })
            if len(kept)>=1000: break
        for r in kept: out.write(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n')
        status.append({'parent_instance_id':pid,'input_candidate_count':len(cands),'fused_candidate_count':len(kept),'fusion_status':'success' if kept else 'success_empty'})
    out.close()
    with open(args.status_csv,'w',newline='') as f:
        import csv
        w=csv.DictWriter(f, fieldnames=['parent_instance_id','input_candidate_count','fused_candidate_count','fusion_status'])
        w.writeheader(); w.writerows(status)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
