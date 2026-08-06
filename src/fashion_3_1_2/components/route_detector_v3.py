#!/usr/bin/env python3
import argparse, csv, gzip, hashlib, json, os, sys, time, traceback
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

POLICY_VERSION = "route_a_live_trisource_policy_v2"

def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for ch in iter(lambda:f.read(1024*1024), b''):
            h.update(ch)
    return h.hexdigest()

def read_manifest(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def open_out(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, 'at', encoding='utf-8') if str(path).endswith('.gz') else open(path, 'a', encoding='utf-8')

def load_done(path):
    done=set()
    p=Path(path)
    if not p.exists(): return done
    opener=gzip.open if str(p).endswith('.gz') else open
    mode='rt' if str(p).endswith('.gz') else 'r'
    with opener(p, mode, encoding='utf-8') as f:
        for line in f:
            try:
                r=json.loads(line); done.add(r.get('parent_instance_id'))
            except Exception:
                pass
    return done

def clip(b,w,h):
    x1,y1,x2,y2=[float(x) for x in b]
    x1=max(0,min(float(w),x1)); x2=max(0,min(float(w),x2))
    y1=max(0,min(float(h),y1)); y2=max(0,min(float(h),y2))
    if x2<=x1 or y2<=y1: return None
    return [x1,y1,x2,y2]

def make_model(source):
    if source == 'detector_v2':
        model=fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=2, min_size=640, max_size=1024)
        model.roi_heads.score_thresh=0.0; model.roi_heads.nms_thresh=0.9; model.roi_heads.detections_per_img=1200
    else:
        model=fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=2, min_size=(800,960,1120), max_size=1536)
        model.roi_heads.score_thresh=0.0; model.roi_heads.nms_thresh=0.9; model.roi_heads.detections_per_img=1500
        model.roi_heads.batch_size_per_image=512; model.roi_heads.positive_fraction=0.35
        model.rpn._pre_nms_top_n={'training':4000,'testing':4000}; model.rpn._post_nms_top_n={'training':2000,'testing':1500}
    return model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--parent-manifest', required=True)
    ap.add_argument('--output-jsonl', required=True)
    ap.add_argument('--status-jsonl', required=True)
    ap.add_argument('--policy-config', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-parents', type=int, default=0)
    ap.add_argument('--log-file', required=True)
    args=ap.parse_args()
    source = "detector_v3"
    policy=json.load(open(args.policy_config))
    ckpt_path=policy[source]['checkpoint']
    device=torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    log={'adapter_version':'v1','source_name':source,'started_at':now(),'checkpoint_path':ckpt_path,'device':str(device)}
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.manual_seed(3122)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(3122)
        t_load=time.time()
        model=make_model(source)
        data=torch.load(ckpt_path, map_location='cpu')
        state=data.get('model_state') or data.get('model') or data
        missing,unexpected=model.load_state_dict(state, strict=False)
        model.to(device); model.eval()
        log.update({'model_load_success':True,'load_sec':time.time()-t_load,'missing_keys_count':len(missing),'unexpected_keys_count':len(unexpected),'missing_keys_sample':list(missing)[:20],'unexpected_keys_sample':list(unexpected)[:20]})
    except Exception:
        log.update({'model_load_success':False,'traceback':traceback.format_exc()})
        Path(args.log_file).write_text(json.dumps(log, indent=2)+'\n')
        return 2
    config_hash=sha256_file(args.policy_config)
    checkpoint_hash=sha256_file(ckpt_path)
    preprocessing_hash=hashlib.sha256(json.dumps({'crop':'parent_bbox_full_image_xyxy','to_tensor':True,'source':source},sort_keys=True).encode()).hexdigest()
    done=load_done(args.output_jsonl) if args.resume else set()
    cand_f=open_out(args.output_jsonl); stat_f=open(args.status_jsonl, 'a', encoding='utf-8')
    processed=0
    for parent in read_manifest(args.parent_manifest):
        if args.max_parents and processed>=args.max_parents: break
        pid=parent['parent_instance_id']
        if pid in done:
            continue
        st={'policy_version':POLICY_VERSION,'adapter_version':'v1','source_name':source,'parent_instance_id':pid,'executed':True,'forward_completed':False,'status':'started','started_at':now()}
        try:
            im=Image.open(parent['image_path']).convert('RGB')
            iw,ih=im.size
            pb=clip(parent['parent_bbox_full_image_xyxy'], iw, ih)
            if not pb: raise RuntimeError('invalid clipped parent bbox')
            crop=im.crop((int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])))
            tensor=F.to_tensor(crop)
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            t0=time.time()
            with torch.no_grad():
                out=model([tensor.to(device)])[0]
            rt=time.time()-t0
            boxes=out['boxes'].detach().cpu().tolist()
            scores=out['scores'].detach().cpu().tolist()
            labels=out['labels'].detach().cpu().tolist()
            order=sorted(range(len(scores)), key=lambda j:scores[j], reverse=True)
            cap=1200 if source=='detector_v2' else 1500
            rows=[]
            rank=0
            for j in order:
                if int(labels[j]) != 1: continue
                cb=boxes[j]
                fb=clip([pb[0]+cb[0], pb[1]+cb[1], pb[0]+cb[2], pb[1]+cb[3]], iw, ih)
                if not fb: continue
                rank += 1
                rows.append({
                    'policy_version':POLICY_VERSION,
                    'source_name':source,
                    'split':parent.get('split',''),
                    'parent_instance_id':pid,
                    'image_id':parent.get('image_id',''),
                    'image_path':parent['image_path'],
                    'parent_bbox_full_image_xyxy':pb,
                    'candidate_source_id':f'{source}:{pid}:cand:{rank:04d}',
                    'candidate_bbox_full_image_xyxy':fb,
                    'candidate_bbox_source_crop_xyxy':[float(x) for x in cb],
                    'candidate_score_raw':float(scores[j]),
                    'candidate_score_normalized':float(scores[j]),
                    'candidate_source_rank':rank,
                    'source_runtime_config_hash':config_hash,
                    'checkpoint_hash':checkpoint_hash,
                    'preprocessing_hash':preprocessing_hash,
                    'adapter_version':'v1',
                    'adapter_command':' '.join(shlex_quote(x) for x in sys.argv),
                    'runtime_environment':{'python':sys.executable,'torch':torch.__version__,'torchvision_imported':True,'device':str(device)},
                    'coordinate_transform_trace':{'source_box_format':'crop_xyxy','source_coordinate_frame':'parent_crop_pixels','target_coordinate_frame':'full_image_xyxy','offset':[pb[0],pb[1]],'scale':[1.0,1.0]},
                })
                if rank>=cap: break
            for r in rows:
                cand_f.write(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n')
            cand_f.flush()
            st.update({'status':'success','forward_completed':True,'image_size':[iw,ih],'input_tensor_shape':list(tensor.shape),'inference_duration_sec':rt,'candidate_count_before_filtering':len(scores),'candidate_count_after_filtering':len(rows),'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1024/1024 if torch.cuda.is_available() else 0})
        except Exception:
            st.update({'status':'failed','error_message':traceback.format_exc()})
        stat_f.write(json.dumps(st, ensure_ascii=False, sort_keys=True)+'\n'); stat_f.flush()
        processed += 1
    cand_f.close(); stat_f.close()
    log.update({'completed_at':now(),'processed_parent_count':processed})
    Path(args.log_file).write_text(json.dumps(log, indent=2, sort_keys=True)+'\n')
    return 0

def shlex_quote(x):
    import shlex
    return shlex.quote(str(x))

if __name__ == '__main__':
    raise SystemExit(main())
