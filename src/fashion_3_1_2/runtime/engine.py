#!/usr/bin/env python3
import gzip,hashlib,importlib.util,json,math,os,subprocess,tempfile,time,uuid,sys
from pathlib import Path
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel,CLIPProcessor
REPO=Path(__file__).resolve().parents[3]
ENV=Path(sys.executable)
ASSET_ROOT=Path(os.environ.get('FASHION_3_1_2_ASSET_ROOT',REPO/'checkpoints/3_1_2')).resolve()
POL=REPO/'configs/3_1_2/route_a_policy.json'
FC=ASSET_ROOT/'fashionclip'
PCK=ASSET_ROOT/'presence_g2/best_model.pth'
SCK=ASSET_ROOT/'smoke_r1/best_model.pth'
SAM_REPO=REPO/'third_party/3_1_2/sam_hq'
SAM_CK=ASSET_ROOT/'sam_hq/sam_hq_vit_l.pth'
from fashion_3_1_2.components import smoke_r1 as SM
from fashion_3_1_2.runtime.constraints import apply_spatial_constraints, constraint_names, parse_spatial_constraints
os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
def view_boxes():
 out=[[0,0,1,1]]
 for g in (2,3):
  step=1/g;pad=.05*step
  for y in range(g):
   for x in range(g):out.append([max(0,x*step-pad),max(0,y*step-pad),min(1,(x+1)*step+pad),min(1,(y+1)*step+pad)])
 return out
class G2(nn.Module):
 def __init__(self,d=512):
  super().__init__();self.qp=nn.Linear(d,256);self.vp=nn.Linear(d,256);self.net=nn.Sequential(nn.Linear(d*3+7,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
 def forward(self,v,q):
  vn=F.normalize(v,dim=-1);qn=F.normalize(q,dim=-1);s=(vn*qn[:,None]).sum(-1);ss=torch.sort(s,dim=1,descending=True).values;ent=-(torch.softmax(s,1)*torch.log_softmax(s,1)).sum(1,keepdim=True);summary=torch.cat([s[:,:1],s.max(1,keepdim=True).values,ss[:,:3].mean(1,keepdim=True),s.mean(1,keepdim=True),s.var(1,keepdim=True),ss[:,:1]-ss[:,1:2],ent],1);qh=self.qp(q);vh=self.vp(v);w=F.softmax((vh*qh[:,None]).sum(-1)/math.sqrt(vh.shape[-1]),1);att=(v*w[:,:,None]).sum(1);return self.net(torch.cat([q,v[:,0],att,summary],1)).squeeze(1)
def clipbox(b,p):return [max(p[0],min(p[2],b[0])),max(p[1],min(p[3],b[1])),max(p[0],min(p[2],b[2])),max(p[1],min(p[3],b[3]))]
def expand(b,p,scale,mn):
 cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2;w=max((b[2]-b[0])*scale,mn);h=max((b[3]-b[1])*scale,mn);return clipbox([cx-w/2,cy-h/2,cx+w/2,cy+h/2],p)
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);it=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return it/(aa+bb-it) if aa+bb-it else 0
def rle(mask):
 flat=np.asarray(mask,dtype=np.uint8).reshape(-1,order='F');counts=[];last=0;n=0
 for v in flat:
  if int(v)==last:n+=1
  else:counts.append(n);n=1;last=int(v)
 counts.append(n);return {'size':[int(mask.shape[0]),int(mask.shape[1])],'counts':counts}
class Fashion312ZeroOneNFunctionalRuntime:
 def __init__(self,device='cuda',work_dir=None):
  self.device=device;self.work=Path(work_dir or tempfile.mkdtemp(prefix='fashion312_'));self.work.mkdir(parents=True,exist_ok=True);self.proc=CLIPProcessor.from_pretrained(str(FC),local_files_only=True);self.clip=CLIPModel.from_pretrained(str(FC),local_files_only=True).eval().to(device);self.pres=G2().to(device).eval();st=torch.load(PCK,map_location=device);self.pres.load_state_dict(st.get('model_state',st),strict=True);self.smoke=SM.SetRanker(414).to(device).eval();self.smoke.load_state_dict(torch.load(SCK,map_location=device),strict=True);self.sam_predictor=None
 def enc_text(self,text):
  z=self.proc(text=[text],return_tensors='pt',padding=True,truncation=True);z={k:v.to(self.device) for k,v in z.items()}
  with torch.no_grad():q=self.clip.get_text_features(**z);return F.normalize(q.float(),dim=-1).cpu().numpy()[0]
 def enc_images(self,imgs,batch=64):
  out=[]
  with torch.no_grad():
   for i in range(0,len(imgs),batch):
    x=self.proc(images=imgs[i:i+batch],return_tensors='pt')['pixel_values'].to(self.device);out.append(F.normalize(self.clip.get_image_features(pixel_values=x).float(),dim=-1).cpu().numpy())
  return np.concatenate(out)
 def presence(self,parent,q):
  w,h=parent.size;views=[parent.crop((int(a*w),int(b*h),int(c*w),int(d*h))) for a,b,c,d in view_boxes()];v=self.enc_images(views);qt=self.enc_text(q)
  with torch.no_grad():logit=self.pres(torch.from_numpy(v[None]).to(self.device),torch.from_numpy(qt[None]).to(self.device))/1.5;pr=float(torch.sigmoid(logit).cpu())
  return pr,pr>=.35,qt
 def route(self,image_path,pbbox,runid):
  d=self.work/f'route_{runid}';d.mkdir(exist_ok=True);policy=json.load(open(POL));policy['detector_v2']['checkpoint']=str(ASSET_ROOT/policy['detector_v2']['checkpoint']);policy['detector_v3']['checkpoint']=str(ASSET_ROOT/policy['detector_v3']['checkpoint']);resolved=d/'resolved_policy.json';resolved.write_text(json.dumps(policy));pid=f'runtime:{runid}';manifest=d/'parent.jsonl';manifest.write_text(json.dumps({'image_id':runid,'image_path':str(image_path),'parent_bbox_full_image_xyxy':pbbox,'parent_instance_id':pid,'split':'runtime'})+'\n');empty=d/'empty.gz'
  with gzip.open(empty,'wt'):pass
  common=['--parent-manifest',str(manifest),'--policy-config',str(resolved),'--device',self.device]
  for s in ('v2','v3'):
   subprocess.run([str(ENV),str(REPO/'src/fashion_3_1_2/components'/f'route_detector_{s}.py'),*common,'--output-jsonl',str(d/f'{s}.gz'),'--status-jsonl',str(d/f'{s}_status.jsonl'),'--log-file',str(d/f'{s}.log')],check=True)
  subprocess.run([str(ENV),str(REPO/'src/fashion_3_1_2/components/candidate_fusion.py'),'--detector-v2-jsonl',str(d/'v2.gz'),'--detector-v3-jsonl',str(d/'v3.gz'),'--samhq-jsonl',str(empty),'--output-jsonl',str(d/'fused.gz'),'--status-csv',str(d/'fusion.csv'),'--policy-config',str(resolved)],check=True)
  with gzip.open(d/'fused.gz','rt') as f:return [json.loads(x) for x in f][:500]
 def candidate_features(self,image,q,qt,pbbox,rows):
  cs=[];tight=[];context=[]
  for r in rows:
   b=r['candidate_bbox_full_image_xyxy'];c={'candidate_id':r['candidate_final_id'],'bbox':b,'parent_bbox':pbbox,'parent_instance_id':r['parent_instance_id'],'image_path':r['image_path'],'rank':r['candidate_final_rank'],'score_route':r['candidate_score_fused'],'score_norm':r.get('candidate_score_normalized',r['candidate_score_fused']),'score_raw':r.get('candidate_score_raw',r['candidate_score_fused']),'source':r['candidate_primary_source'],'supporting_sources':r.get('candidate_supporting_sources',[]),'source_agreement':float(len(r.get('candidate_supporting_sources',[]))>1),'agreement_flag':int(len(r.get('candidate_supporting_sources',[]))>1),'teacher_weighted_score':0,'teacher_selected':0,'mdetr_max_iou':0,'mdetr_weighted_support':0,'gt_label':'ignore','matched_target_iou':0,'matched_target_id':''};cs.append(c);tight.append(image.crop(tuple(expand(b,pbbox,1.1,32))));context.append(image.crop(tuple(expand(b,pbbox,2.5,96))))
  cs=SM.attach_relation_features(cs);zt=self.enc_images(tight);zc=self.enc_images(context);zp=self.enc_images([image.crop(tuple(pbbox))])[0]
  class Store:
   def __init__(self):self.i=0
   def cand(self,cid):i=next(j for j,c in enumerate(cs) if c['candidate_id']==cid);return zt[i],zc[i],True
   def parent(self,pid):return zp,True
   def query(self,qid):return qt,True
  st=Store();x=np.stack([SM.build_feature_vector(c,'runtime_query',st,cs) for c in cs]).astype('float32')
  with torch.no_grad():raw=self.smoke(torch.from_numpy(x).to(self.device))['rank'].cpu().numpy()
  scores=[]
  for c,z in zip(cs,raw):
   route=max(1e-6,min(1-1e-6,float(c['score_route'])));logit=math.log(route/(1-route))+math.tanh(float(z));scores.append(1/(1+math.exp(-max(-30,min(30,logit)))))
  return cs,raw,np.asarray(scores)
 def sam_masks(self,parent,boxes):
  try:
   import sys
   if str(SAM_REPO) not in sys.path:sys.path.insert(0,str(SAM_REPO))
   from segment_anything import sam_model_registry,SamPredictor
   if self.sam_predictor is None:self.sam_predictor=SamPredictor(sam_model_registry['vit_l'](checkpoint=str(SAM_CK)).to(self.device).eval())
   self.sam_predictor.set_image(np.asarray(parent.convert('RGB')));outs=[]
   for b in boxes:
    try:
     masks,scores,_=self.sam_predictor.predict(box=np.asarray(b,dtype=np.float32),multimask_output=False,hq_token_only=True);m=masks[0]
     if not m.any():raise ValueError('empty mask')
     outs.append((m,'sam_hq_bbox_prompt',float(scores[0])))
    except Exception:
     m=np.zeros((parent.height,parent.width),bool);x1,y1,x2,y2=[int(round(v)) for v in b];m[max(0,y1):min(parent.height,y2),max(0,x1):min(parent.width,x2)]=1;outs.append((m,'coarse_bbox_runtime_fallback',0.0))
   return outs
  except Exception:
   outs=[]
   for b in boxes:
    m=np.zeros((parent.height,parent.width),bool);x1,y1,x2,y2=[int(round(v)) for v in b];m[max(0,y1):min(parent.height,y2),max(0,x1):min(parent.width,x2)]=1;outs.append((m,'coarse_bbox_runtime_fallback',0.0))
   return outs
 def predict(self,image_path=None,parent_bbox=None,query_text='',parent_crop_path=None):
  t0=time.time();runid=hashlib.sha256((str(image_path or parent_crop_path)+'|'+json.dumps(parent_bbox)+'|'+query_text).encode()).hexdigest()[:12]
  if parent_crop_path:image_path=parent_crop_path;im=Image.open(image_path).convert('RGB');parent_bbox=[0,0,im.width,im.height]
  else:im=Image.open(image_path).convert('RGB')
  parent=im.crop(tuple(parent_bbox));t1=time.time();pr,present,qt=self.presence(parent,query_text);t2=time.time()
  constraints=parse_spatial_constraints(query_text)
  base={'runtime_name':'3_1_2_zero_one_n_functional_release_v1','query_text':query_text,'supported_scope':'no_target' if not present else ('constrained_subset' if constraints else 'generic_all'),'spatial_constraints':constraint_names(constraints),'presence_probability':pr,'presence_decision':'present' if present else 'empty','maximum_supported_outputs':10,'functional_release':True,'prd_accuracy_target_passed':False,'prd_latency_target_passed':False,'warnings':[]}
  if not present:return {**base,'instances':[],'instance_count':0,'cardinality_status':'empty','latency_ms':{'parent_crop':(t1-t0)*1000,'presence':(t2-t1)*1000,'total':(t2-t0)*1000}}
  rows=self.route(image_path,parent_bbox,runid);t3=time.time()
  if not rows:return {**base,'instances':[],'instance_count':0,'cardinality_status':'empty','warnings':['present_but_no_candidate'],'latency_ms':{'route_a':(t3-t2)*1000,'total':(t3-t0)*1000}}
  cs,raw,scores=self.candidate_features(im,query_text,qt,parent_bbox,rows);t4=time.time();order=sorted(range(len(cs)),key=lambda i:(-float(scores[i]),cs[i]['candidate_id']));keep=[]
  for i in order:
   if scores[i]<.65:continue
   if all(iou(cs[i]['bbox'],cs[j]['bbox'])<.5 for j in keep):keep.append(i)
   if len(keep)>=10:break
  if not keep:keep=[order[0]]
  relboxes=[[cs[i]['bbox'][0]-parent_bbox[0],cs[i]['bbox'][1]-parent_bbox[1],cs[i]['bbox'][2]-parent_bbox[0],cs[i]['bbox'][3]-parent_bbox[1]] for i in keep];t5=time.time();masks=self.sam_masks(parent,relboxes);t6=time.time();inst=[]
  for rank,(i,b,(m,src,ms)) in enumerate(zip(keep,relboxes,masks),1):inst.append({'instance_id':f'{runid}:{rank:02d}','candidate_id':cs[i]['candidate_id'],'rank':rank,'bbox_xyxy_parent':[float(v) for v in b],'bbox_score':float(cs[i]['score_route']),'semantic_score':float(scores[i]),'mask_rle':rle(m),'mask_source':src,'mask_score':ms})
  if constraints:
   inst=apply_spatial_constraints(inst,constraints,parent.width,parent.height)
  return {**base,'instances':inst,'instance_count':len(inst),'cardinality_status':'empty' if len(inst)==0 else ('single' if len(inst)==1 else 'multiple'),'latency_ms':{'parent_crop':(t1-t0)*1000,'presence':(t2-t1)*1000,'route_a':(t3-t2)*1000,'fashionclip_smoke_r1':(t4-t3)*1000,'threshold_nms':(t5-t4)*1000,'sam_hq_rle':(t6-t5)*1000,'total':(t6-t0)*1000},'route_a_raw_candidate_count':len(rows),'route_a_cap500_count':min(500,len(rows))}
