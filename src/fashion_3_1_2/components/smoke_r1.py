import math, numpy as np, torch, torch.nn as nn
SOURCE_VOCAB=['detector_v2','detector_v3','sam_hq','merged','unknown']; EMBED_SLICE=64
def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def biou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);i=max(0,x2-x1)*max(0,y2-y1);u=area(a)+area(b)-i;return i/u if u else 0
def attach_relation_features(cs):
 for c in cs:
  v=sorted([biou(c['bbox'],o['bbox']) for o in cs if o['candidate_id']!=c['candidate_id']],reverse=True);c['relation_max_iou']=v[0] if v else 0.;c['relation_mean_iou']=sum(v)/max(1,len(v));c['relation_top3_iou']=sum(v[:3])/max(1,min(3,len(v)))
 return cs
def numeric(c,cs):
 b,p=c['bbox'],c['parent_bbox'];ba,pa=area(b),area(p);ww,hh=max(0,b[2]-b[0]),max(0,b[3]-b[1]);src=c.get('source','unknown');one=[float(s in src) for s in SOURCE_VOCAB[:-1]]+[float(not any(s in src for s in SOURCE_VOCAB[:-1]))]
 return np.array([c.get('score_route',0),c.get('score_norm',0),c.get('score_raw',0),1/max(1,c.get('rank',999)),min(1,c.get('rank',999)/500),c.get('source_agreement',0),ww/max(1,p[2]-p[0]),hh/max(1,p[3]-p[1]),ba/max(1,pa),math.log1p(ba)/12,(ww/max(1,hh))/10,1,(((b[0]+b[2])/2)-p[0])/max(1,p[2]-p[0]),(((b[1]+b[3])/2)-p[1])/max(1,p[3]-p[1]),c.get('relation_max_iou',0),c.get('relation_mean_iou',0),c.get('relation_top3_iou',0)]+one,dtype=np.float32)
def build_feature_vector(c,qid,store,cs):
 t,co,hc=store.cand(c['candidate_id']);p,hp=store.parent(c['parent_instance_id']);q,hq=store.query(qid);norm=lambda x:x/max(1e-6,np.linalg.norm(x));t,co,p,q=map(norm,[t,co,p,q]);inter=np.array([t@q,co@q,p@q,np.mean(abs(t-q)),np.mean(abs(co-q)),hc,hp,hq],dtype=np.float32);return np.concatenate([t[:64],co[:64],p[:64],q[:64],abs(t-q)[:64],abs(co-q)[:64],inter,numeric(c,cs)]).astype(np.float32)
class SetRanker(nn.Module):
 def __init__(self,input_dim,hidden=256,heads=4,layers=2,dropout=.1):
  super().__init__();self.proj=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout));e=nn.TransformerEncoderLayer(hidden,heads,512,dropout,batch_first=True,activation='gelu');self.encoder=nn.TransformerEncoder(e,layers);self.rank_head=nn.Linear(hidden,1);self.keep_head=nn.Linear(hidden,1);self.count_head=nn.Linear(hidden,5)
 def forward(self,x):
  z=self.encoder(self.proj(x.unsqueeze(0))).squeeze(0);return {'rank':self.rank_head(z).squeeze(-1),'keep':self.keep_head(z).squeeze(-1),'count':self.count_head(z.mean(0))}
