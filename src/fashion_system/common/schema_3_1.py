from __future__ import annotations
def parent_bbox_to_full_image(local_bbox,parent_bbox):
    px1,py1,_,_=[float(v) for v in parent_bbox]; x1,y1,x2,y2=[float(v) for v in local_bbox]; return [x1+px1,y1+py1,x2+px1,y2+py1]
def select_highest_confidence(instances):
    rows=list(instances); return None if not rows else max(rows,key=lambda r:float(r.get('confidence',r.get('score',r.get('bbox_score',0.0)))))
def normalize_312_instance(row,parent_bbox):
    out=dict(row); b=row.get('bbox_xyxy_parent') or row.get('bbox'); out['bbox_xyxy_image']=parent_bbox_to_full_image(b,parent_bbox) if b else None; return out
