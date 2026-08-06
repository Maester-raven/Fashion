"""Natural-language spatial subset filtering for Fashion 3.1.2."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class SpatialConstraint:
    name: str
    mode: str

KEYWORDS={
    'leftmost':['leftmost','最左'], 'rightmost':['rightmost','最右'], 'topmost':['topmost','uppermost','最上','顶部','最顶部'], 'bottommost':['bottommost','lowest','最下','底部','最底部'],
    'left':[' left ','left side','on the left','左侧','左边'], 'right':[' right ','right side','on the right','右侧','右边'],
    'upper':['upper','上方','上部'], 'lower':['lower','下方','下部'], 'top':[' top ','顶部'], 'bottom':[' bottom ','底部'], 'center':['center','middle','中央','中间','靠近中心']
}

def parse_spatial_constraints(query_text: str) -> list[SpatialConstraint]:
    q=' '+(query_text or '').lower().replace('-',' ')+' '
    for name in ('leftmost','rightmost','topmost','bottommost'):
        if any(k in q for k in KEYWORDS[name]): return [SpatialConstraint(name,'extreme')]
    out=[]
    for name in ('left','right','upper','lower','top','bottom','center'):
        if any(k in q for k in KEYWORDS[name]): out.append(SpatialConstraint(name,'region'))
    return out

def _bbox(inst): return inst.get('bbox_xyxy_parent') or inst.get('bbox') or inst.get('bbox_xyxy')
def _center(inst):
    b=_bbox(inst); return ((float(b[0])+float(b[2]))/2.0,(float(b[1])+float(b[3]))/2.0)

def _parent_size(rows, parent_width=None, parent_height=None):
    if parent_width and parent_height: return float(parent_width),float(parent_height)
    return max(max((float(_bbox(r)[2]) for r in rows),default=1.0),1.0), max(max((float(_bbox(r)[3]) for r in rows),default=1.0),1.0)

def apply_spatial_constraints(instances: Iterable[dict], constraints: list[SpatialConstraint], parent_width=None, parent_height=None) -> list[dict]:
    rows=[dict(i) for i in instances]
    if not rows or not constraints: return rows
    width,height=_parent_size(rows,parent_width,parent_height)
    for c in constraints:
        if not rows: break
        centers=[(_center(r),i) for i,r in enumerate(rows)]
        if c.mode=='extreme':
            key={'leftmost':lambda it:(it[0][0],it[0][1]),'rightmost':lambda it:(-it[0][0],it[0][1]),'topmost':lambda it:(it[0][1],it[0][0]),'bottommost':lambda it:(-it[0][1],it[0][0])}[c.name]
            rows=[rows[min(centers,key=key)[1]]]; continue
        kept=[]
        for row in rows:
            cx,cy=_center(row); ok=True
            if c.name=='left': ok=cx<=width*0.5
            elif c.name=='right': ok=cx>=width*0.5
            elif c.name in {'upper','top'}: ok=cy<=height*0.5
            elif c.name in {'lower','bottom'}: ok=cy>=height*0.5
            elif c.name=='center': ok=width*0.25<=cx<=width*0.75 and height*0.25<=cy<=height*0.75
            if ok: kept.append(row)
        if kept: rows=kept
    for rank,row in enumerate(rows,1): row['rank']=rank
    return rows

def constraint_names(constraints: list[SpatialConstraint]) -> list[str]: return [c.name for c in constraints]
