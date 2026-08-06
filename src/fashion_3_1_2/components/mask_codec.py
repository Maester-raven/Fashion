import numpy as np
def encode_uncompressed_rle(mask):
    flat=np.asarray(mask,dtype=np.uint8).reshape(-1,order='F'); counts=[]; last=0; n=0
    for v in flat:
        if int(v)==last:n+=1
        else:counts.append(n);n=1;last=int(v)
    counts.append(n);return {'size':[int(mask.shape[0]),int(mask.shape[1])],'counts':counts}
def validate_rle(rle): return sum(rle['counts'])==rle['size'][0]*rle['size'][1]
