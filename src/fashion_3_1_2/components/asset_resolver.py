import os
from pathlib import Path
class AssetResolver:
    def __init__(self, root=None):
        repo=Path(__file__).resolve().parents[3]
        self.root=Path(root or os.environ.get('FASHION_3_1_2_ASSET_ROOT') or repo/'checkpoints/3_1_2').resolve()
    def get(self, relative):
        p=(self.root/relative).resolve()
        if self.root not in p.parents and p!=self.root: raise ValueError('asset path escapes root')
        if not p.is_file(): raise FileNotFoundError(p)
        if p.stat().st_size<200 and p.read_bytes().startswith(b'version https://git-lfs.github.com/spec'): raise ValueError('Git LFS pointer is not a checkpoint')
        return p
