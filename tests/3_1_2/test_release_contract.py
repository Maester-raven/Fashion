from pathlib import Path
from fashion_3_1_2.components.mask_codec import encode_uncompressed_rle,validate_rle
import numpy as np
def test_rle(): assert validate_rle(encode_uncompressed_rle(np.zeros((4,5),bool)))
def test_profiles():
 from fashion_3_1_2 import Fashion312Runtime,SingleHitBBoxMaskPipeline
 assert Fashion312Runtime and SingleHitBBoxMaskPipeline
def test_no_forbidden_paths():
 root=Path(__file__).resolve().parents[2]
 for p in (root/'src/fashion_3_1_2').rglob('*.py'):
  s=p.read_text();assert '/root/' not in s and 'work_dirs' not in s
