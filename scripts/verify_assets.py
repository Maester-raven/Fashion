import sys
from pathlib import Path
# FASHION313_REPO_SRC_BOOTSTRAP: prefer this checkout when scripts run from repo root.
_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.exists():
    sys.path.insert(0, str(_repo_src))
import argparse, hashlib, sys
EXPECTED={'attribute':'2842eeea66c79cf03ae3b5958859dc150669d8e76914edf6089b64a011853920','region':'06c3711e88721eaa135f1ece750c2911fb55a76b8e2b90b4d489bcefdec12bfb'}
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  [h.update(b) for b in iter(lambda:f.read(1024*1024), b'')]
 return h.hexdigest()
p=argparse.ArgumentParser(); p.add_argument('--attribute-checkpoint', required=True); p.add_argument('--region-family-checkpoint', required=True); a=p.parse_args()
assert sha(a.attribute_checkpoint)==EXPECTED['attribute'], 'attribute checkpoint SHA mismatch'
assert sha(a.region_family_checkpoint)==EXPECTED['region'], 'region-family checkpoint SHA mismatch'
print('asset verification passed')
