import argparse,hashlib,json,shutil,tarfile,tempfile,urllib.request
from pathlib import Path
def sha(p):
 h=hashlib.sha256();f=open(p,'rb')
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--local-asset-dir');p.add_argument('--release-tag',default='3.1.2-zero-one-n-functional-v1');p.add_argument('--asset-root',default='checkpoints/3_1_2');a=p.parse_args();dst=Path(a.asset_root);dst.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parents[1];manifest=json.load(open(repo/'configs/3_1_2/release_asset_upload_manifest.json'));tmp=Path(tempfile.mkdtemp(prefix='fashion312_assets_'))
source=Path(a.local_asset_dir) if a.local_asset_dir else tmp
for row in manifest['packages']:
 archive=source/row['filename']
 if not a.local_asset_dir: urllib.request.urlretrieve(f'https://github.com/Maester-raven/Fashion/releases/download/{a.release_tag}/{row["filename"]}',archive)
 if not archive.is_file() or sha(archive)!=row['sha256']: raise SystemExit(f'package checksum mismatch: {archive}')
 with tarfile.open(archive,'r') as tf: tf.extractall(tmp,filter='data')
 pack=tmp/row['name']
 man=json.load(open(pack/'MANIFEST.json'))
 for x in man['files']:
  src=pack/x['path']; assert sha(src)==x['sha256'];out=dst/x['path'];out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out);assert sha(out)==x['sha256']
print(json.dumps({'installed':True,'asset_root':str(dst.resolve())}))
