#!/usr/bin/env python3
"""Download and verify Fashion 3.1 model assets from GitHub Releases."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tarfile, tempfile, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GITHUB_RELEASE = "https://github.com/Maester-raven/Fashion/releases/download"
STATIC_ASSETS = [
    {"module":"3.1.1","release_tag":"v0.1.2-real-deploy-311","filename":"epoch_5.pth","sha256":"cb5e2ae8916568954882bfc5f5f741e9753c3bfdd969e8441c1f0e6ce3d882da","size_bytes":739509791,"target":"rtmdet/epoch_5.pth","purpose":"RTMDet-Ins-L 8-class instance segmentation checkpoint","extract":False},
    {"module":"3.1.3","release_tag":"v0.1.3-real-deploy-313","filename":"fashion313_attribute_model_v1.pth","sha256":"2842eeea66c79cf03ae3b5958859dc150669d8e76914edf6089b64a011853920","size_bytes":99514225,"target":"fashion313_attribute_model_v1.pth","purpose":"Native Design attribute model","extract":False},
    {"module":"3.1.3","release_tag":"v0.1.3-real-deploy-313","filename":"fashion313_region_family_model_v1.pth","sha256":"06c3711e88721eaa135f1ece750c2911fb55a76b8e2b90b4d489bcefdec12bfb","size_bytes":880509,"target":"fashion313_region_family_model_v1.pth","purpose":"Native Design region-family model","extract":False},
]
ALIASES={"311":"3.1.1","312":"3.1.2","313":"3.1.3"}

def load_assets():
    assets=list(STATIC_ASSETS)
    manifest_path=REPO/'configs/3_1_2/release_asset_upload_manifest.json'
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        for row in manifest['packages']:
            assets.append({"module":"3.1.2","release_tag":manifest['tag'],"filename":row['filename'],"sha256":row['sha256'],"size_bytes":int(row['size_bytes']),"target":f"3_1_2/packages/{row['filename']}","purpose":row.get('component',row['filename']),"extract":True,"package_name":row['name']})
    return assets

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def asset_url(asset):
    return f"{GITHUB_RELEASE}/{asset['release_tag']}/{asset['filename']}"

def select_assets(module: str):
    module=ALIASES.get(module.lower(), module.lower())
    assets=load_assets()
    if module=='all': return assets
    out=[a for a in assets if a['module']==module]
    if not out: raise SystemExit(f'unknown module or no assets: {module}')
    return out

def verify_file(path: Path, asset: dict):
    if not path.is_file(): return False, 'missing'
    size=path.stat().st_size
    if size != int(asset['size_bytes']): return False, f"size_mismatch:{size}!={asset['size_bytes']}"
    digest=sha256_file(path)
    if digest != asset['sha256']: return False, f'sha256_mismatch:{digest}'
    return True, 'ok'

def curl_download(url: str, target: Path, args):
    part=target.with_name(target.name+'.part'); part.parent.mkdir(parents=True, exist_ok=True)
    resume_offset=part.stat().st_size if part.exists() else 0
    cmd=['curl','--http1.1','-L','--fail','--continue-at','-' if args.resume else '0','--retry',str(args.max_retries),'--retry-all-errors','--connect-timeout',str(args.connect_timeout),'--max-time',str(args.transfer_timeout),'-o',str(part),url]
    started=time.time(); proc=subprocess.run(cmd,text=True,capture_output=True)
    return {"backend":"curl","exit_code":proc.returncode,"resume_offset":resume_offset,"downloaded_bytes":part.stat().st_size if part.exists() else 0,"seconds":round(time.time()-started,3),"stderr_tail":proc.stderr[-2000:]}

def urllib_download(url: str, target: Path, args):
    part=target.with_name(target.name+'.part'); part.parent.mkdir(parents=True, exist_ok=True)
    started=time.time()
    try: urllib.request.urlretrieve(url, part); code=0; err=''
    except Exception as exc: code=1; err=repr(exc)
    return {"backend":"public","exit_code":code,"resume_offset":0,"downloaded_bytes":part.stat().st_size if part.exists() else 0,"seconds":round(time.time()-started,3),"stderr_tail":err}

def download_one(asset, model_dir: Path, args):
    target=model_dir/asset['target']; url=asset_url(asset)
    ok, reason=verify_file(target, asset)
    row={"module":asset['module'],"filename":asset['filename'],"target":str(target),"url":url,"expected_bytes":asset['size_bytes'],"expected_sha256":asset['sha256'],"status":"verified_existing" if ok else "pending","verify_reason":reason}
    if ok or args.verify_only:
        row['ok']=ok; return row
    backend='curl' if args.backend in {'auto','curl','gh','token'} and shutil.which('curl') else 'public'
    for attempt in range(max(1,args.max_retries)):
        result=curl_download(url,target,args) if backend=='curl' else urllib_download(url,target,args)
        row.update(result); part=target.with_name(target.name+'.part')
        pok, preason=verify_file(part, asset); row['verify_reason']=preason
        if result['exit_code']==0 and pok:
            os.replace(part,target); row['status']='downloaded_verified'; row['ok']=True; break
        row['status']='download_failed_or_unverified'; row['ok']=False
        time.sleep(min(2**attempt,30))
    return row

def safe_extract_tar(archive: Path, dst: Path):
    tmp=Path(tempfile.mkdtemp(prefix='fashion_asset_extract_'))
    with tarfile.open(archive,'r') as tf:
        base=tmp.resolve()
        for member in tf.getmembers():
            target=(tmp/member.name).resolve()
            if not str(target).startswith(str(base)+os.sep): raise RuntimeError(f'unsafe tar member path: {member.name}')
        tf.extractall(tmp, filter='data')
    package_dirs=[p for p in tmp.iterdir() if p.is_dir()]
    if not package_dirs: raise RuntimeError(f'no package directory in {archive}')
    package=package_dirs[0]; manifest=json.loads((package/'MANIFEST.json').read_text(encoding='utf-8'))
    for file_row in manifest.get('files',[]):
        src=package/file_row['path']
        if sha256_file(src)!=file_row['sha256']: raise RuntimeError(f"extracted file checksum mismatch: {file_row['path']}")
        out=dst/file_row['path']; out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src,out)
    return package

def install_312(model_dir: Path, rows):
    asset_root=model_dir/'3_1_2'; installed=[]; errors=[]
    for row in rows:
        if row.get('module')!='3.1.2' or not row.get('ok'): continue
        try: installed.append({'archive':row['filename'],'package':safe_extract_tar(Path(row['target']),asset_root).name})
        except Exception as exc: errors.append({'archive':row['filename'],'error':repr(exc)})
    manifest={'asset_root':str(asset_root),'installed':installed,'errors':errors}
    if installed or errors:
        (asset_root/'installation_manifest.json').parent.mkdir(parents=True,exist_ok=True)
        (asset_root/'installation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser(description='Download/verify Fashion 3.1 model assets from public GitHub Releases.')
    p.add_argument('--module',default='3.1.3',choices=['3.1.1','3.1.2','3.1.3','311','312','313','all'])
    p.add_argument('--model-dir',default='models'); p.add_argument('--backend',default='auto',choices=['auto','curl','public','gh','token'])
    p.add_argument('--verify-only',action='store_true'); p.add_argument('--resume',action='store_true',default=True)
    p.add_argument('--max-retries',type=int,default=5); p.add_argument('--connect-timeout',type=int,default=30); p.add_argument('--transfer-timeout',type=int,default=3600)
    p.add_argument('--install',action='store_true'); p.add_argument('--report-json')
    args=p.parse_args(argv); model_dir=Path(args.model_dir); model_dir.mkdir(parents=True,exist_ok=True)
    rows=[download_one(asset,model_dir,args) for asset in select_assets(args.module)]
    report={'ok':all(r.get('ok') for r in rows),'assets':rows,'install_3_1_2':install_312(model_dir,rows) if args.install else None}
    text=json.dumps(report,indent=2,ensure_ascii=False)
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True,exist_ok=True); Path(args.report_json).write_text(text+'\n',encoding='utf-8')
    print(text); return 0 if report['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
