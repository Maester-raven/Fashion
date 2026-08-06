#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
MODULES={'3.1.1':Path('envs/3_1_1'),'3.1.2':Path('envs/3_1_2'),'3.1.3':Path('envs/3_1_3')}; ALIASES={'311':'3.1.1','312':'3.1.2','313':'3.1.3'}
def main(argv=None):
    p=argparse.ArgumentParser(description='Create a module-level Fashion 3.1 runtime environment.')
    p.add_argument('--module',required=True,choices=['3.1.1','3.1.2','3.1.3','311','312','313']); p.add_argument('--backend',default='conda',choices=['conda','venv']); p.add_argument('--prefix',required=True); p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv); module=ALIASES.get(args.module,args.module); root=Path(__file__).resolve().parents[1]; env_dir=root/MODULES[module]; prefix=Path(args.prefix)
    cmd=['conda','env','create','--solver','libmamba','--prefix',str(prefix),'-f',str(env_dir/'environment-minimal.yml')] if args.backend=='conda' else [sys.executable,'-m','venv',str(prefix)]
    print(json.dumps({'module':module,'backend':args.backend,'prefix':str(prefix),'command':cmd,'environment_dir':str(env_dir)},indent=2))
    if args.dry_run: return 0
    subprocess.check_call(cmd)
    if args.backend=='venv':
        pip=prefix/('Scripts/pip.exe' if sys.platform=='win32' else 'bin/pip')
        subprocess.check_call([str(pip),'install','-e',str(root)]); subprocess.check_call([str(pip),'install','-r',str(env_dir/'requirements-runtime.txt')])
    return 0
if __name__=='__main__': raise SystemExit(main())
