#!/usr/bin/env python3
import argparse,subprocess,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--onnx',required=True); p.add_argument('--engine',required=True); p.add_argument('--fp16',action='store_true'); p.add_argument('--workspace-gb',type=float,default=8.0); a=p.parse_args(); root=Path(__file__).resolve().parents[2]; cmd=[sys.executable,str(root/'scripts/export/build_rtmdet_tensorrt_engine.py'),'--onnx',a.onnx,'--engine',a.engine,'--workspace-gb',str(a.workspace_gb)];
 if a.fp16: cmd.append('--fp16');
 return subprocess.call(cmd)
if __name__=='__main__': raise SystemExit(main())
