#!/usr/bin/env python3
import argparse,subprocess,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--config',default='configs/instance_segmentation/rtmdet_ins_l_fashionpedia8_copypaste13000_1024_e24_v1.py'); p.add_argument('--a-only',action='store_true',default=True); a=p.parse_args(); root=Path(__file__).resolve().parents[2]; return subprocess.call([sys.executable,str(root/'scripts/export/export_rtmdet_dual_cls_onnx.py'),'--config',a.config,'--a-checkpoint',a.checkpoint,'--output',a.output,'--a-only'])
if __name__=='__main__': raise SystemExit(main())
