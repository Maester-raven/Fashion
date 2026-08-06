#!/usr/bin/env python3
import argparse,subprocess,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--engine',required=True); p.add_argument('--output-json',required=True); p.add_argument('--config',default='configs/instance_segmentation/rtmdet_ins_l_fashionpedia8_copypaste13000_1024_e24_v1.py'); a=p.parse_args(); root=Path(__file__).resolve().parents[2]; return subprocess.call([sys.executable,'-m','fashion_system.instance_segmentation.infer','--config',a.config,'--engine',a.engine,'--image',a.image,'--output-json',a.output_json,'--disable-d1-fusion'],cwd=root)
if __name__=='__main__': raise SystemExit(main())
