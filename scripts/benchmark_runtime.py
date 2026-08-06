#!/usr/bin/env python3
import sys
from pathlib import Path
# FASHION313_REPO_SRC_BOOTSTRAP: prefer this checkout when scripts run from repo root.
_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.exists():
    sys.path.insert(0, str(_repo_src))
import argparse, gc, json, statistics, time
from pathlib import Path
from fashion313_runtime import Fashion313Runtime

EXPECTED_ATTR = "2842eeea66c79cf03ae3b5958859dc150669d8e76914edf6089b64a011853920"
EXPECTED_REGION = "06c3711e88721eaa135f1ece750c2911fb55a76b8e2b90b4d489bcefdec12bfb"

def pct(vals, q):
    vals = sorted(vals)
    if not vals: return None
    idx = (len(vals)-1)*q
    lo = int(idx); hi = min(lo+1, len(vals)-1); frac = idx-lo
    return vals[lo]*(1-frac)+vals[hi]*frac

def main(argv=None):
    p=argparse.ArgumentParser(description="Benchmark Fashion 3.1.3 runtime.")
    p.add_argument("--attribute-checkpoint", required=True)
    p.add_argument("--region-family-checkpoint", required=True)
    p.add_argument("--image", default="examples/3_1_3/example.jpg")
    p.add_argument("--target-mask", default="examples/3_1_3/target_mask.png")
    p.add_argument("--parent-mask", default="examples/3_1_3/parent_mask.png")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--output", default="benchmark_runtime_summary.json")
    p.add_argument("--device", default="cuda")
    args=p.parse_args(argv)
    rt=Fashion313Runtime(args.attribute_checkpoint,args.region_family_checkpoint,device=args.device,expected_attribute_sha=EXPECTED_ATTR,expected_region_family_sha=EXPECTED_REGION)
    for _ in range(args.warmup): rt.predict(args.image,args.target_mask,args.parent_mask)
    gc.disable()
    times=[]
    try:
        for _ in range(args.count):
            t0=time.perf_counter(); rt.predict(args.image,args.target_mask,args.parent_mask); times.append((time.perf_counter()-t0)*1000)
    finally:
        gc.enable()
    out={"count":len(times),"average_ms":statistics.mean(times),"p50_ms":pct(times,0.5),"p95_ms":pct(times,0.95),"p99_ms":pct(times,0.99),"p999_ms":pct(times,0.999),"min_ms":min(times),"max_ms":max(times)}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))
    return 0
if __name__ == "__main__": raise SystemExit(main())
