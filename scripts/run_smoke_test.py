#!/usr/bin/env python3
import sys
from pathlib import Path
# FASHION313_REPO_SRC_BOOTSTRAP: prefer this checkout when scripts run from repo root.
_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.exists():
    sys.path.insert(0, str(_repo_src))
import argparse, json
from pathlib import Path
from fashion313_runtime import Fashion313Runtime

EXPECTED_ATTR = "2842eeea66c79cf03ae3b5958859dc150669d8e76914edf6089b64a011853920"
EXPECTED_REGION = "06c3711e88721eaa135f1ece750c2911fb55a76b8e2b90b4d489bcefdec12bfb"

def main(argv=None):
    p = argparse.ArgumentParser(description="Run a Fashion 3.1.3 real-checkpoint smoke test.")
    p.add_argument("--attribute-checkpoint", required=True)
    p.add_argument("--region-family-checkpoint", required=True)
    p.add_argument("--image", default="examples/3_1_3/example.jpg")
    p.add_argument("--target-mask", default="examples/3_1_3/target_mask.png")
    p.add_argument("--parent-mask", default="examples/3_1_3/parent_mask.png")
    p.add_argument("--output", default="examples/outputs/fashion313_smoke_output.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    rt = Fashion313Runtime(args.attribute_checkpoint, args.region_family_checkpoint, device=args.device, expected_attribute_sha=EXPECTED_ATTR, expected_region_family_sha=EXPECTED_REGION)
    out = rt.predict(args.image, args.target_mask, args.parent_mask)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": out["status"], "region_family": out["region_family"], "prediction_groups": len(out["predictions"])}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
