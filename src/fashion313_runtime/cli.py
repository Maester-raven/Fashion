
import argparse, json
from pathlib import Path
from .runtime import Fashion313Runtime

def main(argv=None):
    p = argparse.ArgumentParser(description="Fashion 3.1.3 Native Design runtime")
    p.add_argument("--image", required=True)
    p.add_argument("--target-mask", required=True)
    p.add_argument("--parent-mask")
    p.add_argument("--attribute-checkpoint", required=True)
    p.add_argument("--region-family-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    rt = Fashion313Runtime(a.attribute_checkpoint, a.region_family_checkpoint, device=a.device)
    out = rt.predict(a.image, a.target_mask, a.parent_mask)
    Path(a.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
