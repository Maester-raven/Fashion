#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from fashion_3_1_2.asset_loader import expand_path, load_config, sha256_file
from fashion_3_1_2.presence_gate import load_presence_g2_checkpoint_strict
from fashion_3_1_2.smoke_r1_selector import load_smoke_r1_checkpoint_strict
from fashion_3_1_2.sam_hq_refiner import SamHQRefiner


def one_report(name, path, expected_sha):
    p = Path(path)
    r = {"name": name, "path": str(p), "exists": p.exists(), "expected_sha256": expected_sha, "strict_load": False, "strict_false_used": False, "errors": []}
    if p.exists():
        r["sha256"] = sha256_file(p)
        r["sha256_match"] = r["sha256"] == expected_sha
    else:
        r["sha256_match"] = False
    return r


def main():
    ap = argparse.ArgumentParser(description="Audit strict checkpoint loading for 3.1.2 Presence, Smoke R1, and SAM-HQ assets.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--sam-hq-repo-root")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    presence_path = expand_path(cfg["presence"]["checkpoint"], model_root=args.model_root)
    smoke_path = expand_path(cfg["selector"]["checkpoint"], model_root=args.model_root)
    sam_path = expand_path(cfg["sam_hq"]["checkpoint"], model_root=args.model_root)
    presence = one_report("presence_gate_g2", presence_path, cfg["presence"]["checkpoint_sha256"])
    try:
        _, audit = load_presence_g2_checkpoint_strict(presence_path, device=args.device)
        presence.update(audit)
        presence["strict_load"] = True
    except Exception as exc:
        presence["errors"].append(repr(exc))
    smoke = one_report("smoke_r1_set_ranker", smoke_path, cfg["selector"]["checkpoint_sha256"])
    try:
        _, audit = load_smoke_r1_checkpoint_strict(smoke_path, device=args.device)
        smoke.update(audit)
        smoke["strict_load"] = True
    except Exception as exc:
        smoke["errors"].append(repr(exc))
    sam_repo = args.sam_hq_repo_root or os.environ.get("SAM_HQ_REPO_ROOT") or str(expand_path(cfg["sam_hq"]["repo"], model_root=args.model_root))
    sam = one_report("sam_hq_vit_l", sam_path, cfg["sam_hq"]["checkpoint_sha256"])
    try:
        ref = SamHQRefiner(sam_repo, sam_path, cfg["sam_hq"]["checkpoint_sha256"], cfg["sam_hq"].get("model_type", "vit_l"), device=args.device, multimask_output=cfg["sam_hq"].get("multimask_output", False)).load()
        sam.update(ref.audit)
        sam["strict_load"] = True
        sam["repo_root"] = sam_repo
    except Exception as exc:
        sam["errors"].append(repr(exc))
    for name, rep in [("presence_strict_loader_audit.json", presence), ("smoke_r1_strict_loader_audit.json", smoke), ("sam_hq_strict_loader_audit.json", sam)]:
        (out / name).write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"presence_strict_load": presence["strict_load"], "smoke_r1_strict_load": smoke["strict_load"], "sam_hq_strict_load": sam["strict_load"], "strict_false_used": any(r.get("strict_false_used") for r in [presence, smoke, sam]), "all_passed": presence["strict_load"] and smoke["strict_load"] and sam["strict_load"] and not any(r.get("strict_false_used") for r in [presence, smoke, sam])}
    (out / "strict_loader_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["all_passed"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
