#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


LOCALIZATION_LATER_SUPERCATEGORIES = {
    "textile pattern",
    "textile finishing, manufacturing techniques",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path(
            "data_manifests/v1/fashionpedia_attribute_audit/"
            "fashionpedia_attribute_coverage.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "data_manifests/v1/fashionpedia_attribute_routing"
        ),
    )
    parser.add_argument("--dominance-threshold", type=float, default=0.95)
    parser.add_argument("--local-part-share-threshold", type=float, default=0.80)
    parser.add_argument("--min-auto-assignments", type=int, default=20)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coverage = load_json(args.coverage)
    routed = []
    route_counts = Counter()
    reason_counts = Counter()
    confidence_counts = Counter()
    local_target_counts = Counter()
    manual_rows = []

    for source in coverage["attributes"]:
        total = source["total_assignments"]
        local = source["local_part_assignments"]
        instance = source["instance_assignments"]
        local_ratio = local / total if total else 0.0
        instance_ratio = instance / total if total else 0.0
        top_part = None
        top_part_count = 0
        top_part_share = 0.0
        if source["candidate_target_parts"]:
            top_part = source["candidate_target_parts"][0]["target_part"]
            top_part_count = source["candidate_target_parts"][0]["count"]
            top_part_share = top_part_count / local if local else 0.0

        route = "manual_review"
        reason = None
        target_part = None
        region_policy = None
        region_supervision = None

        if total < args.min_auto_assignments:
            reason = "insufficient_support"
        elif (
            local_ratio >= args.dominance_threshold
            and top_part is not None
            and top_part_share >= args.local_part_share_threshold
        ):
            route = "local_attribute"
            reason = "local_dominant_with_concentrated_part"
            target_part = top_part
            region_policy = "use_fashionpedia_part_mask"
            region_supervision = "strong"
        elif instance_ratio >= args.dominance_threshold:
            route = "global_attribute"
            reason = "instance_dominant"
            target_part = "whole_garment"
            region_policy = "use_parent_instance_mask"
            region_supervision = "strong_instance_context"
        elif local_ratio >= args.dominance_threshold and top_part_share < args.local_part_share_threshold:
            reason = "local_dominant_but_target_part_ambiguous"
        else:
            reason = "local_global_scope_conflict"

        requires_future_localization = (
            route == "global_attribute"
            and source["source_supercategory"] in LOCALIZATION_LATER_SUPERCATEGORIES
        )

        if route == "manual_review":
            confidence = "manual"
        else:
            dominant_ratio = local_ratio if route == "local_attribute" else instance_ratio
            if (
                total >= 100
                and dominant_ratio >= 0.99
                and (route != "local_attribute" or top_part_share >= 0.90)
            ):
                confidence = "high"
            else:
                confidence = "medium"

        item = {
            "source_attribute_id": source["source_attribute_id"],
            "source_attribute_name": source["source_attribute_name"],
            "source_supercategory": source["source_supercategory"],
            "source_level": source["source_level"],
            "source_taxonomy_id": source["source_taxonomy_id"],
            "route": route,
            "route_reason": reason,
            "routing_confidence": confidence,
            "target_part": target_part,
            "region_policy": region_policy,
            "region_supervision": region_supervision,
            "requires_future_localization": requires_future_localization,
            "evidence": {
                "total_assignments": total,
                "local_part_assignments": local,
                "instance_assignments": instance,
                "local_ratio": local_ratio,
                "instance_ratio": instance_ratio,
                "top_local_part": top_part,
                "top_local_part_count": top_part_count,
                "top_local_part_share": top_part_share,
                "candidate_target_parts": source["candidate_target_parts"],
                "candidate_unified_categories": source[
                    "candidate_unified_categories"
                ],
            },
            "normalized_attribute_id": None,
            "review_status": "pending" if route == "manual_review" else "auto_proposed",
            "review_notes": "",
        }
        routed.append(item)
        route_counts[route] += 1
        reason_counts[reason] += 1
        confidence_counts[confidence] += 1
        if route == "local_attribute":
            local_target_counts[target_part] += 1
        if route == "manual_review":
            manual_rows.append(item)

    routing_document = {
        "routing_version": "fashionpedia_attribute_routing_v1_proposed",
        "status": "proposed",
        "thresholds": {
            "dominance_threshold": args.dominance_threshold,
            "local_part_share_threshold": args.local_part_share_threshold,
            "min_auto_assignments": args.min_auto_assignments,
        },
        "policy": {
            "local_attribute": "Fashionpedia part mask is available as strong region supervision.",
            "global_attribute": "Attribute is supervised at parent instance level.",
            "manual_review": "No automatic target-part decision is accepted.",
            "future_localization": (
                "Pattern and manufacturing attributes may have a global label but no "
                "true localized pattern/decoration mask."
            ),
        },
        "attributes": routed,
    }
    write_json(
        args.out_dir / "fashionpedia_attribute_routing_v1_proposed.json",
        routing_document,
    )

    manual_csv = args.out_dir / "manual_review_attributes.csv"
    with manual_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "source_attribute_id",
            "source_attribute_name",
            "source_supercategory",
            "route_reason",
            "total_assignments",
            "local_ratio",
            "instance_ratio",
            "top_local_part",
            "top_local_part_share",
            "proposed_decision",
            "proposed_target_part",
            "review_notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in manual_rows:
            evidence = item["evidence"]
            writer.writerow(
                {
                    "source_attribute_id": item["source_attribute_id"],
                    "source_attribute_name": item["source_attribute_name"],
                    "source_supercategory": item["source_supercategory"],
                    "route_reason": item["route_reason"],
                    "total_assignments": evidence["total_assignments"],
                    "local_ratio": evidence["local_ratio"],
                    "instance_ratio": evidence["instance_ratio"],
                    "top_local_part": evidence["top_local_part"],
                    "top_local_part_share": evidence["top_local_part_share"],
                    "proposed_decision": "",
                    "proposed_target_part": "",
                    "review_notes": "",
                }
            )

    report = {
        "source_attributes": len(coverage["attributes"]),
        "routed_attributes": len(routed),
        "route_counts": dict(route_counts),
        "route_reason_counts": dict(reason_counts),
        "routing_confidence_counts": dict(confidence_counts),
        "local_target_part_attribute_counts": dict(local_target_counts),
        "future_localization_attributes": sum(
            item["requires_future_localization"] for item in routed
        ),
        "manual_review_csv": str(manual_csv.resolve()),
        "status": "proposed",
        "raw_files_modified": False,
    }
    write_json(args.out_dir / "routing_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
