#!/usr/bin/env python3

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CONFIRMED_DECISIONS = {
    "fashionpedia_train_12399_part_44152": (
        "candidate_1",
        "Ribbon is attached at the skirt waist and belongs to the skirt instance.",
    ),
    "fashionpedia_train_2440_part_171310": (
        "candidate_1",
        "Fringe is located at the skirt hem and belongs to the skirt instance.",
    ),
    "fashionpedia_train_13292_part_113552": (
        "independent_or_missing_parent",
        "The bow tie is an independent neck accessory; none of the shown garment masks is its true parent.",
    ),
    "fashionpedia_train_610_part_144314": (
        "candidate_2",
        "The buckle is installed on the belt accessory, not on the outerwear instance.",
    ),
    "fashionpedia_train_33567_part_218143": (
        "candidate_2",
        "The buckle belongs to the belt accessory worn over the dress.",
    ),
    "fashionpedia_train_22718_part_285876": (
        "candidate_2",
        "The buckle belongs to the belt accessory, not to the coat.",
    ),
    "fashionpedia_train_6167_part_124423": (
        "candidate_1",
        "The sleeve belongs to the dress; the scarf only overlaps spatially.",
    ),
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decision-file",
        type=Path,
        default=Path(
            "data_manifests/v1/fashionpedia_parent_binding_review/"
            "review_decisions.json"
        ),
    )
    args = parser.parse_args()
    decision_path = args.decision_file.resolve()
    data = load_json(decision_path)
    cases = data["cases"]

    if set(cases) != set(CONFIRMED_DECISIONS):
        raise RuntimeError(
            f"Review case mismatch: missing={sorted(set(cases) - set(CONFIRMED_DECISIONS))}, "
            f"unexpected={sorted(set(CONFIRMED_DECISIONS) - set(cases))}"
        )

    backup_path = decision_path.with_name("review_decisions.before_user_confirmation.json")
    if not backup_path.exists():
        shutil.copy2(decision_path, backup_path)

    applied = []
    for binding_id, (decision, notes) in CONFIRMED_DECISIONS.items():
        case = cases[binding_id]
        if decision not in case["allowed_decisions"]:
            raise RuntimeError(f"Invalid decision for {binding_id}: {decision}")
        case["decision"] = decision
        case["review_notes"] = notes
        case["confirmation_source"] = "user_confirmed_in_codex_thread"
        case["confirmed_at_utc"] = datetime.now(timezone.utc).isoformat()
        applied.append(
            {
                "binding_id": binding_id,
                "target_part": case["target_part"],
                "decision": decision,
                "review_notes": notes,
            }
        )

    pending = [binding_id for binding_id, case in cases.items() if case["decision"] == "pending"]
    invalid = [
        binding_id
        for binding_id, case in cases.items()
        if case["decision"] not in case["allowed_decisions"]
    ]
    if pending or invalid:
        raise RuntimeError(f"Decision validation failed: pending={pending}, invalid={invalid}")

    data["review_status"] = "user_confirmed"
    data["confirmed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(decision_path, data)

    counts = Counter(item["decision"] for item in applied)
    report = {
        "review_version": data["review_version"],
        "review_status": "user_confirmed",
        "cases": len(applied),
        "pending": len(pending),
        "invalid": len(invalid),
        "decision_counts": dict(counts),
        "backup_file": str(backup_path),
        "decision_file": str(decision_path),
        "applied_decisions": applied,
        "binding_algorithm_modified": False,
        "full_dataset_modified": False,
    }
    report_path = decision_path.with_name("confirmed_decisions_report.json")
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
