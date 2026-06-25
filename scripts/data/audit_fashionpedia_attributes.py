#!/usr/bin/env python3

import argparse
import csv
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    print(f"Loading: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--category-mapping",
        type=Path,
        default=Path(
            "data_manifests/frozen/fashionpedia_category_mapping_v1/"
            "fashionpedia_category_mapping_v1.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_manifests/v1/fashionpedia_attribute_audit"),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()

    def resolve(path):
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    mapping_path = resolve(args.category_mapping)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = root / "data_raw/fashionpedia/annotations/instances_attributes_train2020.json"
    val_path = root / "data_raw/fashionpedia/annotations/instances_attributes_val2020.json"
    category_mapping = load_json(mapping_path)
    category_by_id = {
        item["source_category_id"]: item
        for item in category_mapping["categories"]
    }

    train_counts = Counter()
    val_counts = Counter()
    category_counts = defaultdict(Counter)
    role_counts = defaultdict(Counter)
    part_counts = defaultdict(Counter)
    unified_category_counts = defaultdict(Counter)
    example_annotations = defaultdict(list)
    unknown_attribute_ids = Counter()
    unknown_category_ids = Counter()
    annotations_without_attributes = Counter()
    total_attribute_assignments = Counter()
    definitions_by_split = {}
    split_stats = {}

    def process_split(path, split):
        data = load_json(path)
        definitions = {item["id"]: item for item in data["attributes"]}
        definitions_by_split[split] = definitions
        target_counter = train_counts if split == "train" else val_counts

        for annotation in data["annotations"]:
            category_id = annotation.get("category_id")
            category_mapping_item = category_by_id.get(category_id)
            if category_mapping_item is None:
                unknown_category_ids[category_id] += 1
                continue

            attribute_ids = annotation.get("attribute_ids") or []
            if not attribute_ids:
                annotations_without_attributes[split] += 1

            for attribute_id in sorted(set(attribute_ids)):
                if attribute_id not in definitions:
                    unknown_attribute_ids[attribute_id] += 1
                    continue

                target_counter[attribute_id] += 1
                total_attribute_assignments[split] += 1
                category_counts[attribute_id][category_id] += 1
                role = category_mapping_item["role"]
                role_counts[attribute_id][role] += 1

                if role == "local_part":
                    part_counts[attribute_id][category_mapping_item["target_part"]] += 1
                else:
                    unified_category_counts[attribute_id][
                        category_mapping_item["unified_category"]
                    ] += 1

                if len(example_annotations[attribute_id]) < 5:
                    example_annotations[attribute_id].append(
                        {
                            "split": split,
                            "annotation_id": annotation.get("id"),
                            "image_id": annotation.get("image_id"),
                            "source_category_id": category_id,
                            "source_category_name": category_mapping_item[
                                "source_category_name"
                            ],
                        }
                    )

        split_stats[split] = {
            "images": len(data["images"]),
            "annotations": len(data["annotations"]),
            "attribute_definitions": len(definitions),
            "attribute_assignments": total_attribute_assignments[split],
            "annotations_without_attributes": annotations_without_attributes[split],
        }
        del data
        gc.collect()

    process_split(train_path, "train")
    process_split(val_path, "val")

    train_definitions = definitions_by_split["train"]
    val_definitions = definitions_by_split["val"]
    definitions_match = train_definitions == val_definitions

    coverage_counts = Counter()
    supercategory_definition_counts = Counter()
    supercategory_assignment_counts = Counter()
    rows = []

    for attribute_id in sorted(train_definitions):
        definition = train_definitions[attribute_id]
        train_count = train_counts[attribute_id]
        val_count = val_counts[attribute_id]
        total_count = train_count + val_count
        local_count = role_counts[attribute_id]["local_part"]
        instance_count = total_count - local_count

        if total_count == 0:
            candidate_scope = "unused"
        elif local_count == total_count:
            candidate_scope = "local_only"
        elif instance_count == total_count:
            candidate_scope = "instance_only"
        else:
            candidate_scope = "mixed"

        coverage_counts[candidate_scope] += 1
        supercategory = definition.get("supercategory")
        supercategory_definition_counts[supercategory] += 1
        supercategory_assignment_counts[supercategory] += total_count

        category_distribution = [
            {
                "source_category_id": category_id,
                "source_category_name": category_by_id[category_id][
                    "source_category_name"
                ],
                "role": category_by_id[category_id]["role"],
                "count": count,
            }
            for category_id, count in category_counts[attribute_id].most_common()
        ]

        row = {
            "source_attribute_id": attribute_id,
            "source_attribute_name": definition.get("name"),
            "source_supercategory": supercategory,
            "source_level": definition.get("level"),
            "source_taxonomy_id": definition.get("taxonomy_id"),
            "train_assignments": train_count,
            "val_assignments": val_count,
            "total_assignments": total_count,
            "local_part_assignments": local_count,
            "instance_assignments": instance_count,
            "candidate_scope": candidate_scope,
            "candidate_target_parts": [
                {"target_part": part, "count": count}
                for part, count in part_counts[attribute_id].most_common()
            ],
            "candidate_unified_categories": [
                {"unified_category": category, "count": count}
                for category, count in unified_category_counts[attribute_id].most_common()
            ],
            "source_category_distribution": category_distribution,
            "example_annotations": example_annotations[attribute_id],
            "mapping_status": "pending",
            "normalized_attribute_id": None,
            "target_part": None,
            "notes": "",
        }
        rows.append(row)

    detailed_document = {
        "audit_version": "fashionpedia_attribute_audit_v1",
        "category_mapping_version": category_mapping["mapping_version"],
        "attribute_definitions_match": definitions_match,
        "attributes": rows,
    }
    write_json(out_dir / "fashionpedia_attribute_coverage.json", detailed_document)

    csv_path = out_dir / "fashionpedia_attribute_coverage.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source_attribute_id",
            "source_attribute_name",
            "source_supercategory",
            "source_level",
            "source_taxonomy_id",
            "train_assignments",
            "val_assignments",
            "total_assignments",
            "local_part_assignments",
            "instance_assignments",
            "candidate_scope",
            "top_target_parts",
            "top_unified_categories",
            "mapping_status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames},
                    "top_target_parts": ";".join(
                        f'{item["target_part"]}:{item["count"]}'
                        for item in row["candidate_target_parts"][:5]
                    ),
                    "top_unified_categories": ";".join(
                        f'{item["unified_category"]}:{item["count"]}'
                        for item in row["candidate_unified_categories"][:5]
                    ),
                }
            )

    report = {
        "train_annotation": str(train_path),
        "val_annotation": str(val_path),
        "category_mapping": str(mapping_path),
        "attribute_definitions": len(train_definitions),
        "attribute_definitions_match": definitions_match,
        "unknown_attribute_ids": dict(unknown_attribute_ids),
        "unknown_category_ids": dict(unknown_category_ids),
        "split_stats": split_stats,
        "coverage_scope_counts": dict(coverage_counts),
        "supercategory_definition_counts": dict(supercategory_definition_counts),
        "supercategory_assignment_counts": dict(supercategory_assignment_counts),
        "unused_attribute_ids": [
            row["source_attribute_id"]
            for row in rows
            if row["candidate_scope"] == "unused"
        ],
        "raw_files_modified": False,
    }
    write_json(out_dir / "fashionpedia_attribute_audit_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
