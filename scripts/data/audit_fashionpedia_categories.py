#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    print(f"正在读取：{path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fashionpedia-root",
        type=Path,
        default=Path(
            "data_raw/fashionpedia"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "data_manifests/v1/fashionpedia_audit"
        ),
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    annotation_dir = (
        args.fashionpedia_root / "annotations"
    )

    train_path = (
        annotation_dir
        / "instances_attributes_train2020.json"
    )
    val_path = (
        annotation_dir
        / "instances_attributes_val2020.json"
    )

    train = load_json(train_path)
    val = load_json(val_path)

    train_categories = {
        item["id"]: item
        for item in train["categories"]
    }

    val_categories = {
        item["id"]: item
        for item in val["categories"]
    }

    category_definitions_match = (
        train_categories == val_categories
    )

    train_instance_counts = Counter()
    val_instance_counts = Counter()

    train_image_sets = defaultdict(set)
    val_image_sets = defaultdict(set)

    missing_segmentation = Counter()
    missing_bbox = Counter()
    unknown_category_ids = Counter()

    def process_annotations(
        annotations,
        category_counts,
        image_sets,
    ):
        for annotation in annotations:
            category_id = annotation.get(
                "category_id"
            )

            if category_id not in train_categories:
                unknown_category_ids[category_id] += 1
                continue

            category_counts[category_id] += 1
            image_sets[category_id].add(
                annotation["image_id"]
            )

            segmentation = annotation.get(
                "segmentation"
            )

            if not segmentation:
                missing_segmentation[
                    category_id
                ] += 1

            bbox = annotation.get("bbox")

            if not bbox or len(bbox) != 4:
                missing_bbox[category_id] += 1

    process_annotations(
        train["annotations"],
        train_instance_counts,
        train_image_sets,
    )

    process_annotations(
        val["annotations"],
        val_instance_counts,
        val_image_sets,
    )

    category_rows = []

    for category_id in sorted(train_categories):
        category = train_categories[category_id]

        train_instances = train_instance_counts[
            category_id
        ]
        val_instances = val_instance_counts[
            category_id
        ]

        category_rows.append(
            {
                "category_id": category_id,
                "name": category.get("name"),
                "supercategory":
                    category.get("supercategory"),
                "train_instances":
                    train_instances,
                "val_instances":
                    val_instances,
                "total_instances":
                    train_instances
                    + val_instances,
                "train_images": len(
                    train_image_sets[category_id]
                ),
                "val_images": len(
                    val_image_sets[category_id]
                ),
                "missing_segmentation":
                    missing_segmentation[
                        category_id
                    ],
                "missing_bbox":
                    missing_bbox[category_id],
                "proposed_role": "pending",
                "unified_category": None,
                "target_part": None,
            }
        )

    csv_path = (
        args.out_dir
        / "fashionpedia_category_counts.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                category_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(category_rows)

    review_template = {
        "allowed_roles": [
            "garment_instance",
            "accessory_instance",
            "local_part",
            "ignore",
        ],
        "instructions": {
            "garment_instance":
                "映射到 top/pants/skirt/outerwear/dress。",
            "accessory_instance":
                "映射到 shoes/bag/accessory。",
            "local_part":
                "作为父服饰实例下的局部区域。",
            "ignore":
                "暂不进入统一训练数据。",
        },
        "categories": category_rows,
    }

    write_json(
        args.out_dir
        / "fashionpedia_category_role_review.json",
        review_template,
    )

    report = {
        "train_annotation": str(
            train_path.resolve()
        ),
        "val_annotation": str(
            val_path.resolve()
        ),
        "train_images": len(train["images"]),
        "val_images": len(val["images"]),
        "train_annotations":
            len(train["annotations"]),
        "val_annotations":
            len(val["annotations"]),
        "categories": len(train_categories),
        "attributes": len(
            train.get("attributes", [])
        ),
        "category_definitions_match":
            category_definitions_match,
        "unknown_category_ids":
            dict(unknown_category_ids),
        "annotations_missing_segmentation":
            sum(missing_segmentation.values()),
        "annotations_missing_bbox":
            sum(missing_bbox.values()),
        "category_table": category_rows,
        "raw_files_modified": False,
    }

    write_json(
        args.out_dir / "fashionpedia_audit_report.json",
        report,
    )

    print(
        json.dumps(
            {
                "train_images":
                    report["train_images"],
                "val_images":
                    report["val_images"],
                "train_annotations":
                    report["train_annotations"],
                "val_annotations":
                    report["val_annotations"],
                "categories":
                    report["categories"],
                "attributes":
                    report["attributes"],
                "category_definitions_match":
                    category_definitions_match,
                "unknown_category_ids":
                    report["unknown_category_ids"],
                "missing_segmentation":
                    report[
                        "annotations_missing_segmentation"
                    ],
                "missing_bbox":
                    report[
                        "annotations_missing_bbox"
                    ],
                "output_csv": str(
                    csv_path.resolve()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
