#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_info(path: Path):
    exists = path.is_file()

    return {
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/root/autodl-tmp/fashion_prd"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_manifests/v1"),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir.is_absolute()
        else (root / args.out_dir).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fashionai_root = (
        root / "data_raw/round1_fashionAI_attributes_test_a"
    )
    fashionai_csv = (
        fashionai_root
        / "Tests/round1_fashionAI_attributes_answer_a.csv"
    )

    deepfashion2_root = root / "data_raw/deepfashion2"

    fashionpedia_root = root / "data_raw/fashionpedia"
    fashionpedia_train_json = (
        fashionpedia_root
        / "annotations/instances_attributes_train2020.json"
    )
    fashionpedia_val_json = (
        fashionpedia_root
        / "annotations/instances_attributes_val2020.json"
    )

    fashionpedia_val600_json = (
        root
        / "data_interim/fashionpedia/fashionpedia_val_600_coco.json"
    )

    required_paths = [
        fashionai_root,
        fashionai_csv,
        deepfashion2_root / "train/annos",
        deepfashion2_root / "train/image",
        deepfashion2_root / "validation/annos",
        deepfashion2_root / "validation/image",
        fashionpedia_train_json,
        fashionpedia_val_json,
        fashionpedia_root / "images/train2020",
        fashionpedia_root / "images/val_test2020",
        fashionpedia_val600_json,
    ]

    missing_required_paths = [
        str(path) for path in required_paths if not path.exists()
    ]

    if missing_required_paths:
        print("以下必要路径不存在：")
        for path in missing_required_paths:
            print(path)
        raise SystemExit(1)

    manifest_path = out_dir / "source_images.jsonl"

    counts = Counter()
    missing_images = []
    duplicate_logical_keys = []
    logical_keys = set()

    def emit_record(f, record):
        logical_key = (
            record["dataset"],
            record["source_split"],
            str(record["image_id"]),
        )

        if logical_key in logical_keys:
            duplicate_logical_keys.append(logical_key)
        else:
            logical_keys.add(logical_key)

        info = image_info(Path(record["image_abs_path"]))
        record.update(info)

        if not info["exists"]:
            missing_images.append(record["image_abs_path"])

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[f'{record["dataset"]}:{record["source_split"]}'] += 1
        counts["total"] += 1

    with manifest_path.open("w", encoding="utf-8") as manifest:

        # -------------------------------------------------
        # FashionAI
        # -------------------------------------------------
        fashionai_paths = set()
        fashionai_csv_rows = 0

        with fashionai_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) != 3:
                    continue

                image_path = row[0].strip()
                fashionai_csv_rows += 1
                fashionai_paths.add(image_path)

        for relative_path in sorted(fashionai_paths):
            image_path = fashionai_root / relative_path

            emit_record(
                manifest,
                {
                    "dataset": "FashionAI",
                    "source_split": "unassigned",
                    "split_locked": False,
                    "image_id": Path(relative_path).stem,
                    "image_path": relative_path,
                    "image_abs_path": str(image_path.resolve()),
                    "annotation_path": str(fashionai_csv.resolve()),
                },
            )

        # -------------------------------------------------
        # DeepFashion2
        # -------------------------------------------------
        for split_name in ["train", "validation"]:
            anno_dir = deepfashion2_root / split_name / "annos"
            image_dir = deepfashion2_root / split_name / "image"

            for anno_path in sorted(anno_dir.glob("*.json")):
                image_id = anno_path.stem

                image_path = image_dir / f"{image_id}.jpg"
                if not image_path.exists():
                    png_path = image_dir / f"{image_id}.png"
                    if png_path.exists():
                        image_path = png_path

                emit_record(
                    manifest,
                    {
                        "dataset": "DeepFashion2",
                        "source_split": split_name,
                        "split_locked": True,
                        "image_id": image_id,
                        "image_path": str(
                            image_path.relative_to(root)
                        ),
                        "image_abs_path": str(image_path.resolve()),
                        "annotation_path": str(anno_path.resolve()),
                    },
                )

        # -------------------------------------------------
        # Fashionpedia
        # -------------------------------------------------
        fashionpedia_val_names = set()
        fashionpedia_val_ids = set()

        fashionpedia_specs = [
            (
                "train",
                fashionpedia_train_json,
                fashionpedia_root / "images/train2020",
            ),
            (
                "validation",
                fashionpedia_val_json,
                fashionpedia_root / "images/val_test2020",
            ),
        ]

        for split_name, annotation_path, image_dir in fashionpedia_specs:
            print(f"正在读取：{annotation_path}")
            coco = load_json(annotation_path)

            for image in coco.get("images", []):
                file_name = image["file_name"]
                image_path = image_dir / Path(file_name).name

                if split_name == "validation":
                    fashionpedia_val_names.add(Path(file_name).name)
                    fashionpedia_val_ids.add(image["id"])

                emit_record(
                    manifest,
                    {
                        "dataset": "Fashionpedia",
                        "source_split": split_name,
                        "split_locked": True,
                        "image_id": image["id"],
                        "image_path": str(
                            image_path.relative_to(root)
                        ),
                        "image_abs_path": str(image_path.resolve()),
                        "annotation_path": str(
                            annotation_path.resolve()
                        ),
                        "width": image.get("width"),
                        "height": image.get("height"),
                    },
                )

            del coco

    # -------------------------------------------------
    # 冻结 Fashionpedia val600
    # -------------------------------------------------
    val600 = load_json(fashionpedia_val600_json)
    frozen_records = []

    val600_names = set()
    val600_source_ids = set()

    for image in val600.get("images", []):
        file_name = Path(image["file_name"]).name
        source_image_id = image.get(
            "source_image_id",
            image.get("id"),
        )

        val600_names.add(file_name)
        val600_source_ids.add(source_image_id)

        image_path = (
            fashionpedia_root
            / "images/val_test2020"
            / file_name
        )

        frozen_records.append(
            {
                "dataset": "Fashionpedia",
                "evaluation_name": "fashionpedia_val600",
                "source_split": "validation",
                "source_image_id": source_image_id,
                "file_name": file_name,
                "image_abs_path": str(image_path.resolve()),
                "exists": image_path.is_file(),
                "never_use_for_training": True,
            }
        )

    frozen_eval = {
        "name": "frozen_eval_images_v1",
        "rules": [
            "Fashionpedia val600 must never enter training.",
            "DeepFashion2 validation must not enter training.",
            "Fashionpedia official validation must not enter training.",
            "Derived regions and queries must inherit the parent image split.",
        ],
        "fashionpedia_val600_count": len(frozen_records),
        "all_names_in_official_val": val600_names.issubset(
            fashionpedia_val_names
        ),
        "all_source_ids_in_official_val": val600_source_ids.issubset(
            fashionpedia_val_ids
        ),
        "images": frozen_records,
    }

    write_json(
        out_dir / "frozen_eval_images.json",
        frozen_eval,
    )

    resource_forks = [
        path
        for path in fashionai_root.rglob("._*")
        if path.is_file()
    ]

    report = {
        "project_root": str(root),
        "manifest": str(manifest_path),
        "counts": dict(counts),
        "fashionai_csv_rows": fashionai_csv_rows,
        "fashionai_unique_images": len(fashionai_paths),
        "fashionai_resource_forks_ignored": len(resource_forks),
        "fashionpedia_val600_count": len(frozen_records),
        "fashionpedia_val600_names_in_official_val":
            frozen_eval["all_names_in_official_val"],
        "fashionpedia_val600_ids_in_official_val":
            frozen_eval["all_source_ids_in_official_val"],
        "missing_required_paths": missing_required_paths,
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images[:100],
        "duplicate_logical_key_count":
            len(duplicate_logical_keys),
        "duplicate_logical_key_sample":
            duplicate_logical_keys[:100],
    }

    write_json(
        out_dir / "source_inventory.json",
        report,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
