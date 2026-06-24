#!/usr/bin/env python3

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def stable_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(
                self.parent[value]
            )
        return self.parent[value]

    def union(self, left, right):
        left = self.find(left)
        right = self.find(right)

        if left == right:
            return False

        if self.rank[left] < self.rank[right]:
            left, right = right, left

        self.parent[right] = left

        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1

        return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "data_manifests/v1/source_images.jsonl"
        ),
    )
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path(
            "data_interim/"
            "fashionai_images_with_attributes.jsonl"
        ),
    )
    parser.add_argument(
        "--exact-groups",
        type=Path,
        default=Path(
            "data_manifests/v1/exact_dedup/"
            "exact_duplicate_groups.json"
        ),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path(
            "data_manifests/v1/"
            "fashionai_visual_review/"
            "review_decisions.json"
        ),
    )
    parser.add_argument(
        "--perceptual-hashes",
        type=Path,
        default=Path(
            "data_manifests/v1/"
            "fashionai_visual_dedup/"
            "fashionai_perceptual_hashes.jsonl"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "data_manifests/v1/fashionai_canonical"
        ),
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_records = [
        record
        for record in read_jsonl(args.source_manifest)
        if record["dataset"] == "FashionAI"
    ]

    source_by_path = {
        record["image_path"]: record
        for record in source_records
    }

    source_paths = sorted(source_by_path)
    path_to_index = {
        path: index
        for index, path in enumerate(source_paths)
    }

    attributes_by_path = {
        record["image_path"]: record.get("attributes", [])
        for record in read_jsonl(args.attributes)
    }

    image_meta_by_path = {}

    for record in read_jsonl(args.perceptual_hashes):
        if not record.get("ok"):
            continue

        image_meta_by_path[record["image_path"]] = {
            "width": record["width"],
            "height": record["height"],
        }

    dsu = DisjointSet(len(source_paths))

    with args.exact_groups.open(
        "r",
        encoding="utf-8",
    ) as f:
        exact_data = json.load(f)

    exact_path_to_group = {}
    exact_union_count = 0

    for group in exact_data["groups"]:
        paths = [
            record["image_path"]
            for record in group["records"]
            if record["dataset"] == "FashionAI"
            and record["image_path"] in path_to_index
        ]

        if len(paths) < 2:
            continue

        for path in paths:
            exact_path_to_group[path] = group["group_id"]

        first = path_to_index[paths[0]]

        for path in paths[1:]:
            if dsu.union(first, path_to_index[path]):
                exact_union_count += 1

    with args.decisions.open(
        "r",
        encoding="utf-8",
    ) as f:
        decision_data = json.load(f)

    pending = [
        group_id
        for group_id, decision
        in decision_data["groups"].items()
        if decision["decision"] == "pending"
    ]

    if pending:
        raise RuntimeError(
            f"仍有未审核组：{pending}"
        )

    preferred_by_path = {}
    near_merge_groups = 0
    near_separate_groups = 0
    near_union_count = 0
    near_path_to_group = {}

    for group_id, decision in decision_data["groups"].items():
        paths = decision["members"]

        if decision["decision"] == "separate":
            near_separate_groups += 1
            continue

        if decision["decision"] != "merge":
            raise ValueError(
                f"未知审核结论：{decision['decision']}"
            )

        near_merge_groups += 1

        for path in paths:
            near_path_to_group[path] = group_id

        first = path_to_index[paths[0]]

        for path in paths[1:]:
            if dsu.union(first, path_to_index[path]):
                near_union_count += 1

        preferred_path = decision.get(
            "preferred_image_path"
        )

        if preferred_path:
            if preferred_path not in paths:
                raise ValueError(
                    f"{group_id} 的 preferred_image_path "
                    f"不在 members 中：{preferred_path}"
                )

            for path in paths:
                preferred_by_path[path] = preferred_path

    components = defaultdict(list)

    for path in source_paths:
        component_root = dsu.find(path_to_index[path])
        components[component_root].append(path)

    canonical_records = []
    path_mapping = []
    conflicts = []

    for component_paths in components.values():
        component_paths = sorted(component_paths)

        preferred_candidates = {
            preferred_by_path[path]
            for path in component_paths
            if path in preferred_by_path
        }

        if len(preferred_candidates) > 1:
            raise ValueError(
                "同一合并组件存在多个 preferred_image_path："
                f"{preferred_candidates}"
            )

        if preferred_candidates:
            canonical_path = next(
                iter(preferred_candidates)
            )
            selection_strategy = "manual_preferred"
        else:
            def image_rank(path):
                meta = image_meta_by_path.get(path, {})
                width = meta.get("width", 0)
                height = meta.get("height", 0)
                size = source_by_path[path].get(
                    "size_bytes", 0
                )

                return (
                    -(width * height),
                    -size,
                    path,
                )

            canonical_path = sorted(
                component_paths,
                key=image_rank,
            )[0]
            selection_strategy = (
                "highest_resolution_then_file_size"
            )

        attribute_items = defaultdict(list)

        for path in component_paths:
            for attribute in attributes_by_path.get(
                path, []
            ):
                attribute_items[
                    attribute["attribute_id"]
                ].append((path, attribute))

        merged_attributes = []

        for attribute_id, items in sorted(
            attribute_items.items()
        ):
            values = {
                item["attribute_value"]
                for _, item in items
            }

            if len(values) > 1:
                conflicts.append(
                    {
                        "source_image_paths":
                            component_paths,
                        "attribute_id": attribute_id,
                        "values": sorted(values),
                    }
                )
                continue

            base = dict(items[0][1])

            base["source_image_paths"] = sorted({
                path for path, _ in items
            })

            base["confidence"] = max(
                float(item.get("confidence", 1.0))
                for _, item in items
            )

            merged_attributes.append(base)

        component_key = "\n".join(component_paths)
        canonical_id = (
            "fashionai_"
            + stable_hash(component_key)[:16]
        )

        canonical_source = source_by_path[
            canonical_path
        ]
        meta = image_meta_by_path.get(
            canonical_path, {}
        )

        exact_group_ids = sorted({
            exact_path_to_group[path]
            for path in component_paths
            if path in exact_path_to_group
        })

        near_group_ids = sorted({
            near_path_to_group[path]
            for path in component_paths
            if path in near_path_to_group
        })

        canonical_record = {
            "image_uid": canonical_id,
            "dataset": "FashionAI",
            "source_split": "unassigned",
            "split": None,
            "image_path": canonical_path,
            "image_abs_path":
                canonical_source["image_abs_path"],
            "width": meta.get("width"),
            "height": meta.get("height"),
            "source_image_paths": component_paths,
            "source_image_count": len(component_paths),
            "selection_strategy": selection_strategy,
            "exact_duplicate_group_ids":
                exact_group_ids,
            "near_duplicate_group_ids":
                near_group_ids,
            "attributes": merged_attributes,
            "attribute_count":
                len(merged_attributes),
        }

        canonical_records.append(canonical_record)

        for path in component_paths:
            path_mapping.append(
                {
                    "source_image_path": path,
                    "canonical_image_uid":
                        canonical_id,
                    "canonical_image_path":
                        canonical_path,
                }
            )

    if conflicts:
        write_json(
            args.out_dir / "attribute_conflicts.json",
            conflicts,
        )
        raise RuntimeError(
            f"发现 {len(conflicts)} 个属性冲突，"
            "已写入 attribute_conflicts.json"
        )

    # --------------------------------------------
    # 多标签分层切分
    # --------------------------------------------
    split_names = ["train", "val", "test"]
    split_ratios = {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
    }

    total = len(canonical_records)
    target_sizes = {
        "train": round(total * 0.8),
        "val": round(total * 0.1),
    }
    target_sizes["test"] = (
        total
        - target_sizes["train"]
        - target_sizes["val"]
    )

    def labels_for(record):
        return sorted({
            (
                attribute["attribute_id"]
                + "="
                + attribute["attribute_value"]
            )
            for attribute in record["attributes"]
        })

    label_totals = Counter()

    for record in canonical_records:
        label_totals.update(labels_for(record))

    label_targets = {
        split: {
            label: count * split_ratios[split]
            for label, count in label_totals.items()
        }
        for split in split_names
    }

    split_counts = Counter()
    split_label_counts = {
        split: Counter()
        for split in split_names
    }

    def rarity_key(record):
        labels = labels_for(record)

        minimum_frequency = min(
            label_totals[label]
            for label in labels
        )

        return (
            minimum_frequency,
            -len(labels),
            stable_hash(record["image_uid"]),
        )

    ordered_records = sorted(
        canonical_records,
        key=rarity_key,
    )

    for record in ordered_records:
        labels = labels_for(record)
        candidates = []

        for split in split_names:
            if split_counts[split] >= target_sizes[split]:
                continue

            label_need = 0.0

            for label in labels:
                target = label_targets[split][label]
                current = split_label_counts[
                    split
                ][label]

                label_need += (
                    target - current
                ) / max(label_totals[label], 1)

            remaining_ratio = (
                target_sizes[split]
                - split_counts[split]
            ) / max(target_sizes[split], 1)

            score = label_need + 0.1 * remaining_ratio

            tie_break = stable_hash(
                record["image_uid"] + ":" + split
            )

            candidates.append(
                (
                    -score,
                    tie_break,
                    split,
                )
            )

        selected_split = sorted(candidates)[0][2]
        record["split"] = selected_split

        split_counts[selected_split] += 1
        split_label_counts[selected_split].update(
            labels
        )

    canonical_records.sort(
        key=lambda record: record["image_uid"]
    )
    path_mapping.sort(
        key=lambda record: record["source_image_path"]
    )

    split_records = {
        split: [
            record
            for record in canonical_records
            if record["split"] == split
        ]
        for split in split_names
    }

    write_jsonl(
        args.out_dir / "fashionai_canonical_all.jsonl",
        canonical_records,
    )

    for split in split_names:
        write_jsonl(
            args.out_dir
            / f"fashionai_{split}.jsonl",
            split_records[split],
        )

    mapping_by_uid = {
        record["image_uid"]: record["split"]
        for record in canonical_records
    }

    for item in path_mapping:
        item["split"] = mapping_by_uid[
            item["canonical_image_uid"]
        ]

    write_jsonl(
        args.out_dir / "source_path_mapping.jsonl",
        path_mapping,
    )

    multi_attribute_records = sum(
        record["attribute_count"] > 1
        for record in canonical_records
    )

    report = {
        "source_paths": len(source_paths),
        "canonical_images": len(canonical_records),
        "collapsed_paths":
            len(source_paths) - len(canonical_records),
        "exact_duplicate_groups":
            len(exact_data["groups"]),
        "exact_unions": exact_union_count,
        "near_merge_groups": near_merge_groups,
        "near_separate_groups": near_separate_groups,
        "near_unions": near_union_count,
        "attribute_conflicts": len(conflicts),
        "multi_attribute_canonical_images":
            multi_attribute_records,
        "split_counts": dict(split_counts),
        "target_split_counts": target_sizes,
        "source_paths_mapped": len(path_mapping),
        "label_counts_total":
            dict(sorted(label_totals.items())),
        "label_counts_by_split": {
            split: dict(
                sorted(
                    split_label_counts[split].items()
                )
            )
            for split in split_names
        },
        "raw_files_modified": False,
    }

    write_json(
        args.out_dir / "split_report.json",
        report,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
