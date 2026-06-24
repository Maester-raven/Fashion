#!/usr/bin/env python3
"""Detailed error analysis for COCO-style RTMDet instance predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pycocotools import mask as mask_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--candidate-score", type=float, default=0.05)
    parser.add_argument("--match-iou", type=float, default=0.05)
    parser.add_argument("--loose-iou", type=float, default=0.01)
    parser.add_argument("--fragment-contain", type=float, default=0.80)
    parser.add_argument("--max-examples-per-reason", type=int, default=20)
    return parser.parse_args()


def load_module(path):
    spec = importlib.util.spec_from_file_location("prd_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def area_bin(area):
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def pair_stats(pred, gt):
    iou = float(mask_utils.iou([pred["rle"]], [gt["rle"]], [False])[0, 0])
    if iou <= 0:
        return 0.0, 0.0, 0.0
    inter = iou * (pred["area"] + gt["area"]) / (1.0 + iou)
    return (
        iou,
        inter / max(1.0, pred["area"]),
        inter / max(1.0, gt["area"]),
    )


def counter_rows(counter, key_names, limit=None):
    rows = []
    for key, count in counter.most_common(limit):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({**dict(zip(key_names, key)), "count": count})
    return rows


def main():
    cli = parse_args()
    module = load_module(cli.evaluator)
    with open(cli.gt_json, "r", encoding="utf-8") as file:
        gt_json = json.load(file)
    with open(cli.pred_json, "r", encoding="utf-8") as file:
        pred_json = json.load(file)

    args = SimpleNamespace(
        min_score=cli.candidate_score,
        match_iou=cli.match_iou,
        thresholds=None,
        resolver="prd_best",
        mask_iou_thr=0.30,
        same_contain_thr=0.85,
        cross_contain_thr=0.92,
        skirt_score_penalty=0.85,
        pants_score_bonus=1.06,
        dress_score_bonus=1.00,
        accessory_cloth_contain_thr=0.98,
        accessory_filter_score_max=0.60,
    )
    evaluator = module.Evaluator(gt_json, pred_json, args)
    filtered = evaluator.filter_predictions()
    image_names = {item["id"]: item.get("file_name", "") for item in gt_json["images"]}

    fp_reasons = Counter()
    fn_reasons = Counter()
    fp_by_class_reason = Counter()
    fn_by_class_reason = Counter()
    confusion_fp = Counter()
    confusion_fn = Counter()
    fp_area = Counter()
    fn_area = Counter()
    resolver_removed = Counter()
    score_bands = defaultdict(list)
    examples = defaultdict(list)
    totals = Counter()
    per_class = defaultdict(Counter)

    image_ids = sorted(set(evaluator.gt_by_img_cat) | set(evaluator.pred_by_img))
    for image_id in image_ids:
        raw_preds = evaluator.pred_by_img.get(image_id, [])
        output_preds = filtered.get(image_id, [])
        output_ids = {pred["idx"] for pred in output_preds}
        raw_by_cat = defaultdict(list)
        for pred in raw_preds:
            raw_by_cat[pred["category_id"]].append(pred)
        output_by_cat = defaultdict(list)
        for pred in output_preds:
            output_by_cat[pred["category_id"]].append(pred)

        matched_pred_ids = set()
        matched_gt_keys = set()
        for cid in evaluator.cat_ids:
            preds = output_by_cat.get(cid, [])
            gts = evaluator.gt_by_img_cat[image_id].get(cid, [])
            if not preds or not gts:
                continue
            matrix = mask_utils.iou(
                [pred["rle"] for pred in preds],
                [gt["rle"] for gt in gts],
                [False] * len(gts),
            )
            used_gt = set()
            for pred_index, pred in sorted(
                enumerate(preds), key=lambda item: item[1]["score"], reverse=True
            ):
                best_gt = -1
                best_iou = 0.0
                for gt_index in range(len(gts)):
                    if gt_index in used_gt:
                        continue
                    value = float(matrix[pred_index, gt_index]) if matrix.size else 0.0
                    if value > best_iou:
                        best_iou = value
                        best_gt = gt_index
                if best_gt >= 0 and best_iou >= cli.match_iou:
                    used_gt.add(best_gt)
                    matched_pred_ids.add(pred["idx"])
                    matched_gt_keys.add((image_id, cid, best_gt))
                    score_bands[(evaluator.cat_id_to_name[cid], "tp")].append(pred["score"])

        all_gts = []
        for cid in evaluator.cat_ids:
            for gt_index, gt in enumerate(evaluator.gt_by_img_cat[image_id].get(cid, [])):
                all_gts.append((cid, gt_index, gt))
                totals["gt"] += 1
                per_class[evaluator.cat_id_to_name[cid]]["gt"] += 1

        totals["pred"] += len(output_preds)
        totals["tp"] += len(matched_pred_ids)
        for pred in output_preds:
            pred_name = evaluator.cat_id_to_name[pred["category_id"]]
            per_class[pred_name]["pred"] += 1
            if pred["idx"] in matched_pred_ids:
                per_class[pred_name]["tp"] += 1
                continue

            best = None
            for gt_cid, gt_index, gt in all_gts:
                iou, pred_contain, gt_cover = pair_stats(pred, gt)
                candidate = (iou, pred_contain, gt_cover, gt_cid, gt_index, gt)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
            best_iou, pred_contain, gt_cover, gt_cid, _, best_gt = best or (0, 0, 0, None, None, None)
            gt_name = evaluator.cat_id_to_name.get(gt_cid, "none")
            if gt_cid == pred["category_id"] and best_iou >= cli.match_iou:
                reason = "same_class_duplicate"
            elif gt_cid is not None and gt_cid != pred["category_id"] and best_iou >= cli.match_iou:
                reason = "wrong_category_overlap"
                confusion_fp[(gt_name, pred_name)] += 1
            elif pred_contain >= cli.fragment_contain:
                reason = "fragment_inside_gt"
            elif best_iou >= cli.loose_iou:
                reason = "loose_overlap_or_bad_mask"
            else:
                reason = "background_or_unmatched"
            fp_reasons[reason] += 1
            fp_by_class_reason[(pred_name, reason)] += 1
            fp_area[(pred_name, area_bin(pred["area"]))] += 1
            score_bands[(pred_name, "fp")].append(pred["score"])
            if len(examples[("fp", reason)]) < cli.max_examples_per_reason:
                examples[("fp", reason)].append(
                    {
                        "image_id": image_id,
                        "file_name": image_names.get(image_id, ""),
                        "pred": pred_name,
                        "score": pred["score"],
                        "best_gt": gt_name,
                        "mask_iou": best_iou,
                        "pred_contain": pred_contain,
                        "gt_cover": gt_cover,
                        "area": pred["area"],
                    }
                )

        for cid in evaluator.cat_ids:
            gt_name = evaluator.cat_id_to_name[cid]
            gts = evaluator.gt_by_img_cat[image_id].get(cid, [])
            for gt_index, gt in enumerate(gts):
                if (image_id, cid, gt_index) in matched_gt_keys:
                    continue
                same_raw = []
                for pred in raw_by_cat.get(cid, []):
                    iou, pred_contain, gt_cover = pair_stats(pred, gt)
                    same_raw.append((iou, pred["score"], pred_contain, gt_cover, pred))
                same_raw.sort(key=lambda item: (item[0], item[1]), reverse=True)
                best_same = same_raw[0] if same_raw else (0, 0, 0, 0, None)

                best_any = None
                for pred in raw_preds:
                    iou, pred_contain, gt_cover = pair_stats(pred, gt)
                    candidate = (iou, gt_cover, pred_contain, pred["score"], pred)
                    if best_any is None or candidate[:4] > best_any[:4]:
                        best_any = candidate
                best_any = best_any or (0, 0, 0, 0, None)
                best_any_pred = best_any[4]
                threshold = evaluator.thresholds[cid]

                if best_same[0] >= cli.match_iou:
                    same_pred = best_same[4]
                    if same_pred["score"] < threshold:
                        reason = "same_class_below_output_threshold"
                    elif same_pred["idx"] not in output_ids:
                        reason = "same_class_removed_by_resolver"
                        resolver_removed[gt_name] += 1
                    else:
                        reason = "same_class_candidate_taken_by_other_gt"
                elif best_any_pred is not None and best_any[0] >= cli.match_iou:
                    pred_name = evaluator.cat_id_to_name[best_any_pred["category_id"]]
                    if pred_name == gt_name:
                        reason = "same_class_bad_mask"
                    else:
                        reason = "wrong_category_candidate"
                        confusion_fn[(gt_name, pred_name)] += 1
                elif best_same[0] >= cli.loose_iou:
                    reason = "same_class_bad_mask"
                elif best_any_pred is not None and best_any[0] >= cli.loose_iou:
                    pred_name = evaluator.cat_id_to_name[best_any_pred["category_id"]]
                    reason = "weak_spatial_candidate"
                    if pred_name != gt_name:
                        confusion_fn[(gt_name, pred_name)] += 1
                else:
                    reason = "no_spatial_candidate"

                fn_reasons[reason] += 1
                fn_by_class_reason[(gt_name, reason)] += 1
                fn_area[(gt_name, area_bin(gt["area"]))] += 1
                if len(examples[("fn", reason)]) < cli.max_examples_per_reason:
                    best_pred_name = (
                        evaluator.cat_id_to_name[best_any_pred["category_id"]]
                        if best_any_pred is not None
                        else "none"
                    )
                    examples[("fn", reason)].append(
                        {
                            "image_id": image_id,
                            "file_name": image_names.get(image_id, ""),
                            "gt": gt_name,
                            "area": gt["area"],
                            "best_same_iou": best_same[0],
                            "best_same_score": best_same[1],
                            "best_any_iou": best_any[0],
                            "best_any_gt_cover": best_any[1],
                            "best_pred": best_pred_name,
                            "best_pred_score": best_any[3],
                            "output_threshold": threshold,
                        }
                    )

    report = {
        "setup": {
            "candidate_score": cli.candidate_score,
            "match_iou": cli.match_iou,
            "thresholds": module.DEFAULT_THRESHOLDS,
            "resolver": "prd_best_original",
        },
        "totals": {
            "gt": totals["gt"],
            "pred": totals["pred"],
            "tp": totals["tp"],
            "fp": totals["pred"] - totals["tp"],
            "fn": totals["gt"] - totals["tp"],
            "recall": totals["tp"] / max(1, totals["gt"]),
            "precision_like": totals["tp"] / max(1, totals["pred"]),
        },
        "per_class": {},
        "fp_reasons": counter_rows(fp_reasons, ["reason"]),
        "fn_reasons": counter_rows(fn_reasons, ["reason"]),
        "fp_by_class_reason": counter_rows(fp_by_class_reason, ["class", "reason"]),
        "fn_by_class_reason": counter_rows(fn_by_class_reason, ["class", "reason"]),
        "fp_area_bins": counter_rows(fp_area, ["class", "area_bin"]),
        "fn_area_bins": counter_rows(fn_area, ["class", "area_bin"]),
        "fp_confusions": counter_rows(confusion_fp, ["gt", "pred"], 50),
        "fn_confusions": counter_rows(confusion_fn, ["gt", "pred"], 50),
        "resolver_removed_fn": counter_rows(resolver_removed, ["class"]),
        "score_summary": {},
        "examples": {
            f"{kind}_{reason}": values
            for (kind, reason), values in examples.items()
        },
    }
    for name, values in per_class.items():
        report["per_class"][name] = {
            **values,
            "fp": values["pred"] - values["tp"],
            "fn": values["gt"] - values["tp"],
            "recall": values["tp"] / max(1, values["gt"]),
            "precision_like": values["tp"] / max(1, values["pred"]),
        }
    for (name, outcome), scores in score_bands.items():
        values = np.asarray(scores, dtype=float)
        report["score_summary"][f"{name}_{outcome}"] = {
            "count": len(scores),
            "mean": float(values.mean()),
            "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
        }

    output = Path(cli.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        "totals": report["totals"],
        "fp_reasons": report["fp_reasons"],
        "fn_reasons": report["fn_reasons"],
        "top_fp_confusions": report["fp_confusions"][:15],
        "top_fn_confusions": report["fn_confusions"][:15],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
