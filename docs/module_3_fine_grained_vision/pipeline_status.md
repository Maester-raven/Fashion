# Pipeline Status

AutoDL workspace:

```bash
/root/autodl-tmp/fashion_prd
```

Conda environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vibe
```

## Completed

1. Unified category taxonomy is defined in `configs/attribute_schema.json`.
2. FashionAI original attribute names and values are preserved.
3. `attribute -> target_part` mapping is defined and exported.
4. DeepFashion2 original categories are mapped to the unified category system.
5. DeepFashion2 bbox and segmentation annotations are converted into JSONL.
6. FashionAI original attribute labels are parsed into the unified attribute format.
7. SAM-HQ and DINOv2 repositories/checkpoints are installed and verified on AutoDL.
8. SAM-HQ was fine-tuned on a 20k DeepFashion2 garment-instance subset.
9. SAM-HQ automatic-mask FashionAI segmentation was tested and stopped because it produced too many part-level candidates.
10. Mask2Former training pipeline has been added for category-aware multi-garment instance segmentation.
11. DeepFashion2 has been converted to COCO instance-segmentation format for Mask2Former.
12. A Mask2Former smoke training run completed successfully.
13. A Mask2Former FashionAI inference script has been added for category-aware multi-instance segmentation.
14. The 20k-image / 3k-step Mask2Former training run completed successfully.
15. A 20-image FashionAI inference smoke test completed successfully.
16. Full FashionAI Mask2Former inference has been started in the background.
17. Full FashionAI Mask2Former inference completed successfully.
18. FashionAI instance category bindings were repaired using unique FashionAI attribute category constraints.
19. FashionAI unified multi-instance, multi-attribute, multi-region output has been generated.
20. DINOv2 global/local FashionAI attribute training manifests have been generated.
21. DINOv2 frozen-backbone global/local attribute heads have been trained.

## Current Taxonomy Output

Generated on AutoDL under:

```bash
/root/autodl-tmp/fashion_prd/data_taxonomy
```

Important files:

- `category_taxonomy.json`
- `fashionai_attribute_taxonomy.json`
- `fashionai_attribute_values.json`
- `attribute_target_part_map.json`
- `attribute_target_part_map.csv`
- `category_attribute_part_rules.jsonl`
- `label_space.json`
- `taxonomy_report.json`

Current counts:

- Categories: 8
- FashionAI attributes: 8
- Attribute values: 54
- Target parts: 6
- Global attributes: 3
- Local attributes: 5
- Valid category-attribute-part rules: 15

## Segmentation Decision Log

### 2026-06-10

FashionAI does not provide garment instance boxes or masks. A first SAM-HQ-only
strategy was implemented and tested:

```text
FashionAI image -> SAM-HQ automatic masks -> filter candidates -> bind attributes
```

This successfully produced multiple masks per image, but it also produced
part-level and fragment-level candidates such as collars, cuffs, pant legs, and
edge fragments. That behavior is not suitable as the primary garment-instance
source because the target dataset needs clothing instances such as top, pants,
skirt, outerwear, and dress.

The partial SAM-HQ automatic-mask output is kept only as a diagnostic baseline:

```bash
/root/autodl-tmp/fashion_prd/data_interim/fashionai_samhq_instances_multi.jsonl
```

It was stopped at 3,580 processed FashionAI images and should not be used as the
final FashionAI instance source.

The segmentation plan is changed to:

```text
DeepFashion2 -> train Mask2Former for garment instance segmentation
FashionAI -> Mask2Former predicts category + bbox + mask instances
Optional -> SAM-HQ refines Mask2Former instance masks or boxes
FashionAI attributes -> bind to the predicted garment instances
```

Mask2Former is now the preferred instance-discovery model because it can produce
category-aware instance masks. SAM-HQ remains useful as an edge-refinement model.

Implementation progress:

- DeepFashion2 has been converted to COCO instance-segmentation format for Mask2Former.
- Full COCO outputs on AutoDL:

```bash
/root/autodl-tmp/fashion_prd/data_mask2former/deepfashion2_coco/deepfashion2_train_coco_instance.json
/root/autodl-tmp/fashion_prd/data_mask2former/deepfashion2_coco/deepfashion2_val_coco_instance.json
```

- Full COCO counts:

```text
train images:      191961
train instances:   312186
val images:         32153
val instances:      52490
```

- Hugging Face `transformers` Mask2Former support is available in the `vibe` environment.
- Direct Hugging Face access is blocked from AutoDL, but `HF_ENDPOINT=https://hf-mirror.com` works.
- A 5-step Mask2Former smoke training run completed successfully and saved a checkpoint under:

```bash
/root/autodl-tmp/fashion_prd/work_dirs/mask2former_fashion_smoke/final
```

- A 20k-image / 3k-step Mask2Former training run completed successfully:

```bash
/root/autodl-tmp/fashion_prd/scripts/run_mask2former_20k_train.sh
```

- Background log:

```bash
/root/autodl-tmp/fashion_prd/logs/mask2former_20k_3ksteps.log
```

- Output directory:

```bash
/root/autodl-tmp/fashion_prd/work_dirs/mask2former_fashion_20k_3ksteps
```

- Current confirmed training events:

```text
step 500:  train_loss 27.5939, val_loss 31.8276
step 1000: train_loss 42.2778, val_loss 29.0846
step 1500: train_loss 21.1212, val_loss 25.3932
step 2000: train_loss 17.6251, val_loss 23.0173
step 2500: train_loss 24.7910, val_loss 22.6833
step 3000: train_loss 22.3850, val_loss 22.1281
```

- `checkpoint-1000`, `checkpoint-2000`, `checkpoint-3000`, `final`, and
  `train_report.json` have been saved successfully.

The 20k run uses `HF_ENDPOINT=https://hf-mirror.com` because direct Hugging Face
access is blocked from AutoDL. It also uses `ignore_index=0` in the Mask2Former
image processor so the rendered background id does not enter the foreground
category label map.

FashionAI inference after training should use:

```bash
/root/autodl-tmp/fashion_prd/scripts/run_fashionai_mask2former_segment.sh
```

Default output:

```bash
/root/autodl-tmp/fashion_prd/data_interim/fashionai_mask2former_instances.jsonl
```

20-image FashionAI smoke inference report:

```text
records_written: 20
instances_total: 30
instances_min: 1
instances_max: 2
instances_mean: 1.5
attributes_bound: 20
missing_image/read_error/empty_prediction: 0
```

Full FashionAI inference log and PID:

```bash
/root/autodl-tmp/fashion_prd/logs/fashionai_mask2former_segment.log
/root/autodl-tmp/fashion_prd/logs/fashionai_mask2former_segment.pid
```

Full FashionAI inference report:

```text
total_seen: 10080
records_written: 10031
empty_prediction: 49
instances_total: 16272
instances_min: 1
instances_max: 5
instances_mean: 1.6222
attributes_bound: 10031
```

The raw Mask2Former category predictions produced 3,919 invalid
category-attribute bindings. Most came from attributes with exactly one valid
category, such as `pant_length -> pants`, `skirt_length -> skirt`, and
`coat_length/lapel_design -> outerwear`. These were repaired without rerunning
inference:

```bash
/root/autodl-tmp/fashion_prd/scripts/repair_fashionai_category_bindings.py
```

Repair report:

```text
already_valid: 6112
invalid_before: 3919
repaired: 3908
unrepaired: 11
```

The repaired FashionAI instance file is:

```bash
/root/autodl-tmp/fashion_prd/data_interim/fashionai_mask2former_instances_repaired.jsonl
```

The current FashionAI unified region file is:

```bash
/root/autodl-tmp/fashion_prd/data_unified/fashionai_unified_regions.jsonl
```

Unified FashionAI region counts:

```text
records: 10031
instances: 16272
attributes: 10031
global_regions: 3536
local_regions: 6495
remaining invalid category-attribute bindings: 11
```

## DINOv2 Attribute Models

### 2026-06-10

FashionAI unified regions were converted into DINOv2 training manifests:

```bash
/root/autodl-tmp/fashion_prd/data_dinov2/fashionai_attributes/fashionai_dinov2_global.jsonl
/root/autodl-tmp/fashion_prd/data_dinov2/fashionai_attributes/fashionai_dinov2_local.jsonl
/root/autodl-tmp/fashion_prd/data_dinov2/fashionai_attributes/dinov2_attribute_data_report.json
```

The remaining 11 invalid category-attribute bindings were skipped.

Manifest counts:

```text
samples: 10020
global samples: 3536
local samples: 6484
global train / val: 3215 / 321
local train / val: 5903 / 581
```

Training used frozen DINOv2 ViT-B/14 features and lightweight per-attribute
classification heads:

```bash
/root/autodl-tmp/fashion_prd/checkpoints/dinov2/hub/checkpoints/dinov2_vitb14_pretrain.pth
/root/autodl-tmp/fashion_prd/scripts/run_dinov2_attribute_heads.sh
```

Model outputs:

```bash
/root/autodl-tmp/fashion_prd/work_dirs/dinov2_attribute_heads/global/attribute_heads.pt
/root/autodl-tmp/fashion_prd/work_dirs/dinov2_attribute_heads/local/attribute_heads.pt
```

Final validation results after 12 epochs:

```text
global:
  attr_coat_length: 0.6029
  attr_pant_length: 0.6824
  attr_skirt_length: 0.6900
  val_loss: 0.9150

local:
  attr_collar_design: 0.7600
  attr_lapel_design: 0.7901
  attr_neck_design: 0.5789
  attr_neckline_design: 0.6053
  attr_sleeve_length: 0.6536
  val_loss: 1.0886
```

## Mask2Former + SAM-HQ Refinement Evaluation

### 2026-06-11

A DeepFashion2 validation subset experiment was added to quantify whether
Mask2Former-guided SAM-HQ refinement improves segmentation quality.

Evaluation script:

```bash
/root/autodl-tmp/fashion_prd/scripts/evaluate_mask2former_samhq_refinement.py
```

Experiment setup:

```text
Dataset: DeepFashion2 validation COCO instance format
Images: 200 shuffled validation images
GT instances: 330
Mask2Former predictions: 391
Matched same-category instances: 214
GT match rate: 0.6485
Prompt: Mask2Former bbox -> SAM-HQ
SAM-HQ mode: multimask output, select highest SAM-HQ score
```

Outputs:

```bash
/root/autodl-tmp/fashion_prd/work_dirs/refinement_eval_val200_shuffle/summary.json
/root/autodl-tmp/fashion_prd/work_dirs/refinement_eval_val200_shuffle/matched_instances.jsonl
/root/autodl-tmp/fashion_prd/work_dirs/refinement_eval_val200_shuffle/image_stats.jsonl
```

Overall results:

```text
Mask2Former mask IoU:        0.7795
SAM-HQ refined mask IoU:     0.7798
mask IoU delta:             +0.0003
mask IoU improved ratio:     0.6308

Mask2Former boundary IoU:    0.5227
SAM-HQ refined boundary IoU: 0.5777
boundary IoU delta:         +0.0550
boundary improved ratio:     0.6729

Mask2Former Dice:            0.8572
SAM-HQ refined Dice:         0.8543
Dice delta:                 -0.0030

Mask2Former bbox IoU:        0.8163
SAM-HQ refined bbox IoU:     0.8399
bbox IoU delta:             +0.0236
bbox improved ratio:         0.8084
```

Per-category boundary IoU deltas:

```text
top:   +0.0709
dress: +0.0359
pants: +0.0442
skirt: +0.0685
```

Interpretation: Mask2Former-guided SAM-HQ refinement mainly improves boundary
alignment and bbox tightness while keeping overall mask IoU essentially flat.
It is therefore useful as an optional boundary-refinement stage, but should
keep Mask2Former's category and instance identity as the source of truth.

### SAM-HQ Only vs Mask2Former-Guided SAM-HQ

A second DeepFashion2 validation subset experiment compared the real solution
choices:

```text
A. SAM-HQ automatic masks only
B. Mask2Former bbox -> SAM-HQ refinement
```

This experiment intentionally does not use DeepFashion2 GT boxes as SAM-HQ
prompts. `SAM-HQ + GT` is only an oracle upper-bound setting and is not
available on FashionAI.

Evaluation script:

```bash
/root/autodl-tmp/fashion_prd/scripts/evaluate_samhq_only_vs_guided.py
```

Outputs:

```bash
/root/autodl-tmp/fashion_prd/work_dirs/samhq_only_vs_guided_val200_shuffle/summary.json
/root/autodl-tmp/fashion_prd/work_dirs/samhq_only_vs_guided_val200_shuffle/image_stats.jsonl
/root/autodl-tmp/fashion_prd/work_dirs/samhq_only_vs_guided_val200_shuffle/samhq_auto_only_matches.jsonl
/root/autodl-tmp/fashion_prd/work_dirs/samhq_only_vs_guided_val200_shuffle/mask2former_guided_samhq_matches.jsonl
```

Experiment setup:

```text
Dataset: DeepFashion2 validation COCO instance format
Images: 200 shuffled validation images
GT instances: 330
SAM-HQ automatic predictions: 6984
Mask2Former predictions: 391
SAM-HQ only matching: category-free oracle matching
Mask2Former-guided matching: same-category matching
```

All matched instances:

```text
SAM-HQ only:
  matched_instances: 329 / 330
  mask IoU:          0.7893
  boundary IoU:      0.5642
  Dice:              0.8681
  bbox IoU:          0.8208

Mask2Former -> SAM-HQ:
  matched_instances: 233 / 330
  mask IoU:          0.7762
  boundary IoU:      0.5773
  Dice:              0.8546
  bbox IoU:          0.8300
```

On the 232 GT instances matched by both methods:

```text
SAM-HQ only:
  mask IoU:          0.7956
  boundary IoU:      0.5669
  Dice:              0.8734
  bbox IoU:          0.8257

Mask2Former -> SAM-HQ:
  mask IoU:          0.7779
  boundary IoU:      0.5794
  Dice:              0.8559
  bbox IoU:          0.8309

Guided minus SAM-HQ only:
  mask IoU:         -0.0178
  boundary IoU:     +0.0125
  Dice:             -0.0175
  bbox IoU:         +0.0052
```

Interpretation: SAM-HQ automatic masks can achieve slightly higher oracle mask
IoU on DeepFashion2 because it produces a very large candidate pool. However,
it has no category and produces many fragment candidates. Mask2Former-guided
SAM-HQ produces far fewer, category-aware garment-level candidates and improves
boundary IoU and bbox tightness on the common matched set. For the FashionAI
pipeline, Mask2Former-guided SAM-HQ remains the more usable solution, while
SAM-HQ only is best treated as a high-recall candidate generator rather than
the final instance source.

### Guided Prompt And Filter Ablation

The first two improvement ideas were tested on the same DeepFashion2 validation
subset:

```text
1. Lower Mask2Former threshold, expand boxes, and add a foreground point prompt.
2. Add geometric mask validation filters and image-level fallback to SAM-HQ automatic masks.
```

Evaluation script:

```bash
/root/autodl-tmp/fashion_prd/scripts/evaluate_guided_prompt_filter_ablation.py
```

Outputs:

```bash
/root/autodl-tmp/fashion_prd/work_dirs/guided_ablation_val200/summary.json
/root/autodl-tmp/fashion_prd/work_dirs/guided_ablation_val200_fallback/summary.json
```

No-fallback 200-image results:

```text
baseline_035_box:
  predictions:      391
  matched:          233 / 330
  match_rate_gt:    0.7061
  precision_like:   0.5959
  mask IoU:         0.7762
  boundary IoU:     0.5773
  Dice:             0.8546
  bbox IoU:         0.8300

low020_box:
  predictions:      467
  matched:          238 / 330
  match_rate_gt:    0.7212
  precision_like:   0.5096
  mask IoU:         0.7456
  boundary IoU:     0.5529
  bbox IoU:         0.7996

low020_expand008:
  predictions:      467
  matched:          234 / 330
  match_rate_gt:    0.7091
  mask IoU:         0.7595
  boundary IoU:     0.5577
  bbox IoU:         0.8264

low020_expand008_point:
  predictions:      467
  matched:          239 / 330
  match_rate_gt:    0.7242
  precision_like:   0.5118
  mask IoU:         0.7841
  boundary IoU:     0.5750
  Dice:             0.8663
  bbox IoU:         0.8314

low020_expand008_point_filter:
  predictions:      333
  matched:          178 / 330
  match_rate_gt:    0.5394
  precision_like:   0.5345
  mask IoU:         0.8103
  boundary IoU:     0.6001
  Dice:             0.8844
  bbox IoU:         0.8487
```

Image-level fallback to SAM-HQ automatic masks triggered rarely:

```text
baseline_035_box fallback images: 2
low020_expand008_point_filter fallback images: 4
low020_expand008_point fallback images: 0
```

Interpretation:

```text
Best balanced setting:
  low020_expand008_point
  score_threshold = 0.20
  mask_threshold = 0.40
  box_expand = 0.08
  positive point = distance-transform center inside the Mask2Former mask

Compared with baseline:
  GT match rate: 0.7061 -> 0.7242
  mask IoU:      0.7762 -> 0.7841
  Dice:          0.8546 -> 0.8663
  bbox IoU:      0.8300 -> 0.8314
  boundary IoU:  0.5773 -> 0.5750
```

Lowering the threshold alone increases candidates but hurts quality. Expanding
the box helps bbox tightness. Adding a robust foreground point is the main
useful prompt improvement. Current hard geometric filtering improves quality
among retained masks but hurts recall too much, so it should be used as a
ranking/downweighting signal or as a high-confidence export mode rather than as
the default hard filter. Image-level fallback does not solve the main recall
gap because most misses are not complete image-level failures.

## DeepFashion2 Mask NMS + Nested Mask Evaluation

Tested the proposed pipeline on the same 200 shuffled DeepFashion2 validation
images (`seed=42`, 330 GT instances):

```text
Mask2Former proposal
  -> bbox expansion
  -> bbox + coarse-mask foreground point prompt
  -> SAM-HQ refinement
  -> mask-IoU NMS
  -> nested small-mask handling as part candidates
```

Output directories:

```text
/root/autodl-tmp/fashion_prd/work_dirs/nms_nested_val200
/root/autodl-tmp/fashion_prd/work_dirs/nms_nested_val200_gentle
/root/autodl-tmp/fashion_prd/work_dirs/nms_nested_val200_ultrag
```

The raw refined masks before NMS are the current best balanced guided setting:

```text
raw_refined_before_nms:
  predictions:      467
  matched:          239 / 330
  match_rate_gt:    0.7242
  precision_like:   0.5118
  mask IoU:         0.7841
  boundary IoU:     0.5750
  Dice:             0.8663
  bbox IoU:         0.8314
```

Aggressive mask NMS / nested handling:

```text
same_category_nms_iou = 0.65
nested_containment    = 0.86
nested_area_ratio     = 0.28

after_mask_nms_nested:
  predictions:      435
  matched:          234 / 330
  match_rate_gt:    0.7091
  precision_like:   0.5379
  mask IoU:         0.7839
  boundary IoU:     0.5730
  Dice:             0.8663
  bbox IoU:         0.8317
  part_candidates:  17
```

Gentle mask NMS / nested handling:

```text
same_category_nms_iou = 0.80
nested_containment    = 0.92
nested_area_ratio     = 0.18

after_mask_nms_nested:
  predictions:      448
  matched:          235 / 330
  match_rate_gt:    0.7121
  precision_like:   0.5246
  mask IoU:         0.7835
  boundary IoU:     0.5727
  Dice:             0.8661
  bbox IoU:         0.8317
  part_candidates:  9
```

Ultra-gentle mask NMS / nested handling:

```text
same_category_nms_iou = 0.90
nested_containment    = 0.98
nested_area_ratio     = 0.08

after_mask_nms_nested:
  predictions:      458
  matched:          238 / 330
  match_rate_gt:    0.7212
  precision_like:   0.5197
  mask IoU:         0.7857
  boundary IoU:     0.5767
  Dice:             0.8675
  bbox IoU:         0.8329
  part_candidates:  2
```

Interpretation:

```text
Mask NMS and nested handling reduce redundant predictions, but every tested
threshold setting loses some GT matches. The ultra-gentle setting is the least
harmful: it removes 9 predictions and loses 1 match, while slightly improving
the average mask/boundary quality of retained matches.

This should be treated as an optional cleanup/export step, not the main recall
improvement path. The main bottleneck is still proposal recall and category
correctness from Mask2Former before SAM-HQ refinement.
```

## DeepFashion2 Proposal Fusion Evaluation

Tested the third proposal strategy on 200 shuffled DeepFashion2 validation
images (`seed=42`, 330 GT instances):

```text
Mask2Former proposal
+ SAM-HQ automatic proposal
+ Mask2Former query/category score for SAM-HQ automatic masks
-> candidate fusion
-> same-category GT evaluation
```

Output directories:

```text
/root/autodl-tmp/fashion_prd/work_dirs/proposal_fusion_val50
/root/autodl-tmp/fashion_prd/work_dirs/proposal_fusion_val200
```

The deployable query-score fusion variants performed as follows:

```text
baseline_guided:
  predictions:      467
  matched:          239 / 330
  match_rate_gt:    0.7242
  precision_like:   0.5118
  mask IoU:         0.7841
  boundary IoU:     0.5750

fusion_query_s030_top5:
  predictions:      620
  matched:          241 / 330
  match_rate_gt:    0.7303
  precision_like:   0.3887
  SAM-HQ auto matched additions: 3

fusion_query_s020_top8:
  predictions:      668
  matched:          241 / 330
  match_rate_gt:    0.7303
  precision_like:   0.3608
  SAM-HQ auto matched additions: 3

fusion_query_s012_top12:
  predictions:      749
  matched:          242 / 330
  match_rate_gt:    0.7333
  precision_like:   0.3231
  SAM-HQ auto matched additions: 4
```

A diagnostic oracle-complement upper bound was also tested. This uses GT only to
answer whether SAM-HQ automatic proposals contain the missing garment instances;
it is not a deployable method:

```text
oracle_samhq_complement_upper_bound:
  predictions:      558
  matched:          330 / 330
  match_rate_gt:    1.0000
  precision_like:   0.5914
  mask IoU:         0.7879
  boundary IoU:     0.5755
  SAM-HQ auto oracle additions: 91
```

Interpretation:

```text
SAM-HQ automatic proposals contain enough masks to recover the missed
DeepFashion2 garment instances. Adding only 91 oracle-selected SAM-HQ automatic
candidates to the 467 Mask2Former-guided predictions reaches 330/330 matches.

However, the deployable Mask2Former query/category score is not strong enough
to select those missing candidates: it adds 153-282 SAM-HQ masks but recovers
only 2-3 extra GT instances over the baseline.

Therefore, proposal fusion is promising, but the missing component is a strong
garment-instance candidate classifier / reranker. Mask2Former query scores
alone are insufficient. The next useful step is to train a DeepFashion2 garment
category/reranking head on SAM-HQ automatic crops/masks, or use DINOv2 to score
candidate category and instance validity before fusion.
```

## DeepFashion2 DINOv2 SAM Candidate Reranker

Tested a first DINOv2-based SAM-HQ automatic candidate reranker:

```text
SAM-HQ automatic masks on DeepFashion2 train subset
-> assign labels from DeepFashion2 GT
   positive: max mask IoU >= 0.30
   negative: max mask IoU <= 0.05
   ambiguous: ignored
-> crop candidate mask region
-> frozen DINOv2 ViT-B/14 feature
-> train lightweight objectness + category heads
-> use reranker to select SAM-HQ automatic candidates for fusion
```

Script:

```text
/root/autodl-tmp/fashion_prd/scripts/train_eval_dinov2_sam_candidate_reranker.py
```

Output directories:

```text
/root/autodl-tmp/fashion_prd/work_dirs/dinov2_sam_reranker_smoke
/root/autodl-tmp/fashion_prd/work_dirs/dinov2_sam_reranker_train300_val200
```

Smoke test (`train=30 images`, `val=20 images`) ran end-to-end:

```text
train candidates: 1024
positive:         80
negative:         944

baseline_guided:
  predictions:    52
  matched:        22 / 30
  match_rate_gt:  0.7333

rerank_obj050_cat030_top5:
  predictions:    75
  matched:        23 / 30
  match_rate_gt:  0.7667
```

Main test (`train=300 images`, `val=200 images`, `seed=42`):

```text
train candidates: 9102
positive:         711
negative:         8391
ambiguous:        340
feature_dim:      768

epoch 5 train object accuracy:              0.9914
epoch 5 train category accuracy on positive: 0.9733
```

DeepFashion2 validation results:

```text
baseline_guided:
  predictions:      467
  matched:          239 / 330
  match_rate_gt:    0.7242
  precision_like:   0.5118
  mask IoU:         0.7841
  boundary IoU:     0.5750

rerank_obj050_cat030_top5:
  predictions:      556
  matched:          250 / 330
  match_rate_gt:    0.7576
  precision_like:   0.4496
  SAM-HQ auto matched additions: 11

rerank_obj035_cat025_top8:
  predictions:      597
  matched:          250 / 330
  match_rate_gt:    0.7576
  precision_like:   0.4188
  SAM-HQ auto matched additions: 12

rerank_obj020_cat020_top12:
  predictions:      653
  matched:          250 / 330
  match_rate_gt:    0.7576
  precision_like:   0.3828
  SAM-HQ auto matched additions: 12

oracle_samhq_complement_upper_bound:
  predictions:      558
  matched:          330 / 330
  match_rate_gt:    1.0000
  precision_like:   0.5914
  SAM-HQ auto oracle additions: 91
```

Interpretation:

```text
The DINOv2 reranker is clearly better than Mask2Former query-score fusion:

Mask2Former query-score fusion:
  +153 predictions -> +2 matched GT

DINOv2 reranker fusion:
  +89 predictions  -> +11 matched GT

This confirms that training a garment-instance candidate classifier/reranker is
the right direction. However, the first 300-image prototype still recovers only
11 of the 91 oracle-recoverable missing instances, so it is not yet close to the
90% recall target.

The likely next improvements are:
1. train with more DeepFashion2 images and more positive SAM-HQ candidates;
2. improve positive labels by adding IoU buckets/regression instead of binary
   positive/negative only;
3. include geometry features, SAM predicted IoU/stability, and Mask2Former
   overlap/category priors beside DINOv2 visual features;
4. tune thresholds using a held-out split instead of train accuracy.
```

## DeepFashion2 DINOv2 Quality Reranker

Updated the SAM candidate reranker to address low Precision-like:

```text
input = DINOv2 visual feature
      + SAM predicted_iou
      + SAM stability_score
      + mask area ratio
      + bbox area ratio
      + mask fill ratio
      + aspect ratio
      + center x/y
      + full-image bbox overlap

heads:
  objectness
  category
  quality / IoU regression

fusion score:
  object_prob * category_prob * quality_prob
```

Output directory:

```text
/root/autodl-tmp/fashion_prd/work_dirs/dinov2_sam_quality_reranker_train300_val200
```

Training setup:

```text
train images:       300
train candidates:   9102
positive:           711
negative:           8391
ambiguous ignored:  340
feature_dim:        768
meta_dim:           9

epoch 5 train object accuracy:               0.9911
epoch 5 train category accuracy on positive: 0.9775
epoch 5 train quality MAE:                   0.0257
```

DeepFashion2 validation results (`val=200`, `330 GT instances`):

```text
baseline_guided:
  predictions:      467
  matched:          239 / 330
  match_rate_gt:    0.7242
  precision_like:   0.5118

previous DINOv2 reranker best recall:
  predictions:      556
  matched:          250 / 330
  match_rate_gt:    0.7576
  precision_like:   0.4496

quality_obj060_cat035_q030_top4:
  predictions:      489
  matched:          248 / 330
  match_rate_gt:    0.7515
  precision_like:   0.5072
  SAM-HQ auto matched additions: 9

quality_obj050_cat030_q020_top5:
  predictions:      506
  matched:          248 / 330
  match_rate_gt:    0.7515
  precision_like:   0.4901
  SAM-HQ auto matched additions: 9

quality_obj035_cat025_q015_top8:
  predictions:      520
  matched:          249 / 330
  match_rate_gt:    0.7545
  precision_like:   0.4788
  SAM-HQ auto matched additions: 10

quality_obj020_cat020_q010_top12:
  predictions:      555
  matched:          249 / 330
  match_rate_gt:    0.7545
  precision_like:   0.4486
  SAM-HQ auto matched additions: 10

oracle_samhq_complement_upper_bound:
  predictions:      558
  matched:          330 / 330
  match_rate_gt:    1.0000
  precision_like:   0.5914
```

Interpretation:

```text
Adding quality regression and geometry/SAM features substantially improves the
precision-recall tradeoff.

The strict quality setting recovers 9 additional GT instances while adding only
22 predictions:
  baseline: 467 predictions, 239 matches, precision_like 0.5118
  quality:  489 predictions, 248 matches, precision_like 0.5072

This is much better than the previous DINOv2 object/category-only reranker:
  previous: +89 predictions -> +11 matches, precision_like 0.4496
  quality:  +22 predictions -> +9 matches,  precision_like 0.5072

The quality reranker does not reach 90% recall yet, but it shows a usable
direction: recall can increase without collapsing Precision-like. The remaining
gap to oracle suggests the next bottleneck is ranking the 91 recoverable missing
candidates higher, not simply generating more candidates.
```

## DeepFashion2 Quality Reranker Threshold Sweep

Added eval-only checkpoint loading and stricter Top-K threshold sweep for the
quality-aware DINOv2 SAM candidate reranker.

Script update:

```text
/root/autodl-tmp/fashion_prd/scripts/train_eval_dinov2_sam_candidate_reranker.py
```

Loaded checkpoint:

```text
/root/autodl-tmp/fashion_prd/work_dirs/dinov2_sam_quality_reranker_train300_val200/dinov2_sam_candidate_reranker.pt
```

Output directory:

```text
/root/autodl-tmp/fashion_prd/work_dirs/dinov2_sam_quality_reranker_sweep_val200
```

Validation results (`val=200`, `330 GT instances`):

```text
baseline_guided:
  predictions:      467
  matched:          239 / 330
  match_rate_gt:    0.7242
  precision_like:   0.5118

quality_obj075_cat045_q045_top1:
  predictions:      481
  matched:          247 / 330
  match_rate_gt:    0.7485
  precision_like:   0.5135
  SAM-HQ auto matched additions: 8

quality_obj070_cat040_q040_top2:
  predictions:      483
  matched:          247 / 330
  match_rate_gt:    0.7485
  precision_like:   0.5114
  SAM-HQ auto matched additions: 8

quality_obj065_cat038_q035_top3:
  predictions:      484
  matched:          248 / 330
  match_rate_gt:    0.7515
  precision_like:   0.5124
  SAM-HQ auto matched additions: 9

quality_obj060_cat035_q030_top4:
  predictions:      489
  matched:          248 / 330
  match_rate_gt:    0.7515
  precision_like:   0.5072
  SAM-HQ auto matched additions: 9

quality_obj035_cat025_q015_top8:
  predictions:      520
  matched:          249 / 330
  match_rate_gt:    0.7545
  precision_like:   0.4788
  SAM-HQ auto matched additions: 10
```

Interpretation:

```text
The stricter sweep found operating points that improve recall without reducing
Precision-like.

Recommended balanced default:
  quality_obj065_cat038_q035_top3

Compared with baseline:
  predictions:      467 -> 484
  matched:          239 -> 248
  match_rate_gt:    0.7242 -> 0.7515
  precision_like:   0.5118 -> 0.5124

Strictest high-precision option:
  quality_obj075_cat045_q045_top1

Compared with baseline:
  predictions:      467 -> 481
  matched:          239 -> 247
  match_rate_gt:    0.7242 -> 0.7485
  precision_like:   0.5118 -> 0.5135

This is the first tested non-oracle setting that raises recall while keeping
Precision-like at or slightly above the original baseline. The 90% recall target
still requires better ranking/classification of the oracle-recoverable SAM-HQ
candidates, but the precision concern is now under control for a conservative
fusion mode.
```

## DeepFashion2 Quality Reranker Train-1000 Check

Expanded the quality-aware DINOv2 SAM candidate reranker training set from 300
to 1000 DeepFashion2 train images to test whether more candidate labels alone
can push recall beyond 80% while also improving Precision-like.

Command:

```bash
python /root/autodl-tmp/fashion_prd/scripts/train_eval_dinov2_sam_candidate_reranker.py \
  --train-json /root/autodl-tmp/fashion_prd/data_mask2former/deepfashion2_coco/deepfashion2_train_coco_instance.json \
  --val-json /root/autodl-tmp/fashion_prd/data_mask2former/deepfashion2_coco/deepfashion2_val_coco_instance.json \
  --mask2former-dir /root/autodl-tmp/fashion_prd/work_dirs/mask2former_fashion_20k_3ksteps/final \
  --sam-hq-repo /root/autodl-tmp/fashion_prd/repos/sam-hq-official \
  --sam-hq-checkpoint /root/autodl-tmp/fashion_prd/work_dirs/sam_hq_fashion_20k_1epoch/sam_hq_epoch_0.pth \
  --dinov2-repo /root/autodl-tmp/fashion_prd/repos/dinov2-official \
  --dinov2-checkpoint /root/autodl-tmp/fashion_prd/checkpoints/dinov2/hub/checkpoints/dinov2_vitb14_pretrain.pth \
  --output-dir /root/autodl-tmp/fashion_prd/work_dirs/dinov2_sam_quality_reranker_train1000_val200 \
  --max-train-images 1000 \
  --max-val-images 200 \
  --shuffle-val \
  --seed 42 \
  --epochs 5 \
  --sam-multimask \
  --device cuda
```

Output directory:

```text
/root/autodl-tmp/fashion_prd/work_dirs/dinov2_sam_quality_reranker_train1000_val200
```

Training setup:

```text
train images:       1000
train candidates:   30464
positive:           2328
negative:           28136
ambiguous ignored:  1161
feature_dim:        768
meta_dim:           9

epoch 5 train loss:                          0.1620
epoch 5 train object accuracy:               0.9925
epoch 5 train category accuracy on positive: 0.9626
epoch 5 train quality MAE:                   0.0198
```

DeepFashion2 validation results (`val=200`, `330 GT instances`):

```text
baseline_guided:
  predictions:      467
  matched:          239 / 330
  match_rate_gt:    0.7242
  precision_like:   0.5118

quality_obj075_cat045_q045_top1:
  predictions:      485
  matched:          246 / 330
  match_rate_gt:    0.7455
  precision_like:   0.5072
  SAM-HQ auto matched additions: 7

quality_obj070_cat040_q040_top2:
  predictions:      490
  matched:          248 / 330
  match_rate_gt:    0.7515
  precision_like:   0.5061
  SAM-HQ auto matched additions: 9

quality_obj060_cat035_q030_top4:
  predictions:      499
  matched:          250 / 330
  match_rate_gt:    0.7576
  precision_like:   0.5010
  SAM-HQ auto matched additions: 11

quality_obj035_cat025_q015_top8:
  predictions:      528
  matched:          251 / 330
  match_rate_gt:    0.7606
  precision_like:   0.4754
  SAM-HQ auto matched additions: 12

quality_obj020_cat020_q010_top12:
  predictions:      558
  matched:          251 / 330
  match_rate_gt:    0.7606
  precision_like:   0.4498
  SAM-HQ auto matched additions: 12

oracle_samhq_complement_upper_bound:
  predictions:      558
  matched:          330 / 330
  match_rate_gt:    1.0000
  precision_like:   0.5914
  SAM-HQ auto oracle additions: 91
```

Comparison with the train-300 strict sweep:

```text
Best train-300 balanced:
  predictions:      484
  matched:          248 / 330
  match_rate_gt:    0.7515
  precision_like:   0.5124

Best train-1000 recall:
  predictions:      528
  matched:          251 / 330
  match_rate_gt:    0.7606
  precision_like:   0.4754

Best train-1000 conservative:
  predictions:      485
  matched:          246 / 330
  match_rate_gt:    0.7455
  precision_like:   0.5072
```

Interpretation:

```text
Training on 1000 images improved the training labels and loss, but it did not
move the deployable validation operating point toward the desired 80%+ recall
and higher Precision-like target.

The best recall setting reaches only 251 / 330 matched instances (76.06%) and
does so with 528 predictions, dropping Precision-like to 47.54%. The conservative
settings preserve Precision-like better but do not beat the train-300 sweep.

This suggests the current bottleneck is not simply reranker training-set size.
The oracle still proves the missing instances exist inside SAM-HQ automatic
proposals, but the deployable model cannot identify most of the 91 recoverable
oracle additions. The next improvement should change candidate assignment and
selection, not only add more images to the same reranker setup.
```

## DeepFashion2 Category-Aware Duplicate Filtering

Added oracle-gap diagnostics to the DINOv2 SAM candidate reranker evaluation.
The diagnosis showed that the train-1000 reranker was not mainly blocked by
low scores or Top-K capacity:

```text
Oracle-recoverable SAM-HQ additions: 91
Scored oracle additions:            91
Category correct:                   58
Category wrong:                     33

Median object_prob:                 0.9998
Median category_prob:               0.9600
Median quality_prob:                0.7322
Median combined-score rank:         2
```

The main blocker was duplicate/nested filtering against the Mask2Former-guided
baseline masks:

```text
Old duplicate filtering, top12:
  eligible oracle candidates:        11 / 91
  blocked by duplicate IoU:          52
  blocked by nested/duplicate rule:  23

Category-aware duplicate filtering, top12:
  eligible oracle candidates:        71 / 91
  blocked by duplicate IoU:           8
  blocked by nested/duplicate rule:   1
```

This indicates that many true garment instances are spatially overlapping with
another predicted instance. Treating cross-category overlap as duplicate is too
aggressive for clothing, because top/outerwear, dress/outerwear, top/pants, and
other layered garments can legitimately overlap.

Implementation update:

```text
scripts/train_eval_dinov2_sam_candidate_reranker.py

New config option:
  duplicate_same_category_only = true

Effect:
  mask-IoU duplicate filtering and containment filtering are applied only when
  candidate.category_id == existing.category_id.
```

Evaluation output directories:

```text
/root/autodl-tmp/fashion_prd/work_dirs/dinov2_sam_quality_catdup_train1000_eval_val200
/root/autodl-tmp/fashion_prd/work_dirs/dinov2_sam_quality_catdup_precision_train1000_eval_val200
```

DeepFashion2 validation results (`val=200`, `330 GT instances`):

```text
baseline_guided:
  predictions:      467
  matched:          239 / 330
  match_rate_gt:    0.7242
  precision_like:   0.5118
  mask IoU:         0.7841
  boundary IoU:     0.5750

old quality reranker, best conservative:
  predictions:      481
  matched:          247 / 330
  match_rate_gt:    0.7485
  precision_like:   0.5135

old quality reranker, best recall:
  predictions:      528
  matched:          251 / 330
  match_rate_gt:    0.7606
  precision_like:   0.4754

category-aware duplicate filtering, high-precision:
  config:           quality_catdup_obj090_cat065_q060_top2
  predictions:      550
  matched:          286 / 330
  match_rate_gt:    0.8667
  precision_like:   0.5200
  mask IoU:         0.7963
  boundary IoU:     0.5809
  SAM-HQ additions: 50 matched

category-aware duplicate filtering, balanced:
  config:           quality_catdup_obj085_cat055_q055_top2
  predictions:      564
  matched:          292 / 330
  match_rate_gt:    0.8848
  precision_like:   0.5177
  mask IoU:         0.7945
  boundary IoU:     0.5792
  SAM-HQ additions: 56 matched

category-aware duplicate filtering, high-recall:
  config:           quality_catdup_obj075_cat045_q045_top2
  predictions:      584
  matched:          296 / 330
  match_rate_gt:    0.8970
  precision_like:   0.5068
  mask IoU:         0.7948
  boundary IoU:     0.5799
  SAM-HQ additions: 61 matched

category-aware duplicate filtering, max tested recall:
  config:           quality_catdup_obj050_cat030_q020_top12
  predictions:      635
  matched:          304 / 330
  match_rate_gt:    0.9212
  precision_like:   0.4787
  mask IoU:         0.7895
  boundary IoU:     0.5729
  SAM-HQ additions: 70 matched
```

Interpretation:

```text
This is the first deployable non-oracle configuration that exceeds 80% recall
while also improving Precision-like over the original Mask2Former -> SAM-HQ
baseline.

Recommended default for downstream FashionAI segmentation:
  quality_catdup_obj090_cat065_q060_top2

Reason:
  baseline recall:        72.42% -> 86.67%
  baseline Precision-like:51.18% -> 52.00%
  mask IoU:               0.7841 -> 0.7963
  boundary IoU:           0.5750 -> 0.5809

Recommended higher-recall option:
  quality_catdup_obj085_cat055_q055_top2

Reason:
  recall reaches 88.48% while Precision-like remains above baseline at 51.77%.

The remaining gap to 90%+ with good precision is now smaller and more concrete:
the next focus should be category correctness for overlapping garments and
per-category duplicate thresholds, not simply larger reranker training.
```

## DeepFashion2 Final Mask Verifier / Quality Scorer

Tested a dedicated final mask verifier to filter redundant masks after the
category-aware proposal fusion stage. The verifier is trained on DeepFashion2
GT with one-to-one labels:

```text
Candidate pool:
  category-aware high-recall fusion
  quality_catdup_obj050_cat030_q020_top12

Positive label:
  for each GT garment, only the same-category candidate with the highest IoU is
  labeled keep=1 if IoU >= 0.30

Negative label:
  other candidates are labeled keep=0, including duplicate masks around the same
  GT instance

Inputs:
  DINOv2 crop feature
  SAM / reranker scores
  mask geometry
  source flags
  same-category and cross-category overlap context with higher-ranked masks
```

Script:

```text
/root/autodl-tmp/fashion_prd/scripts/train_eval_final_mask_verifier.py
```

Output directories:

```text
/root/autodl-tmp/fashion_prd/work_dirs/final_mask_verifier_train300_val200
/root/autodl-tmp/fashion_prd/work_dirs/final_mask_verifier_train300_eval_sweep_val200
```

Training setup:

```text
train images:       300
train candidates:   922
positive keep:      465
negative discard:   457
feature_dim:        768
meta_dim:           25

epoch 1 train loss: 0.7097, accuracy: 0.5727
epoch 5 train loss: 0.4941, accuracy: 0.7722
```

DeepFashion2 validation results (`val=200`, `330 GT instances`):

```text
baseline_guided:
  predictions:      467
  matched:          239 / 330
  match_rate_gt:    0.7242
  precision_like:   0.5118
  mask IoU:         0.7841
  boundary IoU:     0.5750

category-aware high-recall pool:
  predictions:      635
  matched:          304 / 330
  match_rate_gt:    0.9212
  precision_like:   0.4787
  mask IoU:         0.7895
  boundary IoU:     0.5729

score_top3_per_image:
  predictions:      524
  matched:          286 / 330
  match_rate_gt:    0.8667
  precision_like:   0.5458
  mask IoU:         0.7879
  boundary IoU:     0.5772

score_top4_per_image:
  predictions:      589
  matched:          302 / 330
  match_rate_gt:    0.9152
  precision_like:   0.5127
  mask IoU:         0.7878
  boundary IoU:     0.5711

verifier_t005_cap700:
  predictions:      588
  matched:          299 / 330
  match_rate_gt:    0.9061
  precision_like:   0.5085
  mask IoU:         0.7917
  boundary IoU:     0.5750

verifier_t010_cap700:
  predictions:      560
  matched:          294 / 330
  match_rate_gt:    0.8909
  precision_like:   0.5250
  mask IoU:         0.7963
  boundary IoU:     0.5790

verifier_t020_cap600:
  predictions:      512
  matched:          280 / 330
  match_rate_gt:    0.8485
  precision_like:   0.5469
  mask IoU:         0.8016
  boundary IoU:     0.5833

verifier_t050_cap420:
  predictions:      359
  matched:          216 / 330
  match_rate_gt:    0.6545
  precision_like:   0.6017
  mask IoU:         0.8249
  boundary IoU:     0.6156

verifier_t090_cap360:
  predictions:      53
  matched:          42 / 330
  match_rate_gt:    0.1273
  precision_like:   0.7925
  mask IoU:         0.8413
  boundary IoU:     0.6721
```

Interpretation:

```text
The final mask verifier is useful, but the first train-300 version behaves more
like a precision/quality filter than a high-recall selector.

It can raise Precision-like and mask quality:
  verifier_t020: Precision-like 54.69%, mask IoU 0.8016
  verifier_t050: Precision-like 60.17%, mask IoU 0.8249

But higher verifier thresholds remove too many true positives, so it does not
yet reach the desired 90%+ recall region.

Best current operating points:

1. Best 90%+ recall with Precision-like not below baseline:
   score_top4_per_image
   recall 91.52%, Precision-like 51.27%

2. Best precision-improved near-90 recall:
   verifier_t010_cap700
   recall 89.09%, Precision-like 52.50%

3. Best precision/quality diagnostic point:
   verifier_t020_cap600
   recall 84.85%, Precision-like 54.69%

This means a mask quality scorer is feasible and already improves precision in
lower-recall modes, but it needs more training data, better calibration, and
possibly per-category thresholds before it can replace the simpler top-K
selection in the 90%+ recall operating region.

The current deployable recommendation is:
  - Use score_top4_per_image when recall >= 90% is required.
  - Use verifier_t010_cap700 when slightly lower recall is acceptable in return
    for better Precision-like and mask quality.
```

### Combined High-Recall + Verifier Veto Evaluation

Tested whether the effective methods can be combined instead of choosing only
one operating point:

```text
category-aware high-recall pool
-> top-4 per image by combined Mask2Former / SAM / reranker score
-> final verifier veto removes very low-confidence masks
-> same-category mask NMS
```

Script:

```text
/root/autodl-tmp/fashion_prd/scripts/train_eval_final_mask_verifier.py
```

Output directory:

```text
/root/autodl-tmp/fashion_prd/work_dirs/final_mask_verifier_hybrid_sweep_val200
```

DeepFashion2 validation results (`val=200`, `330 GT instances`):

```text
score_top4_per_image:
  predictions:      589
  matched:          302 / 330
  match_rate_gt:    0.9152
  precision_like:   0.5127
  mask IoU:         0.7878
  boundary IoU:     0.5711

hybrid_score_top4_veto_all_t0p02:
  predictions:      577
  matched:          302 / 330
  match_rate_gt:    0.9152
  precision_like:   0.5234
  mask IoU:         0.7888
  boundary IoU:     0.5728

verifier_t010_cap700:
  predictions:      560
  matched:          294 / 330
  match_rate_gt:    0.8909
  precision_like:   0.5250
  mask IoU:         0.7963
  boundary IoU:     0.5790
```

Interpretation:

```text
The combined strategy is the current best 90%+ recall operating point. Compared
with plain score_top4_per_image, it keeps the same 302 matched GT instances and
91.52% recall, while removing 12 false-positive predictions:

  predictions:    589 -> 577
  Precision-like: 51.27% -> 52.34%

This confirms that combining high-recall top-K selection with a conservative
verifier veto is useful. The gain is modest, so it does not solve the 90%+
Precision-like target by itself. A stronger verifier needs more DeepFashion2
training data, per-category calibration, and harder duplicate / background
negative mining.

Current recommendation:
  - Use hybrid_score_top4_veto_all_t0p02 when recall >= 90% is required.
  - Use verifier_t010_cap700 when a slightly lower 89.09% recall is acceptable
    for marginally higher Precision-like and better mask quality.
```

### BCR / MCS / ARC Completeness Evaluation

Added completeness-oriented evaluation metrics to avoid relying only on mask IoU:

```text
BCR: boundary coverage ratio
     fraction of GT boundary covered by predicted boundary

MCS: mask completeness score
     intersection(pred, GT) / area(GT)

ARC: area ratio consistency
     min(area(pred), area(GT)) / max(area(pred), area(GT))
```

These metrics are intended to expose fragment masks: a local clothing fragment
can sometimes pass a loose IoU match threshold, but it should score poorly on
MCS and ARC.

Script:

```text
/root/autodl-tmp/fashion_prd/scripts/evaluate_mask2former_samhq_refinement.py
```

Output directories:

```text
/root/autodl-tmp/fashion_prd/work_dirs/bcr_mcs_arc_guarded_independent_val200
/root/autodl-tmp/fashion_prd/work_dirs/bcr_mcs_arc_guarded_loose_val200
```

Setup:

```text
Dataset: DeepFashion2 validation COCO instance format
Images: 200 shuffled validation images
GT instances: 330
Mask2Former threshold: score 0.20, mask 0.40
SAM-HQ prompt: bbox expand 0.08 + positive foreground point
Predictions: 467 for each method
```

Independent matching results:

```text
Mask2Former raw:
  matched:          195 / 330
  match_rate_gt:    0.5909
  precision_like:   0.4176
  mask IoU:         0.7431
  boundary IoU:     0.5034
  BCR:              0.6058
  MCS:              0.7806
  ARC:              0.7892

SAM-HQ refined:
  matched:          239 / 330
  match_rate_gt:    0.7242
  precision_like:   0.5118
  mask IoU:         0.7841
  boundary IoU:     0.5750
  BCR:              0.7015
  MCS:              0.8165
  ARC:              0.8189

Guarded refined, strict coverage/area fallback:
  guard thresholds: coverage >= 0.70, area consistency >= 0.55
  fallback ratio:   0.1385 on raw-matched proposals
  matched:          195 / 330
  match_rate_gt:    0.5909
  precision_like:   0.4176
  mask IoU:         0.7584
  boundary IoU:     0.5593
  BCR:              0.6744
  MCS:              0.7925
  ARC:              0.7930

Guarded refined, loose coverage/area fallback:
  guard thresholds: coverage >= 0.40, area consistency >= 0.30
  fallback ratio:   0.0718 on raw-matched proposals
  matched:          196 / 330
  match_rate_gt:    0.5939
  precision_like:   0.4197
  mask IoU:         0.7666
  boundary IoU:     0.5663
  BCR:              0.6827
  MCS:              0.7977
  ARC:              0.8003
```

Per-category signal from the common raw-matched diagnostic:

```text
top:
  SAM-HQ improves mask IoU 0.7589 -> 0.8258
  SAM-HQ improves MCS      0.7957 -> 0.8520
  SAM-HQ improves ARC      0.8019 -> 0.8575

dress:
  SAM-HQ improves mask IoU 0.6975 -> 0.7698
  SAM-HQ improves MCS      0.7393 -> 0.8213
  SAM-HQ improves ARC      0.7370 -> 0.8149

pants:
  SAM-HQ lowers mask IoU   0.7772 -> 0.7496
  SAM-HQ lowers MCS        0.8069 -> 0.7792
  SAM-HQ lowers ARC        0.8332 -> 0.7913
  boundary still improves  0.4999 -> 0.5500
```

Interpretation:

```text
The claim "SAM-HQ cuts Mask2Former masks into fragments" is not true globally
on this DeepFashion2 val200 subset. With bbox expansion and a foreground point,
SAM-HQ improves recall, Precision-like, IoU, BCR, MCS, and ARC overall.

However, the claim is partly true for pants: SAM-HQ improves boundary alignment
but reduces completeness and area consistency. This suggests the failure mode is
category-dependent rather than universal.

The naive global guarded fallback is not a good solution. It prevents some
fragment-like outputs, but it also removes many SAM-HQ fixes that are needed for
recall. Even the loose guard drops matched instances from 239 to 196.

Current recommendation:
  - Keep SAM-HQ refinement as the default for top/dress/skirt-like cases.
  - Add category-aware selection for pants and possibly other long/thin
    categories: compare raw Mask2Former and SAM-HQ refined masks with BCR/MCS/
    ARC-style self-consistency features, then choose per instance.
  - Add BCR/MCS/ARC to future verifier training as explicit quality targets,
    rather than using a hard global fallback rule.
```

### BCR / MCS / ARC Final Verifier

Extended the final mask verifier from two heads:

```text
keep probability
IoU quality
```

to five heads:

```text
keep probability
IoU quality
BCR quality
MCS quality
ARC quality
```

The verifier now trains on DeepFashion2 GT-derived metric targets and the final
evaluation summary includes BCR, MCS, and ARC for every method.

Scripts:

```text
/root/autodl-tmp/fashion_prd/scripts/train_eval_final_mask_verifier.py
/root/autodl-tmp/fashion_prd/scripts/train_eval_dinov2_sam_candidate_reranker.py
```

Completed train-300 experiment:

```text
/root/autodl-tmp/fashion_prd/work_dirs/final_mask_verifier_bcr_mcs_arc_train300_val200
```

Low-threshold completeness veto sweep:

```text
/root/autodl-tmp/fashion_prd/work_dirs/final_mask_verifier_bcr_mcs_arc_low_veto_eval_val200
```

Training setup:

```text
train images:       300
train candidates:   922
positive keep:      465
negative discard:   457
feature_dim:        768
meta_dim:           25

epoch 1 train loss: 0.7610, accuracy: 0.5727
epoch 5 train loss: 0.5408, accuracy: 0.7636
```

DeepFashion2 validation results (`val=200`, `330 GT instances`):

```text
score_top4_per_image:
  predictions:      589
  matched:          302 / 330
  match_rate_gt:    0.9152
  precision_like:   0.5127
  mask IoU:         0.7878
  BCR / MCS / ARC:  0.7023 / 0.8283 / 0.8240

previous hybrid_score_top4_veto_all_t0p02:
  predictions:      579
  matched:          302 / 330
  match_rate_gt:    0.9152
  precision_like:   0.5216
  mask IoU:         0.7888
  BCR / MCS / ARC:  0.7036 / 0.8288 / 0.8238

hybrid_score_top4_complete_veto_all_c0p15:
  predictions:      566
  matched:          300 / 330
  match_rate_gt:    0.9091
  precision_like:   0.5300
  mask IoU:         0.7910
  BCR / MCS / ARC:  0.7058 / 0.8315 / 0.8256

complete_t000_c020_cap700:
  predictions:      562
  matched:          295 / 330
  match_rate_gt:    0.8939
  precision_like:   0.5249
  mask IoU:         0.7954
  BCR / MCS / ARC:  0.7099 / 0.8363 / 0.8285

complete_t000_c030_cap700:
  predictions:      478
  matched:          268 / 330
  match_rate_gt:    0.8121
  precision_like:   0.5607
  mask IoU:         0.8065
  BCR / MCS / ARC:  0.7192 / 0.8459 / 0.8371

complete_t000_c040_cap700:
  predictions:      347
  matched:          208 / 330
  match_rate_gt:    0.6303
  precision_like:   0.5994
  mask IoU:         0.8250
  BCR / MCS / ARC:  0.7419 / 0.8563 / 0.8465
```

Interpretation:

```text
BCR / MCS / ARC are useful quality targets. As completeness thresholds increase,
mask IoU, BCR, MCS, ARC, and Precision-like all improve.

However, simple thresholding still trades away recall too quickly. The best
90%+ recall point from this run is:

  hybrid_score_top4_complete_veto_all_c0p15
  recall 90.91%, Precision-like 53.00%

This improves over the previous 90%+ recall operating point:

  previous best: recall 91.52%, Precision-like 52.16%
  new best:      recall 90.91%, Precision-like 53.00%

The gain is real but modest. It does not approach 90% Precision-like. The main
remaining problem is not mask boundary quality alone; it is still false garment
proposal selection and duplicate/background suppression.
```

Started a larger train-1000 BCR/MCS/ARC verifier run in the background:

```text
script: /root/autodl-tmp/fashion_prd/scripts/run_final_mask_verifier_bcr_1000.sh
log:    /root/autodl-tmp/fashion_prd/logs/final_mask_verifier_bcr_1000.log
pid:    /root/autodl-tmp/fashion_prd/logs/final_mask_verifier_bcr_1000.pid
output: /root/autodl-tmp/fashion_prd/work_dirs/final_mask_verifier_bcr_mcs_arc_train1000_val200
```

## Next Steps

The next implementation stages are:

1. Finish the 20k / 3k-step Mask2Former training run and inspect checkpoints.
2. Run Mask2Former on FashionAI to get category-aware garment instances.
3. Optionally refine Mask2Former masks with the fine-tuned SAM-HQ model.
4. Bind FashionAI attributes to detected garment instances.
5. Generate local part regions for local attributes.
6. Train DINOv2 global and local attribute classifiers.
7. Predict FashionAI-style pseudo labels for DeepFashion2.
8. Clean and merge the final multi-instance, multi-attribute, multi-region dataset.
