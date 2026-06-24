
_base_ = "/root/autodl-tmp/fashion_prd/checkpoints/rtmdet_ins_l/rtmdet-ins_l_8xb32-300e_coco.py"

classes = ("top", "pants", "skirt", "outerwear", "dress", "shoes", "bag", "accessory")
metainfo = dict(classes=classes)
data_root = "/root/autodl-tmp/fashion_prd/"

load_from = "/root/autodl-tmp/fashion_prd/checkpoints/rtmdet_ins_l/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth"

base_lr = 2.5e-4
max_epochs = 24
stage2_num_epochs = 4
interval = 2
img_size = (1024, 1024)

model = dict(
    backbone=dict(norm_cfg=dict(type="BN")),
    neck=dict(norm_cfg=dict(type="BN")),
    bbox_head=dict(num_classes=8, norm_cfg=dict(type="BN")),
    test_cfg=dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.6), max_per_img=150),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    loss_scale="dynamic",
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
    dict(type="CosineAnnealingLR", eta_min=base_lr * 0.05, begin=0, end=max_epochs, T_max=max_epochs, by_epoch=True, convert_to_iter_based=True),
]

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=False),
    dict(type="CachedMosaic", img_scale=img_size, pad_val=114.0),
    dict(type="RandomResize", scale=(2048, 2048), ratio_range=(0.1, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=img_size, recompute_bbox=True, allow_negative_crop=True),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=img_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="CachedMixUp", img_scale=img_size, ratio_range=(1.0, 1.0), max_cached_images=20, pad_val=(114, 114, 114)),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
    dict(type="PackDetInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=False),
    dict(type="RandomResize", scale=img_size, ratio_range=(0.1, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=img_size, recompute_bbox=True, allow_negative_crop=True),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=img_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=img_size, keep_ratio=True),
    dict(type="Pad", size=img_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="data_interim/fashionpedia/fashionpedia_train_balanced8_copypaste_small_v1_13000img_coco.json",
        data_prefix=dict(img=""),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="data_interim/fashionpedia/fashionpedia_val_600_coco.json",
        data_prefix=dict(img=""),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "data_interim/fashionpedia/fashionpedia_val_600_coco.json",
    metric=["bbox", "segm"],
    format_only=False,
    classwise=True,
)
test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=2, max_keep_ckpts=3, save_best="coco/segm_mAP"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

custom_hooks = [
    dict(type="EMAHook", ema_type="ExpMomentumEMA", momentum=0.0002, update_buffers=True, priority=49),
    dict(type="PipelineSwitchHook", switch_epoch=max_epochs - stage2_num_epochs, switch_pipeline=train_pipeline_stage2),
]

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
randomness = dict(seed=42, deterministic=False)
auto_scale_lr = dict(enable=False, base_batch_size=16)
