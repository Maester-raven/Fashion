_base_ = "/root/autodl-tmp/fashion_prd/configs/rtmdet/rtmdet_ins_l_fashionpedia8_copypaste13000_1024_e24_v1.py"

classes = ("top", "pants", "skirt", "outerwear", "dress", "shoes", "bag", "accessory")
metainfo = dict(classes=classes)
data_root = "/root/autodl-tmp/fashion_prd/"
img_size = (1024, 1024)

load_from = (
    "/root/autodl-tmp/fashion_prd/work_dirs/"
    "rtmdet_ins_l_fashionpedia8_copypaste13000_1024_e24_v1/"
    "best_coco_segm_mAP_epoch_24.pth"
)

base_lr = 2.5e-5
max_epochs = 6

model = dict(
    backbone=dict(frozen_stages=4, norm_eval=True),
)

optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0, decay_mult=0.0),
            "neck": dict(lr_mult=0.3),
        },
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale="dynamic",
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.1,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# Keep difficult image context intact during the short fine-tune.
train_pipeline_clean = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=False),
    dict(type="RandomResize", scale=img_size, ratio_range=(0.5, 1.5), keep_ratio=True),
    dict(type="RandomCrop", crop_size=img_size, recompute_bbox=True, allow_negative_crop=True),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=img_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="PackDetInputs"),
]

original_dataset = dict(
    type="CocoDataset",
    data_root=data_root,
    ann_file=(
        "data_interim/fashionpedia/"
        "fashionpedia_train_balanced8_copypaste_small_v1_13000img_coco.json"
    ),
    data_prefix=dict(img=""),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=True, min_size=1),
    pipeline=train_pipeline_clean,
)

hard_dataset = dict(
    type="CocoDataset",
    data_root=data_root,
    ann_file="data_interim/fashionpedia/fashionpedia_train_hard_images_v1_coco.json",
    data_prefix=dict(img=""),
    metainfo=metainfo,
    filter_cfg=dict(filter_empty_gt=True, min_size=1),
    pipeline=train_pipeline_clean,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        _delete_=True,
        type="ConcatDataset",
        datasets=[
            original_dataset,
            dict(type="RepeatDataset", times=2, dataset=hard_dataset),
        ],
    ),
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=6,
        save_best="coco/segm_mAP",
    ),
)

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
]

randomness = dict(seed=42, deterministic=False)
