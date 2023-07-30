_base_ = [
    "checkpoint/rtmdet_x_8xb32-300e_coco.py"
]


# # # # #
# Model #
# # # # #
model = dict(
    # backbone=dict(
    #     # type="mmpretrain.ConvNeXt"
    #     norm_cfg = dict(type='SyncBN', requires_grad=False),
    #     # frozen_stages=-1,
    # ),
    # neck=dict(
    #     type='CSPNeXtPAFPNBA',
    #     # norm_cfg = dict(type='SyncBN', requires_grad=False),
    # ),
    # backbone=dict(
    #     _delete_=True,
    #     type='mmpretrain.SwinTransformerV2',
    #     arch="base",
    #     img_size=384,
    #     out_indices=[1, 2, 3],
    #     window_size=[16, 16, 16, 8],
    #     drop_path_rate=0.2,
    #     pretrained_window_sizes=[12, 12, 12, 6],
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w16_in21k-pre_3rdparty_in1k-256px_20220803-8d7aa8ad.pth',
    #         prefix='backbone.'
    #     )
    # ),
    # neck=dict(
        # in_channels=[256, 512, 1024],
    # ),
    bbox_head=dict(
        num_classes=10,
        # loss_cls=dict(_delete_=True, type="GHMC"),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.),
        # norm_cfg = dict(type='SyncBN', requires_grad=False),
    ),
    # train_cfg=dict(
    #     assigner=dict(type='DynamicSoftLabelAssigner', topk=7),
    # ),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        # score_thr=0.001,
        score_thr=0.02,
        nms=dict(type='soft_nms', iou_threshold=0.5),
        max_per_img=300
    )
)

# # # # # # # #
# DataLoader #
# # # # # # #



# # # # # # #
# Runtime #
# # # # #
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best="auto"
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/valid_coco.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10)
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0,
        end=60),
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        begin=60,
        end=300,
        T_max=150,
        by_epoch=True,
        convert_to_iter_based=True)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    accumulative_counts=1,
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=10,
    dynamic_intervals=[(280, 1)]
)

load_from = "checkpoint/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
work_dir = "work_dir"

