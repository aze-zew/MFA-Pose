_base_ = ['mmpose::_base_/default_runtime.py']

# 使用CrowdPose数据集配置
dataset_type = 'CrowdPoseDataset'
data_mode = 'topdown'
data_root = '/root/autodl-tmp/'  # CrowdPose数据路径

# 关键点数量改为CrowdPose的14个
num_keypoints = 14
input_size = (192, 256)

# 训练超参数（参考官方CrowdPose配置）
max_epochs = 210
stage2_num_epochs = 30
base_lr = 5e-4  # CrowdPose官方学习率
train_batch_size = 128   # 根据显存调整
val_batch_size = 64

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# 优化器（保持L模型配置）
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),###
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 学习率策略（保持L模型配置）
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

auto_scale_lr = dict(base_batch_size=512)  # CrowdPose官方设置

# codec设置（保持L模型配置）
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# 模型配置（保持L模型结构）
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        #_scope_ = 'mmdet',
        type='CSPNeXt_gai',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        out_indices=(3, 4),
        dca_attention=True,
        dca_reduction=8,
        dca_groups=4,
        use_input_dcn=True,
        use_bifpn=False,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
                       'rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'
        )),
    neck=dict(
        type='Bi_A',
        in_channels=[512, 1024],
        out_channels=1024,
        num_layers=2,
        attention=True),
    head=dict(
        type='RTMCCHead',
        in_channels=1024,
        out_channels=num_keypoints,  # 自动使用14个关键点
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        use_efficient_lka=True,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

backend_args = dict(backend='local')

# 数据管道（使用CrowdPose官方增强配置）
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),  # 保持L模型强度
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),  # 保持L模型强度
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# 数据加载器（使用CrowdPose路径）
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='crowdpose/crowdpose_trainval.json',
        data_prefix=dict(img='crowdpose/images'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='crowdpose/crowdpose_test.json',
        #bbox_file='data/crowdpose/annotations/det_for_crowd_test_0.1_0.5.json',
        data_prefix=dict(img='crowdpose/images'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

# 钩子配置（保持L模型设置）
default_hooks = dict(
    checkpoint=dict(
        save_best='crowdpose/AP',  # 改为CrowdPose指标
        rule='greater',
        max_keep_ckpts=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# 评估器（使用CrowdPose专用配置）
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'crowdpose/crowdpose_test.json',
        use_area=False,
        iou_type='keypoints_crowd',
        prefix='crowdpose'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[1, 2])
]

test_evaluator = val_evaluator