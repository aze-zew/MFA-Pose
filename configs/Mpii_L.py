_base_ = ['mmpose::_base_/default_runtime.py']

# ================== 数据集设置 ==================
dataset_type = 'MpiiDataset'  # 改为MPII数据集
data_mode = 'topdown'
data_root = '/root/autodl-tmp/'  # MPII数据路径

# ================== 模型参数 ==================
num_keypoints = 16  # MPII只有16个关键点
input_size = (192, 256)  # 或改为(256,256)看需求
# runtime 训练超参数
max_epochs = 220    # 参考MPII官方设置
stage2_num_epochs = 30
base_lr = 3e-4      # 保持L模型的学习率

train_cfg = dict(max_epochs=max_epochs, val_interval=10)  # 验证间隔
randomness = dict(seed=21)

# ================== 优化器设置 ==================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# ================== 学习率策略 ==================
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
auto_scale_lr = dict(base_batch_size=1024)

# ================== 编码器设置 ==================
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# ================== 模型架构 ==================
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt_gai',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        out_indices=(3,4),
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
        out_channels=num_keypoints,  # 自动使用16
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
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

# ================== 数据管道 ==================
backend_args = dict(backend='local')

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    # MPII专用增强参数
    dict(
        type='RandomBBoxTransform', 
        scale_factor=[0.6, 1.4],  # 缩放范围调整
        rotate_factor=80),         # 旋转角度调整
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
                p=1.0),
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

# ================== 数据加载 ==================
train_dataloader = dict(
    batch_size=64,  # 减小batch_size适应MPII
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='MPII/mpii_train.json',
        data_prefix=dict(img='MPII/images'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='MPII/mpii_val.json',
        headbox_file=f'{data_root}/MPII/mpii_gt_val.mat',  # MPII必需
        data_prefix=dict(img='MPII/images'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader
# ================== 评估设置 ==================
val_evaluator = dict(type='MpiiPCKAccuracy')  # MPII专用评估器

test_evaluator = val_evaluator

# ================== 训练钩子 ==================
default_hooks = dict(
    checkpoint=dict(
        save_best='PCK',  # 监控PCK指标
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