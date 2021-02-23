#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2021/1/15 16:34 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/15 16:34   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

Config Name Style:
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}

config/_base_
dataset, model, schedule, default_runtime.

"""
_base_ = 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

num_classes=1
total_epochs = 12                                 # 最大epoch数
samples_per_gpu=2                                 # 每个gpu计算的图像数量:


# 如果出现类别数量不匹配的错误，人为指定数据集的类别，并且在train、val、test中传入参数classes
# classes = ('seal')
# work_dir = './output/seal_detection_train/faster_rcnn_r50_fpn_1x'   # log文件和模型文件存储路径


# 1. dataset settings
# configs/_base_/datasets/coco_detection.py 修改数据路径
dataset_type = 'CocoDataset'                    # 数据集类型
data_root = 'data_generator/data/bak/'  # 数据集根目录

data = dict(
    samples_per_gpu=samples_per_gpu,              # 每个gpu计算的图像数量: v1 imgs_per_gpu=2, 需要根据自己的gpu情况进行调整训练的batch大小
    workers_per_gpu=2,                           # 每个gpu分配的线程数量
    train=dict(
        type=dataset_type,                       # 数据集类型
        # classes=classes, # 传入类别参数
        ann_file= data_root + 'annotations/train.json',    # 数据集annotation路径
        img_prefix=  data_root + 'train/',     # 数据集的图片路径
        # pipeline=train_pipeline
    ),
    val=dict(                                    # val、test 同上
        type=dataset_type,
        # classes=classes, # 传入类别参数
        ann_file=data_root +'annotations/dev.json',
        img_prefix= data_root +'dev/',
        # pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        # classes=classes, # 传入类别参数
        ann_file= data_root +'annotations/dev.json',
        img_prefix= data_root +'dev/',
        # pipeline=test_pipeline
    ))

# 2. model settings
# configs/_base_/models/faster_rcnn_r50_fpn.py  num_classes 默认为 81
pretrained ='checkpoints/pretrained/resnet50/resnet50-19c8e357.pth' # backbone使用 resnet50的预训练模型
model = dict(
    pretrained= pretrained,                        # 训练时backbone（例如resnet）可能需要下载 pretrained 模型来预加载
    roi_head =dict(
        bbox_head=dict(
        num_classes=num_classes,                   # V2: 分类器的类别数量，V1:+1是因为背景类,需要对类别数进行一下修改
    ))
)

# 3. schedule
# configs/_base_/schedules/schedule_1x.py 修改 optimizer 参数
# 这里注意配置都是默认8块gpu的训练，如果用一块gpu训练，需要在lr/8
# 学习率lr和总的batch size数目成正比，例如：8卡GPU  samples_per_gpu = 2的情况（相当于总的batch size = 8*2）,学习率lr = 0.02
# 如果我是单卡GPU samples_per_gpu = 4的情况，学习率lr应该设置为:0.02*(4/16) = 0.005
optimizer = dict(type='SGD', lr=0.02*(samples_per_gpu/16), momentum=0.9, weight_decay=0.0001)


# 4. runtime settings
# configs/_base_/default_runtime.py 修改配置
dist_params = None                                # 分布式参数
# mmdet默认加载权重优先级别是resume_from(断点加载)，load_from，pretrained 的顺序，所以需要从load_from加载预训练权重
resume_from = None                                # 恢复训练模型的路径
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_1.pth'     # 加载模型的路径，None表示从预训练模型加载，需要修改 num_classes 对应的参数

# workflow = [('train', 1)]                         # 当前工作区名称

# logger
log_level = 'INFO'                                # 输出信息的完整度级别
log_config = dict(
    interval=50,                                  # 每50个batch输出一次信息
    hooks=[
        dict(type='TextLoggerHook'),              # 控制台输出信息的风格
        dict(type='TensorboardLoggerHook')
    ])


"""
# model settings
model = dict(
    type='FasterRCNN',                          # model类型
    pretrained='torchvision://resnet50',        # 预训练模型
    backbone=dict(
        type='ResNet',                          # backbone类型
        depth=50,                               # 网络层数
        num_stages=4,                           # rsenet的stage数量
        out_indices=(0, 1, 2, 3),               # 输出的stage序号
        frozen_stages=1,                        # 冻结的stage数量，即该stage不更新参数。-1表示所有的stage都更新参数
        norm_cfg=dict(type='BN', requires_grad=True),     # 传递梯度，更新weight和bias
        style='pytorch'),                       # 网络风格：如果设置为pytorch，那么stride为2的层是conv3*3卷积。如果设置为caffe，则stride为2的层是第一个conv1x1的卷积层
    neck=dict(
        type='FPN',                             # neck类型
        in_channels=[256, 512, 1024, 2048],     # 输入的各个stage的通道数
        out_channels=256,                       # 输出的特征层的通道数
        num_outs=5),                            # 输出的特征层的数量
    rpn_head=dict(
        type='RPNHead',                         # RPN网络类型
        in_channels=256,                        # RPN网络的通道数
        feat_channels=256,                      # 特征层的通道数
        anchor_scales=[8],                      # 生成的anchor的baselen，baselen=sqrt(w*h);w,h为anchor的width，height。
        anchor_ratios=[0.5, 1.0, 2.0],          # anchor的宽高比
        anchor_strides=[4, 8, 16, 32, 64],      # 对应于原图，每个特征层上的 anchor的步长
        target_means=[.0, .0, .0, .0],          # 均值
        target_stds=[1.0, 1.0, 1.0, 1.0],       # 方差
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),    # 是否使用sigmoid来进行分类，如果False则使用softmax来分类
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',              # RoIExtractor类型
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),    # RoI具体参数：ROI类型为ROIalign，输出尺寸为7，sample数为2
        out_channels=256,                       # 输出通道数
        featmap_strides=[4, 8, 16, 32]),        # 特征图步长
    bbox_head=dict(
        type='SharedFCBBoxHead',                # 全连接层类型
        num_fcs=2,                              # 全连接层数量
        in_channels=256,                        # 输入通道数
        fc_out_channels=1024,                   # 输出通道数
        roi_feat_size=7,                        # ROI特征层size
        num_classes=81,                         # 分类器的类别数量+1，+1是因为背景类
        target_means=[0., 0., 0., 0.],          # 均值
        target_stds=[0.1, 0.1, 0.2, 0.2],       # 方差
        reg_class_agnostic=False,               # 是否采用class-agnostic的方式来预测；class_agnostic表示输出bbox时只考虑其是否为前景，后续分类的时候再根据该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',              # RPN网络的正负样本划分
            pos_iou_thr=0.7,                    # 正样本的iou阈值
            neg_iou_thr=0.3,                    # 负样本的iou最小值
            min_pos_iou=0.3,                    # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1),                 # 忽略bbox的阈值。当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='RandomSampler',               # 正负样本提取器类型
            num=256,                            # 需提取的正负样本数量
            pos_fraction=0.5,                   # 正样本比例
            neg_pos_ub=-1,                      # 最大负样本比例，大于该比例的负样本忽略。-1表示不忽略
            add_gt_as_proposals=False),         # 把ground truth 加入 proposal作为正样本
        allowed_border=0,                       # 允许bbox周围外扩一定的像素
        pos_weight=-1,                          # 正样本权重，-1表示不改变原始权重
        debug=False),                           # debug模式
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',              # rcnn网络正负样本划分
            pos_iou_thr=0.5,                    # 正样本的iou阈值
            neg_iou_thr=0.5,                    # 负样本的iou阈值
            min_pos_iou=0.5,                    # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1),                 # 忽略bbox的阈值。当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='RandomSampler',               # 正负样本提取器类型
            num=512,                            # 需要提取的正负样本数量
            pos_fraction=0.25,                  # 正样本比例
            neg_pos_ub=-1,                      # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
            add_gt_as_proposals=True),          # 把 ground truth加入proposal作为正样本
        pos_weight=-1,                          # 正样本权重，-1表示不改变原始的权重
        debug=False))                           # debug模式

test_cfg = dict(
    rpn=dict(                                   # 推断时的rpn参数
        nms_across_levels=False,                # 在所有的fpn层内做nms
        nms_pre=1000,                           # 在nms之前保留的的分最高的proposal数量
        nms_post=1000,                          # 在nms之后保留的得分最高的proposal数量
        max_num=1000,                           # 在后处理完成之后保留的proposal数量
        nms_thr=0.7,                            # nms阈值
        min_bbox_size=0),                       # 最小bbox的size
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)   # max_per_img表示最终输出的det bbox数量
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)            # soft-nms参数
)


evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)    # 优化参数
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))    # 梯度均衡参数

# learning policy
lr_config = dict(
    policy='step',                                # 优化策略
    warmup='linear',                              # 初始化的learning-rate增加的策略，linear为线性增加
    warmup_iters=500,                             # 在初始的500次迭代中learning-rate 逐渐增加
    warmup_ratio=1.0 / 3,                         # original learning-rate
    step=[8, 11])                                 # 在第8和11个epoch时降低learning-rate
checkpoint_config = dict(interval=1)              # 每1个epoch存储一次模型


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#输入图像初始化，减去均值mean并处以方差std，to_rgb表示将bgr转为rgb

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),    # 输入图片尺寸，最大边1333，最小边800。
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),                              # 对图像进行resize时的最小单位，32表示所有的图像都会被resize成32的倍数
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

"""



