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
from tools.config.configs import *

_base_ = 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 1. dataset settings
# configs/_base_/datasets/coco_detection.py 修改数据路径


data = dict(
    samples_per_gpu=samples_per_gpu,              # 每个gpu计算的图像数量: v1 imgs_per_gpu=2, 需要根据自己的gpu情况进行调整训练的batch大小
    workers_per_gpu=1,                           # 每个gpu分配的线程数量
    train=dict(
        type=dataset_type,                       # 数据集类型
        # classes=classes, # 传入类别参数
        ann_file= data_root + 'annotations/train.json',    # 数据集annotation路径
        img_prefix= data_root + 'train/',     # 数据集的图片路径
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
        ann_file= test_data_root +'annotations/test.json',
        img_prefix= test_data_root +'test/',
        # pipeline=test_pipeline
    ))

# 2. model settings
# configs/_base_/models/faster_rcnn_r50_fpn.py  num_classes 默认为 81
model = dict(
    pretrained= pretrained,                        # 训练时 backbone（例如resnet）可能需要下载 pretrained 模型来预加载
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
optimizer = dict(type='SGD', lr=0.02*(samples_per_gpu*gpus/16), momentum=0.9, weight_decay=0.0001)


# 4. runtime settings
# configs/_base_/default_runtime.py 修改配置
dist_params = dict(backend='nccl')                   # 分布式参数

# 使用work_flow来进行控制,在训练过程中进行eval
# workflow = [('train', 1), ('val' , 1)]                         # 当前工作区名称

# logger
log_level = 'INFO'                                # 输出信息的完整度级别
log_config = dict(
    interval=50,                                  # 每50个batch输出一次信息
    hooks=[
        dict(type='TextLoggerHook'),              # 控制台输出信息的风格
        # dict(type='TensorboardLoggerHook')
    ])



