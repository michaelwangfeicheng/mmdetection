#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/1/21 17:18

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/21 17:18   wangfc      1.0         None


"""
import os
# 使用 __name__ 为导入包的时候的路径，根据导入包的关系来确定，具有继承关系
# 比如： from object_detection_server.load_model import model   该文件的  __name__ = object_detection_server.load_model
# 比如： from load_model import model  该文件的 __name__ = load_model （load_model 加入的 sys.path 当中）
import logging
logger = logging.getLogger(__name__)
from mmdet.apis import init_detector
# 使用相对导入
from .config.configs import *
from .load_model_config import load_model_config

# 将相对路径组装为绝对路径
logger.info(f"加载模型 on device={device},config_file={config_file_absolute_path}，checkpoint_file={checkpoint_file_absolute_path},")

# 使用自定义的加载模型配置文件，读取 转换过来的text格式 模型配置文件： faster_rcnn_r50_fpn_1x_seal_002.txt
mmcv_config = load_model_config(filename=config_file_absolute_path)
# 直接在 参数 config 传入 mmcv.Config 对象
model = init_detector(config=mmcv_config,checkpoint= checkpoint_file_absolute_path, device=device)

