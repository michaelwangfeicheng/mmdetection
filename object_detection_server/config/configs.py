#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/1/19 11:20

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/19 11:20   wangfc      1.0         None


"""
import os
from configparser import ConfigParser


def _get_module_path(path):
    return os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__), path))


config_parser = ConfigParser()
config_path = _get_module_path('./configs.cfg')
config_parser.read(config_path, encoding='utf-8')

# [main]
gpu_no = config_parser.get('main', 'gpu_no')
device = config_parser.get('main', 'device')

# [infer]
# 目标识别阈值
score_thr = config_parser.getfloat('infer', 'score_thr')

num_classes = config_parser.getint('infer', 'num_classes')

test_no= config_parser.get('infer', 'test_no')
checkpoint_epoch = config_parser.getint('infer', 'checkpoint_epoch')
# 模型配置文件: 使用相对路径（相对于当前目录 object_detection_server ）
config_file = config_parser.get('infer', 'config_file')
# 模型的目录

model_dir = os.path.join('../model',f'model_{num_classes}_classes_{test_no}')
config_path = os.path.join(model_dir,config_file)
config_file_absolute_path = _get_module_path(config_path)
# 模型
checkpoint_path = os.path.join(model_dir,f"epoch_{checkpoint_epoch}.pth")
checkpoint_file_absolute_path = _get_module_path(checkpoint_path)
# api识别结果是否返回处理过的图片
if_ger_processed_image = config_parser.getboolean('infer', 'if_ger_processed_image')

# [url]
# 目标识别web服务的url
object_detection_server_url = config_parser.get('url', 'object_detection_server_url')
# 目标识别web服务的url 端口
port = config_parser.getint('url', 'port')
num_processes = config_parser.getint('url', 'num_processes')



