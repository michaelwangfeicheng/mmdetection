#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2021/1/19 11:20 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/19 11:20   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

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
# 模型配置文件: 使用相对路径（相对于当前目录 object_detection_server ）
num_classes = config_parser.getint('infer', 'num_classes')
config_file = config_parser.get('infer', 'config_file')
config_file_absolute_path = _get_module_path(config_file)
# 模型
checkpoint_file = config_parser.get('infer', 'checkpoint_file')
checkpoint_file_absolute_path = _get_module_path(checkpoint_file)
# api识别结果是否返回处理过的图片
if_ger_processed_image = config_parser.getboolean('infer', 'if_ger_processed_image')

# [url]
# 目标识别web服务的url
object_detection_server_url = config_parser.get('url', 'object_detection_server_url')
# 目标识别web服务的url 端口
port = config_parser.getint('url', 'port')
num_processes = config_parser.getint('url', 'num_processes')
