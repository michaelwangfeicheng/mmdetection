#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2021/1/28 17:00 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/28 17:00   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""
num_classes = 5
test_num = "001"
dataset_type = 'CocoDataset'                    # 数据集类型
data_root = f'data_generator/test_data_{num_classes}_classes_{test_num}/'  # 数据集根目录
test_data_root = "object_detection_server/test_data/seal_data_real/"
# DETECTION_CLASSES = ("back_idcard",  "ellipse_seal","front_idcard", "rectangle_name_seal","round_seal","square_name_seal")
# DETECTION_CLASSES = ("back_idcard",  "ellipse_seal","front_idcard","qrcode", "rectangle_name_seal","round_seal","square_name_seal")

model_config_file_path = f"configs/faster_rcnn_r50_fpn_1x_003.py"
output_dir = f'output/faster_rcnn_r50_fpn_1x_{num_classes}_classes_{test_num}/'

total_epochs = 12                                # 最大epoch数
samples_per_gpu=8                                 # 每个gpu计算的图像数量:
gpus =2
device_ids= 1

test_epoch = 12
checkpoint = output_dir + f'epoch_{test_epoch}.pth'
eval_result_file_path = output_dir + 'eval_result.pkl'
eval_save_image_dir =  output_dir + f'eval_images_epoch_{test_epoch}'
eval_metrics = 'bbox'


# mmdet 默认加载权重优先级别是 resume_from (断点加载) # 恢复训练模型的路径 # resume_from = output_dir+ 'epoch_4.pth'
# load_from，pretrained 的顺序，所以需要从load_from加载预训练权重
# 加载模型的路径，None表示从预训练模型加载，需要修改 num_classes 对应的参数
pretrained_model='faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# load_from = f'checkpoints/faster_rcnn_r50_fpn_1x_coco_{num_classes}.pth'
load_from = f'checkpoints/{pretrained_model}'
# backbone使用 resnet50 的预训练模型
pretrained = 'checkpoints/pretrained/resnet50/resnet50-19c8e357.pth'
