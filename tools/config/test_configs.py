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
dataset_type = 'CocoDataset'                    # 数据集类型
data_root = 'data_generator/test_data_6_classes_003/'  # 数据集根目录
test_data_root = "object_detection_server/test_data/seal_data_real/"
num_classes = 6

model_config_file_path = "configs/faster_rcnn_r50_fpn_1x_003.py"
output_dir = 'output/faster_rcnn_r50_fpn_1x_{}_classes_003/'.format(num_classes)

total_epochs = 12                                # 最大epoch数
samples_per_gpu=2                                 # 每个gpu计算的图像数量:
gpus =1

device_ids=1

checkpoint = output_dir + 'epoch_12.pth'
eval_result_file_path = output_dir + 'eval_result.pkl'
eval_save_image_dir =  output_dir + 'eval_images'
eval_metrics = 'bbox'

pretrained = 'checkpoints/pretrained/resnet50/resnet50-19c8e357.pth' # backbone使用 resnet50的预训练模型
# mmdet默认加载权重优先级别是resume_from(断点加载)，load_from，pretrained 的顺序，所以需要从load_from加载预训练权重
# resume_from = output_dir+ 'epoch_4.pth'                               # 恢复训练模型的路径
load_from = f'checkpoints/faster_rcnn_r50_fpn_1x_coco_{num_classes}.pth'     # 加载模型的路径，None表示从预训练模型加载，需要修改 num_classes 对应的参数
