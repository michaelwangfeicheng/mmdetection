#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  

@time: 2021/1/18 13:53 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/18 13:53   wangfc      1.0         None

"""

import torch
from tools.config.configs import num_classes


def convert_pretrained_model(load_from,num_class,save_path):
      pretrained_weights  = torch.load(load_from)

      print("roi_head.bbox_head.fc_cls.weight.size={}"
            .format(pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.weight'].size()))

      print("roi_head.bbox_head.fc_cls.bias.size={}"
            .format(pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.bias'].size()))
      print("roi_head.bbox_head.fc_reg.weight.size={}"
            .format(pretrained_weights['state_dict']['roi_head.bbox_head.fc_reg.weight'].size()))
      print("roi_head.bbox_head.fc_reg.bias.size={}"
            .format(pretrained_weights['state_dict']['roi_head.bbox_head.fc_reg.bias'].size()))

      pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.weight'].resize_(num_class+1, 1024)
      pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.bias'].resize_(num_class+1)
      pretrained_weights['state_dict']['roi_head.bbox_head.fc_reg.weight'].resize_(num_class*4, 1024)
      pretrained_weights['state_dict']['roi_head.bbox_head.fc_reg.bias'].resize_(num_class*4)

      torch.save(pretrained_weights,save_path )
if __name__ == '__main__':
      load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
      save_path = "checkpoints/faster_rcnn_r50_fpn_1x_coco_%d.pth"%num_classes
      convert_pretrained_model(load_from=load_from,num_class=num_classes,save_path=save_path)