#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/1/15 14:30

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/15 14:30   wangfc      1.0         None


"""
import os
from PIL import Image, ImageEnhance

from convert_synthetic_to_coco import *

if __name__ == '__main__':
    output_dir = "data/train_images/"
    date_type ='train'
    instances_json_file = "{}_{}.json".format(SEAL,date_type)
    instances_json_path = os.path.join(output_dir,instances_json_file)
    # 生成 COCO 标注数据

    with open(instances_json_path, "r",encoding='utf-8') as f:
        instances_json = json.load(f)

    print(f"instances_json type={type(instances_json)}\n"
          f"keys={instances_json.keys()}")

    print(f"\ninfo={instances_json.get('info')}\n"
          f"licenses={instances_json.get('licenses')}")

    images = instances_json.get('images')
    annotations = instances_json.get('annotations')
    categories = instances_json.get('categories')
    print(f"images={images.__len__()}\n"
          f"annotations={annotations.__len__()}\n"
          f"categories={categories.__len__()}")

    print(f"images[0]={images[0]}\n"
          f"annotations[0]={annotations[0]}\n"
          f"categories[0]={categories[0]}")

    test_image_info = images[0]
    test_image_filename = test_image_info['file_name']
    test_image_path = os.path.join(output_dir,test_image_filename)

    test_annotation = annotations[0]
    test_annotation_bbox = test_annotation['bbox']
    test_img = Image.open(test_image_path)
