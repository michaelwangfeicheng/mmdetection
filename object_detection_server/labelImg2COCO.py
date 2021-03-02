#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/2/2 8:56

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/2 8:56   wangfc      1.0         None

"""
import re
import os
import ast
from collections import Counter, OrderedDict
import json
from datetime import datetime
from PIL import Image
import shutil

from data_generator.synthetic_data_generator import get_annotation_info, get_image_info

# info字段
INFO = {
    "year": datetime.now().strftime("%Y"),  # 年份
    "version": 'seal-detection-v0.1',  # 版本
    "description": 'labelImg标注数据',  # 数据集描述
    "contributor": 'wangfc27441',  # 提供者
    "date_created": datetime.now().strftime("%Y-%m-%d")
}

# licenses字段
LICENSE = {
    "url": "http://www.hundsun.com",
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License"
}
LICENSES = [LICENSE]

LABEL2CATEGORY_MAPPING = {'红色圆形印章': "round_seal",
                          '黑色圆形印章': "round_seal",
                          '红色半圆形印章': "round_seal",
                          '黑色半圆形印章': "round_seal",
                          '红色椭圆形印章': 'ellipse_seal',
                          '黑色椭圆形印章': 'ellipse_seal',
                          '黑色正方形印章': 'square_name_seal',
                          '红色正方形印章': 'square_name_seal',
                          '红色长方形印章': 'rectangle_company_seal',
                          '黑色长方形印章': 'rectangle_company_seal',
                          '蓝色长方形印章': 'rectangle_company_seal',
                          '红色三角形印章': 'triangle_seal',
                          '二维码': 'qrcode',
                          }


def get_labels(image_dir,label_lines):
    total_labels = []
    for index, label_line in enumerate(label_lines):
        # label_line = label_lines[0]
        label_line = re.sub(pattern='false', repl='False', string=label_line)
        label_line_split = re.split(pattern='\t|\n', string=label_line)
        image_name = label_line_split[0].split('/')[-1]
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue
            # raise FileNotFoundError(f"image_name={image_name} in image_dir={image_dir} 没有找到！")
        image = Image.open(image_path)
        image_size = image.size
        label_annotation_ls = ast.literal_eval(label_line_split[1])
        labels = []
        # if label_annotation_ls.__len__() > 1:
        #     print(f"index={index},image_name={image_name}，label_annotation_ls={label_annotation_ls}")
        for label_annotation_dict in label_annotation_ls:
            assert set(label_annotation_dict.keys()) == {'transcription', 'points', 'difficult'}
            points = label_annotation_dict['points']
            assert points.__len__() == 4
            # 提取标注的标签
            label = label_annotation_dict['transcription']
            new_label = re.sub(pattern='\r', repl='', string=label)
            labels.append(new_label)

        total_labels.extend(labels)

    counter = Counter()
    counter.update(total_labels)
    print(f"counter: length={counter.__len__()},{counter}")
    return sorted(counter.keys())


def labelImg_annotations2COCO(data_type, categories_path, image_dir, output_annotations_dir, labelImg_annotations_path):
    with open(categories_path, mode='r') as f:
        coco_format_categories = json.load(f)
    print(f"使用标注 coco_format_categories={coco_format_categories}")

    category2id_mapping = {category_info['name']: category_info['id'] for category_info in coco_format_categories}

    with open(labelImg_annotations_path, mode='r') as f:
        label_lines = f.readlines()
        print(f"labelImg_annotations 共有{label_lines.__len__()}条标注数据")

    # labels = get_labels(image_dir,label_lines)

    license = 1
    image_id = 1
    annotation_id = 1
    segmentation = None
    iscrowd = 0
    images_info = []
    annotations_info = []

    image_have_unvalid_category_num =0
    for index, label_line in enumerate(label_lines):
        # label_line = label_lines[0]
        label_line = re.sub(pattern='false', repl='False', string=label_line)
        label_line_split = re.split(pattern='\t|\n', string=label_line)
        image_name = label_line_split[0].split('/')[-1]
        image_path = os.path.join(image_dir, image_name)
        unfounded_image_num =0
        if not os.path.exists(image_path):
            print(f"image_name={image_name} in image_dir={image_dir} 没有找到！")
            unfounded_image_num+=1
            continue

        image = Image.open(image_path)
        image_size = image.size
        label_annotation_ls = ast.literal_eval(label_line_split[1])
        labels = []
        # 标记 该图片是否已经具有有效标注
        image_have_unvalid_category = False
        for label_annotation_dict in label_annotation_ls:
            assert set(label_annotation_dict.keys()) == {'transcription', 'points', 'difficult'}
            points = label_annotation_dict['points']
            assert points.__len__() == 4
            # 提取标注的标签
            label = label_annotation_dict['transcription']
            new_label = re.sub(pattern='\r', repl='', string=label)
            labels.append(new_label)
            # 转换为 id
            category = LABEL2CATEGORY_MAPPING[new_label]
            category_id = category2id_mapping.get(category)
            if category_id is not None:
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = points
                x, y = x1, y1
                w = x3 - x1
                h = y3 - y1
                obj_bbox = (x, y, w, h)
                obj_img_area = w * h
                obj_segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                annotation_info = get_annotation_info(annotation_id=annotation_id, image_id=image_id,
                                                      category_id=category_id,
                                                      segmentation=obj_segmentation,
                                                      area=obj_img_area, bbox=obj_bbox,
                                                      iscrowd=iscrowd)
                annotations_info.append(annotation_info)
                annotation_id += 1
            else:
                image_have_unvalid_category_num += 1
                print(f"第{image_have_unvalid_category_num}张 label={new_label},category={category}不支持图片,image_name={image_name}")

                # 将该图片移动到 untest
                # untest_images_dir = os.path.join(os.path.dirname(image_dir), 'untest_images')
                # os.makedirs(untest_images_dir, exist_ok=True)
                # dst = os.path.join(untest_images_dir, image_name)
                # if os.path.exists(image_path):
                #     shutil.move(src=image_path, dst=dst)
                # break
                image_have_unvalid_category =True



        # if not image_have_unvalid_category:
        # image_id 使用 int 格式
        image_info = get_image_info(image_id=image_id, width=image_size[0], height=image_size[1],
                                    file_name=image_name, license=license)
        images_info.append(image_info)

        image_id += 1


    # 保存的 标注数据文件名称
    instances_json_file = "{}.json".format(data_type)
    # annotations 输出的路径
    output_annotations_json_path = os.path.join(output_annotations_dir, instances_json_file)
    # 生成 COCO 标注数据
    instances_json = OrderedDict(info=INFO, licenses=LICENSES, categories=coco_format_categories,
                                 images=images_info, annotations=annotations_info)

    # # Save annotations
    with open(output_annotations_json_path, "w", encoding='utf-8') as f:
        json.dump(instances_json, f, indent=4, ensure_ascii=False)
    print(f'unfounded_image_num={unfounded_image_num},有效的图片共{image_id-1}张，共有{annotation_id-1}标注')
    print("Saving out instances Annotations of {} in {}".format(data_type, output_annotations_json_path))
    return instances_json


def get_coco_annotations(train_data_root, test_data_root,img_prefix='test', data_type='test'):
    """
    @author:wangfc27441
    @desc:
      # 读取人工标注的信息，转换为 coco格式
    @version：
    @time:2021/2/25 9:18

    Parameters
    ----------

    Returns
    -------
    """
    # 训练数据中的类型信息
    categories_path = os.path.join(train_data_root, 'annotations', 'coco_format_categories.json')
    # 测试数据的图片路径
    image_dir = os.path.join(test_data_root, img_prefix)
    # 测试数据中的人工标注信息 labelImg标注
    output_annotations_dir = os.path.join(test_data_root, 'annotations')
    labelImg_annotations_path = os.path.join(output_annotations_dir, 'Label.txt')
    # 转换为 COCO
    labelImg_annotations2COCO(data_type, categories_path, image_dir, output_annotations_dir, labelImg_annotations_path)



if __name__ == '__main__':
    from tools.config.configs import data_root
    data_type = 'test'
    # 不包含 二维码的数据
    categories_path = os.path.join(data_root,'annotations', 'coco_format_categories.json')
    image_dir = os.path.join(os.getcwd(), 'object_detection_server', 'test_data', 'seal_data_real', 'test')
    output_annotations_dir = os.path.join(os.path.dirname(image_dir), 'annotations')
    labelImg_annotations_path = os.path.join(output_annotations_dir, 'Label.txt')
    labelImg_annotations2COCO(data_type, categories_path, image_dir, output_annotations_dir, labelImg_annotations_path)
