#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2021/1/15 10:12 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/15 10:12   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""
import os
from datetime import datetime
import json
current_dir = os.path.dirname(__name__)

"""
5个字段信息：info, licenses, images, annotations，categories。
上面3种标注类型共享的字段信息有：info、image、license。
不共享的是annotation和category这两种字段，他们在不同类型的JSON文件中是不一样的。
{
    "info": info,
    "licenses": [license],
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}

"""
# info字段
INFO = {
    "year": datetime.now().strftime("%Y"),  # 年份
    "version": 'seal-detection-v0.1',  # 版本
    "description": '印章识别的自动生成数据',  # 数据集描述
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

# categories字段
# categories 是一个包含多个 category 实例的列表，而一个category结构体描述如下：
#
# SEAL= 'seal'
# CATEGORY_ID = 1
# CATEGORY= {
# 	"supercategory":SEAL ,# 主类别
#     "id": CATEGORY_ID,# 类对应的id （0 默认为背景）
#     "name": SEAL # 子类别
# }
# CATEGORIES= [CATEGORY]

def generator_coco_format_categories(subobject_dirs,categories_path =None ,if_save=True):
    """
    @author:wangfc27441
    @desc:
    @version：
    @time:2021/1/28 13:32

    Parameters
    ----------

    Returns
    -------
    """
    coco_format_categories = []
    subobject_dir2category_dict = {}
    category_id =1
    for subobject_dir in sorted(subobject_dirs):
        category = {
            "supercategory": subobject_dir.split('_')[-1],  # 主类别
            "id": category_id,  # 类对应的id （0 默认为背景）
            "name": subobject_dir  # 子类别
        }
        coco_format_categories.append(category)
        subobject_dir2category_dict.update({subobject_dir:category})
        category_id +=1
    if if_save:
        os.makedirs(os.path.dirname(categories_path),exist_ok=True)
        with open(categories_path,mode='w',encoding='utf-8') as f:
            json.dump(coco_format_categories,f,indent=4)

    return coco_format_categories,subobject_dir2category_dict





# images字段
# image= {
#     "id": int,# 图片的ID编号（每张图片ID是唯一的）
#     "width": int,#宽
#     "height": int,#高
#     "file_name": str,# 图片名
#     "license": int,
#     "flickr_url": str,# flickr网路地址
#     "coco_url": str,# 网路地址路径
#     "date_captured": datetime # 数据获取日期
# }

def get_image_info(image_id, width, height, file_name, license):
    # image 信息
    image_info = {
        "id": image_id,  # 图片的ID编号（每张图片ID是唯一的）
        "width": width,  # 宽
        "height": height,  # 高
        "file_name": file_name,  # 图片名
        "license": license,
        # "flickr_url": str,# flickr网路地址
        # "coco_url": str,# 网路地址路径
        "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 数据获取日期
    }
    return image_info


# annotation{
#     "id": int, # 对象ID，因为每一个图像有不止一个对象，所以要对每一个对象编号（每个对象的ID是唯一的）
#     "image_id": int,# 对应的图片ID（与images中的ID对应）
#     "category_id": int,# 类别ID（与categories中的ID对应）
#     "segmentation": RLE or [polygon],# 对象的边界点（边界多边形，此时iscrowd=0）。
#     #segmentation格式取决于这个实例是一个单个的对象（即iscrowd=0，将使用polygons格式）还是一组对象（即iscrowd=1，将使用RLE格式）
#     "area": float,# 区域面积
#     "bbox": [x,y,width,height], # 定位边框 [x,y,w,h]
#     "iscrowd": 0 or 1 #见下
# }

def get_annotation_info(annotation_id,image_id,category_id,segmentation,area,bbox,iscrowd,):
    annotation_info = {
        "id": annotation_id,  # 对象ID，因为每一个图像有不止一个对象，所以要对每一个对象编号（每个对象的ID是唯一的）
        "image_id": image_id,  # 对应的图片ID（与images中的ID对应）
        "category_id": category_id,  # 类别ID（与categories中的ID对应）
        "segmentation": segmentation, #RLE or [polygon],  # 对象的边界点（边界多边形，此时iscrowd=0）。
        # segmentation格式取决于这个实例是一个单个的对象（即iscrowd=0，将使用polygons格式）还是一组对象（即iscrowd=1，将使用RLE格式）
        "area": area,#float,  # 区域面积
        "bbox": bbox,#  [x, y, width, height],  # 定位边框 [x,y,w,h]
        "iscrowd": iscrowd #0 or 1  # 见下
    }
    return annotation_info





if __name__ == '__main__':
    cwd = os.getcwd()
    coco_instances_val_2017_json_path = os.path.join(cwd,"data","COCO","instances_val2017.json")

    with open(coco_instances_val_2017_json_path,mode='r') as f:
        coco_instances_val_2017_json = json.load(f)
    print(f"coco_instances_val_2017_json type={type(coco_instances_val_2017_json)}\n"
          f"keys={coco_instances_val_2017_json.keys()}")

    print(f"\ninfo={coco_instances_val_2017_json.get('info')}\n"
          f"licenses={coco_instances_val_2017_json.get('licenses')}")

    images = coco_instances_val_2017_json.get('images')
    annotations = coco_instances_val_2017_json.get('annotations')
    categories = coco_instances_val_2017_json.get('categories')
    print(f"images={images.__len__()}\n"
          f"annotations={annotations.__len__()}\n"
          f"categories={categories.__len__()}")

    print(f"images[0]={images[0]}\n"
          f"annotations[0]={annotations[0]}\n"
          f"categories[0]={categories[0]}")