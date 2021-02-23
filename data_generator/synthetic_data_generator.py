# from __future__ import absolute_import
import os
import sys

sys.path.append(os.getcwd())
import time
from mmdet.utils import get_root_logger

LOG_LEVEL = 'INFO'
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
LOG_FILE = f'synthetic_data_generator-{timestamp}.log'
log_dir = os.path.join(os.path.dirname(__file__), 'log')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, LOG_FILE)
logger = get_root_logger(log_file=log_file_path, log_level=LOG_LEVEL)

import json
import argparse
import numpy as np
import random
import math
from PIL import Image, ImageEnhance, ImageDraw, ImageChops, ImageFont
from collections import OrderedDict
from pycocotools.coco import COCO
# 根据项目的工作目录
from data_generator.convert_synthetic_to_coco import *
from data_generator.image_preprocess import *


def parse_args():
    # Entrypoint Args
    parser = argparse.ArgumentParser(description='Create synthetic training data for object detection algorithms.')
    parser.add_argument("-bkg", "--backgrounds", type=str, default="backgrounds/",
                        help="Path to background images folder.")
    parser.add_argument("-obj", "--objects", type=str, default="objects/",
                        help="Path to object images folder.")
    parser.add_argument("-o", "--output", type=str, default="test_data_6_classes_003",
                        help="Path to output images folder.")
    parser.add_argument("-ann", "--annotate", type=bool, default=True,
                        help="Include annotations in the data augmentation steps?")
    parser.add_argument("-s", "--sframe", type=bool, default=False,
                        help="Convert dataset to an sframe?")
    parser.add_argument("-g", "--groups", type=bool, default=False,
                        help="Include groups of objects in training set?")
    parser.add_argument("-mut", "--mutate", type=bool, default=False,
                        help="Perform mutatuons to objects (rotation, brightness, shapness, contrast)")
    args = parser.parse_args()
    return args


def get_box(obj_w, obj_h, max_x, max_y):
    x1, y1 = np.random.randint(0, max_x, 1), np.random.randint(0, max_y, 1)
    x2, y2 = x1 + obj_w, y1 + obj_h
    return [x1[0], y1[0], x2[0], y2[0]]


def get_group_obj_positions(obj_group, bkg):
    bkg_w, bkg_h = bkg.size
    boxes = []
    objs = [Image.open(objs_path + obj_images[i]) for i in obj_group]
    obj_sizes = [tuple([int(0.6 * x) for x in i.size]) for i in objs]
    for w, h in obj_sizes:
        # set background image boundaries
        max_x, max_y = bkg_w - w, bkg_h - h
        # get new box coordinates for the obj on the bkg
        while True:
            new_box = get_box(w, h, max_x, max_y)
            for box in boxes:
                res = intersects(box, new_box)
                if res:
                    break

            else:
                break  # only executed if the inner loop did NOT break
            # logger.info("retrying a new obj box")
            continue  # only executed if the inner loop DID break
        # append our new box
        boxes.append(new_box)
    return obj_sizes, boxes


# 使用 COCO API
def get_annotations_with_COCOAPI(data_dir='data', data_type='train', if_show_demo=False):
    ann_path = f'{data_dir}/annotations/{data_type}.json'
    ann_json = COCO(ann_path)
    category_size = len(ann_json.dataset['categories'])
    annotation_size = len(ann_json.dataset['annotations'])
    image_size = len(ann_json.dataset['images'])
    print(f'image_size={image_size},annotation_size={annotation_size},category_size={category_size}')
    images = ann_json.dataset['images']
    annotations = ann_json.dataset['annotations']
    categories = ann_json.dataset['categories']
    if if_show_demo:
        image_index = 0
        print(f"images[{image_index}]\n{images[image_index]}")
        image_index = image_size - 1
        print(f"images[{image_index}]\n{images[image_index]}")

        annotation_index = 0
        print(f"annotations[{annotation_index}]\n{annotations[annotation_index]}")
        annotation_index = annotation_size - 1
        print(f"annotations[{annotation_index}]\n{annotations[annotation_index]}")

    return ann_json, images, annotations, categories


def generator_synthetic_data(obj_images_names_and_dir_ls, bkg_images,
                             coco_format_categories, subobject_dir2category_dict,
                             sizes, count_per_size,
                             image_num, data_type='train', raw_data_dir='data',
                             image_id_size=12,
                             if_rotate=True, if_enhance=True, max_rotate_angle=30
                             ):
    """
    @author:wangfc27441
    @desc:
    @version：
    @time:2021/1/18 16:24

    Parameters
    ----------
    obj_images_names_and_dir_ls：目标图片名称 + 目录名称
    bkg_images： 背景图片
    obj_label : 使用 subobject_dir 作为 标注 object的label
    sizes:      图片伸缩的不同大小比例
    image_num:  标记处理每张图片时候的 num
    data_type: 数据类型
    image_id_size :  image_id 字符串的长度
    instances_json: instances_json

    Returns
    -------
    """
    base_bkgs_path = os.path.join(raw_data_dir, 'backgrounds')
    #  objects 目录
    obj_dir = os.path.join(raw_data_dir, 'objects')
    # 输出图片的路径： data/train : data/dev: data/test
    output_image_dir = os.path.join(output_dir, data_type)
    # annotations 输出的路径
    output_annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    # images 标注
    images_info = []
    annotations_info = []  # store annots here
    categories_info = []
    license = 1
    annotation_id = 1  # annotation_id 起始的id

    bkg_images_size = bkg_images.__len__()
    obj_images_size = obj_images_names_and_dir_ls.__len__()
    transformer_size = sizes.__len__() * count_per_size
    total_images_size = bkg_images_size * obj_images_size * transformer_size

    # Make synthetic training data
    logger.info("Making synthetic {} images of {}".format(data_type, total_images_size))
    # 遍历所有的 背景图片
    for bkg in bkg_images:
        # Load the background image
        bkg_path = base_bkgs_path + bkg
        bkg_img = Image.open(bkg_path)
        # 获取背景图片的尺寸
        bkg_x, bkg_y = bkg_img.size

        # 遍历所有的目标识别图片 Do single objs first
        for obj_image_name, obj_image_dir_name in obj_images_names_and_dir_ls:
            # Load the single obj
            obj_image_path = os.path.join(obj_dir, obj_image_dir_name, obj_image_name)
            obj_img = Image.open(obj_image_path)
            # Get an array of random obj positions (from top-left corner)
            # 随机选择不同的位置并且进行缩放
            obj_h, obj_w, x_pos, y_pos = get_obj_positions(obj=obj_img, bkg=bkg_img, sizes=sizes, count=count_per_size)

            # Create synthetic images based on positions
            for h, w, x, y in zip(obj_h, obj_w, x_pos, y_pos):
                if image_num % 10 == 1:
                    logger.info(f"开始处理 {data_type} image_id={image_num}/{total_images_size}")

                # 原来的 obj_bbox
                obj_bbox = list([int(x), int(y), int(w), int(h)])
                bkg_w_obj, obj_bbox, obj_mask = combine_background_and_object(bkg_img, obj_img, obj_bbox,
                                                                              if_rotate=if_rotate,
                                                                              if_enhance=if_enhance,
                                                                              max_rotate_angle=max_rotate_angle)
                # 计算 obj_img
                x, y, w, h = obj_bbox
                obj_img_area = w * h

                # 实例是一个单个的对象（即iscrowd=0，将使用polygons格式）
                iscrowd = 0
                obj_segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]

                # 输出图片的id + 名称 + 路径
                category_id = subobject_dir2category_dict[obj_image_dir_name]['id']
                image_id = int(str(category_id) + "0" * 8) + image_num
                output_image_name = str(image_id).zfill(image_id_size) + ".png"
                # 输出图片的路径： data/train : data/dev: data/test
                output_image_path = os.path.join(output_image_dir, output_image_name)
                # image_id 使用 int 格式
                image_info = get_image_info(image_id=image_id, width=bkg_x, height=bkg_y,
                                            file_name=output_image_name, license=license)

                images_info.append(image_info)
                # Save the image
                bkg_w_obj.save(fp=output_image_path, format="png")

                # 默认 annotation 都只有一个

                # Make annotation
                # ann = [{'coordinates': {'height': h, 'width': w, 'x': x+(0.5*w), 'y': y+(0.5*h)}, 'label': i.split(".png")[0]}]
                annotation_info = get_annotation_info(annotation_id=annotation_id, image_id=image_id,
                                                      category_id=category_id,
                                                      segmentation=obj_segmentation, area=obj_img_area, bbox=obj_bbox,
                                                      iscrowd=iscrowd)
                annotation_id += 1
                # Save the annotation data
                annotations_info.append(annotation_info)
                # logger.info(n)
                image_num += 1

    # 保存的 标注数据文件名称
    instances_json_file = "{}.json".format(data_type)
    # annotations 输出的路径
    output_annotations_json_path = os.path.join(output_annotations_dir, instances_json_file)
    # 生成 COCO 标注数据
    instances_json = OrderedDict(info=INFO, licenses=LICENSES, categories=coco_format_categories,
                                 images=images_info, annotations=annotations_info)
    logger.info("Saving out instances Annotations of {} in {}".format(data_type, output_annotations_json_path))
    # # Save annotations
    with open(output_annotations_json_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(instances_json, indent=4, ensure_ascii=False))

    total_images = len([f for f in os.listdir(output_image_dir) if f.endswith(".png")])
    logger.info("Done! Created {} synthetic {} images in {}".format(total_images, data_type, output_image_dir))
    return instances_json, image_num


def generator_synthetic_data_with_objects_group(base_bkgs_dir, base_objects_dir,
                                                objects_group_ls, bkg_images,
                                                coco_format_categories, subobject_dir2category_dict,
                                                sizes, count_per_size, rotate_num=2,
                                                image_id=1, annotation_id=1,
                                                data_type='train', output_dir='data',
                                                image_id_size=12,
                                                if_rotate=True, if_enhance=True, max_rotate_angle=90
                                                ):
    """
    @author:wangfc27441
    @desc:
    @version：
    @time:2021/1/18 16:24

    Parameters
    ----------
    obj_images_names_and_dir_ls：目标图片名称 + 目录名称
    bkg_images： 背景图片
    obj_label : 使用 subobject_dir 作为 标注 object的label
    sizes:      图片伸缩的不同大小比例
    count_per_size： 随机生成的位置数量

    image_num:  标记处理每张图片时候的 num
    data_type: 数据类型
    image_id_size :  image_id 字符串的长度
    instances_json: instances_json

    Returns
    -------
    """
    # base_bkgs_path = os.path.join(output_dir, 'backgrounds')
    # #  objects 目录
    # base_objects_dir = os.path.join(output_dir, 'objects')

    # 输出图片的路径： data/train : data/dev: data/test
    output_image_dir = os.path.join(output_dir, data_type)
    # annotations 输出的路径
    output_annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    # images 标注
    images_info = []
    annotations_info = []  # store annots here
    categories_info = []
    license = 1
    # annotation_id = 1  # annotation_id 起始的id

    bkg_images_size = bkg_images.__len__()
    objects_group_size = objects_group_ls.__len__()

    # 每个 object 变换的个数：
    transformer_size = sizes.__len__() * count_per_size * rotate_num if if_rotate else sizes.__len__() * count_per_size
    total_images_size = bkg_images_size * objects_group_size * transformer_size
    logger.info(f"total_images_size={total_images_size},sizes.len={ sizes.__len__()},count_per_size={count_per_size},rotate_num={rotate_num}")

    # Make synthetic training data
    logger.info("Making synthetic {} images of {}".format(data_type, total_images_size))
    image_num = 1  # 记录生成图片的数量
    # 遍历所有的 背景图片
    for bkg_index, bkg in enumerate(bkg_images):
        # 每次生成图片的时候，选择一张背景图片
        # Load the background image
        bkg_path = os.path.join(base_bkgs_dir, bkg)
        bkg_img = Image.open(bkg_path)
        # 获取背景图片的尺寸
        bkg_x, bkg_y = bkg_img.size

        # 遍历所有的目标识别图片 Do single objs first
        for object_index, objects_group in enumerate(objects_group_ls):
            sample_num = 0
            # 这里每个 objects_group 可以同时有多个 objects：每个 object 由两部分组成 obj_image_name +obj_image_dir_name
            while sample_num < transformer_size:
                # 对 每个 objects_group 中每一种变换：进行放缩和排布，我们需要重新设置背景
                bkg_w_obj = bkg_img.copy()
                if image_num % 10 == 1:
                    logger.info(
                        f"开始处理 {data_type} image_num={image_num}/{total_images_size},image_id={image_id},annotation_id={annotation_id}")

                # 初始化生成的box 和所有 object 生成的 boxes
                boxes = []
                for obj_image_name, obj_image_dir_name in objects_group:
                    # 根据 sample_num 确定放缩的 size
                    size = sizes[sample_num % sizes.__len__()]
                    obj_image_path = os.path.join(base_objects_dir, obj_image_dir_name, obj_image_name)
                    obj_img = Image.open(obj_image_path)
                    # 对每个 object 生成 box

                    # 增加图片渐变处理: 70%的图片进行腐蚀处理
                    if_erode=False
                    random_num = np.random.random()
                    if random_num >0.5:
                        if_erode=True
                        logger.info(f"image_id={image_id},obj_image_dir_name={obj_image_dir_name},if_erode={if_erode}")
                    bkg_w_obj, obj_bbox, new_box = synthesize_background_and_object(bkg_w_obj, bkg_img, obj_img,
                                                                                    obj_image_dir_name,
                                                                                    boxes,
                                                                                    sizes = [size],
                                                                                    if_erode=if_erode,
                                                                                    if_rotate=if_rotate,
                                                                                    max_rotate_angle=max_rotate_angle,
                                                                                    if_enhance=if_enhance)
                    if new_box is None:
                        logger.warning(f'当背景图片较小，object图片较大的时候，可能无法合成图片或者box 必然会交叉等情况:bkg={bkg},bkg.size={bkg_img.size},'
                                       f'obj_image_dir_name/obj_image_name={obj_image_dir_name}/{obj_image_name},obj_img.size={obj_img.size}')
                        continue

                    boxes.append(new_box)

                    # 3. 计算 annotation
                    x, y, w, h = obj_bbox
                    obj_img_area = w * h
                    # 4. 实例是一个单个的对象（即iscrowd=0，将使用polygons格式）
                    iscrowd = 0
                    # 5. 计算 obj_segmentation
                    obj_segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                    # 6. annotation 中的 category_id  + image_id
                    category_id = subobject_dir2category_dict[obj_image_dir_name]['id']
                    image_id = image_num
                    annotation_info = get_annotation_info(annotation_id=annotation_id, image_id=image_id,
                                                          category_id=category_id,
                                                          segmentation=obj_segmentation, area=obj_img_area,
                                                          bbox=obj_bbox,
                                                          iscrowd=iscrowd)
                    # Save the annotation data
                    annotations_info.append(annotation_info)
                    annotation_id += 1

                # 输出图片的id + 名称 + 路径
                output_image_name = str(image_id).zfill(image_id_size) + ".png"

                # image_id 使用 int 格式
                image_info = get_image_info(image_id=image_id, width=bkg_x, height=bkg_y,
                                            file_name=output_image_name, license=license)

                # 输出图片的路径： data/train : data/dev: data/test
                output_image_path = os.path.join(output_image_dir, output_image_name)

                images_info.append(image_info)
                # Save the image
                bkg_w_obj.save(fp=output_image_path, format="png")
                # 对于每组的 object_group 设置某个伸缩
                sample_num += 1
                image_num += 1
                image_id += 1

    # 保存的 标注数据文件名称
    instances_json_file = "{}.json".format(data_type)
    # annotations 输出的路径
    output_annotations_json_path = os.path.join(output_annotations_dir, instances_json_file)
    # 生成 COCO 标注数据
    instances_json = OrderedDict(info=INFO, licenses=LICENSES, categories=coco_format_categories,
                                 images=images_info, annotations=annotations_info)
    logger.info("Saving out instances Annotations of {} in {}".format(data_type, output_annotations_json_path))
    # # Save annotations
    with open(output_annotations_json_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(instances_json, indent=4, ensure_ascii=False))

    total_images = len([f for f in os.listdir(output_image_dir) if f.endswith(".png")])
    logger.info("Done! Created {} synthetic {} images in {}".format(total_images, data_type, output_image_dir))
    return instances_json, image_id, annotation_id


def main(output, max_bkg_num=-1, max_obj_num=20, generator_single_object_data=False,
         selected_subobject_dir_names = None
         ):
    # SEED = 1234
    # random.seed(SEED)
    # np.random.seed(SEED)
    args = parse_args()

    cwd = os.path.abspath(os.path.dirname(__file__))
    rawdata_dir = os.path.join(cwd, 'data')
    base_bkgs_path = os.path.join(rawdata_dir, args.backgrounds)
    objs_dir = os.path.join(rawdata_dir, args.objects)

    output_dir = os.path.join(cwd, output)
    os.makedirs(output_dir, exist_ok=True)
    # 进行测试
    logger.info("开始生成人工图片")
    logger.info(
        f"rawdata_dir={rawdata_dir},base_bkgs_path={base_bkgs_path},objs_path={objs_dir},output_dir={output_dir}")

    # Prepare data creation pipeline
    if max_bkg_num == -1:
        total_bkg_images_names = [f for f in os.listdir(base_bkgs_path) if not f.startswith(".")][:]
    else:
        total_bkg_images_names = [f for f in os.listdir(base_bkgs_path) if not f.startswith(".")][:max_bkg_num]

    # 根据 train:dev 分割比例分割图片
    train_size = 0.8
    dev_size = 0.2

    total_bkg_images_size = len(total_bkg_images_names)
    bkg_train_size = int(total_bkg_images_size * train_size)
    bkg_dev_size = int(total_bkg_images_size * dev_size)

    random.shuffle(total_bkg_images_names)

    bkg_train_images_names = total_bkg_images_names
    logger.info("使用所有的背景图片生成训练数据")
    bkg_dev_images_names = total_bkg_images_names[bkg_train_size:bkg_train_size+bkg_dev_size]

    test_size = 1 - train_size - dev_size
    bkg_test_size = int(total_bkg_images_size * test_size)
    bkg_test_images_names = total_bkg_images_names[bkg_train_size+bkg_dev_size:]

    # assert bkg_dev_images_names.__len__() + bkg_train_images_names.__len__() + bkg_test_images_names.__len__() == total_bkg_images_size
    logger.info(f"背景图片共{total_bkg_images_size}张，按照 train_size:dev_size:test_size = {train_size}:{dev_size}:{test_size} 划分，"
                f"分别为{bkg_train_images_names.__len__()}：{bkg_dev_images_names.__len__()}：{bkg_test_images_names.__len__()}")

    # 设置 图片伸缩的不同大小比例 [0.4, 0.6, 0.8, 1, 1.2]
    sizes = np.arange(1, 1.3, 0.2).tolist()  # different obj sizes to use TODO make configurable
    count_per_size = 1  # number of locations for each obj size TODO make configurable
    rotate_num =1

    # 选择 object 子目录进行图片合成
    subobject_dir_names = os.listdir(objs_dir)
    if selected_subobject_dir_names:
        subobject_dir_names= [sub for sub in subobject_dir_names if sub in selected_subobject_dir_names]
    subobject_dir_names = sorted(subobject_dir_names)
    logger.info(f"目标数据文件夹名称{subobject_dir_names}")

    # 生成 coco 数据集标注的 categories
    coco_format_categories_path = os.path.join(output_dir, 'annotations', 'coco_format_categories.json')
    coco_format_categories, subobject_dir2category_dict = generator_coco_format_categories(subobject_dir_names,
                                                                                           categories_path=coco_format_categories_path)

    # 生成 object 数据
    total_obj_train_names_and_dir_ls = []
    total_obj_dev_names_and_dir_ls = []
    total_obj_test_names_and_dir_ls = []
    for subobject_dir_name in subobject_dir_names:
        # 选择 子目录
        subobject_path = os.path.join(objs_dir, subobject_dir_name)

        obj_images_names = [f for f in os.listdir(subobject_path) if not f.startswith(".")][:]
        if obj_images_names.__len__() > max_obj_num:
            obj_images_names = obj_images_names[:max_obj_num]
        logger.info(f"subobject_dir_name={subobject_dir_name},size = {obj_images_names.__len__()}")

        # 将 obj_image_name + subobject_dir 构成一个tuple
        obj_images_names_and_dir_ls = [(obj_image_name, subobject_dir_name) for obj_image_name in
                                       obj_images_names]
        obj_images_size = len(obj_images_names_and_dir_ls)
        obj_train_size = int(obj_images_size * train_size)
        obj_dev_size = int(obj_images_size * dev_size)
        obj_test_size = int(obj_images_size * test_size)

        random.shuffle(obj_images_names_and_dir_ls)
        obj_train_images_names_and_dir_ls = obj_images_names_and_dir_ls[:obj_train_size]
        obj_dev_images_names_and_dir_ls = obj_images_names_and_dir_ls[
                                          obj_train_size:obj_dev_size + obj_train_size]
        obj_test_images_names_and_dir_ls = obj_images_names_and_dir_ls[obj_dev_size + obj_train_size:]
        assert obj_dev_images_names_and_dir_ls.__len__() + obj_train_images_names_and_dir_ls.__len__() + obj_test_images_names_and_dir_ls.__len__() == obj_images_size

        total_obj_train_names_and_dir_ls.extend(obj_train_images_names_and_dir_ls)
        total_obj_dev_names_and_dir_ls.extend(obj_dev_images_names_and_dir_ls)
        total_obj_test_names_and_dir_ls.extend(obj_test_images_names_and_dir_ls)

    logger.info(f"object 按照 train_size:dev_size:test_size = {train_size}:{dev_size}:{test_size} 划分，"
                f"分别为{total_obj_train_names_and_dir_ls.__len__()}：{total_obj_dev_names_and_dir_ls.__len__()}：{total_obj_test_names_and_dir_ls.__len__()}")

    # 对 train 和 dev 数据集分别进行数据合成
    image_id = 1
    annotation_id = 1
    for data_type, obj_images_names_and_dir_ls, bkg_images in zip(['train', 'dev', 'test'],
                                                                  [total_obj_train_names_and_dir_ls,
                                                                   total_obj_dev_names_and_dir_ls,
                                                                   total_obj_test_names_and_dir_ls],
                                                                  [bkg_train_images_names, bkg_dev_images_names,
                                                                   bkg_test_images_names]):
        # 生成多个 object
        objects_group_ls = []
        obj_size = obj_images_names_and_dir_ls.__len__()
        for i in range(obj_size):
            anchor = obj_images_names_and_dir_ls[i]
            other_index = random.randint(0, obj_size - 1)
            other = obj_images_names_and_dir_ls[other_index]
            objects_group_ls.append((anchor, other))

        instances_json, image_id, annotation_id = generator_synthetic_data_with_objects_group(
            base_bkgs_dir=base_bkgs_path, base_objects_dir=objs_dir,
            objects_group_ls=objects_group_ls,
            bkg_images=bkg_images, data_type=data_type,
            coco_format_categories=coco_format_categories,
            subobject_dir2category_dict=subobject_dir2category_dict,
            image_id=image_id, annotation_id=annotation_id,
            sizes=sizes, count_per_size=count_per_size,rotate_num=rotate_num,
            output_dir=output_dir)


def test_synthesis_image_annotation(data_dir, data_type, image_index, ann_json, images, annotations, categories):
    # image_index 对应的 image_info
    test_image_info = images[image_index]
    test_image_id = test_image_info['id']
    test_image_filename = test_image_info['file_name']

    image_dir = f'{data_dir}/{data_type}'
    test_image_path = os.path.join(image_dir, test_image_filename)
    test_img = Image.open(test_image_path)

    # 找到 image 对应的 annotation
    test_annotations = []
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id == test_image_id:
            test_annotations.append(annotation)

    print(f'image_index={image_index}\nimage_info={test_image_info}\nannotations={test_annotations}')
    return test_img, test_image_path, test_annotations




if __name__ == "__main__":
    main(output="test_data_5_classes_002",max_bkg_num= -1, max_obj_num=5, generator_single_object_data=False,
         selected_subobject_dir_names=['round_seal','ellipse_seal', 'rectangle_name_seal','square_name_seal','rectangle_company_seal'])
                                        # 'front_idcard','back_idcard','guohui'
