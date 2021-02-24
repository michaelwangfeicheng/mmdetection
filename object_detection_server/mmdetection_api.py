#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2020/11/18 9:24 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/18 9:24   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""
import asyncio
import json
import os
import random
import re
import sys
import time
import math

import torch
from matplotlib import pyplot as plt
from PIL import Image
from mmcv.utils import get_logger
import logging

from data_generator.synthetic_data_generator import draw_bbox
from mmdet.utils.contextmanagers import concurrent

logger = logging.getLogger(__name__)
from typing import List
import numpy as np
import cv2 as cv
from mmdet.apis import init_detector, inference_detector, async_inference_detector


def build_logger(log_dir, logger_name, log_level='INFO'):
    log_dir = os.path.join(log_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    LOG_FILE = f'{logger_name}_{timestamp}.log'
    log_file_path = os.path.join(log_dir, LOG_FILE)
    # name = "object_detection_server",启动后该目录下包加载的 logger 使用 __name__ = object_detection_server.package 都会有继承关系，可以沿用该 logger
    logger = get_logger(name=logger_name, log_file=log_file_path, log_level=log_level)
    return logger


def seal_detection(model, image: np.ndarray, score_thr: float = 0.3, if_ger_processed_image=False,
                   only_seal_detection=False):
    """
    @author:wangfc27441
    @desc:
    输入: 一张图片 image = cv.imread(test_img_path,1) # load image as bgr
    输出：识别出 印章所在的位置信息 list[json]  （+ processed_image as bgr）
    @version：
    @time:2020/11/18 10:53

    Parameters
    ----------
    only_seal_detection: 对 印章的子类进行合并为 seal

    Returns
    -------
    """
    # transform image to rgb
    # image_rbg = image[:, :, ::-1]
    # 使用模型进行预测
    result = inference_detector(model, img=image)
    # 获取识别结果
    output_result = get_result(result=result, class_names=model.CLASSES, score_thr=score_thr,
                               only_seal_detection=only_seal_detection)

    # 是否返回目标识别处理过的图片
    processed_image = None
    if if_ger_processed_image:
        processed_image = model.show_result(image, result)
    return output_result, processed_image



# def async_seal_detection_main(model, image: np.ndarray, score_thr: float = 0.3, if_ger_processed_image=False,
#                    only_seal_detection=True,streamqueue_size = 3
#                          ):
#     """
#     @author:wangfc27441
#     @desc:  异步请求
#     @version：
#     @time:2021/2/5 11:31
#
#     Parameters
#     ----------
#     streamqueue_size :  queue size defines concurrency level
#
#     Returns
#     -------
#     """
#     # queue is used for concurrent inference of multiple images
#     streamqueue = asyncio.Queue()
#
#     for _ in range(streamqueue_size):
#         streamqueue.put_nowait(torch.cuda.Stream(device=device))
#
#     # test a single image and show the results
#     # img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
#     async with concurrent(streamqueue):
#         result = await async_inference_detector(model, image)





def get_result(result,
               class_names, score_thr=0.3, only_seal_detection=True, drop_overlaped_bbox=True):
    """
    @author:wangfc27441
    @desc:
    输入 mmdetection 预测结果，转变为我们需要的格式
    1. 增加 only_seal_detection 参数： 可以多个不同的印章子类型中 返回 印章父类型
    2.  drop_duplicate_size：
    @version：
    @time:2021/2/3 10:03

    Parameters
    ----------

    Returns
    -------
    """

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    # bbox_result = num_classes * [num,5] = 80 * [num,5]
    bboxes = np.vstack(bbox_result)
    # 根据每个 bbox的index和shape 生成 label
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        scores = scores[inds]

    if drop_overlaped_bbox and bboxes.shape[0] > 0:
        # 如果 x,y 顶点在距离较近的点，我们需要去除重复的
        bboxes, scores, labels = nms_after_detection(dets=bboxes, scores=scores, labels=labels)

    output_result = []
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        # 转换为普通的整数类型
        bbox_int = [int(dot) for dot in bbox_int]

        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_probability = f'{bbox[-1]:.03f}'

        right_top = (bbox_int[2], bbox_int[1])
        left_bottom = (bbox_int[0], bbox_int[3])
        bbox_result = {"position": list(left_top + right_top + right_bottom + left_bottom),
                       "label": label_text, 'probability': label_probability,
                       'value': None}
        output_result.append(bbox_result)

    if only_seal_detection:
        only_seal_detection_result = []
        # 对识别的结果进行过滤
        for bbox_result in output_result:
            label = bbox_result['label']
            if re.match(pattern=".*_seal$", string=label):
                bbox_result.update({'label': 'seal'})
                only_seal_detection_result.append(bbox_result)
        return only_seal_detection_result
    else:
        # 因为训练的时候还包括了非印章的类型，所以需要过滤非 印章的类型:如国徽，身份证等
        seal_detection_result = []
        # 对识别的结果进行过滤
        for bbox_result in output_result:
            label = bbox_result['label']
            if re.match(pattern=".*seal$", string=label):
                seal_detection_result.append(bbox_result)
        return seal_detection_result


def show_result(img_path, bbox_results, annotation=None, show=True, fig_size=(20, 15)):
    bboxes = []
    labels = []
    probabilities = []
    for result in bbox_results:
        position = result['position']
        x1, y1, x2, y2, x3, y3, x4, y4 = position
        x, y = x1, y1
        w = x3 - x1
        h = y3 - y1
        bbox = (x, y, w, h)
        label = result['label']
        probability = result['probability']
        bboxes.append(bbox)
        labels.append(label)
        probabilities.append(probability)

    img = Image.open(img_path)
    # 灰度图 转换为 RGB
    if img.mode == 'L':
        img = img.convert("RGB")

    img_drawed = None
    if show and img_path.split('.')[-1] != 'tif':
        img_drawed = draw_bbox(img=img, bboxes=bboxes, labels=labels, probabilities=probabilities)
        if annotation:
            anno_bboxes = [ann['bbox'] for ann in annotation]
            anno_labels = [ann['category_id'] for ann in annotation]
            img_drawed = draw_bbox(img=img_drawed, bboxes=anno_bboxes, labels=anno_labels, fill='red', width=3)

        plt.figure(figsize=fig_size)
        plt.imshow(img_drawed)
        plt.show()
    return img_drawed


def nms_after_detection(dets, scores, labels, thresh=0.5):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]  # pred bbox top_x
    y1 = dets[:, 1]  # pred bbox top_y
    x2 = dets[:, 2]  # pred bbox bottom_x
    y2 = dets[:, 3]  # pred bbox bottom_y
    # scores = dets[:, 4]  # pred bbox cls score

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # pred bbox areas
    order = scores.argsort()[::-1]  # 对pred bbox按score做降序排序，对应(2)-1

    keep = []  # NMS后，保留的 pred bbox
    while order.size > 0:
        i = order[0]  # top-1 score bbox
        keep.append(i)  # top-1 score的话，自然就保留了
        xx1 = np.maximum(x1[i], x1[order[1:]])  # top-1 bbox（score最大）与order中剩余bbox计算NMS
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # IoU计算

        inds = np.where(ovr <= thresh)[
            0]  # 这个操作可以对代码断点调试理解下，结合(2)-2，我们希望剔除所有与当前top-1 bbox IoU > thresh的冗余bbox，那么保留下来的bbox，自然就是ovr <= thresh的非冗余bbox，其inds保留下来，作进一步筛选
        order = order[
            inds + 1]  # 保留有效bbox，就是这轮NMS未被抑制掉的幸运儿，为什么 + 1？因为ind = 0就是这轮NMS的top-1，剩余有效bbox在IoU计算中与top-1做的计算，inds对应回原数组，自然要做 +1 的映射，接下来就是的循环

    keep_dets = dets[keep]
    keep_scores = scores[keep]
    keep_labels = labels[keep]
    return keep_dets, keep_scores, keep_labels  # 最终NMS结果返回


def test_api_speed(test_num, test_data_dir, test_data_filename, model):
    i = 1
    while i < test_num:
        # for image_index in range():
        image_index = random.randint(0, test_image_size - 1)
        test_img_path = os.path.join(test_data_dir, test_data_filename)  # test_data_filenames[image_index])
        test_img_bgr = cv.imread(test_img_path)
        start_time = time.time()
        result, processed_image = seal_detection(model=model, image=test_img_bgr, if_ger_processed_image=False)
        # logger.info(f"i={i},image_name = {test_data_filenames[image_index]}\nresult={result}")
        # if i % 100==0:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"i={i},speed={duration}")
        i += 1


def seal_similarity(det_res_1,det_res_2,class_key = 'label',position_key='position'):
    """
    @author:wangfc27441
    @desc:
    两个印章的相似度： 根据印章的形状和大小计算
    相似度取值范围： [0,1]
    计算方式： 在形状（返回类型）相同的情况下：similarity = 1- (max_area-min_area)/max_area

    @version：
    @time:2021/2/20 9:04

    Parameters
    ----------

    Returns
    -------
    """
    if det_res_1.get(class_key)!=det_res_2.get(class_key):
        return 0
    else:
        det_res_areas = []
        for det_res in zip(det_res_1,det_res_2):
            det_res_pos = det_res.get(position_key)
            x1,y1,x2,y2,x3,y3,x4,y4 = det_res_pos
            w= x3-x1
            h =y3-y1
            area = w*h
            det_res_areas.append(area)
        sorted_det_res_areas = sorted(det_res_areas)
        min_area = sorted_det_res_areas[0]
        max_area = sorted_det_res_areas[-1]
        similarity = 1 - (max_area-min_area)/max_area
        return similarity









if __name__ == '__main__':
    """
    需求：
    1. 任意的位置启动 mmdetection_api.py 进行测试
    2. 将 object_detection_server 作为一个 package 使用
    """
    from object_detection_server.config.configs import *
    from object_detection_server.evaluation import COCODatasetEvaluation

    # 当启动当前文件为主路口的时候，指定工作目录
    working_dir = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # os.chdir(working_dir)
    # 增加当前路径为 包搜索路径
    sys.path.append(working_dir)

    # 当从任意位置启动该脚本的时候，需要加入其对应的搜索路径
    from object_detection_server.load_model import model

    logger = build_logger(log_dir=os.path.dirname(__file__), logger_name='object_detection_server')

    test_data_dir = os.path.join(working_dir, 'data', 'test_data_20210224')
    test_data_filenames = [image_name for image_name in os.listdir(test_data_dir) if
                           image_name.split(".")[-1] in ['png', 'jpg', 'tif', 'bmp']]
    test_image_size = test_data_filenames.__len__()
    logger.info(f"测试 mmdetection_api on device {device}:test_image_size={test_image_size}")
    # test_filename = "券商-83_2.jpg"# "0_09.jpg"#"0_1.jpg" #'0_06.jpg'#"page_6.png"
    results = []
    test_num = 10000
    for index, test_image_name in enumerate(test_data_filenames):
    # index =0
    # while index < test_num:
        image_index = index % test_image_size
        test_image_name = test_data_filenames[image_index]
        test_img_path = os.path.join(test_data_dir, test_image_name)
        print(f'index={index},test_img_path={test_img_path}')
        test_img_bgr = cv.imread(test_img_path, 1)  # load image as bgr
        # 使用模型进行预测
        result, processed_image = seal_detection(model=model, image=test_img_bgr,only_seal_detection=False,
                                                 if_ger_processed_image=True)
        print(f'result={result}')
        eval_images_dir = os.path.join(test_data_dir,f'eval_images_model_{num_classes}_classes_epoch_{test_epoch}_{test_no}')
        os.makedirs(eval_images_dir,exist_ok=True)
        eval_image_path = os.path.join(eval_images_dir,test_image_name)
        cv.imwrite(eval_image_path,processed_image)
        # img_drawed = show_result(img_path=test_img_path, output_result=output_result, fig_size=(20, 15))
        # results.append({'image_name': test_image_name, 'result': result})
        # index += 1
