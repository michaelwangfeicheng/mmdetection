#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2020/11/16 14:03 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/16 14:03   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""
import logging
logger = logging.getLogger(__name__)
import tornado.web
import os
import cv2 as cv
import base64
import numpy as np
import json

from PIL import Image
from mmdet.apis import inference_detector
from object_detection_server.mmdetection_api import get_result
from object_detection_server.load_model import model

def image_transform_byes2bgr(img_body):
    return img_body


class IndexHandler(tornado.web.RequestHandler):
    """
     Home page for user,photo feeds 主页----所关注的用户图片流
    """
    def get(self,*args,**kwargs):
        self.render('upload.html',img_error=None,path=None)  #打开index.html网页


class UploadFileHandler(tornado.web.RequestHandler):
    @staticmethod
    def get_processed_image_path(ori_image_path, prefix='processed'):
        ori_image_path_ls = ori_image_path.split(os.sep)
        image_name_full = ori_image_path_ls[-1]
        image_name_splited = image_name_full.split('.')
        image_name = ''.join(image_name_splited[:-1])
        img_type = image_name_splited[-1]
        processed_image_name = '{}-{}.{}'.format(image_name, prefix, img_type)
        processed_image_path = os.path.join( os.path.dirname(ori_image_path),processed_image_name)
        return processed_image_path

    def get(self):
        path = os.listdir('static/images')
        self.render('upload.html', img_error=None,path=path)

    def post(self):
        image = self.request.files.get('image', '')
        # files里面部分为[{filename: xx, body: 'xx', content_type:'xx'}, {},....]结构组成
        # 因为只有一个照片
        if image != '':
            img = image[0]
            img_name = img.get('filename', '')
            img_body = img.get('body', '')

            # 保存为二进制流
            image_dir='static/images'
            os.makedirs(image_dir,exist_ok=True)
            image_path = os.path.join(image_dir,img_name)
            with open(image_path, 'wb') as f:
                f.write(img_body)
                print("保存图片{}到{}".format(img_name,image_path))
            # self.set_header('Content-Type', img_type)
            # self.write(img_body)

            # 进行目标识别
            result = inference_detector(model, img=image_path)
            processed_image_name, processed_image_path = self.get_processed_image_path(img_name,image_dir)
            processed_image = model.show_result(image_path, result, out_file=processed_image_path)
            with open(processed_image_path, 'wb') as f:
                f.write(processed_image)
                print("保存图片{}到{}".format(img_name,image_path))
            # Thumbnail
            # im = Image.open(image_path)  # Open picture
            # im.thumbnail((600, 600))  # Set image size
            # im.save(image_path, aa)
            self.redirect('/upload')
        else:
            self.render('upload_form.html', img_error="图片不能为空")


class ObjectDetectionHandler(tornado.web.RequestHandler):
    """
    @author:wangfc27441
    @desc:
        通过 web request 请求获取图片，返回 result 和处理后的图片
    @version：
    @time:2021/1/22 9:04

    Parameters
    ----------

    Returns
    -------
    """
    # def __init__(self,score_thr=0.3,if_ger_processed_image=True):
        # super(ObjectDetectionHandler,self).__init__()
        # self.score_thr = score_thr
        # self.if_ger_processed_image = if_ger_processed_image

    # 定义 post 方法
    @tornado.gen.coroutine
    def post(self):
        # 获取上传的文件: files里面部分为[{filename: xx, body: 'xx', content_type:'xx'}, {},....]结构组成
        files = self.request.files
        # 获取指定文件列表
        imgs = files.get('image', [])

        # 因为只有一个照片
        output_result_ls = []
        for img in imgs:
            img_name = img.get('filename', '')
            content_type = img.get('content_type')
            img_bytes = img.get('body', '')

            # 将img_body 转换为 numpy.ndarray

            # img_decode_ = img_bytes.encode('ascii')  # ascii编码
            # img_decode = base64.b64decode(img_decode_)  # base64解码
            # img_np = np.frombuffer(img_decode, np.uint8)  # 从byte数据读取为np.array形式
            # image_bgr = cv.imdecode(img_np, cv.COLOR_RGB2BGR)  # 转为OpenCV形式

            # convert string of image data to uint8
            nparr = np.fromstring(img_bytes, np.uint8)
            # decode image
            image_bgr = cv.imdecode(nparr, cv.IMREAD_COLOR)

            # image_bgr = image_transform_byes2bgr(img_body)
            # transform image to rgb
            image_rbg = image_bgr[:, :, ::-1]

            # 进行目标识别
            result = inference_detector(model=model, img=image_rbg)
            # 获取识别结果
            output_result = get_result(result=result,class_names=model.CLASSES)
            # 是否返回目标识别处理过的图片
            processed_image=None
            processed_image = model.show_result(image_bgr, result)
            output_result_ls.append({img_name:output_result})
        response = {'code': '00', 'msg': 'Success', 'result': output_result_ls}
        logger.info(f"response={response}")
        self.write(response)




class HelloWorld(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")








