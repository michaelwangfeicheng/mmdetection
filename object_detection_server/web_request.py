#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/2/26 15:52

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/26 15:52   wangfc      1.0         None

"""
import json
import os
import requests
from object_detection_server.labelImg2COCO import LABEL2CATEGORY_MAPPING
import re



class OCRWebRequest():
    def __init__(self,url,label2category_mapping):
        # self.ip = ip
        self.url = url
        # self.port =port
        self.label2category_mapping = label2category_mapping
        self.category2label_mapping = self.get_category2label_mapping()

    def get_category2label_mapping(self):
        category2label_mapping = {}
        for label,category in self.label2category_mapping.items():
            seal_name_without_color_matched = re.match(pattern=r'.{0,2}色(.*)$', string=label)
            if seal_name_without_color_matched is not None:
                category2label_mapping.update({category: seal_name_without_color_matched.groups()[0]})
        return category2label_mapping



    def post_image(self,img_dir=None,img_file_name=None,img_path=None,ppocrlabel_out=True):
        if img_path is not None:
            with open(img_path,mode='rb') as f:
                img_bytes = f.read()
            img_file_name = os.path.basename(img_path)
        files_ = {'file':(img_file_name,img_bytes)}

        response = requests.post(self.url,files=files_)
        response_dict = json.loads(response.text)
        response_data = response_dict['data']
        if ppocrlabel_out:
            # 转换为 ppocrlabel 标注工具的输出格式
            results = response_data['result']
            output= []
            for result_each_page in results:
                # [position, [label,probability]]
                extract_each_page = result_each_page['extract']
                for extract in extract_each_page:
                    position = extract['position']
                    category = extract['label']
                    # 转换为中文
                    label = self.category2label_mapping.get(category)
                    probability = extract['probability']
                    output_result = [position,[label,probability]]
                    output.append(output_result)
        else:
            output = response_data
        return output


if __name__ == '__main__':
    URL = 'http://10.20.33.11:10121/hsnlp/ocr/seal_recognize'
    ocr_web_request = OCRWebRequest(url=URL,label2category_mapping=LABEL2CATEGORY_MAPPING)

    current_dir = os.path.dirname(__file__)
    test_img_dir = os.path.join(os.path.dirname(current_dir),'data','seal_data_real','test')
    img_names = os.listdir(test_img_dir)
    img_index =0
    img_path = os.path.join(test_img_dir,img_names[img_index])
    result = ocr_web_request.post_image(img_path=img_path)



