#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2020/11/23 16:58 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/23 16:58   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""
import os
import json
import requests

def test_ocr_api(url,file):
    files = {'file': open(file, 'rb')}
    # values = {'upload_file': 'file.txt', 'DB': 'photcat', 'OUT': 'csv', 'SHORT': 'short'}
    json_response = requests.post(url, files=files)
    predictions = json.loads(json_response.text)["data"]
    # print(predictions)
    return predictions

if __name__ =="__main__":
    url ="http://10.20.32.187:30222/hsnlp/ocr/parse_image"
    cwd = os.getcwd()
    table_name = "fullinetable_20200917092928_001.png"
    image_path = os.path.join(cwd, 'tests', 'table', table_name)
    test_ocr_api(url=url, file=image_path)
