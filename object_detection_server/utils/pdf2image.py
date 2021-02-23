#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2021/2/5 10:03 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/5 10:03   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""

import os
from typing import List
import time
import fitz
import logging

from mmdet.utils import get_root_logger


timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
log_file = os.path.join(os.path.dirname(__file__),'log', f'{"pdf2image"}-{timestamp}.log')
logger = get_root_logger(log_file=log_file, log_level="INFO")


def pdf2image(pdf_path:str,output_dir:str=None,data_byte:bytes=None,page_name_pattern:str = 'page-num.png',
               pdf_page_nums:List[int] =None, zoom_x=2,zoom_y=2):
    pdf_name = os.path.basename(pdf_path)[:-4]
    try:
        if pdf_path:
            doc= fitz.open(pdf_path)
        elif data_byte:
            doc = fitz.open(stream= data_byte,filetype='bytes')
    except IOError as e:
        raise (f"[erro in {__name__}],文件打开失败:{e}")

    if output_dir is None:
        base_name = os.path.basename(pdf_path)
        base_name = base_name.split('.')[0]
        output_dir = os.path.join(os.path.dirname(pdf_path),base_name)
    os.makedirs(output_dir,exist_ok=True)
    mat = fitz.Matrix(zoom_x, zoom_y)

    if pdf_page_nums is not None:
        page_range =[page -1 for page in pdf_page_nums if page -1 >=0]
    else:
        page_range = range(doc.pageCount)

    i=1
    for page_i in page_range:
        pdf_page_i = page_i +1
        if i % 10==1:
            logger.info(f"开始处理第{i}张图片")
        page = doc[page_i]
        pix = page.getPixmap(matrix=mat)
        page_path = os.path.join(output_dir,f'{pdf_name}_page_{pdf_page_i}.png')
        pix.writePNG(page_path)
        i+=1
    logger.info(f"PDF转换为图片结束，图片保存在{output_dir}")

if __name__ == '__main__':
    # pdf_name = '3878005567758457551011715.pdf'
    pdf_dir = os.path.join(os.getcwd(),'data','pdf_001')
    output_dir = os.path.join(pdf_dir,'images')
    pdf_names = os.listdir(pdf_dir)
    logger.info(f"共有{pdf_names.__len__()}个pdf文件")
    for pdf_name in pdf_names:
        if pdf_name[-3:] =='pdf':
            pdf_path = os.path.join(pdf_dir,pdf_name)
            pdf2image(pdf_path=pdf_path,output_dir=output_dir)