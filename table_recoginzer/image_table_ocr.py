#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2020/11/18 13:51 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/18 13:51   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""
import logging

logger = logging.getLogger(__file__)
import os
import sys
import re
from PIL import Image
from copy import deepcopy
import cv2 as cv
import numpy as np
from table_recoginzer.ROI_selection import get_table_structure, ocr_on_cells, transform2df, ocr2cells,ocr_on_block
from chineseocr_lite.config import *
from chineseocr_lite.model import OcrHandle


class ImageTableOCR(object):
    """
    @author:wangfc27441
    @desc:
    @version：
    @time:2020/11/18 13:52

    Parameters
        self.image为RGB模块的图片
        self.gray为灰度模式的图片

    Returns
    -------
    """

    # 初始化
    def __init__(self, ImagePath):
        # 读取图片
        self.image = cv.imread(ImagePath, 1)
        # 把图片转换为灰度模式
        self.gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

    # 横向直线检测
    def HorizontalLineDetect(self):
        """
        @author:wangfc27441
        @desc: 首先对图片进行二值化处理，再进行两次中值滤波，这样是为了使相邻两条记录之间的像素区别尽可能大。然后对该图片中的每一行的像素进行检测， 以相邻两行的平均像素差大于120为标准， 识别出分割两条记录的横线。
        @version：
        @time:2020/11/18 13:54

        Parameters
        ----------

        Returns
        -------
        """
        # 图像二值化
        ret, thresh1 = cv.threshold(self.gray, 240, 255, cv.THRESH_BINARY)
        # 进行两次中值滤波
        blur = cv.medianBlur(thresh1, 3)  # 模板大小3*3
        blur = cv.medianBlur(blur, 3)  # 模板大小3*3

        h, w = self.gray.shape

        # 横向直线列表
        horizontal_lines = []
        for i in range(h - 1):
            # 找到两条记录的分隔线段，以相邻两行的平均像素差大于120为标准
            if abs(np.mean(blur[i, :]) - np.mean(blur[i + 1, :])) > 120:
                # 在图像上绘制线段
                horizontal_lines.append([0, i, w, i])
                cv.line(self.image, (0, i), (w, i), (0, 255, 0), 2)

        horizontal_lines = horizontal_lines[1:]
        # print(horizontal_lines)
        return horizontal_lines

    #  纵向直线检测
    def VerticalLineDetect(self):
        """
        @author:wangfc27441
        @desc: 利用opencv中的Hough直线检测方法来检测图片中的竖线
        首先我们对灰度图片进行Canny边缘检测，在此基础上再利用Hough直线检测方法识别图片中的直线，要求识别的最大间距为30，线段长度最小为500，并且为竖直直线（x1 == x2）
        @version：
        @time:2020/11/18 13:54

        Parameters
        ----------

        Returns
        -------
        """
        # Canny边缘检测
        edges = cv.Canny(self.gray, 30, 240)

        # Hough直线检测
        minLineLength = 500
        maxLineGap = 30
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap).tolist()
        lines.append([[13, 937, 13, 102]])
        lines.append([[756, 937, 756, 102]])
        sorted_lines = sorted(lines, key=lambda x: x[0])

        # 纵向直线列表
        vertical_lines = []
        for line in sorted_lines:
            for x1, y1, x2, y2 in line:
                # 在图片上绘制纵向直线
                if x1 == x2:
                    print(line)
                    vertical_lines.append((x1, y1, x2, y2))
                    cv.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return vertical_lines

    # 顶点检测
    def VertexDetect(self):
        vertical_lines = self.VerticalLineDetect()
        horizontal_lines = self.HorizontalLineDetect()
        # 顶点列表
        vertex = []
        for v_line in vertical_lines:
            for h_line in horizontal_lines:
                vertex.append((v_line[0], h_line[1]))
        # print(vertex)
        # 绘制顶点
        for point in vertex:
            cv.circle(self.image, point, 1, (255, 0, 0), 2)
        return vertex

    # 寻找单元格区域
    def CellDetect(self):
        vertical_lines = self.VerticalLineDetect()
        horizontal_lines = self.HorizontalLineDetect()
        # 顶点列表
        rects = []
        for i in range(0, len(vertical_lines) - 1, 2):
            for j in range(len(horizontal_lines) - 1):
                rects.append((vertical_lines[i][0], horizontal_lines[j][1], \
                              vertical_lines[i + 1][0], horizontal_lines[j + 1][1]))
        # print(rects)
        return rects

    # 识别单元格中的文字
    def OCR(self):
        rects = self.CellDetect()
        thresh = self.gray

        # 特殊字符列表
        special_char_list = ' `~!@#$%^&*()-_=+[]{}|\\;:‘’，。《》/？ˇ'
        for i in range(20):
            rect1 = rects[i]
            DetectImage1 = thresh[rect1[1]:rect1[3], rect1[0]:rect1[2]]

            # Tesseract所在的路径
            pytesseract.pytesseract.tesseract_cmd = 'C://Program Files (x86)/Tesseract-OCR/tesseract.exe'
            # 识别数字（每行第一列）
            text1 = pytesseract.image_to_string(DetectImage1, config="--psm 10")
            print(text1, end='-->')

            # 识别汉字（每行第二列）
            rect2 = rects[i + 20]
            DetectImage2 = thresh[rect2[1]:rect2[3], rect2[0]:rect2[2]]
            text2 = pytesseract.image_to_string(DetectImage2, config='--psm 7', lang='chi_sim')
            text2 = ''.join([char for char in text2 if char not in special_char_list])
            print(text2, end='-->')

            # 识别汉字（每行第三列）
            rect3 = rects[i + 40]
            DetectImage3 = thresh[rect3[1]:rect3[3], rect3[0]:rect3[2]]
            text3 = pytesseract.image_to_string(DetectImage3, config='--psm 7', lang='chi_sim')
            text3 = ''.join([char for char in text3 if char not in special_char_list])
            print(text3)

    # 显示图像
    def ShowImage(self):
        cv.imshow('AI', self.image)
        cv.waitKey(0)
        # cv.imwrite('E://Horizontal.png', self.image)


#
# ImagePath = 'E://AI.png'
# imageOCR = ImageTableOCR(ImagePath)
# imageOCR.OCR()
def get_horizontal_lines(image_path):
    # 读取图片
    image = cv.imread(image_path, 1)
    ori_image = deepcopy(image)
    # 把图片转换为灰度模式
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 灰度图像对于Canny边缘检测
    canny = cv.Canny(gray, 50, 150)

    # 图像二值化
    ret, thresh1 = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

    # 进行两次中值滤波
    blur = cv.medianBlur(thresh1, 1)  # 模板大小3*3
    blur = cv.medianBlur(blur, 1)  # 模板大小3*3

    logger.info(
        "cv.THRESH_BINARY={},thresh1.shape={},ret.shape={},blur.shape={}"
            .format(cv.THRESH_BINARY, thresh1.shape, ret, blur.shape))

    h, w = gray.shape

    image_with_line = deepcopy(blur)
    # 横向直线列表
    horizontal_lines = []
    for i in range(h - 1):
        # 找到两条记录的分隔线段，以相邻两行的平均像素差大于120为标准
        if abs(np.mean(blur[i, :]) - np.mean(blur[i + 1, :])) > 100:
            horizontal_lines.append([0, i, w, i])
            # 在图像上绘制线段
            cv.line(image_with_line, (0, i), (w, i), (255, 0, 0), 2)

    # horizontal_lines = horizontal_lines[1:]
    print(len(horizontal_lines))


def parse_pic_to_excel_data(src):
    raw = cv.imread(src, 1)
    # 灰度图片
    gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
    # 二值化
    binary = cv.adaptiveThreshold(~gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, -5)
    cv.imshow("binary_picture", binary)  # 展示图片
    rows, cols = binary.shape
    scale = 40
    # 自适应获取核值 识别横线
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (cols // scale, 1))
    eroded = cv.erode(binary, kernel, iterations=1)

    dilated_col = cv.dilate(eroded, kernel, iterations=1)
    cv.imshow("excel_horizontal_line", dilated_col)
    # cv.waitKey(0)
    # 识别竖线
    scale = 20
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, rows // scale))
    eroded = cv.erode(binary, kernel, iterations=1)
    dilated_row = cv.dilate(eroded, kernel, iterations=1)
    cv.imshow("excel_vertical_line", dilated_row)
    # cv.waitKey(0)
    # 标识交点
    bitwise_and = cv.bitwise_and(dilated_col, dilated_row)
    cv.imshow("excel_bitwise_and", bitwise_and)
    # cv.waitKey(0)
    # 标识表格
    merge = cv.add(dilated_col, dilated_row)
    cv.imshow("entire_excel_contour", merge)
    # cv.waitKey(0)
    # 两张图片进行减法运算，去掉表格框线
    merge2 = cv.subtract(binary, merge)
    cv.imshow("binary_sub_excel_rect", merge2)

    new_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    erode_image = cv.morphologyEx(merge2, cv.MORPH_OPEN, new_kernel)
    cv.imshow('erode_image2', erode_image)
    merge3 = cv.add(erode_image, bitwise_and)
    cv.imshow('merge3', merge3)
    # cv.waitKey(0)
    # 识别黑白图中的白色交叉点，将横纵坐标取出
    ys, xs = np.where(bitwise_and > 0)
    # 纵坐标
    y_point_arr = []
    # 横坐标
    x_point_arr = []
    # 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值值相近，我只取相近值的最后一点
    # 这个10的跳变不是固定的，根据不同的图片会有微调，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
    i = 0
    sort_x_point = np.sort(xs)
    for i in range(len(sort_x_point) - 1):
        if sort_x_point[i + 1] - sort_x_point[i] > 10:
            x_point_arr.append(sort_x_point[i])
        i = i + 1
    x_point_arr.append(sort_x_point[i])  # 要将最后一个点加入

    i = 0
    sort_y_point = np.sort(ys)
    # print(np.sort(ys))
    for i in range(len(sort_y_point) - 1):
        if (sort_y_point[i + 1] - sort_y_point[i] > 10):
            y_point_arr.append(sort_y_point[i])
        i = i + 1
    # 要将最后一个点加入
    y_point_arr.append(sort_y_point[i])
    print('y_point_arr', y_point_arr)
    print('x_point_arr', x_point_arr)
    # 循环y坐标，x坐标分割表格
    data = [[] for i in range(len(y_point_arr))]
    for i in range(len(y_point_arr) - 1):
        for j in range(len(x_point_arr) - 1):
            # 在分割时，第一个参数为y坐标，第二个参数为x坐标
            cell = raw[y_point_arr[i]:y_point_arr[i + 1], x_point_arr[j]:x_point_arr[j + 1]]
            cv.imshow("sub_pic" + str(i) + str(j), cell)

            # 读取文字，此为默认英文
            # pytesseract.pytesseract.tesseract_cmd = 'E:/Tesseract-OCR/tesseract.exe'
            text1 = '1'  # pytesseract.image_to_string(cell, lang="chi_sim")

            # 去除特殊字符
            # text1 = re.findall(r'[^\*"/:?\\|<>″′‖ 〈\n]', text1, re.S)
            # text1 = "".join(text1)
            print('单元格图片信息：' + text1)
            data[i].append(text1)
            j = j + 1
        i = i + 1
    # cv.waitKey(0)
    return data


def load_ocr_model(ocr_name):
    if ocr_name == 'chineseocr_lite':
        ocr_model = OcrHandle()
    return ocr_model


if __name__ == "__main__":
    cwd = os.getcwd()
    image_file_name = "fullinetable_20200917092928_001.png"
    # image_file_name = "lADPDhmOtvS8onbNC1LNCAA_2048_2898.jpg"
    image_path = os.path.join(cwd, 'tests', 'table', image_file_name)
    OCR_NAMES = ['pytesseract', 'chineseocr_lite']
    OCR_MODEL = load_ocr_model(ocr_name=OCR_NAMES[1])
    URL = "http://10.20.32.187:30222/hsnlp/ocr/parse_image"

    image_gray, x_coordinates, y_coordinates,block = get_table_structure(image_path)

    # 对整个表格使用OCR 识别，然后根据顶点的位置信息依次还原各个cell的内容
    # block = ocr2cells(image=image, x_val=x_coordinates, y_val=y_coordinates,
    #                   ocr_name=OCR_NAMES[1], ocr_model=None, ocr_api=URL )

    # 利用表格的顶点，分割表格为一个个cell，再对每个cell依次使用OCR 识别
    # final_block = ocr_on_cells(image_path=image_path, x_val=x_coordinates, y_val=y_coordinates,
    #                            ocr_name='paddle_ocr', ocr_api=URL,save=True)
    
    final_block = ocr_on_block(image_path=image_path,block=block,
                               ocr_name='paddle_ocr', ocr_api=URL,save=True)

    table_df = transform2df(final_block)
    excel_file_name = 'paddleOCR_' + image_file_name.split('.')[0] + ".xlsx"
    excel_path = os.path.join(os.path.dirname(image_path), excel_file_name)
    table_df.to_excel(excel_path, encoding="utf_8_sig", index=False, header=False)
