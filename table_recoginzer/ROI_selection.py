# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
import re
import os
import sys
import math
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import requests
import json
from io import BytesIO,BufferedReader
from PIL import Image, ImageDraw,ImageFont
import base64


def is_vertical(line):
    return line[0] == line[2]


def is_horizontal(line):
    return line[1] == line[3]


def overlapping_filter(lines, sorting_index):
    filtered_lines = []

    lines = sorted(lines, key=lambda lines: lines[sorting_index])

    for i in range(len(lines)):
        l_curr = lines[i]
        if (i > 0):
            l_prev = lines[i - 1]
            if ((l_curr[sorting_index] - l_prev[sorting_index]) > 5):
                filtered_lines.append(l_curr)
        else:
            filtered_lines.append(l_curr)

    return filtered_lines


def detect_lines(image, title='default', rho=1, theta=np.pi / 180, threshold=50, minLinLength=350, maxLineGap=6,
                 display=False, write=False):
    """
    HoughLinesP函数，有如下几个输入参数：
        image -8位单通道二进制源图像。该图像可以通过该功能进行修改。
        rho —累加器的距离分辨率，以像素为单位。
        theta —弧度的累加器角度分辨率。
        threshold-累加器阈值参数。仅返回那些获得足够投票的行
        line — 线的输出向量。这里设置为无，该值保存到linesP
        minLineLength —最小行长。短于此的线段将被拒绝。
        maxLineGap —同一线上的点之间允许链接的最大间隙。
    """

    if gray is None:
        print('Error opening image!')
        return -1
    # 灰度图像对于Canny边缘检测
    dst = cv.Canny(gray, 50, 150, None, 3)

    # Copy edges to the images that will display the results in BGR
    cImage = np.copy(image)

    # linesP = cv.HoughLinesP(dst, 1 , np.pi / 180, 50, None, 290, 6)
    linesP = cv.HoughLinesP(dst, rho, theta, threshold, None, minLinLength, maxLineGap)

    horizontal_lines = []
    vertical_lines = []

    if linesP is not None:
        # for i in range(40, nb_lines):
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            if (is_vertical(l)):
                vertical_lines.append(l)

            elif (is_horizontal(l)):
                horizontal_lines.append(l)

        horizontal_lines = overlapping_filter(horizontal_lines, 1)
        vertical_lines = overlapping_filter(vertical_lines, 0)

    if (display):
        for i, line in enumerate(horizontal_lines):
            cv.line(cImage, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3, cv.LINE_AA)

            cv.putText(cImage, str(i) + "h", (line[0] + 5, line[1]), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 1, cv.LINE_AA)

        for i, line in enumerate(vertical_lines):
            cv.line(cImage, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv.LINE_AA)
            cv.putText(cImage, str(i) + "v", (line[0], line[1] + 5), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.imshow("Source", cImage)
        # cv.imshow("Canny", cdstP)
        cv.waitKey(0)
        cv.destroyAllWindows()

    if (write):
        image_dir = os.path.join(os.path.dirname(__file__), 'images')
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, title + ".png")
        cv.imwrite(image_path, cImage)

    return (horizontal_lines, vertical_lines)


def get_cropped_image(image, x, y, w, h):
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image


def get_ROI(image, horizontal, vertical, left_line_index, right_line_index, top_line_index, bottom_line_index,
            offset=4):
    x1 = vertical[left_line_index][2] + offset
    y1 = horizontal[top_line_index][3] + offset
    x2 = vertical[right_line_index][2] - offset
    y2 = horizontal[bottom_line_index][3] - offset

    w = x2 - x1
    h = y2 - y1

    cropped_image = get_cropped_image(image, x1, y1, w, h)

    return cropped_image, (x1, y1, w, h)


def main(argv):
    default_file = '../Images/source6.png'
    filename = argv[0] if len(argv) > 0 else default_file

    src = cv.imread(cv.samples.findFile(filename))

    # Loads an image
    horizontal, vertical = detect_lines(src, display=True)

    return 0


def combine_lines(lines, flag, image_width, image_height, width_scale = 20, height_scale = 30):
    """
    @author:wangfc27441
    @desc: #求交点坐标
    @version：
    @time:2020/11/19 16:19

    Parameters
    ----------

    Returns
    -------
    """
    if lines is None:
        return []
    # 求竖线的横坐标
    elif flag == "col":
        lines_x = np.sort(lines[:, :, 0], axis=None)
        list_x = list(lines_x)

        # 合并距离相近的点
        for i in range(len(list_x) - 1):
            if (list_x[i] - list_x[i + 1]) ** 2 <= (image_width /width_scale) ** 2:
                list_x[i + 1] = list_x[i]

        list_x = list(set(list_x))  # 去重
        list_x.sort()  # 排序
        return list_x
    # 求横线的纵坐标
    elif flag == "row":
        # 两个 横线 line 的 y 坐标之间的最小距离
        height_gap  = image_height / height_scale
        # lines.shape =(num_rows,1,4)
        # 获取所有的 横线 line 的 y 坐标，
        lines_y = np.sort(lines[:, :, 1], axis=None)
        list_y = list(lines_y)
        # 合并距离相近的点
        for i in range(len(list_y) - 1):
            if (list_y[i] - list_y[i + 1]) ** 2 <= (height_gap) ** 2:
                list_y[i + 1] = list_y[i]

        list_y = list(set(list_y))  # 去重
        list_y.sort()  # 排序
        print("原有横线共{}条，去重后横线共{}条".format(lines.__len__(),list_y.__len__()))
        return list_y

def get_table_coordinates(image,threshold_binary=200):
    img_height, img_width, _ = image.shape
    #  把图片转换为灰度模式
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 二值化阈值选为100，大于100的置0，小于100的置255
    # threshold_binary = 100
    ret, img_bin = cv.threshold(image_gray, threshold_binary, 255, cv.THRESH_BINARY_INV)

    # 使用不同的核对对二值化后的图像进行开运算(先腐蚀后膨胀)，分别检测出二值图像中的横线和竖线。
    # opencv中的 morphologyEx() 函数可以用自定义的核对图像进行开、闭运算。根据应用场景不同，可灵活调整核的形状和大小。


    # 自定义检测横线的核 : 滤波器的长度设为9，是为了避免较粗线条的干扰
    kernel_row = np.ones((1, 9))
    # kernel_row = cv.getStructuringElement(cv.MORPH_RECT, (1, 5)) # 修改内核为(15,1)可以提起横线
    # kernel_row = np.ones((3, 9))
    kernel_col = np.ones((9, 1))
    # 开运算检测横线 + 竖线
    img_open_row = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel_row)
    img_open_col = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel_col)

    # 检测出横线和竖线后，可以对两张图片分别使用霍夫寻线，获得每条线两端点的坐标。
    # 但在实际操作过程中，发现寻竖线时效果总是不好，经测试后发现由于图片高度较低，竖线普遍很短，不易寻找。
    # 因此可以通过resize()将img_open_col的高度拔高后，再进行霍夫寻线，效果显著。
    # 图片高度较低，为了方便霍夫寻纵线，将图片的高度拉高5倍
    # if img_height*2 < img_width:
    height_resize_scale = 10
    width_resize_scale = 10
    img_open_col = cv.resize(img_open_col, (img_width, height_resize_scale* img_height))
    img_open_row = cv.resize(img_open_row, (width_resize_scale * img_width, img_height))

    # 事实上经过开运算后的img_open_col和img_open_row中已经清晰地呈现出来所有组成表格的横线和纵线，
    # 但要想进一步分割表格，只找到线是不够的，还必须获取线在图片中的位置。
    # 霍夫寻线可以帮助我们完成这一操作，将img_open_col和img_open_row作为参数传递给从cv2.HoughLinesP()，可返回每条线段两端点的坐标(x1, y1, x2, y2)
    maxLineGap = 5
    minLineLength_height = int(0.52 * height_resize_scale * img_height)
    lines_col = cv.HoughLinesP(img_open_col, 1, np.pi / 180, 100, minLineLength=minLineLength_height,
                               maxLineGap=5)
    minLineLength_width = int(0.52  * width_resize_scale * img_width)
    # minLineLength_width = int(0.75 * img_width)
    lines_row = cv.HoughLinesP(img_open_row, 1, np.pi / 180, 100, minLineLength=minLineLength_width,maxLineGap=maxLineGap)

    x_coordinates = combine_lines(lines=lines_col, flag='col', image_width=img_width, image_height=img_height)
    y_coordinates = combine_lines(lines=lines_row, flag='row', image_width=img_width, image_height=img_height)
    print("横线共{}条，竖线共{}条".format(len(y_coordinates), len(x_coordinates)))
    return x_coordinates,y_coordinates


def draw_cells(image,x_coordinates,y_coordinates,
               image_path=None,image_name=None, save=True):
    """
    @author:wangfc27441
    @desc:  在原图上会是 cell 的 正方形
    @version：
    @time:2020/11/27 10:48

    Parameters
    ----------

    Returns
    -------
    """
    # 在原图上绘制 顶点
    if image is not None:
        im = Image.fromarray(image)
    elif image_path is not None:
        im = Image.open(image_path)
    img_detected = im.copy()
    img_draw = ImageDraw.Draw(img_detected)
    colors = ['red', 'green', 'blue', "purple"]
    num_rows = len(y_coordinates)-1
    num_columns = len(x_coordinates)-1
    cells_data = []
    for i in range(num_rows):
        for j in range(num_columns):
            # rect: 文本框的四个点
            x1= x4= x_coordinates[j]
            y1= y2 = y_coordinates[i]
            x2= x3= x_coordinates[j+1]
            y3 = y4= y_coordinates[i+1]
            rect_coordinates = (x1, y1, x2, y2, x3, y3, x4, y4)
            rect_na = np.array(rect_coordinates).reshape(-1,2)
            width = 5
            shrink_size = width
            shrink_na = np.array([(shrink_size,shrink_size),(-shrink_size,shrink_size),
                                  (-shrink_size,-shrink_size),(shrink_size,-shrink_size)])
            rect_shrink = rect_na + shrink_na

            size = max(min(x2 - x1, y3 - y2) // 2, 20)
            fillcolor = colors[(i+j) % len(colors)]
            # 在原图上添加 数字
            cell_num = 'R{}_C{}'.format(i,j)
            # img_draw.text((x1, y1 - size), cell_num, fill=fillcolor)
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = rect_shrink.tolist()
            for xy in [(x1, y1, x2, y2), (x2, y2, x3, y3), (x3, y3, x4, y4), (x4, y4, x1, y1)]:
                img_draw.line(xy=xy, fill=fillcolor, width=width)

            cell_data = {
                "position": rect_na,
                "text": "",
                "start_row": i,
                "end_row": i,
                "start_column": j,
                "end_column": j,
            }
            cells_data.append(cell_data)

    block = {"is_table": True, "num_columns": num_columns, "num_rows": num_rows, "cells": cells_data,
             "position": [], "text": ""}

    img_detected_na = np.array(img_detected)
    if save:
        image_dir = os.path.dirname(image_path)
        image_name = image_path.split(os.sep)[-1].split('.')[0]
        save_image_dir = os.path.join(image_dir, image_name)
        os.makedirs(save_image_dir, exist_ok=True)
        cells_detected_image_name = 'all_cells_detected.png'
        save_image_path = os.path.join(save_image_dir,cells_detected_image_name)
        img_detected.save(save_image_path)

    return block, img_detected_na


def get_table_structure(image_path,threshold_binary = 150):
    """
    @author:wangfc27441
    @desc:
    @version：
    @time:2020/11/23 10:53

    Parameters
    ----------

    Returns
    -------
    image_gray: image的灰度图
    x_coordinates: 有线表格的所有顶点的 x 坐标
    y_coordinates：有线表格的所有顶点的 y 坐标
    """

    # Check if image is loaded fine
    image = cv.imread(image_path)

    x_coordinates,y_coordinates = get_table_coordinates(image, threshold_binary=threshold_binary)

    block,img_detected_na = draw_cells(image,x_coordinates,y_coordinates,
                                                   save=True,image_path=image_path)

    return image,x_coordinates,y_coordinates,block


def imageArray2buffer(array):
    ret, img_encode = cv.imencode('.png', array)
    str_encode = img_encode.tostring()  # 将 array转化为二进制类型
    bytes_io = BytesIO(str_encode)  # 转化为_io.BytesIO类型
    bytes_io.name = 'test.png'  # 名称赋值
    buffer = BufferedReader(bytes_io)
    return buffer,bytes_io


def save_imageArray(array,path):
    pass


def ocr_on_cells(image_path, x_val, y_val, block=None,
                 ocr_name=None, ocr_model=None,
                 ocr_api=None,
                 save=False):
    """
    @author:wangfc27441
    @desc:
    思路1： 利用表格的顶点，分割表格为一个个cell，再对每个cell依次使用OCR 识别
    思路2：
    1） 用 block 中各个cell的信息，我们将图片分割为各个cell，放大后放入一个新的图片，将这个新的图片一起放入 ocr 中进行识别
    2） 记录每个 cell 在新的 人工构造处理的图片中的位置信息
    3） 利用返回得到的 文本框的位置信息，计算文本框的中心点，定位属于哪个 cell

    @version：
    @time:2020/11/23 10:55

    Parameters
    ----------

    Returns
    -------
    """
    image_dir = os.path.dirname(image_path)
    image_name = image_path.split(os.sep)[-1].split('.')[0]

    image = cv.imread(image_path)

    kernel_small = np.ones((3, 3))
    cells_data = []
    pattern = r"\n|\x0c|\f'"
    compiled_pattern = re.compile(pattern=pattern)
    for i in range(len(y_val) - 1):
        for j in range(len(x_val) - 1):
            print("表格共{}行，{}列，正在处理第 {}行-{}列".format(len(y_val)-1, len(x_val)-1,i,j))
            # 截取对应的区域
            left_top_x = x_val[j]
            left_top_y = y_val[i]
            right_down_x = x_val[j + 1]
            right_down_y = y_val[i + 1]
            area = image[left_top_y + 3:right_down_y - 3, left_top_x + 3:right_down_x - 3]
            # 二值化
            area_ret, area_bin = cv.threshold(area, 190, 255, cv.THRESH_BINARY)
            # 放大三倍
            area_resize = cv.resize(area_bin, (0, 0), fx=10, fy=10)
            # 腐蚀两次，加粗字体
            # area_bin = cv.erode(area_bin, kernel_small, iterations=2)

            # 保存图片
            if save:
                save_image_dir = os.path.join(image_dir, image_name)
                os.makedirs(save_image_dir,exist_ok=True)
                save_image_path = os.path.join(save_image_dir,"cell_row_{}_column_{}.png".format(i,j))
                cv.imwrite(save_image_path, area_resize)

            # 送入OCR识别
            text = None
            if ocr_name == "pytesseract":
                text = pytesseract.image_to_string(Image.fromarray(area_bin), lang="chi_sim+eng", config="--psm 7")
            elif ocr_name == "chineseocr_lite":
                area_height, area_width, num_channels = area_resize.shape
                short_size = (min(area_height, area_width) // 32 + 1) * 32
                res = ocr_model.text_predict(area_resize, short_size=short_size)
                for i, r in enumerate(res):
                    # rect: 文本框的四个点，
                    rect, text, confidence = r
                    x1, y1, x2, y2, x3, y3, x4, y4 = rect.reshape(-1)
            elif ocr_name =="paddle_ocr" and ocr_api is not None:
                buffer,bytes_io = imageArray2buffer(area_resize)
                files = {'file': buffer}
                json_response = requests.post(url=ocr_api, files=files)
                response= json.loads(json_response.text)
                code = response.get('code')
                text = None
                if code == '00':
                    data = response.get('data')
                    if data is not None and data != []:
                        texts = [text_pred[0] for rect, text_pred  in data]
                        text = ' '.join(texts)

            if text is not None:
                text = compiled_pattern.sub(repl="", string=text)
            cell_data = {
                "position": {"left_top_x": left_top_x, "left_top_y": left_top_y, "right_down_x": right_down_x,
                             "right_down_y": right_down_y},
                "text": text,
                "start_row": i,
                "end_row": i,
                "start_column": j,
                "end_column": j,
            }
            cells_data.append(cell_data)

    block = {"is_table": True, "num_columns": len(x_val) - 1, "num_rows": len(y_val) - 1, "cells": cells_data,
             "position": [], "text": ""}

    return block


def paddle_api_post(ocr_api,image):
    buffer, bytes_io = imageArray2buffer(image)
    files = {'file': buffer}
    json_response = requests.post(url=ocr_api, files=files)
    response = json.loads(json_response.text)
    code = response.get('code')
    if code == '00':
        data = response.get('data')
        return data


def create_image_by_block(image,block,scale =5):
    """
    @author:wangfc27441
    @desc:  用 block 中各个cell的信息，我们将图片分割为各个cell，放大后放入一个新的图片，记录每个 cell 在新的 人工构造处理的图片中的位置信息
    @version：
    @time:2020/11/27 16:44

    Parameters
    ----------

    Returns
    -------
    """
    cells_data = block.get('cells')
    num_columns = block.get("num_columns")
    num_rows = block.get("num_rows")

    cell_data = {
        "position": rect_na,
        "text": "",
        "start_row": i,
        "end_row": i,
        "start_column": j,
        "end_column": j,
    }
    # 生成一张空白的图片

    for cell_data in cells_data:
        rect_na = cell_data.get('positision')
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = rect_na.tolist()
        # 截取对应的区域
        left_top_x = x1
        left_top_y = y1
        right_down_x = x3
        right_down_y = y3
        # 截取 cell 的位置
        cell_area = image[left_top_y:right_down_y +1, left_top_x :right_down_x+1]
        
        # 将 cell 放大

        # 放入 空白的图片中




def get_text_for_cells(new_block,data):
    pass



def ocr_on_block(image_path, block=None,
                 ocr_name=None, ocr_model=None,
                 ocr_api=None,
                 save=False):
    """
    @author:wangfc27441
    @desc:
    思路1： 利用表格的顶点，分割表格为一个个cell，再对每个cell依次使用OCR 识别
    思路2：
    1） 用 block 中各个cell的信息，我们将图片分割为各个cell，放大后放入一个新的图片，记录每个 cell 在新的 人工构造处理的图片中的位置信息
    2） 将这个新的图片一起放入 ocr 中进行识别
    3） 利用返回得到的 文本框的位置信息，计算文本框的中心点，定位属于哪个 cell

    @version：
    @time:2020/11/23 10:55

    Parameters
    ----------

    Returns
    -------
    """
    image_dir = os.path.dirname(image_path)
    image_name = image_path.split(os.sep)[-1].split('.')[0]
    image = cv.imread(image_path)

    #  1） 用 block 中各个cell的信息，我们将图片分割为各个cell，放大后放入一个新的图片，将这个新的图片一起放入 ocr 中进行识别
    new_image,new_block = create_image_by_block(image,block,scale =5)

    #   2） 将这个新的图片一起放入 ocr 中进行识别
    data = paddle_api_post(ocr_api, image=new_image)

    # 3） 利用返回得到的 文本框的位置信息，计算文本框的中心点，定位属于哪个 cell
    final_block = get_text_for_cells(new_block,data)


    return block



class Cell(object):
    def __init__(self,left_top_x, left_top_y, right_down_x, right_down_y,text):
        self.left_top_x = left_top_x
        self.left_top_y = left_top_y
        self.right_down_x = right_down_x
        self.right_down_y = right_down_y
        self.text = text

    def isIn(self,cell,gap =5):
        left_top_x_bool = self.left_top_x >= cell.left_top_x -gap
        left_top_y_bool = self.left_top_y <= cell.left_top_y +gap
        right_down_x_bool = self.right_down_x <= cell.right_down_x +gap
        right_down_y_bool = self.right_down_y >= cell.right_down_y -gap
        if left_top_x_bool and left_top_y_bool and right_down_x_bool and right_down_y_bool:
            return True
        else:
            return False



def ocr2cells(image,x_val,y_val,ocr_name,ocr_model,short_size=320,ocr_api=None):
    """
    @author:wangfc27441
    @desc: 对整个表格使用OCR 识别，然后根据顶点的位置信息依次还原各个cell的内容
    @version：
    @time:2020/11/26 8:50

    Parameters
    ----------

    Returns
    -------
    """

    if ocr_model is not None and ocr_model == "chineseocr_lite":
        height, width, num_channels = image.shape
        # short_size = (min(height,width) // 32 + 1) * 32
        # 通过 chineseocr_lite 得到 ocr 结果
        results = ocr_model.text_predict(image, short_size=short_size)
    elif ocr_model is None and ocr_api is not None:
        buffer = imageArray2buffer(image)
        files = {'file': buffer}
        json_response = requests.post(url=ocr_api, files=files)
        response = json.loads(json_response.text)
        code = response.get('code')
        if code == '00':
            data = response.get('data')
            if data is not None and data != []:
                results = data

    ocr_cells =[]
    pattern = r'\d{1,3}、\s{0,5}'
    compiled_pattern = re.compile(pattern)
    for i, r in enumerate(results):
        if ocr_model is not None and ocr_model == "chineseocr_lite":
            # rect: 文本框的四个点，
            rect, text, confidence = r
            x1, y1, x2, y2, x3, y3, x4, y4 = rect.reshape(-1)
        elif ocr_model is None and ocr_api is not None:
            rect, [text, confidence] = r
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = rect
        clean_text = compiled_pattern.sub(repl='',string=text)
        ocr_cell = Cell(left_top_x=x1, left_top_y=y1, right_down_x=x3, right_down_y=y3, text=clean_text)
        ocr_cells.append(ocr_cell)

    # 判断 table_cell 的位置 是否包含 ocr 的识别框
    cells_data = []
    for i in range(len(y_val) - 1):
        for j in range(len(x_val) - 1):
            # 截取对应的区域
            left_top_x = x_val[j]
            left_top_y = y_val[i]
            right_down_x = x_val[j + 1]
            right_down_y = y_val[i + 1]
            # 获取所有的 table_cells
            text = ""
            table_cell = Cell(left_top_x=left_top_x, left_top_y=left_top_y, right_down_x=right_down_x, right_down_y=right_down_y, text=text)

            # 判断 tabel_cell 的位置 是否包含 ocr 的识别框
            print("table_cell: row={},column={}".format(i,j))
            for ocr_cell in ocr_cells:
                # print("ocr_text={},left_top_x={},left_top_y={},right_down_x={},right_down_y={}".format(ocr_cell.text,ocr_cell.left_top_x,ocr_cell.left_top_y,ocr_cell.right_down_x, ocr_cell.right_down_y))
                if ocr_cell.isIn(table_cell):
                    ocr_text = ocr_cell.text
                    text = text + ocr_text
                    print("ocr_cell.text={},text={}".format(ocr_text, text))


            cell_data = {
                "position": {"left_top_x": left_top_x, "left_top_y": left_top_y, "right_down_x": right_down_x,
                             "right_down_y": right_down_y},
                "text": text,
                "start_row": i,
                "end_row": i,
                "start_column": j,
                "end_column": j,
            }

            cells_data.append(cell_data)

    block = {"is_table": True, "num_columns": len(x_val) - 1, "num_rows": len(y_val) - 1, "cells": cells_data,
             "position": [], "text": ""}
    return block



def transform2df(block):
    num_rows = block.get('num_rows')
    num_columns = block.get('num_columns')
    empty_array = np.empty((num_rows, num_columns), dtype=str)
    table_df = pd.DataFrame(empty_array)
    for cell in block.get('cells'):
        start_row = cell.get('start_row')
        end_row = cell.get('end_row')
        start_column = cell.get('start_column')
        end_column = cell.get('end_column')
        text = cell.get('text')
        table_df.loc[start_row:end_row, start_column:end_column] = text
    return table_df


if __name__ == "__main__":
    cwd = os.getcwd()
    table_name = "fullinetable_20200917092928_001.png"
    image_path = os.path.join(cwd, 'tests', 'table', table_name)
    image = cv.imread(image_path)
    image_gray, x_coordinates, y_coordinates = get_table_structure(image)

    OCR_NAMES = ['pytesseract', 'chineseocr_lite']
    OCR_MODEL = None
    block = ocr_on_cells(img_gray=image_gray, x_val=x_coordinates, y_val=y_coordinates,
                         ocr_name=OCR_NAMES[1], ocr_model=OCR_MODEL,
                         save=False)
    table_df = transform2df(block)
    excel_path = os.path.join(os.path.dirname(image_path), 'fullinetable_20200917092928_001.csv')
    table_df.to_csv(excel_path, encoding="utf_8_sig", index=False, header=False)
