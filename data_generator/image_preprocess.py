#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2021/2/1 9:58 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/1 9:58   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""
import os
import platform
import re
import random
import math
import json
from collections import OrderedDict
import numpy as np

import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageEnhance, ImageDraw, ImageChops, ImageFont
import cv2 as cv


def get_processed_image_path(img_path, subdir='transparent'):
    dir = os.path.dirname(img_path)
    img_filename = os.path.basename(img_path)
    #     img_name, img_exname = img_filename.split('.')
    matched = re.match(pattern=r"([\s\S]*)\.(\w{3,4})$", string=img_filename)
    img_name = matched.groups()[0]
    img_exname = matched.groups()[1]
    transparent_image_path = os.path.join(dir, subdir, f'{img_name}.png')
    os.makedirs(os.path.dirname(transparent_image_path), exist_ok=True)
    return transparent_image_path


def imread_cv(img_path, flag=cv.IMREAD_UNCHANGED):
    """
    @author:wangfc27441
    @desc: 
    @version：
    @time:2021/2/22 14:32

    cv.IMREAD_COLOR： 加载彩色图像。任何图像的透明度都会被忽视。它是默认标志。
    cv.IMREAD_GRAYSCALE：以灰度模式加载图像
    cv.IMREAD_UNCHANGED：加载图像，包括alpha通道
    也可以简单表述为：1、0、-1
    注意：opencv读取图片尽量不要带有中文路径，否则程序没报错，但返回结果是空的
    Parameters
    ----------
    
    Returns
    -------
    """
    # opencv中opencv不接受non-ascii的路径
    np_array = np.fromfile(file=img_path, dtype=np.uint8)
    img = cv.imdecode(np_array, flags=flag)
    return img


def print_image_info(img):
    """
    img.shape 返回(行数、列数、通道数)
    img.size 返回像素总值 = 行数 x 列数 x 通道数
    img.dtype 返回类型
    """
    print("图像维度：", img.shape)
    print("图像像素总数：", img.size)
    print("图像数据类型：", img.dtype)


def imshow_cv(img_path, flag=cv.IMREAD_UNCHANGED):
    """
    @author:wangfc27441
    @desc:
    一般都是rgb，opencv默认是bgr
    @version：
    @time:2021/2/22 14:43

    Parameters
    ----------

    Returns
    -------
    """
    img = imread_cv(img_path, flag=flag)
    if flag == cv.IMREAD_COLOR:
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif flag == cv.IMREAD_UNCHANGED:
        img_rgb = cv.cvtColor(img, cv.COLOR_BGRA2RGB)
    plt.imshow(img_rgb)
    plt.show()
    return img


def im_processed_write(img_processed, img_path, processed='eroded'):
    img_processed_path = get_processed_image_path(img_path, subdir=processed)
    # cv.imwrite(img_processed_path, img_processed)
    cv.imencode('.png', img_processed)[1].tofile(img_processed_path)


def draw_bbox(img, bboxes, labels=None, probabilities=None, category_ids=None, class_names=None, obj_mask=None,
              fill='green', width=5, font_size=20):
    if platform.system()=='Windows':
        font = ImageFont.truetype('timesbi', font_size)
    else:
        font = ImageFont.truetype('DejaVuSerif', font_size)
    img_copyed = img.copy()

    if labels is not None and isinstance(labels,str):
        text_labels = [labels]* bboxes.__len__()
    elif labels is not None and not isinstance(labels.str):
        assert labels.__len__() == bboxes.__len__()
        # text_labels = [class_names[label] for label in labels]
        text_labels = labels
    elif category_ids is not None and class_names is not None:
        assert category_ids.__len__() == bboxes.__len__()
        text_labels = [class_names[category_id - 1] for category_id in category_ids]
    else:
        text_labels = [''] * bboxes.__len__()

    img_draw = ImageDraw.Draw(img_copyed)
    index = 0
    for bbox, text_label in zip(bboxes, text_labels):
        x, y, w, h = bbox
        x1, y1 = x, y
        x2, y2 = x + w, y
        x3, y3 = x + w, y + h
        x4, y4 = x, y + h
        for xy in [(x1, y1, x2, y2), (x2, y2, x3, y3), (x3, y3, x4, y4), (x4, y4, x1, y1)]:
            img_draw.line(xy=xy, fill=fill, width=width)
        if probabilities:
            text = f'{text_label}|{probabilities[index]}'
        else:
            text = f'{text_label}'
        print(f'bbox={bbox},text_label={text}')
        img_draw.text(xy=(x, y - font_size - 5), text=text, file=(0, 255, 0), font=font)
        index += 1
    return img_copyed


def transparent_image(img_path: str, save=True):
    """
    @author:wangfc27441
    @desc:
    RGB没有透明选项，将RGBA 设置为0%即可变成无色透明。RGBA中alpha通道一般用作不透明度参数。
    如果一个像素的alpha通道数值为0%，那它就是完全透明的（也就是看不见的），
    而数值为100%则意味着一个完全不透明的像素。

    RGBA在RGB的基础上多了控制alpha透明度的参数。以上R、G、B三个参数，正整数值的取值范围为：0 - 255。
    百分数值的取值范围为：0.0% - 100.0%。超出范围的数值将被截至其最接近的取值极限。并非所有浏览器都支持使用百分数值。A参数，取值在0~1之间，不可为负值。
    @version：
    @time:2021/2/7 14:29
    
    Parameters
    ----------
    
    Returns
    -------
    """
    img = Image.open(img_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((2, 2))
    img
    for h in range(H):
        for l in range(L):
            dot = (l, h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0 or l == 0 or h == 0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot, (0, 0, 0, 0))
    if save:
        transparent_image_path = get_processed_image_path(img_path)
        img.save(transparent_image_path)
    return img


def transparent_image_by_threshold(img_path=None,img=None, save=True, threshold=100, alpha=0):
    """
    @author:wangfc27441
    @desc:
    使用函数cv2.imread(filepath,flags)读入一副图片
        filepath：要读入图片的完整路径
        flags：读入图片的标志
        cv.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
        cv.IMREAD_GRAYSCALE：读入灰度图片
        cv.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道
    @version：
    @time:2021/2/7 16:41

    Parameters
    ----------

    Returns
    -------
    """
    if img_path is not None:
        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        img_bgr = img[:, :, :3]
        # b_channel, g_channel, r_channel,alpha_channel = cv.split(img)
    else:
        img_bgr = img
    # "Almost white" color range
    # lower = np.array([192, 192, 192])
    upper = np.array([255, 255, 255])
    lower = upper - threshold
    # Create mask
    mask = cv.inRange(img_bgr, lower, upper)  # Extraction mask for "almost white" only
    inverted_mask = cv.bitwise_not(mask, mask)  # Inversion = Extraction mask other than "almost white"
    # Extract all except "almost white" → make "almost white" black
    img_masked = cv.bitwise_and(img_bgr, img_bgr, mask=inverted_mask)

    # b_channel, g_channel, r_channel = cv.split(img_masked)
    channels = cv.split(img_masked)
    *_, alpha_channel = channels
    # alpha_channel = np.ones(r_channel.shape, dtype=r_channel.dtype) * 255
    # alpha_channel[inverted_mask > 0] = alpha
    img_BGRA = cv.merge((img_masked, alpha_channel))

    if save:
        transparent_image_path = get_processed_image_path(img_path)
        cv.imwrite(transparent_image_path, img_BGRA)
    return img_BGRA


def transparent_image_by_hsv(img_path=None,img=None,save=False, hMin=0, hMax=179, sMin=30, sMax=255, vMin=0, vMax=255):
    """
    @author:wangfc27441
    @desc:
    openCV中经常将RGB图像转换成HSV图像然后进行颜色的辨别和处理
    我们能够查到一般HSV的范围是
        H: [0,360]
        S: [0,100]
        V: [0,100]
    在openCV中，HSV的范围却是
        H: [0,180]
        S: [0,255]
        V: [0,255]
        # THESE VALUES CAN BE PLAYED AROUND FOR DIFFERENT COLORS
        # Hue minimum vs maximum
        # Saturation minimum vs maximum
        # Value minimum vs maximum (Also referred to as brightness)
    @version：
    @time:2021/2/7 17:42

    Parameters
    ----------

    Returns
    -------
    """
    if img is None and img_path is not None:
        img = cv.imread(img_path)
    # Load in the image using the typical imread function using our watch_folder path, and the fileName passed in, then set the final output image to our current image for now
    # img = cv.imread(img_path)
    # Set thresholds. Here, we are using the Hue, Saturation, Value color space model.
    # We will be using these values to decide what values to show in the ranges using a minimum and maximum value.

    # Set the minimum and max HSV values to display in the output image using numpys' array function.
    # We need the numpy array since OpenCVs' inRange function will use those.
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    # Create HSV Image and threshold it into the proper range.
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # Converting color space from BGR to HSV
    mask = cv.inRange(hsv, lower, upper)  # Create a mask based on the lower and upper range, using the new HSV image
    # Create the output image, using the mask created above.
    # This will perform the removal of all unneeded colors, but will keep a black background.
    output = cv.bitwise_and(img, img, mask=mask)
    # Add an alpha channel, and update the output image variable
    channels = cv.split(output)
    *_, alpha = channels
    dst = cv.merge((output, alpha))
    # Generate a random file name using a mini helper function called randomString to write the image data to, and then save it in the processed_folder path, using the generated filename.
    if save and img_path:
        transparent_image_path = get_processed_image_path(img_path)
        cv.imwrite(transparent_image_path, dst)
    return dst


def mutate_image(img, sizes):
    # resize image for random value
    resize_rate = random.choice(sizes)
    img = img.resize([int(img.width * resize_rate), int(img.height * resize_rate)], Image.BILINEAR)

    # rotate image for random andle and generate exclusion mask
    rotate_angle = random.randint(0, 360)
    mask = Image.new('L', img.size, 255)
    img = img.rotate(rotate_angle, expand=True)
    mask = mask.rotate(rotate_angle, expand=True)
    # perform some enhancements on image
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Color, ImageEnhance.Contrast, ImageEnhance.Sharpness]
    enhancers_count = random.randint(0, 3)
    for i in range(0, enhancers_count):
        enhancer = random.choice(enhancers)
        enhancers.remove(enhancer)
        img = enhancer(img).enhance(random.uniform(0.5, 1.5))
    return img, mask


def enhance_image(img):
    # 图片增强
    # # perform some enhancements on image
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Color, ImageEnhance.Contrast, ImageEnhance.Sharpness]
    enhancers_count = random.randint(0, 3)
    for i in range(0, enhancers_count):
        enhancer = random.choice(enhancers)
        enhancers.remove(enhancer)
        img = enhancer(img).enhance(random.uniform(0.5, 1.5))
    return img


def autoCrop(image, backgroundColor=None):
    '''
      对 img 进行 边缘的处理，保证 bbox的边正好框住 图像
      Intelligent automatic image cropping.
      This functions removes the usless "white" space around an image.

      If the image has an alpha (tranparency) channel, it will be used
      to choose what to crop.

      Otherwise, this function will try to find the most popular color
      on the edges of the image and consider this color "whitespace".
      (You can override this color with the backgroundColor parameter)

      Input:
        image (a PIL Image object): The image to crop.
        backgroundColor (3 integers tuple): eg. (0,0,255)
           The color to consider "background to crop".
           If the image is transparent, this parameters will be ignored.
           If the image is not transparent and this parameter is not
           provided, it will be automatically calculated.

      Output:
        a PIL Image object : The cropped image.
    '''

    def mostPopularEdgeColor(image):
        ''' Compute who's the most popular color on the edges of an image.
          (left,right,top,bottom)

          Input:
            image: a PIL Image object

          Ouput:
            The most popular color (A tuple of integers (R,G,B))
        '''
        im = image
        if im.mode != 'RGB':
            im = image.convert("RGB")

        # Get pixels from the edges of the image:
        width, height = im.size
        left = im.crop((0, 1, 1, height - 1))
        right = im.crop((width - 1, 1, width, height - 1))
        top = im.crop((0, 0, width, 1))
        bottom = im.crop((0, height - 1, width, height))
        pixels = left.tostring() + right.tostring() + top.tostring() + bottom.tostring()

        # Compute who's the most popular RGB triplet
        counts = {}
        for i in range(0, len(pixels), 3):
            RGB = pixels[i] + pixels[i + 1] + pixels[i + 2]
            if RGB in counts:
                counts[RGB] += 1
            else:
                counts[RGB] = 1

                # Get the colour which is the most popular:
        mostPopularColor = sorted([(count, rgba) for (rgba, count) in counts.items()], reverse=True)[0][1]
        return ord(mostPopularColor[0]), ord(mostPopularColor[1]), ord(mostPopularColor[2])

    bbox = None

    # If the image has an alpha (tranparency) layer, we use it to crop the image.
    # Otherwise, we look at the pixels around the image (top, left, bottom and right)
    # and use the most used color as the color to crop.

    # --- For transparent images -----------------------------------------------
    if 'A' in image.getbands():  # If the image has a transparency layer, use it.  ('R', 'G', 'B', 'A')
        # This works for all modes which have transparency layer
        bbox = image.split()[list(image.getbands()).index('A')].getbbox()
    # --- For non-transparent images -------------------------------------------
    elif image.mode == 'RGB':
        if not backgroundColor:
            backgroundColor = mostPopularEdgeColor(image)
        # Crop a non-transparent image.
        # .getbbox() always crops the black color.
        # So we need to substract the "background" color from our image.
        bg = Image.new("RGB", image.size, backgroundColor)
        diff = ImageChops.difference(image, bg)  # Substract background color from image
        bbox = diff.getbbox()  # Try to find the real bounding box of the image.
    else:
        raise NotImplementedError("Sorry, this function is not implemented yet for images in mode '%s'." % image.mode)

    if bbox:
        image = image.crop(bbox)

    return image

def object_resize(obj_img, obj_image_dir_name):
    """
    身份证 = 85.6mm* 宽54mm
    :param obj_img:
    :param obj_image_dir_name:
    :return:
    """
    IDCARD_WIDTH = 8.56
    IDCARD_HEIGHT = 5.4
    inchesPerCentimeter = 0.393700787
    IDCARD_WIDTH_inch = IDCARD_WIDTH * inchesPerCentimeter
    IDCARD_HEIGHT_inch = IDCARD_HEIGHT * inchesPerCentimeter
    # print(obj_img.info)
    dpi = obj_img.info.get('dpi')
    if dpi is None:
        dpi = (96,96)

    # 对身份证做 resize 处理
    if obj_image_dir_name.split("_")[-1] == "idcard":
        size = (IDCARD_WIDTH_inch * dpi[0], IDCARD_HEIGHT_inch * dpi[1])
        size = (int(size[0]), int(size[1]))
        obj_img = obj_img.resize(size=size)
    elif obj_image_dir_name.split("_")[-1] == "qrcode":
        obj_img = obj_img.resize(size=(60, 60))
    elif obj_image_dir_name == "guohui":
        size = (IDCARD_WIDTH_inch *0.2* dpi[0], IDCARD_HEIGHT_inch * 0.35* dpi[1])
        size = (int(size[0]), int(size[1]))
        obj_img = obj_img.resize(size=size)
    elif obj_image_dir_name == 'rectangle_comany_name':
        obj_img = obj_img.resize(obj_img.size*0.6)
    return obj_img


def rotate_image(img=None, img_path=None, max_rotate_angle=90):
    """
    @author:wangfc27441
    @desc:
    使用PIL先对图片进行旋转，但是发现旋转后像素是用黑色填充的。
    于是接下来介绍一种方法在使用PIL旋转图片时，指定图像的填充的颜色。
    当原始的图像没有alpha图层时，就可以使用alpha图层作为掩码将背景转换为白色。当旋转创建”背景”时，它使其完全透明。
    @version：
    @time:2021/2/8 13:42

    Parameters
    ----------

    Returns
    -------
    """
    if img_path is not None:
        img = Image.open(img_path)
    # rotate image for random andle and generate exclusion mask
    rotate_angle = random.randint(-max_rotate_angle, max_rotate_angle)
    # if img.mode != 'RGBA':
    #     raise NotImplementedError(f"非RGBA的图像的旋转变换还未实现")
        # img = img.convert('RGBA')
        # # 图片旋转 某个角度,图片在expand的时候会变大
        # img_rot = img.rotate(rotate_angle, expand=True)
        # # 生成一个图片同样大小的 mask
        # alpha = Image.new('RGBA', img_rot.size, (255,) * 4)
        # # 使用alpha层的 rot作为掩码创建一个复合图像
        # img_out = Image.composite(img_rot, alpha, img_rot)
    img = img.convert('RGBA')
    r_channel, g_channel, b_channel, alpha_channel = img.split()
    # # # 图片旋转 某个角度,图片在expand的时候会变大
    img_rot = img.rotate(rotate_angle, expand=True)
    # # # 生成一个图片同样大小的 mask
    alpha_rot = alpha_channel.rotate(rotate_angle, expand=True)
    # # # 使用alpha层的 rot作为掩码创建一个复合图像
    img_out = Image.composite(img_rot, img_rot, alpha_rot)
    return img_out


def test_rotate_image():
    """"
    rotate_image 还存在问题
    """
    objects_dir = f'data_generator/data/objects/'
    backgrounds_dir = "data_generator/data/backgrounds"
    subobject_dir_names = os.listdir(objects_dir)
    subobject_dir = os.path.join(objects_dir, subobject_dir_names[0])
    object_names = os.listdir(subobject_dir)
    object_path = os.path.join(subobject_dir, object_names[0])
    obj_img = Image.open(object_path)

    background_names = os.listdir(backgrounds_dir)
    background_path = os.path.join(backgrounds_dir, background_names[0])
    bkg_img = Image.open(background_path)

    sizes = [2]
    # mutate_image(img=test_img)
    count_per_size = 1
    # Get an array of random obj positions (from top-left corner)
    obj_h, obj_w, x_pos, y_pos = get_obj_positions(obj=obj_img, bkg=bkg_img, sizes=sizes, count=count_per_size)
    print(f"obj_h={obj_h}, obj_w={obj_w}, x_pos={x_pos}, y_pos={y_pos}")
    x = x_pos[0]
    y = y_pos[0]
    w = obj_w[0]
    h = obj_h[0]

    obj_bbox = [int(x), int(y), int(w), int(h)]
    # resize image for random value
    img = obj_img

    seal_max_rotate_angle = 45

    bkg_w_obj, obj_bbox, obj_mask = combine_background_and_object(bkg_img=bkg_img, obj_img=obj_img, obj_bbox=obj_bbox,
                                                                  max_rotate_angle=30, with_mask=True,
                                                                  )


def rotate_image_with_bbox(img, bbox, max_rotate_angle=30, with_mask=False):
    # rotate image for random andle and generate exclusion mask
    rotate_angle = random.randint(-max_rotate_angle, max_rotate_angle)
    rotate_angle_pi = math.radians(rotate_angle)  # 转化为弧度角度， 逆时针这里到底要不要乘以-1？？？

    h = img.size[1]
    w = img.size[0]
    x, y, _, _ = bbox

    # 计算中心点 ： 旋转后的 中心点坐标是否变化
    center_x = int(np.floor(w / 2)) + x
    center_y = int(np.floor(h / 2)) + y

    #     print('center_x,center_y', center_x, center_y)

    def rotate_transform(x, y, center_x, center_y, angle_pi):  # angle 必须是弧度
        return (x - center_x) * round(math.cos(angle_pi), 15) + \
               (y - center_y) * round(math.sin(angle_pi), 15) + center_x, \
               -(x - center_x) * round(math.sin(angle_pi), 15) + \
               (y - center_y) * round(math.cos(angle_pi), 15) + center_y

    # 原来的四个顶点对于原来的中心进行旋转变换
    xx = []
    yy = []
    for x, y in [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]:
        x_, y_ = rotate_transform(x, y, center_x, center_y, rotate_angle_pi)
        xx.append(x_)
        yy.append(y_)

    rot_x = int(min(xx))
    rot_y = int(min(yy))
    # rot_width = int(max(xx)) - rot_x
    # rot_height = int(max(yy)) - rot_y
    # 图片旋转 某个角度,图片在expand的时候会变大
    img = img.rotate(rotate_angle, expand=True)
    rot_width, rot_height = img.size
    bbox_rot = [rot_x, rot_y, rot_width, rot_height]

    mask = None
    if with_mask:
        # 生成一个图片同样大小的 mask
        mask = Image.new('L', img.size, 255)
        mask = mask.rotate(rotate_angle, expand=True)
        # img.paste(new_obj, (x, y), mask=mask)
    return img, bbox_rot, mask


def combine_background_and_object(bkg_img, obj_img, obj_bbox):
    """
    @author:wangfc27441
    @desc:
    不在这里做旋转变换
    @version：
    @time:2021/2/8 14:39

    Parameters
    ----------

    Returns
    -------
    """
    x, y, w, h = obj_bbox
    # Copy background
    bkg_w_obj = bkg_img.copy()
    # Adjust obj size
    new_obj = obj_img.resize(size=(w, h))
    # 分离 alpha 通道
    r, g, b, a = new_obj.split()
    # Paste on the obj: (x, y) 表示左顶点 ,一般这是numpy的int类型无法被json化,所以需要将numpy的int转为原生类型.
    bkg_w_obj.paste(new_obj, (x, y), mask=a)
    return bkg_w_obj, obj_bbox


# check if two boxes intersect
def intersects(box, new_box):
    box_x1, box_y1, box_x2, box_y2 = box
    x1, y1, x2, y2 = new_box
    return not (box_x2 < x1 or box_x1 > x2 or box_y1 > y2 or box_y2 < y1)


def intertset_with_box(new_box, boxes):
    """
    @author:wangfc27441
    @desc: 比较  new_box 和所有的box之间是否存在
    @version：
    @time:2021/1/30 22:19

    Parameters
    ----------

    Returns
    -------
    """
    if new_box is None:
        return True
    elif boxes == []:
        return False
    else:
        if_interset = False
        for box in boxes:
            if intersects(box, new_box):
                if_interset = True
                break
        return if_interset


# Helper functions
def get_obj_positions(obj, bkg, sizes=[1], count=4, seed=1):
    """
    @author:wangfc27441
    @desc:
    1. 根据  sizes 进行缩放
    2. 随机生成 count 个位置
    最后一对 (obj,bkg) 生成的 图片数量 = sizes * count = 5 * 5
    @version：
    @time:2021/1/15 11:04

    Parameters
    ----------
    sizes： object放缩的比例
    count: 放置的不同位置格式

    Returns
    -------
    """
    # np.random.seed(seed=seed)
    obj_w, obj_h = [], []
    x_positions, y_positions = [], []
    bkg_w, bkg_h = bkg.size
    # Rescale our obj to have a couple different sizes
    obj_sizes = [tuple([int(s * x) for x in obj.size]) for s in sizes]
    for w, h in obj_sizes:
        obj_w.extend([w] * count)
        obj_h.extend([h] * count)
        max_x, max_y = bkg_w - w, bkg_h - h
        x_positions.extend(list(np.random.randint(0, max_x, count)))
        y_positions.extend(list(np.random.randint(0, max_y, count)))

    return obj_h, obj_w, x_positions, y_positions


def synthesize_background_and_object(bkg_w_obj, bkg_img, obj_img, obj_image_dir_name,
                                     boxes, if_erode=True,if_rotate=True,max_rotate_angle=90, sizes=[1], count_per_size=1,
                                     if_enhance=True):
    """
    @author:wangfc27441
    @desc:  输入 bkg_img, 和 obj_img
    @version：
    @time:2021/2/8 14:58

    Parameters
    ----------

    Returns
    -------
    """
    # 0. 根据object类型进行调整大小:
    obj_img = object_resize(obj_img,obj_image_dir_name)
    if if_erode:
        obj_img = erode_image(image=obj_img)

    # 1. Rescale our obj to have a couple different sizes
    obj_sizes = [tuple([int(s * x) for x in obj_img.size]) for s in sizes]
    obj_img = obj_img.resize(size=obj_sizes[0])

    # 2. 进行旋转
    if if_rotate:
        obj_img = rotate_image(img=obj_img, max_rotate_angle=max_rotate_angle)
    # 3. 去除图形边缘
    obj_img = autoCrop(obj_img, obj_image_dir_name)
    # 4. 图片增强
    if if_enhance:
        obj_img = enhance_image(img=obj_img)

    # 初始化 new_box 为 None
    new_box = None
    seed =0
    unmatched_bkg_object_pair=False
    # 判断是否和之前的box存在重合
    while intertset_with_box(new_box, boxes):
        # 5. 获取 bbox 和图片增强
        if obj_img.size[0] > bkg_img.size[0] or obj_img.size[1] > bkg_img.size[1]:
            unmatched_bkg_object_pair =True
            break
        elif seed>10:
            break
        else:
            obj_h, obj_w, x_pos, y_pos = get_obj_positions(obj=obj_img, bkg=bkg_img, count=count_per_size,
                                                           seed=seed)
            x = x_pos[0]
            y = y_pos[0]
            w = obj_w[0]
            h = obj_h[0]
            new_box = [x, y, x + w, y + h]
            seed+=1


    if new_box is None:
        # 如果超过10次以上，则停止尝试
        return bkg_w_obj,None,None
    else:
        # 将单个object 合并
        # 6. 获取 obj_bbox
        obj_bbox = list([int(x), int(y), int(w), int(h)])
        # 7. 合并图片 ： bkg_w_obj 作为 bkg_w 参数输入
        bkg_w_obj, obj_bbox = combine_background_and_object(bkg_w_obj, obj_img, obj_bbox)
        return bkg_w_obj, obj_bbox, new_box



def get_detection_classes(data_root):
    # 如果只有一个类别，需要加上一个逗号，否则将会报错，例如只有一个类别，如下：
    coco_format_categories_json_path = os.path.join(data_root, 'annotations', 'coco_format_categories.json')
    with open(coco_format_categories_json_path, mode='r', encoding='utf-8') as f:
        coco_format_categories_json = json.load(f)
    detection_classes = [c['name'] for c in coco_format_categories_json]
    return detection_classes


def cv_blur(img, mode='guassian', kernal_height=None, sigmaX=None, kernel_ratio=0.1, sigmaX_default=0):
    h, w = img.shape[:2]
    if mode == 'guassian':
        # mask 进行高斯模糊
        # 内核的宽度和高度，该宽度和高度应为正数和奇数。我们还应指定X和Y方向的标准偏差，分别为sigmaX和sigmaY。
        # 如果仅指定sigmaX，则将sigmaY与sigmaX相同。如果两个都为零，则根据内核大小进行计算。
        if kernal_height is None:
            kernal_height = int(h * kernel_ratio)
            if kernal_height % 2 == 0:
                kernal_height += 1

        kennel_height, kernel_width = kernal_height, kernal_height

        if sigmaX is None:
            sigmaX = sigmaX_default

        kernel = (kennel_height, kernel_width)

        blured = cv.GaussianBlur(img, kernel, sigmaX)
    elif mode == 'bilateral':
        # 双边滤波：cv.bilateralFilter() 在去除噪声的同时保持边缘清晰锐利非常有效。但是，与其他过滤器相比，该操作速度较慢。
        # 我们已经看到，高斯滤波器采用像素周围的邻域并找到其高斯加权平均值。高斯滤波器仅是空间的函数，也就是说，滤波时会考虑附近的像素。
        # 它不考虑像素是否具有几乎相同的强度。它不考虑像素是否是边缘像素。因此它也模糊了边缘，这是我们不想做的。
        blured = cv.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)  # 双边滤波

    # 扩张维度
    # blured_exp = np.expand_dims(blured,axis=2)
    # 在维度2上进行复制
    #  blured_rep = blured_exp.repeat(repeats=4,axis=2)
    return blured


def cv_erode(img, threshold=10, maxval=255, kernel=None, kernel_size=2, iterations=1):
    """
    它计算给定内核区域的局部最小值。

    当内核在图像上扫描时，我们计算由重叠的最小像素值，并用该最小值替换锚点下的图像像素。
    参数1：源图；参数2：核大小；参数3：腐蚀次数
    """
    #     assert img.shape[-1]==4
    if img.shape[-1] == 4:
        img_bgr = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    elif img.shape[-1] == 3:
        img_bgr = img.copy()

    # 生成灰度图
    gray_img = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    # # 进行二值化处理
    # //大于阈值的像素点将变为maxVal（白色部分），小于阈值的像素将变为0（黑色部分）。

    ret, th = cv.threshold(gray_img, threshold, maxval, cv.THRESH_BINARY)  # +cv.THRESH_OTSU

    # # 定义巻积核
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # # 对二值化的矩阵进行侵蚀
    erode = cv.erode(th, kernel, iterations=iterations)
    # # 转换为0/1
    erode_normal = erode / 255
    # # 扩张维度
    eroded_exp = np.expand_dims(erode_normal, axis=2)
    # # 在维度2上进行复制
    erode_rep = eroded_exp.repeat(repeats=3, axis=2)
    # # 将原始图像与侵蚀的图像相乘
    result = img_bgr * erode_rep
    return result


def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim == 3 and mask.shape[-1] == 3:
        alpha = mask / 255.0
    else:
        alpha = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) / 255.0
    blended = cv.convertScaleAbs(img1 * alpha + img2 * (1 - alpha))
    return blended


def image2cv(image):
    assert isinstance(image, PIL.PngImagePlugin.PngImageFile)==True
    img = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGRA)
    return img

def cvimg2image(img):
    if img.shape[-1]==4:
        image = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGRA2RGBA))
    return image


def erode_image(image, mask_height_ratios =  list(np.arange(7,10)/10),show=False):
    img = image2cv(image)

    if img.shape[-1] == 4:
        img_bgr = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # 对上半部分进行处理: mask_height_ratios 表示不同的高度进行（高斯）模糊（渐变）处理
    random_index = random.randint(0,mask_height_ratios.__len__()-1)
    mask_height_ratio = mask_height_ratios[random_index]
    mask[0:int(mask_height_ratio * h), :] = 255

    # 对 mask进行 高斯模糊处理
    mask_blured = cv_blur(mask, kernel_ratio=0.5)

    # 对原始图像做 腐蚀（侵蚀）处理
    img_eroded = cv_erode(img=img, kernel_size=5)

    # 根据 mask_blured 将原始图像和 侵蚀的图片进行合并
    # blended1 = alphaBlend(img, img_eroded, mask_blured)
    blended = alphaBlend(img_bgr, img_eroded, mask_blured)

    img_eroded_alpha = transparent_image_by_hsv(img=blended)
    image_eroded_alpha = cvimg2image(img_eroded_alpha)
    if show:
        plt.figure(figsize=(30, 20))
        plt.subplot(321)
        plt.imshow(img)
        plt.title('img')

        plt.subplot(322)
        plt.imshow(img_eroded)
        plt.title('img_eroded')

        plt.subplot(323)
        plt.imshow(mask)
        plt.title('mask')

        plt.subplot(324)
        plt.imshow(mask_blured)
        plt.title('mask_blured')

        plt.subplot(325)
        plt.imshow(blended)
        plt.title('blended')

        plt.subplot(326)
        plt.imshow(img_eroded_alpha)
        plt.title('img_eroded_alpha')
        plt.show()
    return image_eroded_alpha


if __name__ == '__main__':
    test_path = os.path.join('data_generator', 'data', 'objects', 'round_seal',
                             'TCL科技集团股份有限公司_www.395.net.cn_alpha.png')
    img_path = os.path.join('object_detection_server', 'test_data', 'seal_data_real', 'untest_images', '3_5-bak.jpg')
    transparent_image_by_hsv(img_path=img_path)
