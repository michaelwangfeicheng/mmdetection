#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/2/7 11:36

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/7 11:36   wangfc      1.0         None

"""
import asyncio
import torch
from mmdet.apis import async_inference_detector
from mmdet.utils.contextmanagers import concurrent



async def main():
    from object_detection_server.config.configs import device
    from object_detection_server.load_model import model

    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    # test a single image and show the results
    img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once

    async with concurrent(streamqueue):
        result = await async_inference_detector(model, img)

    # visualize the results in a new window
    model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file='result.jpg')

if __name__ == '__main__':
    asyncio.run(main())
