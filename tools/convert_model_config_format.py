#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.hundsun.com
@time: 2021/1/25 15:53 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/25 15:53   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：恒生电子股份有限公司 2019
 * 注意：本内容仅限于恒生电子股份有限公司内部传阅，禁止外泄以及用于其他的商业目的

"""
import json
import os
from pathlib import Path
import mmcv

def convert_model_config_format(mmcv_config:mmcv.Config=None,config_file_path:Path=None,convert_to_format:str='txt'):
    """
    @author:wangfc27441
    @desc:  尝试将 model_config.py 文件转换为 json格式，用在服务代码当中
    但是，在加载模型的时候，发生错误 assert mmcv.is_list_of(self.img_scale, tuple)

    @version：
    @time:2021/1/25 16:58

    Parameters
    ----------

    Returns
    -------
    """
    file_dir = os.path.dirname(os.path.abspath(config_file_path))
    file_basename = os.path.basename(config_file_path)
    file_exname = file_basename.split('.')[-1]
    file_name = file_basename.split('.')[0]
    if not mmcv_config and config_file_path:
        # 从 py 格式的文件中读取
        config_path = os.path.join(file_dir,file_basename)
        if file_exname != 'py':
            raise ValueError("我们需要将 py 格式的 model_config转换为json，但是当前的 config file={}".format(config_file_path))
        mmcv_config = mmcv.Config.fromfile(config_path)

    if convert_to_format =='json':
        # 移出  'gpu_ids': range(0, 1)
        mmcv_config.pop('gpu_ids')
        output_file_path = os.path.join(file_dir,f"{file_name}.json")
        cfg_dict = mmcv_config.__getattribute__('_cfg_dict').to_dict()
        with open(output_file_path,encoding='utf-8',mode='w') as f:
            json.dump(cfg_dict,f,indent=4)

    elif convert_to_format == 'txt':
        output_file_path = os.path.join(file_dir, f"{file_name}.txt")
        config_text = mmcv_config.text.split('\n')
        filter_config_text_ls = config_text[1:-1]
        filter_config_text = "\n".join(filter_config_text_ls)
        with open(output_file_path,encoding='utf-8',mode='w') as f:
            f.write(filter_config_text)
    print(f"将 py 格式的 model_config={config_file_path}转换为json={output_file_path}")

    # cfg_dict.dump(file=file_json_path)

if __name__ == '__main__':
    from object_detection_server.config.configs import *
    convert_model_config_format(config_file_path=config_file_absolute_path)


