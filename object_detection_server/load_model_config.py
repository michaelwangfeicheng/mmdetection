#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/1/25 17:01

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/25 17:01   wangfc      1.0         None

"""
import os.path as osp
import shutil
import sys
from importlib import import_module
from pathlib import Path
import tempfile
import platform
if platform.system() == 'Windows':
    import regex as re
else:
    import re


from mmcv import Config, import_modules_from_strings, check_file_exist
from mmcv.utils.config import BASE_KEY

#办法一：
import py_compile
def change_py2pyc(py_file_path):
    #加r前缀进行转义
    py_compile.compile(py_file_path) #py文件完整的路径 ./__pycache__/faster_rcnn_r50_fpn_1x_seal_002.cpython-37.pyc'


def load_model_config(filename:Path,use_predefined_variables=True,
                 import_custom_modules=True)-> Config:
    """
    @author:wangfc27441
    @desc:  使用自定义的加载模型配置文件，读取 转换过来的text格式 模型配置文件： faster_rcnn_r50_fpn_1x_seal_002.txt

    @version：
    @time:2021/1/26 9:52

    Parameters
    ----------

    :param filename:
    :param use_predefined_variables:
    :param import_custom_modules:
    :return:

    """
    cfg_dict, cfg_text = file2dict(filename,use_predefined_variables)
    if import_custom_modules and cfg_dict.get('custom_imports', None):
        import_modules_from_strings(**cfg_dict['custom_imports'])
    return Config(cfg_dict, cfg_text=cfg_text, filename=filename)


def file2dict(filename, use_predefined_variables=True):
    filename = osp.abspath(osp.expanduser(filename))
    check_file_exist(filename)
    fileExtname = osp.splitext(filename)[1]
    # 增加可以读取 '.txt' 格式的配置文件： 使用   tools/convert_model_config_format.py 转换过来
    if fileExtname not in ['.py', '.pyc','.txt', '.json', '.yaml', '.yml']:
        raise IOError('Only py/yml/yaml/json type are supported now!')

    with tempfile.TemporaryDirectory() as temp_config_dir:
        # 临时文件强制设置为 .py
        temp_config_file_exname = '.py'
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix=temp_config_file_exname)
        if platform.system() == 'Windows':
            temp_config_file.close()
        temp_config_name = osp.basename(temp_config_file.name)
        # Substitute predefined variables
        if use_predefined_variables:
            substitute_predefined_vars(filename,
                                               temp_config_file.name,fileExtname=fileExtname)
        else:
            shutil.copyfile(filename, temp_config_file.name)

        if filename.endswith('.py') or filename.endswith('.txt') :
            temp_module_name = osp.splitext(temp_config_name)[0]
            sys.path.insert(0, temp_config_dir)
            # 使用 ast.parse 验证 filename 的语法
            Config._validate_py_syntax(filename)
            # 动态导入对象
            mod = import_module(temp_module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
            # delete imported module
            del sys.modules[temp_module_name]
        elif filename.endswith(('.yml', '.yaml', '.json')):
            import mmcv
            cfg_dict = mmcv.load(temp_config_file.name)
        # close temp file
        temp_config_file.close()

    cfg_text = filename + '\n'
    with open(filename, 'r') as f:
        cfg_text += f.read()

    if BASE_KEY in cfg_dict:
        cfg_dir = osp.dirname(filename)
        base_filename = cfg_dict.pop(BASE_KEY)
        base_filename = base_filename if isinstance(
            base_filename, list) else [base_filename]

        cfg_dict_list = list()
        cfg_text_list = list()
        for f in base_filename:
            _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
            cfg_dict_list.append(_cfg_dict)
            cfg_text_list.append(_cfg_text)

        base_cfg_dict = dict()
        for c in cfg_dict_list:
            if len(base_cfg_dict.keys() & c.keys()) > 0:
                raise KeyError('Duplicate key is not allowed among bases')
            base_cfg_dict.update(c)

        base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = base_cfg_dict

        # merge cfg_text
        cfg_text_list.append(cfg_text)
        cfg_text = '\n'.join(cfg_text_list)

    return cfg_dict, cfg_text


def substitute_predefined_vars(filename, temp_config_name,fileExtname='pyc'):
    file_dirname = osp.dirname(filename)
    file_basename = osp.basename(filename)
    file_basename_no_extension = osp.splitext(file_basename)[0]
    file_extname = osp.splitext(filename)[1]
    support_templates = dict(
        fileDirname=file_dirname,
        fileBasename=file_basename,
        fileBasenameNoExtension=file_basename_no_extension,
        fileExtname=file_extname)
    if fileExtname in ['.py','.txt'] :
        mode = 'r'
    elif fileExtname == '.pyc':
        mode = 'rb'

    with open(filename,mode) as f:
        config_file = f.read()

    for key, value in support_templates.items():
        regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
        value = value.replace('\\', '/')
        config_file = re.sub(regexp, value, config_file)
    with open(temp_config_name, 'w') as tmp_config_file:
        tmp_config_file.write(config_file)

if __name__ == '__main__':
    from object_detection_server.config.configs import config_file_absolute_path
    # change_py2pyc(py_file_path=config_file_absolute_path)
    mmcv_config = load_model_config(filename=config_file_absolute_path)