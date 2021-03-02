#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2020/11/16 13:54

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/16 13:54   wangfc      1.0         None


"""
import os
import sys
# sys.path.append("..")
# os.chdir("..")

import logging
logger = logging.getLogger(__name__)

from mmdet.utils import get_root_logger
LOG_LEVEL = 'INFO'
LOG_FILE = 'mmdetection_api'
# log_dir = os.path.join(os.path.dirname(__file__), 'log')
log_dir = os.path.join(os.getcwd(), 'log')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, LOG_FILE)
logger = get_root_logger(log_file=log_file_path, log_level=LOG_LEVEL)

import socket
import tornado.ioloop  # 开启循环，让服务一直等待请求的到来
import tornado.web  # web服务基本功能都封装在此模块中
import tornado.options  # 从命令行中读取设置
from tornado.options import define, options  # 导入包
from object_detection_server.handlers import mmdetection_handlers
from object_detection_server.config.configs import *




class Application(tornado.web.Application):  # 引入Application类，重写方法，这样做的好处在于可以自定义，添加另一些功能
    """
    @author:wangfc27441
    @desc:
    Tornado Web框架的核心应用类，是与服务器对接的接口，里面保存了路由信息表，其初始化接收的第一个参数就是一个路由信息映射元组的列表；
    其listen(端口)方法用来创建一个http服务器实例，并绑定到给定端口（注意：此时服务器并未开启监听）。
    @version：
    @time:2021/1/22 15:55

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, handlers, settings):
        # 用super方法将父类的init方法重新执行一遍，然后将handlers和settings传进去，完成初始化
        super(Application, self).__init__(handlers, **settings)


def main():
    """
    @author:wangfc27441
    @desc:
    Tornado Web程序编写思路
        创建web应用实例对象，第一个初始化参数为路由映射列表。
        定义实现路由映射列表中的handler类。
        创建服务器实例，绑定服务器端口。
        启动当前线程的IOLoop。
    @version：
    @time:2021/1/22 15:56

    Parameters
    ----------

    Returns
    -------
    """
    # 定义 HTTPServer 监听HTTP请求的端口
    define('port', default=port, help='run on the given port', type=int)
    # 关闭 tornado 日志
    options.logging = None
    # 使用 tornado 的 options 模块解析命令行：转换命令行参数，并将转换后的值对应的设置到全局options对象相关属性上
    tornado.options.parse_command_line()
    # 获取本机计算机名称
    hostname = socket.gethostname()
    # 获取本机ip
    ip = socket.gethostbyname(hostname)

    # 启动单进程服务
    mode = 'cpu' if gpu_no == '-1' else 'gpu:' + gpu_no

    # 1. 初始化 路由信息映射元组的列表
    handlers = [
        # (r'/', HelloWorld)
        (object_detection_server_url, mmdetection_handlers.ObjectDetectionHandler)
        # score_thr=score_thr,if_ger_processed_image=if_ger_processed_image))
        # (r'/', mmdetection_handlers.IndexHandler),
        # (r'/upload', mmdetection_handlers.UploadFileHandler)
        # (r'/explore',main.ExploreHandler),
        # (r'/post/(?P<post_id>[0-9]+)',main.PostHandler), #命名组写法,使用关键字，路由与handler方法不一定顺序一致
    ]
    # 初始化 web.Application 的配置
    settings = dict(
        autoreload=True,  # 自动加载主要用于开发和测试阶段，要不每次修改，都重启tornado服务 ,
        debug=True,  # 调试模式，修改后自动重启服务，不需要自动重启，生产情况下切勿开启，安全性
        # template_path=os.path.join(os.path.dirname(__file__), "templates"),
        # 模板文件目录,想要Tornado能够正确的找到html文件，需要在 Application 中指定文件的位置
        # static_path=os.path.join(os.path.dirname(__file__), "static")
        # 静态文件目录,可用于用于访问js,css,图片之类的添加此配置之后，tornado就能自己找到静态文件
    )

    # 2. 构建一个 Application 对象的实例
    # 初始化接收的第一个参数就是一个路由信息映射元组的列表
    application = Application(handlers, settings)

    # 3. 创建HTTPServer 对象：将 Application实例传递给 Tornado 的  HTTPServer 对象
    http_server = tornado.httpserver.HTTPServer(application)

    # 4. 启动IOLoop
    logger.info(f"当前工作路径为：{os.getcwd()}")
    logger.info('开始启动目标识别服务 num_processes={} with mode={} at url={}:{}{}'
                .format(num_processes, mode, ip, options.port, object_detection_server_url))
    # 服务器绑定到指定端口
    http_server.bind(options.port)  # http_server.listen()这个方法只能在单进程模式中使用
    # 开启num_processes进程（默认值为1）
    """
    开启几个进程，参数num_processes默认值为1，即默认仅开启一个进程；如果num_processes为None或者<=0，
    则自动根据机器硬件的cpu核芯数创建同等数目的子进程；如果num_processes>0，则创建num_processes个子进程。
    """
    http_server.start(num_processes=num_processes)  # num_processes=num_processes
    # 启动当前线程的IOLoop
    tornado.ioloop.IOLoop.current().start()
    logger.info("启动目标识别服务成功！")

    # 启动高级多进程 HTTPServer 服务
    # logger.info('启动 bert 高级多进程服务 with mode={} at url={}:{}{}'.format(mode, ip, options.port, SERVICE_URL_DIR))
    # sockets = tornado.netutil.bind_sockets( options.port)
    # # 启动多进程
    # tornado.process.fork_processes(num_processes = num_processes)
    # http_server.add_sockets(sockets)
    # # IOLoop.current() : 返回当前线程的IOLoop实例
    # # IOLoop.start():  启动IOLoop实例的I / O循环, 同时服务器监听被打开。
    # tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(e, exc_info=True)
