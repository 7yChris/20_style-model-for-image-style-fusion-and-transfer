# -*- coding: UTF-8 -*-
import os
from .test import *
import random


def deleteResult(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if ("jpg" in c_path):
            os.remove(c_path)

def delte(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if ("result" in c_path):
            os.remove(c_path)

def main(arg, result_path, data_path, style_path, model_path):

    # 初始化变量
    args.LABEL_1 = int(arg[0])
    args.LABEL_2 = int(arg[1])
    args.LABEL_3 = int(arg[2])
    args.LABEL_4 = int(arg[3])
    args.PATH_IMG = arg[4]
    args.PATH_RESULTS = result_path
    args.PATH_STYLE = style_path
    args.PATH_MODEL = model_path

    # 删除之前所产生的图片
    deleteResult(args.PATH_RESULTS)

    # 初始化Stylizer类
    stylizer = Stylizer(args)
    # 读取图片
    stylizer.read_image()
    # 设置风格参数
    stylizer.set_style()
    # 产生风格融合图片
    stylizer.generate_result()

    # 存储5*5图像矩阵
    stylizer.img25.save(args.PATH_RESULTS + 'result_25' + str(random.uniform(1, 10)) + '.jpg')

    # 删除之前所产生的图片
    delte(data_path)

    # 存储5*5+4风格图像矩阵
    url = data_path + "result25" + str(random.uniform(1, 10)) + ".jpg"
    stylizer.img25_4.save(url)

    del stylizer
    tf.reset_default_graph()