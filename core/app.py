# -*- coding: UTF-8 -*-
import os
from .test import *
import random


def deleteResult():
    path = "./static/results/"
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if ("jpg" in c_path):
            os.remove(c_path)

def delte():
    path = "./static/data/"
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if ("result" in c_path):
            os.remove(c_path)

def main(style1, style2, style3, style4, imgpath):
    deleteResult()
    # 初始化变量
    args.LABEL_1 = style1
    args.LABEL_2 = style2
    args.LABEL_3 = style3
    args.LABEL_4 = style4
    args.PATH_IMG = imgpath
    args.PATH_RESULTS = "./static/results/"
    args.PATH_STYLE = "./core/style_imgs/"
    args.PATH_MODEL = "./core/model/"

    stylizer0 = Stylizer(args)
    stylizer0.read_image()
    stylizer0.set_style()
    stylizer0.generate_result()

    # 存储5*5图像矩阵
    stylizer0.img25.save(args.PATH_RESULTS + 'result_25' + str(random.uniform(1, 10)) + '.jpg')
    # 存储5*5+4风格图像矩阵
    delte()
    url = "./static/data/result25" + str(random.uniform(1, 10)) + ".jpg"
    stylizer0.img25_4.save(url)
    del stylizer0
    tf.reset_default_graph()