# -*- coding: UTF-8 -*-

import tensorflow as tf  # 导入tensorflow模块
import numpy as np  # 导入numpy模块
from PIL import Image  # 导入PIL模块
from PIL import ImageOps
from .forward import forward  # 导入前向网络模块
import argparse  # 导入参数选择模块
from .generateds import center_crop_img

# 设置参数
parser = argparse.ArgumentParser()  # 定义一个参数设置器

# 修改以下5个参数以开启训练
parser.add_argument("--PATH_IMG", type=str, default="./test_imgs/ruanwei1.jpeg")  # 参数：选择测试图像
parser.add_argument("--LABEL_1", type=int, default=2)  # 参数：风格1
parser.add_argument("--LABEL_2", type=int, default=8)  # 参数：风格2
parser.add_argument("--LABEL_3", type=int, default=10)  # 参数：风格3
parser.add_argument("--LABEL_4", type=int, default=19)  # 参数：风格4

# 固定参数
parser.add_argument("--LABELS_NUMS", type=int, default=20)  # 参数：风格数量
parser.add_argument("--PATH_MODEL", type=str, default="./model/")  # 参数：模型存储路径
parser.add_argument("--PATH_RESULTS", type=str, default="./results/")  # 参数：测试结果存储路径
parser.add_argument("--PATH_STYLE", type=str, default="./style_imgs/")  # 参数：风格图片路径
parser.add_argument("--ALPHA1", type=float, default=0.25)  # 参数：Alpha1，风格权重，默认为0.25
parser.add_argument("--ALPHA2", type=float, default=0.25)  # 参数：Alpha2，风格权重，默认为0.25
parser.add_argument("--ALPHA3", type=float, default=0.25)  # 参数：Alpha3，风格权重，默认为0.25
parser.add_argument("--ALPHA4", type=float, default=0.25)  # 参数：Alpha3，风格权重，默认为0.25

args = parser.parse_args()  # 定义参数集合args


class Stylizer(object):

    def __init__(self, stylizer_arg):
        self.stylizer_arg = stylizer_arg
        self.content = tf.placeholder(tf.float32, [1, None, None, 3])  # 图片输入定义
        self.weight = tf.placeholder(tf.float32, [1, stylizer_arg.LABELS_NUMS])  # 风格权重向量，用于存储用户选择的风格
        self.target = forward(self.content, self.weight)  # 定义将要生成的图片
        self.sess = tf.Session()  # 定义一个sess
        self.sess.run(tf.global_variables_initializer())  # 模型初始化
        self.img_path = stylizer_arg.PATH_IMG
        self.img = None
        self.label_list = None
        self.input_weight = None
        self.img25 = None
        self.img25_4 = None
        saver = tf.train.Saver()  # 模型存储器定义
        ckpt = tf.train.get_checkpoint_state(stylizer_arg.PATH_MODEL)  # 从模型存储路径中获取模型
        if ckpt and ckpt.model_checkpoint_path:  # 从检查点中恢复模型
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def __del__(self):
        self.sess.close()

    def read_image(self):
        # 读取图像
        img_input = Image.open(self.img_path)
        # 不断对图片进行缩小，直到符合尺寸限制
        # 若图片太大，生成速度将会非常慢，若需生成原图，可将此段注释掉
        while img_input.width * img_input.height > 500000:
            img_input = img_input.resize((int(img_input.width / 1.5), int(img_input.height / 1.5)))
        # 转成数组
        self.img = np.array(img_input)

    def set_style(self):
        # 将用户选取的4个风格存储在tuple中
        self.label_list = (self.stylizer_arg.LABEL_1, self.stylizer_arg.LABEL_2,
                           self.stylizer_arg.LABEL_3, self.stylizer_arg.LABEL_4)

    def set_weight(self, weight_dict):
        # 初始化input_weight
        self.input_weight = np.zeros([1, self.stylizer_arg.LABELS_NUMS])
        # 传入权重字典，键为权重编号，键值为权重，并存到input_weight中
        for k, v in weight_dict.items():
            self.input_weight[0, k] = v

    def stylize(self, alpha_list=None):
        # 将风格列表及对应的权重放入字典中
        weight_dict = dict(zip(self.label_list, alpha_list))
        self.set_weight(weight_dict)
        # 打印提示信息
        print('Generating', self.img_path.split('/')[-1], '--style:', str(self.label_list),
              '--alpha:', str(alpha_list), )
        # 进行风格融合与迁移
        img = self.sess.run(self.target,
                            feed_dict={self.content: self.img[np.newaxis, :, :, :], self.weight: self.input_weight})
        # 保存单张图片
        # 直接str(tuple)会产生空格，js无法读取
        Image.fromarray(np.uint8(img[0, :, :, :])).save(
            self.stylizer_arg.PATH_RESULTS + self.img_path.split('/')[-1].split('.')[0] + '_' + '(' +
            str(self.label_list[0]) + ',' +
            str(self.label_list[1]) + ',' +
            str(self.label_list[2]) + ',' +
            str(self.label_list[3]) + ',' + ')' + '_' + '(' +
            str(alpha_list[0]) + ',' +
            str(alpha_list[1]) + ',' +
            str(alpha_list[2]) + ',' +
            str(alpha_list[3]) + ',' + ')' + '.jpg')
        return img

    def generate_result(self):
        size = 5
        i = 0
        # 按行生成
        while i < size:
            # 1、2风格权重之和
            x_sum = 100 - i * 25.0
            # 3、4风格权重之和
            y_sum = i * 25
            # 1、2风格之和进行五等分，计算权重step值
            x_step = x_sum / 4.0
            # 3、4风格之和进行五等分，计算权重step值
            y_step = y_sum / 4.0

            # 按列生成
            j = 0
            while j < size:
                # 计算1、2风格的权重
                ap1 = x_sum - j * x_step
                ap2 = j * x_step
                # 计算3风格权重
                ap3 = y_sum - j * y_step
                ap4 = j * y_step
                # 归一化后存到alphas中
                alphas = (float('%.2f' % (ap1 / 100.0)),
                          float('%.2f' % (ap2 / 100.0)),
                          float('%.2f' % (ap3 / 100.0)),
                          float('%.2f' % (ap4 / 100.0)))
                # 返回融合后图像
                img_return = self.stylize(alphas)
                # array转IMG
                img_return = Image.fromarray(np.uint8(img_return[0, :, :, :]))
                # 将5个图像按行拼接
                if j == 0:
                    width, height = img_return.size
                    img_5 = Image.new(img_return.mode, (width * 5, height))
                img_5.paste(img_return, (width * j, 0, width * (j + 1), height))
                j = j + 1

            # 将多个行拼接图像，拼接成5*5矩阵
            if i == 0:
                img_25 = Image.new(img_return.mode, (width * 5, height * 5))
            img_25.paste(img_5, (0, height * i, width * 5, height * (i + 1)))

            i = i + 1

        # 将5*5矩阵图像的4个角加上4个风格图像，以作对比
        img25_4 = Image.new(img_25.mode, (width * 7, height * 5))
        img25_4 = ImageOps.invert(img25_4)
        img25_4.paste(
            center_crop_img(Image.open(
                self.stylizer_arg.PATH_STYLE + str(self.label_list[0] + 1) + '.png')).resize((width, height)),
            (0, 0, width, height))
        img25_4.paste(
            center_crop_img(Image.open(
                self.stylizer_arg.PATH_STYLE + str(self.label_list[1] + 1) + '.png')).resize((width, height)),
            (width * 6, 0, width * 7, height))
        img25_4.paste(
            center_crop_img(Image.open(
                self.stylizer_arg.PATH_STYLE + str(self.label_list[2] + 1) + '.png')).resize((width, height)),
            (0, height * 4, width, height * 5))
        img25_4.paste(
            center_crop_img(Image.open(
                self.stylizer_arg.PATH_STYLE + str(self.label_list[3] + 1) + '.png')).resize((width, height)),
            (width * 6, height * 4, width * 7, height * 5))
        img25_4.paste(img_25, [width, 0, width * 6, height * 5])

        self.img25 = img_25
        self.img25_4 = img25_4

    def save_result(self):
        # 存储5*5图像矩阵
        self.img25.save(self.stylizer_arg.PATH_RESULTS + self.img_path.split('/')[-1].split('.')[0] + '_' + str(
            self.label_list) + '_result_25' + '.jpg')
        self.img25.show()
        # 存储5*5+4风格图像矩阵
        self.img25_4.save(self.stylizer_arg.PATH_RESULTS + self.img_path.split('/')[-1].split('.')[0] + '_' + str(
            self.label_list) + '_result_25_4' + '.jpg')
        self.img25_4.show()
        print('Image Matrix Saved Successfully!')


def get_image_matrix():
    # 初始化
    stylizer0 = Stylizer(args)
    # 读取图像文件
    stylizer0.read_image()
    # 存储风格标签
    stylizer0.set_style()
    # 生成融合图片
    stylizer0.generate_result()
    # 保存融合图片
    stylizer0.save_result()
    # 释放对象
    del stylizer0
    # 重置图
    tf.reset_default_graph()


# 主程序
def main():
    get_image_matrix()


# 主程序入口
if __name__ == '__main__':
    main()
