# -*- coding: UTF-8 -*-

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.misc as misc

flags = tf.flags
flags.DEFINE_string('path_data', './data', 'tfRecord save path.')
flags.DEFINE_string('path_style', "./style_imgs", 'Row style images path')
flags.DEFINE_string('path_content', "./MSCOCO", 'Row style images path')

flags.DEFINE_string('record_style_name', 'styles.tfrecords', 'Style tfrecord name')
flags.DEFINE_string('record_dataset_name', 'coco_train.tfrecords', 'Data set tfrecord name')

flags.DEFINE_integer('img_h', 256, 'Train images\' height')
flags.DEFINE_integer('img_w', 256, 'Train images\' width')
flags.DEFINE_integer('img_c', 3, 'Train images\' channels num')
flags.DEFINE_integer('style_h', 512, 'Style images\' height')
flags.DEFINE_integer('style_w', 512, 'Style images\' width')

FLAGS = flags.FLAGS


def generate_content_tfrecord():
    """
    制作coco数据集的tfRecord文件
    """
    # 若文件夹不存在，则生成文件夹，并打印相关信息
    if not os.path.exists(FLAGS.path_data):
        os.makedirs(FLAGS.path_data)
        print("the directory was created successful")
    else:
        print("directory already exists")
    write_content_tfrecord()


def write_content_tfrecord():
    # 定义writer对象
    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.path_data, FLAGS.record_dataset_name))
    num_pic = 0
    example_list = list()
    file_path_list = []
    # 读入coco原始数据集中文件路径集合
    for root, _, files in os.walk(FLAGS.path_content):
        for file in files:
            # 检查是否为图像文件
            if os.path.splitext(file)[1] not in ['.jpg', '.png', '.jpeg']:
                continue
            # 若是，则加入文件路径集合
            file_path = os.path.join(root, file)
            file_path_list.append(file_path)

    # 对路径集合进行打乱
    np.random.shuffle(file_path_list)
    for file_path in file_path_list:
        with Image.open(file_path) as img:
            # 对coco数据集图片剪裁为正方形
            img = center_crop_img(img)
            # resize图片大小
            img = img.resize((FLAGS.img_w, FLAGS.img_h))
            img = img.convert('RGB')
            img_raw = img.tobytes()
            # 为图像建Example
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            # 写入tfrecord文件
            num_pic += 1
            writer.write(example.SerializeToString())
            print('the number of picture:', num_pic)
    print('write tfrecord successful')


def read_content_tfrecord(path_tfrecord, image_size):
    # 创建文件队列
    filename_queue = tf.train.string_input_producer([path_tfrecord])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'img_raw': tf.FixedLenFeature([], tf.string)
    })
    # 解码图片数据
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 设置图片shape
    img.set_shape([image_size * image_size * 3])
    return img


def get_content_tfrecord(batch_size, path_tfrecord, image_size):
    """
    获取content_batch，用于训练
    :param batch_size:
    :param path_tfrecord: tfrecord存储路径
    :param image_size: 图片尺寸
    :return: content_batch op
    """
    img = read_content_tfrecord(path_tfrecord, image_size)
    img_batch = tf.train.shuffle_batch([img, ], batch_size=batch_size, num_threads=2, capacity=10, min_after_dequeue=1)
    return img_batch


def center_crop_img(img):
    """
    对图片按中心进行剪裁位正方形
    :param img: 原始图片
    :return:
    """
    width = img.size[0]
    height = img.size[1]
    offset = (width if width < height else height) / 2
    img = img.crop((
        width / 2 - offset,
        height / 2 - offset,
        width / 2 + offset,
        height / 2 + offset
    ))
    return img


# 风格图片随机提取
def random_select_style(path, batch_size, shape, c_nums):
    # 列出风格图片文件名
    filenames = os.listdir(path)
    # 随机选出一张风格图片
    rand_sample = np.random.randint(0, len(filenames))
    # 读取风格图片，并进行裁剪、resize
    img = misc.imresize(crop(np.array(Image.open(path + str(rand_sample + 1) + ".png"))), [shape[0], shape[1]])
    # 初始化风格图片存储矩阵
    batch = np.zeros([batch_size, shape[0], shape[1], shape[2]])
    # 标记选中了哪张风格图片
    y = np.zeros([1, c_nums])
    y[0, rand_sample] = 1
    # 风格图片存储矩阵，存储batch_size个相同的风格图片
    for i in range(batch_size):
        batch[i, :, :, :] = img[:, :, :3]
    return batch, y


# 图片剪裁
def crop(img):
    # 得到图像的高
    h = img.shape[0]
    # 得到图像的宽
    w = img.shape[1]

    # 如果是长方形，则进行随机裁剪，使之介于正方形与原始图像尺寸之间
    if h < w:
        x = 0
        y = np.random.randint(0, w - h + 1)
        length = h
    elif h > w:
        x = np.random.randint(0, h - w + 1)
        y = 0
        length = w

    # 如果是正方形，则不进行裁剪
    else:
        x = 0
        y = 0
        length = h
    return img[x:x + length, y:y + length, :]


def main():
    generate_content_tfrecord()


if __name__ == '__main__':
    main()
