# -*- coding: UTF-8 -*-

from forward import vggnet, forward, get_content_loss, get_style_loss
import tensorflow as tf

from generateds import get_content_tfrecord
from generateds import random_select_style
from PIL import Image
import numpy as np
import scipy.misc as misc
import argparse
import time
import os

# 初始化各种参数
parser = argparse.ArgumentParser()
# 输入图像尺寸
parser.add_argument("--IMG_H", type=int, default=256)
parser.add_argument("--IMG_W", type=int, default=256)
parser.add_argument("--IMG_C", type=int, default=3)
# 风格图像尺寸
parser.add_argument("--STYLE_H", type=int, default=512)
parser.add_argument("--STYLE_W", type=int, default=512)
# 风格图像张数
parser.add_argument("--LABELS_NUMS", type=int, default=20)
# Batch大小，默认为2
parser.add_argument("--BATCH_SIZE", type=int, default=2)
# 学习率
parser.add_argument("--LEARNING_RATE", type=float, default=0.001)
# 内容权重和风格权重
parser.add_argument("--CONTENT_WEIGHT", type=float, default=1.0)
parser.add_argument("--STYLE_WEIGHT", type=float, default=5.0)
# 训练内容图像路径，train2014
parser.add_argument("--PATH_CONTENT", type=str, default="./MSCOCO/")
# 风格图像路径
parser.add_argument("--PATH_STYLE", type=str, default="./style_imgs/")
# 生成模型路径
parser.add_argument("--PATH_MODEL", type=str, default="./model/")
# VGG16路径
parser.add_argument("--PATH_VGG16", type=str, default="./vggnet/")
# 数据集路径
parser.add_argument("--PATH_DATA", type=str, default="./data/")
# 数据集名称
parser.add_argument("--DATASET_NAME", type=str, default="coco_train.tfrecords")
# 训练轮数
parser.add_argument("--steps", type=int, default=50000)
args = parser.parse_args()


def backward(img_h=args.IMG_H, img_w=args.IMG_W, img_c=args.IMG_C, style_h=args.STYLE_H, style_w=args.STYLE_W,
             c_nums=args.LABELS_NUMS, batch_size=args.BATCH_SIZE, learning_rate=args.LEARNING_RATE,
             content_weight=args.CONTENT_WEIGHT, style_weight=args.STYLE_WEIGHT, path_style=args.PATH_STYLE,
             model_path=args.PATH_MODEL, vgg_path=args.PATH_VGG16, path_data=args.PATH_DATA,
             dataset_name=args.DATASET_NAME):
    # 内容图像：batch为2，图像大小为256*256*3
    content = tf.placeholder(tf.float32, [batch_size, img_h, img_w, img_c])
    # 风格图像：batch为2，图像大小为512*512*3
    style = tf.placeholder(tf.float32, [batch_size, style_h, style_w, img_c])
    # 风格1：训练风格的标签
    weight = tf.placeholder(tf.float32, [1, c_nums])

    # 图像生成网络：前向传播
    target = forward(content, weight)
    # 生成图像、内容图像、风格图像特征提取
    Phi_T = vggnet(target, vgg_path)
    Phi_C = vggnet(content, vgg_path)
    Phi_S = vggnet(style, vgg_path)
    # Loss计算
    # 内容Loss
    content_loss = get_content_loss(Phi_C, Phi_T)
    # 风格Loss
    style_loss = get_style_loss(Phi_S, Phi_T)
    # 总Loss
    loss = content_loss * content_weight + style_loss * style_weight

    # 定义当前训练轮数变量
    global_step = tf.Variable(0, trainable=False)

    # 优化器：Adam优化器，损失最小化
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 读取训练数据
    content_batch = get_content_tfrecord(batch_size, os.path.join(path_data, dataset_name), img_h)

    # 实例化saver对象，便于之后保存模型
    saver = tf.train.Saver()
    # 开始计时
    time_start = time.time()

    with tf.Session() as sess:
        # 初始化全局变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 在路径中查询有无checkpoint
        ckpt = tf.train.get_checkpoint_state(model_path)
        # 从checkpoint恢复模型
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restore Model Successfully')
        else:
            print('No Checkpoint Found')

        # 开启多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 开始训练，共进行5万轮
        for itr in range(args.steps):
            # 计时
            time_step_start = time.time()

            # 随机读取batch_size张内容图片，存储在四维矩阵中（batch_size*h*w*c）
            # batch_content = random_batch(path_content, batch_size, [IMG_H, IMG_W, IMG_C])
            batch_content = sess.run(content_batch)
            batch_content = np.reshape(batch_content, [batch_size, img_w, img_h, img_c])
            # 随机选择1个风格图片，并返回风格图片存储矩阵（batch_size*h*w*c，每个h*w*c都相同），y_labels为风格标签
            batch_style, y_labels = random_select_style(path_style, batch_size, [style_h, style_w, img_c], c_nums)

            # 喂数据，开始训练
            sess.run(opt, feed_dict={content: batch_content, style: batch_style, weight: y_labels})
            step = sess.run(global_step)

            # 打印相关信息
            if itr % 500 == 0:
                # 为之后打印信息进行相关计算
                [loss, target, content_loss_res, style_loss_res] = sess.run([loss, target, content_loss, style_loss],
                                                                            feed_dict={content: batch_content,
                                                                                       style: batch_style,
                                                                                       weight: y_labels})
                # 连接3张图片（内容图片、风格图片、生成图片）
                save_img = np.concatenate((batch_content[0, :, :, :],
                                           misc.imresize(batch_style[0, :, :, :], [img_h, img_w]),
                                           target[0, :, :, :]), axis=1)
                # 打印轮数、总loss、内容loss、风格loss
                print("Iteration: %d, Loss: %e, Content_loss: %e, Style_loss: %e" %
                      (step, loss, content_loss_res, style_loss_res))
                # 展示训练效果：打印3张图片，内容图+风格图+风格迁移图
                Image.fromarray(np.uint8(save_img)).save(
                    "save_training_imgs/" + str(step) + "_" + str(np.argmax(y_labels[0, :])) + ".jpg")

            time_step_stop = time.time()
            # 存储模型
            a = 500
            if itr % a == 0:
                saver.save(sess, model_path + "model", global_step=global_step)
                print('Iteration: %d, Save Model Successfully, single step time = %.2fs, total time = %.2fs' % (
                    step, time_step_stop - time_step_start, time_step_stop - time_start))

        # 关闭多线程
        coord.request_stop()
        coord.join(threads)


def main():
    backward()


if __name__ == '__main__':
    main()
