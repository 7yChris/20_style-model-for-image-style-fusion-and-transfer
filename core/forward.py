# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf

"""
***********************************************************************************
网络结构由Johnson 2016提出
详见论文 Perceptual Losses for Real-Time Style Transfer and Super-Resolution  arxiv:1603.08155

                                                                   ^ style loss ========+
                                                                   |                    |
        +----------+                             |*****************|*******|            |
        |  style   +============================->=================+=======>===>        |w1
        +----------+   +********************+    |* Discriminator  |      *|            |
        +----------+   |* generator        *|    |* ( vgg-16 )     |      *|            |
        |  target  +===> (using conditional +====>====+============+=======>===>        + ===== total loss
        +---+------+   |*   normalization) *|    |*   |                   *|            |
            |          +********************+    |*   |                   *|            |
            +====================================>====+====================>===>        |w2
                                                 |****|********************|            |
                                                      |                                 |
                                                      v content loss ===================+

**************************************************************************************
"""

# ***********************************Generative Model***********************************

"""
前向传播网络，也就是generate的过程

input：target图片
y...：风格标签的one-hot编码（tensor)
alpha...：不同风格的权重（并保证alpha之和为1）
return：风格融合后的图片
"""


def forward(inputs, weight):
    # 预处理'BGR'->'RGB',同时减去vgg_mean，以契合"vgg"模型。
    inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])

    # 3层卷积 ,每次卷积后均 +条件归一化并使用rule激活函数，也就是三次 downsample 操作
    inputs = conv("conv1", inputs, 9, 3, 32, 1)  # 卷积
    inputs = conditional_normalization(inputs, "cin1", weight)  # 条件归一化
    inputs = tf.nn.relu(inputs)  # relu

    inputs = conv("conv2", inputs, 3, 32, 64, 2)
    inputs = conditional_normalization(inputs, "cin2", weight)
    inputs = tf.nn.relu(inputs)

    inputs = conv("conv3", inputs, 3, 64, 128, 2)
    inputs = conditional_normalization(inputs, "cin3", weight)
    inputs = tf.nn.relu(inputs)

    # 5层resBlock
    inputs = res_block("res1", inputs, 3, 128, 128, weight)
    inputs = res_block("res2", inputs, 3, 128, 128, weight)
    inputs = res_block("res3", inputs, 3, 128, 128, weight)
    inputs = res_block("res4", inputs, 3, 128, 128, weight)
    inputs = res_block("res5", inputs, 3, 128, 128, weight)

    # 2层upsampling
    inputs = upsampling("up1", inputs, 128, 64, weight)
    inputs = upsampling("up2", inputs, 64, 32, weight)

    # 最后一层卷积将channel限制在3
    inputs = conv("last", inputs, 9, 32, 3, 1)
    inputs = conditional_normalization(inputs, "cinout", weight)
    # 并使用255* sigmoid函数保证结果在 (  0,255 ) 内
    inputs = tf.nn.sigmoid(inputs) * 255

    return inputs


"""
一个封装过的卷积层

name：名称
k_size：核大小
nums_in：上一层的channel
nums_out：本层核的个数

return：tensor
"""


def conv(name, inputs, k_size, nums_in, nums_out, strides):
    # 使用正态分布初始化核
    kernel = tf.get_variable(name + "W", [k_size, k_size, nums_in, nums_out],
                             initializer=tf.truncated_normal_initializer(stddev=0.01))
    # 0初始化bias
    bias = tf.get_variable(name + "B", [nums_out], initializer=tf.constant_initializer(0.))

    # 为了保证输入输出图像大小一致，对原始输入首先按"REFLECT"方式填充。
    pad_size = k_size // 2
    inputs = tf.pad(inputs, [[0, 0], [pad_size, pad_size], [
        pad_size, pad_size], [0, 0]], mode="REFLECT")
    # 使用 valid方式进行卷积 ，避免输出图像边缘发黑。这与REFLECT填充一致
    input = tf.nn.conv2d(
        inputs, kernel, [1, strides, strides, 1], "VALID") + bias

    return input


"""
条件归一化层
详见论文：
1、A LEARNED REPRESENTATION FOR ARTISTIC STYLE
2、The Missing Ingredient for Fast Stylization

它训练z的分布gamma和beta，并将其作为vector以实现风格融合。

return: the normalized, scaled, offset tensor.
"""


def conditional_normalization(x, scope_bn, weight=None):
    # 获取beta、gamma参数变量
    beta_matrix = tf.get_variable(name=scope_bn + 'beta',
                                  shape=[weight.shape[-1], x.shape[-1]],
                                  initializer=tf.constant_initializer([0.]),
                                  trainable=True)  # label_nums x C
    gamma_matrix = tf.get_variable(name=scope_bn + 'gamma',
                                   shape=[weight.shape[-1], x.shape[-1]],
                                   initializer=tf.constant_initializer([1.]),
                                   trainable=True)  # label_nums x C

    # 根据输入权重获取instance normalization beta、gamma值
    # 在train过程当中weight为one hot编码用于抽取当前训练风格beta、gamma
    # 在test过程当中weight可以融合多个风格的参数，从而达到特征融合效果（具体可参考上述论文）
    beta = tf.matmul(weight, beta_matrix)
    gamma = tf.matmul(weight, gamma_matrix)

    # 获取单个feature map上mean和variance用作instance normalization
    mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    # 按照beta、gamma做归一化
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-10)
    return x


"""
上采样层
在Johnson 2016中使用deconverlution
为了减少"棋盘格" 使用插值算法并卷积来生成图片。
"""


def upsampling(name, inputs, nums_in, nums_out, weight):
    # 插值算法 。
    inputs = tf.image.resize_nearest_neighbor(inputs, [tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])
    # 卷积
    inputs = conv(name, inputs, 3, nums_in, nums_out, 1)
    # 条件归一化
    inputs = conditional_normalization(inputs, "cin" + name, weight)
    return inputs


"""
一层残差网络
1、网址：http://torch.ch/blog/2016/02/04/resnets.html
2、论文：He, Kaiming, et al. “Deep Residual Learning for Image Recognition.”  arXiv:1512.03385 (2015).

name
inputs：tensor
k_size：核大小
nums_in = input.shape[3]
nums_out

"""


def res_block(name, inputs, k_size, nums_in, nums_out, weight):
    # 暂存输入
    temp = inputs * 1.0
    # 一层卷积
    inputs = conv("conv1_" + name, inputs, k_size, nums_in, nums_out, 1)
    inputs = conditional_normalization(inputs, "cin1" + name, weight)
    # 激活函数relu
    inputs = tf.nn.relu(inputs)
    # 一层卷积
    inputs = conv("conv2_" + name, inputs, k_size, nums_in, nums_out, 1)
    inputs = conditional_normalization(inputs, "cin2" + name, weight)
    # 返回输入和网络输出之和
    return inputs + temp


# ***********************************Discriminative Model ***********************************

"""
vgg16:
详见vgg论文：
Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. Computer Science, 2014.

inputs：vgg输入
vgg_path：存储vgg参数的路径

return：包含指定层特征的字典，以供计算损失
"""


def vggnet(inputs, vgg_path):
    # VGG16卷积操作
    def vgg_conv(inputs, w, b):
        # 卷积函数，inputs：输入，卷积核，步长，前后形状相同
        return tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME") + b

    # VGG16池化操作
    def vgg_max_pooling(inputs):
        # 池化函数：输入，池化窗口大小，滑动步长，前后形状
        return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

    inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])  # 处理输入图片
    para = np.load(vgg_path + "vgg16.npy", encoding="latin1").item()  # 从vgg模型路径获得vgg模型
    F = {}  # 定义返回值为字典F

    inputs = vgg_conv(inputs, para["conv1_1"][0], para["conv1_1"][1])  # 第一层卷积提取特征
    inputs = tf.nn.relu(inputs)  # 激活函数relu，即将矩阵中每行的非最大值置0。
    inputs = vgg_conv(inputs, para["conv1_2"][0], para["conv1_2"][1])
    inputs = tf.nn.relu(inputs)
    F["conv1_2"] = inputs

    inputs = vgg_max_pooling(inputs)  # 池化
    inputs = vgg_conv(inputs, para["conv2_1"][0], para["conv2_1"][1])  # 第二层卷积提取特征
    inputs = tf.nn.relu(inputs)  # 激活函数relu
    inputs = vgg_conv(inputs, para["conv2_2"][0], para["conv2_2"][1])
    inputs = tf.nn.relu(inputs)
    F["conv2_2"] = inputs

    inputs = vgg_max_pooling(inputs)  # 池化
    inputs = vgg_conv(inputs, para["conv3_1"][0], para["conv3_1"][1])  # 第三层卷积提取特征
    inputs = tf.nn.relu(inputs)  # relu激活函数
    inputs = vgg_conv(inputs, para["conv3_2"][0], para["conv3_2"][1])
    inputs = tf.nn.relu(inputs)
    inputs = vgg_conv(inputs, para["conv3_3"][0], para["conv3_3"][1])
    inputs = tf.nn.relu(inputs)
    F["conv3_3"] = inputs

    inputs = vgg_max_pooling(inputs)  # 池化
    inputs = vgg_conv(inputs, para["conv4_1"][0], para["conv4_1"][1])  # 第四层卷积提取特征
    inputs = tf.nn.relu(inputs)  # relu激活函数
    inputs = vgg_conv(inputs, para["conv4_2"][0], para["conv4_2"][1])
    inputs = tf.nn.relu(inputs)
    inputs = vgg_conv(inputs, para["conv4_3"][0], para["conv4_3"][1])
    inputs = tf.nn.relu(inputs)
    F["conv4_3"] = inputs

    return F


# ***********************************损失函数***********************************

"""
以L2距离计算内容损失
"""


def get_content_loss(content_img, target_img):
    return tf.nn.l2_loss(content_img["conv2_2"] - target_img["conv2_2"]) * 2 / tf.cast(tf.size(content_img["conv2_2"]),
                                                                                       dtype=tf.float32)


"""
计算风格损失

实际上是计算vgg中间输出gram矩阵的mse
详见：Gatys`s A Neural Algorithm of Artistic Style arxiv:1508.06576v2

phi_style
phi_target: 一个tensor的字典，包含两个图经过vgg后的中间输出

"""


def get_style_loss(style_img, target_img):
    # 声明一个字典
    layers = ["conv1_2", "conv2_2", "conv3_3", "conv4_3"]
    # 初始化loss
    loss = 0
    for layer in layers:
        # 取style_output
        style_layer = style_img[layer]
        style_gram = get_gram_matrix(style_layer)
        # 取target_output
        target_layer = target_img[layer]
        target_gram = get_gram_matrix(target_layer)
        # 计算mse
        loss += tf.nn.l2_loss(style_gram - target_gram) * 2 / tf.cast(tf.size(target_gram), dtype=tf.float32)
    return loss


"""
计算gram矩阵(tensor)
"""


def get_gram_matrix(layer):
    shape = tf.shape(layer)
    # 第一维是batch_size
    batch_size = shape[0]
    # 宽
    width = shape[1]
    # 长
    height = shape[2]
    # 通道数
    channel_num = shape[3]
    # 只对 第二三维度进行点乘，因此把1，4维度提前
    filters = tf.reshape(layer, tf.stack([batch_size, -1, channel_num]))
    # 点乘得到grams矩阵, 仅在维度宽和长上做点乘
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * channel_num)
    return grams
