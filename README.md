# 20-Style Model for Image Style Fusion and Transfer
图像风格融合与快速迁移

Simple implementation of the paper "A Learned Representation for Artistic Style"

[A Learned Representation for Artistic Style.](http://cn.arxiv.org/pdf/1610.07629.pdf)

## Instruction
### 一、core文件夹
##### 项目核心代码<br>
generateds：生成tfrecords训练集<br>
forward：定义图像生成网络<br>
backward：主要训练过程<br>
test：测试部分，完成图像风格融合和快速迁移<br>
app：JS网页应用调用test的接口文件

### 二、main.py
运行JS网页应用的入口，在网页上完成可视化操作

## Prepare
##### 一、运行环境
 python3.6<br>
<br>
##### 二、下载coco数据集
http://images.cocodataset.org/zips/train2014.zip <br>
放在 ./core/MSCOCO/ 文件夹中<br>
<br>
##### 三、下载vgg16预训练模型
百度网盘（链接来自网络）：https://pan.baidu.com/s/1gg9jLw3  密码:umce，<br>
放在 ./core/vggnet 文件夹中<br>
<br>
##### 四、制作coco数据集tfRecords
```
$ python generateds.py
```

## Training Model
```
$ python backward.py
```
1、提供断点续训功能<br>
2、提供已经训练好的模型（5万轮），可在5万轮的基础上继续训练，也可删掉checkpoint从0开始训练<br>
3、在训练过程中，每500轮将会保存一张训练图片在./core/save_training_imgs/中，可从中查看训练效果<br>

## Eval Model
1、将测试图片放在./core/test_imgs/中<br>
2、在test.py中，修改测试文件路径PATH_IMG<br>
3、在test.py中，修改风格编号LABEL_1至LABEL_4
```
$ python test.py
```
4、生成图片存储在./core/results/文件夹中

## Run APP
##### 目前仅对windows系统Edge浏览器做过全面测试，其他浏览器存在不同程度的不兼容性，之后会尽快完善
1、将测试图片放在./app_img/中<br>
2、在main.py中修改host地址<br>
#for windows<br>
HOST = '127.0.0.3'<br>
#for mac<br>
HOST = '127.0.0.1'<br>
```
$ python main.py
```
3、在浏览器中打开host地址<br>
4、在网页上完成可视化操作<br>
5、选择测试图片时，请从/app_img/中选择<br>
6、若提示"OSError: [Errno 48] Address already in use"，请按以下步骤处理：<br>
```
$ sudo lsof -i:5000
```
找到占用5000端口进程的PID编号，然后kill掉此进程
```
$ sudo kill pid
```
重新运行main.py
```
$ python main.py
```