from flask import Flask, jsonify, render_template, request
from core import app as model
import random
import os
import base64
import argparse

parser = argparse.ArgumentParser()  # 定义一个参数设置器
# 固定参数
parser.add_argument("--HOST", type=str, default='127.0.0.1')
parser.add_argument("--PATH_RESULTS", type=str, default="./static/results/")
parser.add_argument("--PATH_DATA", type=str, default="./static/data/")
parser.add_argument("--PATH_MODEL", type=str, default="./core/model/")
parser.add_argument("--PATH_STYLE", type=str, default="./core/style_imgs/")
args = parser.parse_args()  # 定义参数集合args

app = Flask(__name__)


@app.route("/api/getPicture", methods=['post'])
def getPicture():
    output1 = model.trainPicture(request.json)
    print(output1)
    return jsonify(results=[output1])


def safe_base64_decode(s):
    b = base64.b64encode(s.encode('utf-8'))  # 因为python3.x中字符都为unicode编码，而b64encode函数的参数为byte类型，所以必须先转码
    bstr_tmp = str(b, 'utf-8')  # 把byte类型的数据转换为utf-8的数据
    b_str = bstr_tmp.strip(r'=+')  # 用正则把 = 去掉
    return b_str


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


@app.route("/filePath", methods=['post'])
def filePath():
    data = request.json
    imgurl = data[4]
    dataPath = imgurl[22:]
    imagedata = base64.b64decode(dataPath)
    if (os.path.exists(args.PATH_DATA + 'picture.jpg')):
        os.remove(args.PATH_DATA + 'picture.jpg')
    file = open(args.PATH_DATA + 'picture.jpg', "wb")
    file.write(imagedata)
    file.close()
    data[4] = args.PATH_DATA + 'picture.jpg'
    print(data)
    model.main(data, args.PATH_RESULTS, args.PATH_DATA, args.PATH_STYLE, args.PATH_MODEL)
    imgPath = [1]
    return jsonify(results=[imgPath])


@app.route("/photograph", methods=['post'])
def photograph():
    data = request.json

    imgurl = data[5]
    dataPath = imgurl[22:]
    imagedata = base64.b64decode(dataPath)
    if (os.path.exists(args.PATH_DATA + 'touxiang.jpg')):
        os.remove(args.PATH_DATA + 'touxiang.jpg')
    file = open(args.PATH_DATA + 'touxiang.jpg', 'wb')
    file.write(imagedata)
    file.close()
    data[4] = args.PATH_DATA + 'touxiang.jpg'
    model.main(data, args.PATH_RESULTS, args.PATH_DATA, args.PATH_STYLE, args.PATH_MODEL)
    imgPath = [1]
    return jsonify(results=[imgPath])


@app.route("/")
def start():
    return render_template("start.html")


@app.route("/result")
def result():
    url = getPath()
    resultPath = getResultPath()
    return render_template("result.html", url=url, resultPath=resultPath)


@app.route("/index", methods=['post', 'get'])
def main():
    return render_template("index.html")


def getPath():
    path = args.PATH_DATA
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if ("result" in c_path):
            url = "http://"+args.HOST+":5000" + c_path[1:]
            print(url)
            return url


def getResultPath():
    path = args.PATH_RESULTS
    ls = os.listdir(path)
    url = []
    for i in ls:
        c_path = os.path.join(path, i)
        if ("jpg" in c_path):
            url.append("http://"+args.HOST+":5000" + c_path[1:])
    return url


@app.route("/hello")
def hello():
    return render_template("hello.html")


if __name__ == "__main__":
    app.debug = False
    app.run()
