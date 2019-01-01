from flask import Flask, jsonify, render_template, request
from core import app as model

import os
import base64

# for windows
# HOST = '127.0.0.3'

# for mac
HOST = '127.0.0.1'

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


@app.route("/filePath", methods=['post', 'get'])
def filePath():
    data = request.json

    print(data)
    model.main(int(data[0]), int(data[1]), int(data[2]), int(data[3]), data[4])
    imgPath = [1]
    return jsonify(results=[imgPath])


@app.route("/photograph", methods=['post'])
def photograph():
    data = request.json

    imgurl = data[5]
    dataPath = imgurl[22:]
    imagedata = base64.b64decode(dataPath)
    if (os.path.exists('./static/data/touxiang.jpg')):
        os.remove("./static/data/touxiang.jpg")
    file = open('./static/data/touxiang.jpg', "wb")
    file.write(imagedata)
    file.close()
    data[4] = './static/data/touxiang.jpg'
    model.main(int(data[0]), int(data[1]), int(data[2]), int(data[3]), './static/data/touxiang.jpg')
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
    path = "./static/data/"
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if ("jpg" in c_path):
            url = "http://127.0.0.1:5000" + c_path[1:]
            print(url)
            return url


def getResultPath():
    path = "./static/results/"
    ls = os.listdir(path)
    url = []
    for i in ls:
        c_path = os.path.join(path, i)
        if ("jpg" in c_path):
            url.append("http://"+HOST+":5000" + c_path[1:])
    return url


@app.route("/hello")
def hello():
    return render_template("hello.html")


if __name__ == "__main__":
    app.debug = False
    app.run(host=HOST)
