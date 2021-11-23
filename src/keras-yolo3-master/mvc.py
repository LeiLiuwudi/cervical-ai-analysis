import base64
import json
import sys
import datetime

import cv2
import pymysql
from PIL import Image
from flask import Flask, request, make_response, jsonify, Response
from flask_cors import *

from cosine import cosine
from yolo import predict, YOLO, getBaseMse, getEuclideanDistance, cutImage
from cos_staut import cos_staut

app = Flask(__name__)
diseases = [
    "正常",
    "颈椎疲劳",
    "颈椎劳损",
    "颈椎间盘突出",
    "颈椎强行性病变"
]
yolo = YOLO()
base_array = getBaseMse("base.jpg", yolo)


@app.route('/similarRecord', methods=["POST"])
@cross_origin()
def similarRecord():
    # id = request.form.get("id")
    data = json.loads(request.get_data(as_text=True))
    id = data["id"]
    print(request)
    print(id)
    # 打开数据库连接
    db = pymysql.connect("10.115.113.58", "root", "lab205", "cervical_spondy_medical", charset="utf8")

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    # 使用 execute()  方法执行 SQL 查询
    cursor.execute("select chief_complaint,present_history,past_history from patient where id=" + str(id))
    original_data = cursor.fetchall()
    cursor.execute("select chief_complaint,present_history,past_history,id from patient where id!=" + str(id))
    docs_data = cursor.fetchall()
    now_time = str(datetime.datetime.now()).split('.')[0]
    now_time = datetime.datetime.strptime(now_time, '%Y-%m-%d %H:%M:%S')
    sql = "insert into analysis (function, create_time) values('%s', '%s')" % ("电子病历对比分析", now_time)
    print(sql)
    cursor.execute(sql)
    db.commit()
    base = original_data[0][0] + "," + original_data[0][1] + "," + original_data[0][2]
    docs = []
    for each in docs_data:
        docs.append(each[0] + "," + each[1] + "," + each[2])
    result = {
        "id": docs_data[cosine(docs, base)][3],
        "name": "liulei"
    }
    resp = make_response(jsonify(result))
    # resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'POST'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp

@app.route('/cutResult', methods=["POST"])
@cross_origin()
def cutResult():
    data = json.loads(request.get_data(as_text=True))
    id = data["id"]

    db = pymysql.connect("10.115.113.58", "root", "lab205", "cervical_spondy_medical", charset="utf8")
    cursor = db.cursor()
    cursor.execute("select infrared_path from recognize where id=" + str(id))
    recognize_data = cursor.fetchall()
    infrared_path = recognize_data[0][0]
    print(infrared_path)
    croped_region = cutImage(infrared_path, yolo)
    base64_str = cv2.imencode('.jpg',croped_region)[1].tostring()
    base64_str = base64.b64encode(base64_str).decode()
    # cut_image = croped_region.tobytes()
    # img_stream = base64.b64encode(cut_image).decode()

    result = {
        "result": base64_str
    }
    resp = make_response(jsonify(result))
    # resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'POST'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp

@app.route('/recognizeResult', methods=["POST"])
@cross_origin()
def recognizeResult():
    data = json.loads(request.get_data(as_text=True))
    id = data["id"]

    db = pymysql.connect("10.115.113.58", "root", "lab205", "cervical_spondy_medical", charset="utf8")
    cursor = db.cursor()
    cursor.execute("select infrared_path from recognize where id=" + str(id))
    recognize_data = cursor.fetchall()
    infrared_path = recognize_data[0][0]
    print(infrared_path)
    croped_region = cutImage(infrared_path, yolo)
    result = predict(croped_region)
    sql = "update recognize set recognize_result='" + diseases[result] + "' where id=" + str(id)
    print(sql)
    cursor.execute(sql)
    db.commit()
    now_time = str(datetime.datetime.now()).split('.')[0]
    now_time = datetime.datetime.strptime(now_time, '%Y-%m-%d %H:%M:%S')
    sql = "insert into analysis (function, create_time) values('%s', '%s')" % ("病种识别结果分析", now_time)
    print(sql)
    cursor.execute(sql)
    db.commit()
    result = {
        "result": str(result)
    }
    resp = make_response(jsonify(result))
    # resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'POST'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp


@app.route('/effectEvaluation', methods=["POST"])
@cross_origin()
def effectEvaluation():
    data = json.loads(request.get_data(as_text=True))
    id = data["id"]

    db = pymysql.connect(host = "10.115.113.58", user = "root", password = "lab205", database = "cervical_spondy_medical", charset="utf8")
    cursor = db.cursor()
    cursor.execute("select infrared_path from recognize where patient_id=" + str(id))
    path_list = cursor.fetchall()
    print(path_list)

    distanceList = []
    for ele in path_list:
        print(ele)
        try:
            distance = getEuclideanDistance(ele[0], yolo, base_array)
            distanceList.append(distance)
        except IndexError:
            distanceList.append(sys.maxsize)
    trend = cos_staut(distanceList)
    result = {
        "result": distanceList,
        "count": len(path_list),
        "trend": trend
    }

    now_time = str(datetime.datetime.now()).split('.')[0]
    now_time = datetime.datetime.strptime(now_time, '%Y-%m-%d %H:%M:%S')
    sql = "insert into analysis (function, evaluation_result, patient_id, create_time) values('%s', '%s', '%d', '%s')" % (
    "治疗效果评估", json.dumps(result), id, now_time)
    print(sql)
    cursor.execute(sql)
    record_id = db.insert_id()
    db.commit()

    result["record_id"] = record_id
    resp = make_response(jsonify(result))
    # resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'POST'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp


if __name__ == '__main__':
    app.run(host="0.0.0.0")
