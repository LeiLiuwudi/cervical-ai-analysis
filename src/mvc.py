from flask import Flask, request
from flask_cors import cross_origin
import pymysql
from tfidf import tfidf
from cosine import cosine

app = Flask(__name__)


@app.route('/similarRecord', methods=['POST'])
@cross_origin()
def similarRecord():
    id = request.form.get("id")
    # 打开数据库连接
    db = pymysql.connect("10.115.113.58", "root", "lab205", "cervical_spondy_medical", charset="utf8")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    # 使用 execute()  方法执行 SQL 查询
    cursor.execute("select chief_complaint,present_history,past_history from patient where id=" + str(id))
    original_data = cursor.fetchall()
    cursor.execute("select chief_complaint,present_history,past_history from patient where id!=" + str(id))
    docs_data = cursor.fetchall()
    base = original_data[0][0] + "," + original_data[0][1] + "," + original_data[0][2]
    docs = []
    for each in docs_data:
        docs.append(each[0] + "," + each[1] + "," + each[2])
    print(tfidf(docs, base))
    print(cosine(docs, base))
    return {"id": 1}


if __name__ == '__main__':
    app.run()
