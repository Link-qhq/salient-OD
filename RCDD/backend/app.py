import base64
import datetime
import io
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # 允许跨域
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# 模拟用户数据库
users = {
    "admin": {"password": "admin123"}
}


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if users.get(username) and users[username]['password'] == password:
        print("登录成功")
    response_data = {
        "code": 200,
        "msg": "Success",
        "data": {
        }
    }
    return jsonify(response_data)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    # 检查文件是否存在
    if 'image' not in request.files:
        return jsonify({"code": 400, "msg": "未接收到文件"}), 400

    file = request.files['image']

    # 验证文件名
    if file.filename == '':
        return jsonify({"code": 400, "msg": "无效文件名"}), 400

    # 验证文件类型
    if not allowed_file(file.filename):
        return jsonify({"code": 400, "msg": "不支持的文件类型"}), 400

    try:
        # 安全保存文件
        # img_base64 = process_image(file)
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        # 调用检测函数（示例）
        # result = process_image(save_path)
        img_base64 = load_image(file.filename)
        print(img_base64)
        return jsonify({
            "code": 200,
            "msg": "检测成功",
            "data": {
                # 'result_image': f"data:image/jpeg;base64,{img_base64}"
                'result_image': img_base64
            }
        })
    except Exception as e:
        app.logger.error(f'处理失败: {str(e)}')
        return jsonify({"code": 500, "msg": "服务器处理错误"}), 500


@app.route('/api/result', methods=['POST'])
def get_upload_image():
    """返回原始图片"""
    data = request.get_json()
    # return send_from_directory(
    #     app.config['UPLOAD_FOLDER'],
    #     data['name'],
    #     mimetype='image/jpeg'  # 根据实际类型调整
    # )


def process_image(file):
    # 接收文件（保持原有验证逻辑）
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # 执行检测处理（示例：添加红色矩形框）
    processed_img = img.copy()
    cv2.rectangle(processed_img, (100, 100), (300, 300), (0, 0, 255), 2)

    # 将图片转为base64
    _, buffer = cv2.imencode('.jpg', processed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64
    # return jsonify({
    #     "code": 200,
    #     "msg": "检测成功",
    #     "data": {
    #         "result_image": f"data:image/jpeg;base64,{img_base64}",
    #         "analysis": {
    #             "defect_type": "裂纹",
    #             "confidence": 0.92
    #         }
    #     }
    # })


def load_image(filename):
    try:
        # 本地图片路径（示例图片）
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 验证图片是否存在
        if not os.path.exists(image_path):
            return jsonify({"code": 404, "msg": "图片不存在"}), 404

        # 读取图片并编码
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # 获取文件类型
        file_ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        mime_type = f"image/{file_ext}" if file_ext != "jpg" else "image/jpeg"

        return f"data:{mime_type};base64,{encoded_string}"
        # return jsonify({
        #     "code": 200,
        #     "data": {
        #         "image": f"data:{mime_type};base64,{encoded_string}",
        #         "width": 800,  # 可选：实际图片尺寸
        #         "height": 600
        #     }
        # })

    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
