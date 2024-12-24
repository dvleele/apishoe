from flask import Flask, request, jsonify
from db import get_vectors_from_db
from search import extract_vector, get_extract_model
import numpy as np
from PIL import Image
import pyodbc

app = Flask(__name__)

# Hàm kết nối tới cơ sở dữ liệu
def create_connection():
    conn = pyodbc.connect(
        driver='{ODBC Driver 17 for SQL Server}',
        server='DESKTOP-H3EBT9V',
        database='BanHangOnline',
        user='readdata',
        password='123456'
    )
    return conn

# Lấy danh sách mã sản phẩm từ cơ sở dữ liệu
def get_product_ids():
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT maSP FROM sanPham")  # Chỉ lấy mã sản phẩm
    result = cursor.fetchall()
    connection.close()

    product_ids = [row[0] for row in result]  # Trả về danh sách mã sản phẩm
    return product_ids

# API tìm kiếm ảnh
@app.route('/search', methods=['POST'])
def search_image():
    # Kiểm tra file ảnh trong request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Đọc file ảnh từ request (không cần lưu)
    image_file = request.files['image']
    img = Image.open(image_file)

    # Trích xuất vector đặc trưng của ảnh tải lên
    model = get_extract_model()
    search_vector = extract_vector(model, img)

    # Lấy danh sách mã sản phẩm từ cơ sở dữ liệu
    product_ids = get_product_ids()

    # Truy vấn các vector từ cơ sở dữ liệu
    vectors, _ = get_vectors_from_db()

    # Tính khoảng cách từ vector ảnh tìm kiếm đến tất cả các vector trong dataset
    distances = np.linalg.norm(vectors - search_vector, axis=1)

    # Ngưỡng để quyết định có khớp hay khô.venv\Scripts\activateng
    distance_threshold = 1  # Thay đổi giá trị này dựa trên thử nghiệm của bạn

    # Tìm khoảng cách nhỏ nhất và mã sản phẩm tương ứng
    min_distance = np.min(distances)
    if min_distance > distance_threshold:
        return jsonify({"message": "Không tìm thấy sản phẩm"}), 404

    # Lấy 5 mã sản phẩm gần nhất nếu có
    K = 5
    ids = np.argsort(distances)[:K]
    nearest_product_ids = [product_ids[id] for id in ids]

    return jsonify({"product_ids": nearest_product_ids})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
