import pyodbc
import numpy as np
import pickle


# Hàm kết nối tới SQL Server
def create_connection():
    conn = pyodbc.connect(
        driver='{ODBC Driver 17 for SQL Server}',  # Tên driver ODBC
        server='DESKTOP-H3EBT9V',  # Tên máy chủ (hoặc localhost)
        database='BanHangOnline',  # Tên cơ sở dữ liệu
        user='readdata',  # Tên đăng nhập
        password='123456'  # Mật khẩu
    )
    return conn


# Lấy đường dẫn ảnh và product_id từ cơ sở dữ liệu
def get_image_paths_and_product_ids():
    connection = create_connection()
    if connection is None:
        return [], []
    cursor = connection.cursor()
    cursor.execute("SELECT hinhDD, maSP FROM sanPham")  # Truy vấn đường dẫn ảnh và mã sản phẩm
    result = cursor.fetchall()
    connection.close()

    paths = [row[0] for row in result]  # Đường dẫn ảnh
    product_ids = [row[1] for row in result]  # Mã sản phẩm
    return paths, product_ids


# Lưu vector đặc trưng và product_id vào cơ sở dữ liệu
def save_vectors_to_db(vectors, paths, product_ids):
    connection = create_connection()
    cursor = connection.cursor()

    for vector, path, product_id in zip(vectors, paths, product_ids):
        # Chuyển vector thành nhị phân (BLOB)
        vector_blob = vector.tobytes()

        # Lưu ảnh và vector vào bảng image_features, kèm theo maSP (mã sản phẩm)
        cursor.execute(
            "INSERT INTO image_features (image_path, feature_vector, product_id) VALUES (?, ?, ?)",
            (path, vector_blob, product_id)
        )

    connection.commit()
    connection.close()


# Truy vấn các vector từ cơ sở dữ liệu và lấy thông tin sản phẩm
def get_vectors_from_db():
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT feature_vector, product_id FROM image_features")  # Truy vấn cả vector và maSP
    result = cursor.fetchall()
    connection.close()

    vectors = [np.frombuffer(row[0], dtype=np.float32) for row in
               result]  # Chuyển đổi dữ liệu nhị phân thành numpy array
    product_ids = [row[1] for row in result]  # Lấy maSP (mã sản phẩm)
    return np.array(vectors), product_ids
