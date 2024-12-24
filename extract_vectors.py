import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import pyodbc

# Tạo mô hình VGG16 để trích xuất đặc trưng ảnh
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model

# Tiền xử lý ảnh: resize ảnh, chuyển đổi thành tensor và chuẩn hóa
def image_preprocess(img):
    img = img.resize((224, 224))  # VGG16 yêu cầu kích thước 224x224
    img = img.convert("RGB")  # Đảm bảo ảnh là 3 kênh màu (RGB)
    x = image.img_to_array(img)  # Chuyển ảnh thành mảng numpy
    x = np.expand_dims(x, axis=0)  # Thêm một chiều cho batch (vì VGG16 yêu cầu input 4 chiều)
    x = preprocess_input(x)  # Chuẩn hóa ảnh (theo cách mà VGG16 yêu cầu)
    return x

# Trích xuất vector đặc trưng từ một ảnh
def extract_vector(model, img):
    img_tensor = image_preprocess(img)  # Tiền xử lý ảnh
    vector = model.predict(img_tensor)[0]  # Dự đoán đặc trưng (vector) của ảnh
    vector = vector / np.linalg.norm(vector)  # Chuẩn hóa vector theo chuẩn L2
    return vector

# Hàm kết nối tới SQL Server
def create_connection():
    try:
        conn = pyodbc.connect(
            driver='{ODBC Driver 17 for SQL Server}',  # Tên driver ODBC
            server='DESKTOP-H3EBT9V',  # Tên máy chủ (hoặc localhost)
            database='BanHangOnline',  # Tên cơ sở dữ liệu
            user='readdata',  # Tên đăng nhập
            password='123456'  # Mật khẩu
        )
        print("Kết nối thành công!")
        return conn
    except Exception as e:
        print(f"Không thể kết nối đến SQL Server: {e}")
        return None

# Lấy đường dẫn ảnh và mã sản phẩm từ bảng sanPham
def get_image_paths_and_ids_from_db():
    connection = create_connection()
    if connection is None:
        return [], []

    cursor = connection.cursor()
    # Truy vấn cột maSP và hinhDD
    cursor.execute("SELECT maSP, hinhDD FROM sanPham")
    result = cursor.fetchall()
    connection.close()

    product_ids = [row[0] for row in result]  # Lấy danh sách maSP
    image_paths = [row[1] for row in result]  # Lấy danh sách đường dẫn ảnh
    return product_ids, image_paths

# Lưu vector đặc trưng vào cơ sở dữ liệu
def save_vectors_to_db(vectors, paths, product_ids):
    connection = create_connection()
    if connection is None:
        return

    cursor = connection.cursor()

    for vector, path, product_id in zip(vectors, paths, product_ids):
        try:
            # Chuyển vector thành nhị phân
            vector_blob = vector.tobytes()
            cursor.execute(
                "INSERT INTO image_features (image_path, feature_vector, product_id) VALUES (?, ?, ?)",
                (path, vector_blob, product_id)
            )
        except Exception as e:
            print(f"Lỗi khi lưu vector vào cơ sở dữ liệu: {e}")

    connection.commit()
    connection.close()

# Duyệt qua các ảnh từ đường dẫn trong SQL và lưu vector vào cơ sở dữ liệu
def process_and_store_images():
    model = get_extract_model()

    vectors = []
    paths = []

    # Lấy maSP và đường dẫn ảnh từ SQL
    product_ids, image_paths = get_image_paths_and_ids_from_db()

    base_dir = r"D:\NTTU\LapTrinhWeb\Nộp bài\23DTH2B_LTW2_Nhom3_LeChiDinh[2100011621]\WEBSHOE\WEBSHOE"

    for product_id, image_path in zip(product_ids, image_paths):
        full_image_path = os.path.join(base_dir, image_path.replace('/', '\\'))

        if not os.path.exists(full_image_path):
            print(f"Lỗi: Không tìm thấy ảnh {full_image_path}")
            continue

        try:
            img = Image.open(full_image_path)
            vector = extract_vector(model, img)
            vectors.append(vector)
            paths.append(image_path)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_path}: {e}")

    # Lưu các vector, đường dẫn và maSP vào cơ sở dữ liệu
    save_vectors_to_db(vectors, paths, product_ids)

# Gọi hàm để trích xuất và lưu vector vào cơ sở dữ liệu
if __name__ == "__main__":
    process_and_store_images()
