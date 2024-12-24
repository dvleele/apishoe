import os
import numpy as np
import pyodbc
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

app = Flask(__name__)

# ====== Database Connection ======
def create_connection():
    try:
        conn = pyodbc.connect(
            driver='{SQL Server}',
            server='SQL1001.site4now.net,1433',
            database='db_ab0ecd_admin',
            user='db_ab0ecd_admin_admin',
            password='Dinh23032003'
        )
        print("Kết nối thành công!")
        return conn
    except Exception as e:
        print(f"Không thể kết nối đến SQL Server: {e}")
        return None

# ====== VGG16 Model Setup ======
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model

def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, img):
    img_tensor = image_preprocess(img)
    vector = model.predict(img_tensor)[0]
    vector = vector / np.linalg.norm(vector)
    return vector

# ====== Database Operations ======
def get_product_ids():
    connection = create_connection()
    if not connection:
        return []
    cursor = connection.cursor()
    cursor.execute("SELECT maSP FROM sanPham")
    result = cursor.fetchall()
    connection.close()
    return [row[0] for row in result]

def get_vectors_from_db():
    connection = create_connection()
    if not connection:
        return np.array([]), []
    cursor = connection.cursor()
    cursor.execute("SELECT feature_vector, product_id FROM image_features")
    result = cursor.fetchall()
    connection.close()

    vectors = [np.frombuffer(row[0], dtype=np.float32) for row in result]
    product_ids = [row[1] for row in result]
    return np.array(vectors), product_ids

# ====== Search API ======
@app.route('/search', methods=['POST'])
def search_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    img = Image.open(image_file)

    model = get_extract_model()
    search_vector = extract_vector(model, img)

    product_ids = get_product_ids()
    vectors, _ = get_vectors_from_db()

    if vectors.size == 0:
        return jsonify({"error": "No vectors available in database"}), 500

    distances = np.linalg.norm(vectors - search_vector, axis=1)
    distance_threshold = 1.0

    min_distance = np.min(distances)
    if min_distance > distance_threshold:
        return jsonify({"message": "Không tìm thấy sản phẩm"}), 404

    K = 5
    ids = np.argsort(distances)[:K]
    nearest_product_ids = [product_ids[id] for id in ids]

    return jsonify({"product_ids": nearest_product_ids})

if __name__ == '__main__':
    app.run()
