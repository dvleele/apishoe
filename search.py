import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image

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
