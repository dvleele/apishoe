import numpy as np
from PIL import Image
from search import get_extract_model, extract_vector
from db import get_vectors_from_db


# Hàm load tập dữ liệu kiểm thử
def load_test_images_and_labels():
    """
    Load ảnh kiểm thử và ground truth từ file.
    Returns:
        list of PIL.Image: Danh sách ảnh kiểm thử.
        list of str: Danh sách mã sản phẩm đúng.
    """
    test_images = []
    ground_truths = []

    # Đọc file labels.txt để load ảnh và mã sản phẩm đúng
    with open("labels.txt", "r") as f:
        for line in f:
            image_name, product_id = line.strip().split()
            img = Image.open(f"test_images/{image_name}")
            test_images.append(img)
            ground_truths.append(product_id)

    return test_images, ground_truths


# Hàm kiểm thử hệ thống
def evaluate_system(model, test_images, ground_truths, database_vectors, product_ids, top_k=5):
    """
    Kiểm thử hệ thống tìm kiếm ảnh.
    Args:
        model: Mô hình trích xuất đặc trưng.
        test_images (list of PIL.Image): Danh sách ảnh kiểm thử.
        ground_truths (list of str): Danh sách mã sản phẩm đúng.
        database_vectors (numpy.ndarray): Vector đặc trưng của sản phẩm trong database.
        product_ids (list of str): Mã sản phẩm tương ứng với các vector.
        top_k (int): Số kết quả trả về để kiểm tra.
    Returns:
        float: Độ chính xác (accuracy) của hệ thống.
    """
    correct_predictions = 0
    total_images = len(test_images)

    for img, true_product_id in zip(test_images, ground_truths):
        # Trích xuất vector đặc trưng từ ảnh kiểm thử
        search_vector = extract_vector(model, img)

        # Tính khoảng cách từ vector ảnh kiểm thử đến vector trong cơ sở dữ liệu
        distances = np.linalg.norm(database_vectors - search_vector, axis=1)
        nearest_ids = np.argsort(distances)[:top_k]
        nearest_products = [product_ids[i] for i in nearest_ids]

        # Kiểm tra mã sản phẩm đúng có nằm trong top_k kết quả không
        if true_product_id in nearest_products:
            correct_predictions += 1

    accuracy = correct_predictions / total_images
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# Chạy kiểm thử
if __name__ == "__main__":
    # Load dữ liệu kiểm thử
    test_images, ground_truths = load_test_images_and_labels()

    # Lấy vector và mã sản phẩm từ cơ sở dữ liệu
    database_vectors, product_ids = get_vectors_from_db()

    # Load mô hình trích xuất đặc trưng
    model = get_extract_model()

    # Kiểm thử hệ thống
    evaluate_system(model, test_images, ground_truths, database_vectors, product_ids, top_k=5)
