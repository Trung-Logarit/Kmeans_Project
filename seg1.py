import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint

def kmeans(data, k, max_iterations=100, tolerance=1e-4):
    """
    Performs K-Means clustering on the given data with k-means++ initialization.
    """
    # Initialize first centroid randomly
    centroids = [data[randint(0, data.shape[0] - 1)]]

    # Initialize other centroids using k-means++ initialization
    for _ in range(k - 1):
        distances = np.sqrt(((data - np.array(centroids)[:, np.newaxis])**2).sum(axis=2))
        min_distances = np.min(distances, axis=0)
        probabilities = min_distances / min_distances.sum()
        centroids.append(data[np.random.choice(data.shape[0], p=probabilities)])

    centroids = np.array(centroids)

    for _ in range(max_iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)

        new_centroids = np.array([data[cluster_assignments == i].mean(axis=0) for i in range(k)])
        
        if np.abs(centroids - new_centroids).sum() < tolerance:
            break

        centroids = new_centroids

    return cluster_assignments, centroids

def detect_ripe_fruits(image):
    """
    Trả về ảnh result_image với bounding box vẽ quanh quả dâu chín
    và trả về danh sách bounding box [x, y, w, h] của mỗi quả dâu chín.
    """
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = rgb_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    # K-Means
    num_clusters = 6
    labels, centers = kmeans(pixel_values, num_clusters)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(rgb_image.shape)

    # Tạo mask cho vùng đỏ + vàng (tùy ý)
    red_mask = cv2.inRange(segmented_image, (160, 0, 0), (255, 100, 100))
    yellow_mask = cv2.inRange(segmented_image, (150, 150, 0), (255, 255, 100))
    combined_mask = cv2.bitwise_or(red_mask, yellow_mask)

    # Tìm contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc các contour nhỏ
    def remove_small_boxes(cnts, min_area=1500):
        return [cnt for cnt in cnts if cv2.contourArea(cnt) > min_area]

    contours = remove_small_boxes(contours, min_area=1500)

    # Lấy bounding box
    boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # (Tuỳ chọn) gộp các bounding box gần nhau
    def merge_nearby_boxes(boxes, max_distance=10):
        merged_boxes = []
        for box in boxes:
            x, y, w, h = box
            merged = False
            for i, mb in enumerate(merged_boxes):
                mx, my, mw, mh = mb
                # Kiểm tra xem box có gần box đã có trong merged_boxes hay không
                if (abs(x - mx) < max_distance or abs((x + w) - (mx + mw)) < max_distance) \
                   and (abs(y - my) < max_distance or abs((y + h) - (my + mh)) < max_distance):
                    x1 = min(x, mx)
                    y1 = min(y, my)
                    x2 = max(x + w, mx + mw)
                    y2 = max(y + h, my + mh)
                    merged_boxes[i] = [x1, y1, x2 - x1, y2 - y1]
                    merged = True
                    break
            if not merged:
                merged_boxes.append([x, y, w, h])
        return merged_boxes

    merged_boxes = merge_nearby_boxes(boxes, max_distance=10)

    # (Tuỳ chọn) bỏ bounding box nằm gọn bên trong bounding box lớn hơn
    def remove_inner_boxes(boxes):
        outer_boxes = []
        for box in boxes:
            x, y, w, h = box
            is_inner = False
            for ob in outer_boxes:
                ox, oy, ow, oh = ob
                if x >= ox and (x + w) <= (ox + ow) and y >= oy and (y + h) <= (oy + oh):
                    is_inner = True
                    break
            if not is_inner:
                outer_boxes.append(box)
        return outer_boxes

    final_boxes = remove_inner_boxes(merged_boxes)

    # Vẽ bounding box lên ảnh để trực quan
    result_image = rgb_image.copy()
    for (x, y, w, h) in final_boxes:
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_image, 'Ripe Fruit', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    return result_image, final_boxes

def segment_strawberry_roi(image, x, y, w, h):
    """
    Hàm này sẽ phân đoạn (segmentation) chi tiết vùng quả dâu
    trong bounding box [x, y, w, h] bằng cách:
    1. Crop ROI
    2. Chuyển HSV (hoặc dùng K-Means cục bộ)
    3. Tạo mask, dùng morphological operations
    4. Trả về mask đã được canh đúng kích thước với ảnh gốc
    """

    # Cắt ROI từ ảnh gốc (chuyển sang RGB để dễ xử lý)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi = rgb_image[y:y+h, x:x+w]

    # Chuyển ROI sang HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # Ngưỡng màu đỏ trong HSV (có 2 dải cho màu đỏ)
    #  (H: 0-10 hoặc 160-180), S: > 100, V: > 50 (tuỳ chỉnh)
    lower_red1 = np.array([0, 100, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([160, 100, 50], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    # Tạo mask cho 2 dải đỏ
    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # (Tuỳ chọn) Tạo mask cho dải vàng, xanh, v.v. nếu muốn “bắt” thêm vùng dâu chưa chín
    #  ...

    # Dùng morphological operations để lọc nhiễu và lấp lỗ trống
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Ở đây ta chỉ quan tâm mask màu đỏ, nên final_mask = closed
    final_mask_roi = closed

    # Tạo mask full-size cho ảnh gốc, ban đầu toàn 0
    mask_full = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # Gắn ROI mask vào vị trí bounding box
    mask_full[y:y+h, x:x+w] = final_mask_roi

    return mask_full

def main():
    image_path = r"D:\KY_8\TRUNG\T4_AI\CODE\K_MEAN\straw1.jpeg"
    image = cv2.imread(image_path)

    # 1) Lấy bounding box quả dâu chín
    detected_image, final_boxes = detect_ripe_fruits(image)

    # 2) Duyệt qua từng bounding box, phân đoạn chi tiết ROI
    full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for (x, y, w, h) in final_boxes:
        mask_roi = segment_strawberry_roi(image, x, y, w, h)
        full_mask = cv2.bitwise_or(full_mask, mask_roi)  # gộp mask ROI vào mask tổng

    # 3) Tô màu lên ảnh gốc để highlight vùng dâu chín
    #    Tạo 1 lớp overlay, sau đó blend với ảnh gốc
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    overlay = rgb_image.copy()

    # Màu tô (VD: xanh lá)
    overlay[full_mask == 255] = (0, 0, 255)

    # Blend (50% opacity)
    alpha = 0.5
    highlighted = cv2.addWeighted(rgb_image, 1 - alpha, overlay, alpha, 0)

    # 4) Hiển thị kết quả
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(detected_image)
    plt.title('Bounding box quả dâu chín')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(highlighted)
    plt.title('Segmentation Object')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
