import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
import math

def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    for _ in range(max_iters):
        # Tính toán gán cụm
        clusters = np.argmin(
            np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)
        # Tính lại tâm cụm
        new_centroids = np.array(
            [X[clusters == j].mean(axis=0) for j in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

def find_regions_with_color(segmented_img, target_color, tolerance=0):
    """
    Tìm các vùng có màu trùng với target_color (nếu tolerance=0),
    hoặc gần với target_color trong khoảng tolerance > 0 (nếu cần).
    Ở đây giữ nguyên so khớp 100% như code gốc.
    """
    regions = []
    rows, cols, _ = segmented_img.shape
    visited = np.zeros((rows, cols), dtype=bool)

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j]:
                pixel = segmented_img[i][j]
                if np.array_equal(pixel, target_color):
                    top_left = (j, i)
                    bottom_right = (j, i)
                    stack = [(i, j)]
                    while stack:
                        y, x = stack.pop()
                        visited[y][x] = True
                        top_left = (min(top_left[0], x), min(top_left[1], y))
                        bottom_right = (max(bottom_right[0], x), max(bottom_right[1], y))
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                nx, ny = x + dx, y + dy
                                if (0 <= nx < cols and 0 <= ny < rows and 
                                    not visited[ny][nx] and 
                                    np.array_equal(segmented_img[ny][nx], target_color)):
                                    stack.append((ny, nx))
                    regions.append((top_left, bottom_right))
    return regions

def check_overlap(rect1, rect2):
    top_left1, bottom_right1 = rect1
    top_left2, bottom_right2 = rect2

    # Kiểm tra xem rect1 có nằm bên trái hoặc bên phải rect2 không
    if bottom_right1[0] < top_left2[0] or top_left1[0] > bottom_right2[0]:
        return False
    # Kiểm tra xem rect1 có nằm phía trên hoặc phía dưới rect2 không
    if bottom_right1[1] < top_left2[1] or top_left1[1] > bottom_right2[1]:
        return False
    return True

def contains(rect1, rect2):
    top_left1, bottom_right1 = rect1
    top_left2, bottom_right2 = rect2
    return (top_left1[0] <= top_left2[0] and top_left1[1] <= top_left2[1] and
            bottom_right1[0] >= bottom_right2[0] and bottom_right1[1] >= bottom_right2[1])

# Đọc ảnh
url = r"D:\KY_8\TRUNG\T4_AI\CODE\K_MEAN\straw2.jpeg"
img = io.imread(url)

plt.figure(figsize=(15, 15))
plt.subplot(221)
plt.imshow(img)
plt.title("Original Image")

# Sao lưu ảnh
img_segmented = img.copy()
img_ori = img.copy()

# Reshape ảnh để đưa vào KMeans
h, w, c = img.shape
flat_img = img.reshape((h * w, c))
k = 5

# Thực hiện KMeans
centroids, labels = kmeans(flat_img.astype(float), k)

# Mảng màu của các cụm
colorArray = [centroids[i].astype("uint8").tolist() for i in range(k)]
print(" ___________________ Dãy mã màu của các trung tâm cụm sau khi áp dụng KMeans:")
for index, color in enumerate(colorArray):
    print(f"Color Segment {index}: {color}")

# Tạo ảnh phân đoạn
for i in range(h):
    for j in range(w):
        label_index = labels[i * w + j]
        img_segmented[i][j] = centroids[label_index].astype("uint8")

plt.subplot(222)
plt.imshow(img_segmented)
plt.title("Segmented Image")

# Tìm tất cả cụm có màu gần với [255, 0, 0] dưới ngưỡng distance
red_color = np.array([255, 0, 0])
threshold = 180  # tuỳ chỉnh, bạn có thể thử 130-200
red_like_indices = []
for index, color in enumerate(colorArray):
    color_np = np.array(color)
    distance = np.linalg.norm(color_np - red_color)
    if distance < threshold:
        red_like_indices.append(index)

print(f"Các cụm (index) có màu gần đỏ với ngưỡng {threshold}: {red_like_indices}")

# Lấy tất cả region từ các cụm “đỏ”
all_regions = []
for idx in red_like_indices:
    # Lấy màu của cụm idx
    cluster_color = colorArray[idx]
    # Tìm vùng có màu cụm này
    regions = find_regions_with_color(img_segmented, cluster_color)
    all_regions.extend(regions)

# Gộp các bounding box bị chồng lấn
combined_rectangle = []
all_regions = list(set(all_regions))  # loại bỏ trùng lặp nếu có

for i in range(len(all_regions)):
    rect1 = all_regions[i]
    overlapped = False
    for j in range(len(combined_rectangle)):
        rect2 = combined_rectangle[j]
        if check_overlap(rect1, rect2):
            # Gộp rect1 và rect2
            top_left = (min(rect1[0][0], rect2[0][0]), 
                        min(rect1[0][1], rect2[0][1]))
            bottom_right = (max(rect1[1][0], rect2[1][0]), 
                            max(rect1[1][1], rect2[1][1]))
            combined_rectangle[j] = (top_left, bottom_right)
            overlapped = True
            break
    if not overlapped:
        combined_rectangle.append(rect1)

# Lọc bỏ những hình chữ nhật nằm hoàn toàn bên trong hình chữ nhật lớn hơn
filtered_rectangles = []
for rect in combined_rectangle:
    if not any(contains(other_rect, rect) for other_rect in combined_rectangle if other_rect != rect):
        filtered_rectangles.append(rect)

# Vẽ bounding box
for top_left, bottom_right in filtered_rectangles:
    cv2.rectangle(img_ori, top_left, bottom_right, (0, 255, 0), thickness=3)

plt.subplot(223)
plt.imshow(img_ori)
plt.title("Detected Image")
plt.show()
