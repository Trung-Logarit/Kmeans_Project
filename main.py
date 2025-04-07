import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
import math


def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    for _ in range(max_iters):
        clusters = np.argmin(
            np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)
        new_centroids = np.array(
            [X[clusters == j].mean(axis=0) for j in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters


def find_regions_with_color(image, color):
    regions = []
    rows, cols, _ = image.shape
    visited = np.zeros((rows, cols), dtype=bool)

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and np.array_equal(image[i][j], color):
                top_left = (j, i)
                bottom_right = (j, i)
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    visited[y][x] = True
                    top_left = (min(top_left[0], x), min(top_left[1], y))
                    bottom_right = (
                        max(bottom_right[0], x), max(bottom_right[1], y))
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < cols and 0 <= ny < rows and not visited[ny][nx] and np.array_equal(image[ny][nx], color):
                                stack.append((ny, nx))
                regions.append((top_left, bottom_right))
    return regions


def check_overlap(rect1, rect2):
    # rect1 và rect2 là các tuple (top_left, bottom_right)
    top_left1, bottom_right1 = rect1
    top_left2, bottom_right2 = rect2

    # Kiểm tra xem rect1 có nằm bên trái hoặc bên phải rect2 không
    if bottom_right1[0] < top_left2[0] or top_left1[0] > bottom_right2[0]:
        return False

    # Kiểm tra xem rect1 có nằm phía trên hoặc phía dưới rect2 không
    if bottom_right1[1] < top_left2[1] or top_left1[1] > bottom_right2[1]:
        return False

    # Nếu không thỏa mãn điều kiện trên, tức là hai hình chữ nhật chồng lấn nhau
    return True


def contains(rect1, rect2):
    top_left1, bottom_right1 = rect1
    top_left2, bottom_right2 = rect2
    return (top_left1[0] <= top_left2[0] and top_left1[1] <= top_left2[1] and
            bottom_right1[0] >= bottom_right2[0] and bottom_right1[1] >= bottom_right2[1])


# 1. _____________________________________________Read image____________________________________________________

# Thay đổi đường dẫn ảnh phù hợp
url = r"D:\KY_8\TRUNG\T4_AI\CODE\K_MEAN\straw1.jpeg"
img = io.imread(url)

plt.figure(figsize=(15, 15))
plt.subplot(221)
plt.imshow(img)
plt.title("Original Image")

img_segmented = img.copy()
img_ori = img.copy()

img = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
k = 5

# 2. ________________________________________ kmean clustering ____________________________________________________
centroids, labels = kmeans(img.astype(float), k)

# Chuyển các trung tâm cụm về kiểu uint8 và in ra dãy mã màu
colorArray = [centroids[i].astype("uint8").tolist() for i in range(k)]
print(" ___________________ Dãy mã màu của các trung tâm cụm sau khi áp dụng KMeans:")
for index, color in enumerate(colorArray):
    print(f"Color Segment {index}: {color}")

# 3. ___________________ Thay thế các pixel bằng màu gần với một phần tử mã màu trong colorArray nhất ______________
for i in range(len(img_segmented)):
    for j in range(len(img_segmented[0])):
        pixel = img_segmented[i][j]
        nearest_color_index = labels[i * len(img_segmented[0]) + j]
        img_segmented[i][j] = centroids[nearest_color_index].astype("uint8")

plt.subplot(222)
plt.imshow(img_segmented)
plt.title("Segmented Image")


# 4. ____________________ Xác định mã màu nào trong colorArray gần với màu đỏ nhất (thay đổi tùy theo quả) ________
red_color = [255, 0, 0]
closest_color_index = None
closest_color_distance = float('inf')

for index, color in enumerate(colorArray):
    # Tính khoảng cách Euclid giữa mỗi màu trong colorArray và màu đỏ
    distance = math.sqrt(sum([(x - y) ** 2 for x, y in zip(color, red_color)]))
    # So sánh khoảng cách với khoảng cách gần nhất đã biết
    if distance < closest_color_distance:
        closest_color_distance = distance
        closest_color_index = index

closest_color = colorArray[closest_color_index]
print(f"Mã màu gần với màu đỏ nhất: Color Segment {
      closest_color_index}: {closest_color}")

# 5. __________________ Tìm trong ảnh đã được phân vùng, vùng nào có màu giống màu đỏ nhất (trả về tọa độ) ________
regions = find_regions_with_color(img_segmented, closest_color)

# 6. ___________________ Xử lí các tọa độ gây chồng lấn khi trước khi vẽ rectangle ________________________________
combined_rectangle = []

for i, rect1 in enumerate(regions):
    for j, rect2 in enumerate(regions[i + 1:]):
        if check_overlap(rect1, rect2):
            next_overlap = True
            # Hai hình chữ nhật chồng lấn nhau
            # Tạo hình chữ nhật lớn hơn bao bọc cả hai
            combined_top_left = (
                min(rect1[0][0], rect2[0][0]), min(rect1[0][1], rect2[0][1]))
            combined_bottom_right = (
                max(rect1[1][0], rect2[1][0]), max(rect1[1][1], rect2[1][1]))
            combined_rectangle.append(
                (combined_top_left, combined_bottom_right))

# Tìm và loại bỏ các hình chữ nhật nhỏ bên trong hình chữ nhật lớn
filtered_rectangles = []

for rect in combined_rectangle:
    if not any(contains(other_rect, rect) for other_rect in combined_rectangle if other_rect != rect):
        filtered_rectangles.append(rect)

# Vẽ các hình chữ nhật đã lọc
for top_left, bottom_right in filtered_rectangles:
    cv2.rectangle(img_ori, top_left, bottom_right, (0, 255, 0), thickness=3)


plt.subplot(223)
plt.imshow(img_ori)
plt.title("Detected Image")

plt.show()
