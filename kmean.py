import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
import numpy as np

# Đọc ảnh
image = io.imread(r'D:\KY_8\TRUNG\T4_AI\CODE\K_MEAN\straw1.jpeg')

# Kiểm tra số kênh màu và loại bỏ kênh alpha nếu có
if image.shape[2] == 4:
    image = image[:, :, :3]  # Chỉ giữ lại 3 kênh màu đầu tiên

image_lab = rgb2lab(image)  # Chuyển đổi không gian màu sang LAB

# Chuẩn bị dữ liệu cho K-means
# Reshape ảnh thành một mảng 2D mà mỗi hàng là một điểm ảnh
reshape_img = image_lab.reshape((-1, 3))

# Áp dụng K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(reshape_img)
# Đưa nhãn về hình dạng ban đầu của ảnh
segmented_img = kmeans.labels_.reshape(image.shape[:2])

# Tạo mặt nạ cho bông hoa
# Chọn nhóm 0 làm bông hoa, bạn có thể thử đổi thành 1 nếu kết quả không đúng
flower_mask = segmented_img == 0

# Áp dụng mặt nạ lên ảnh gốc
segmented_flower = np.zeros_like(image)
for i in range(3):  # Áp dụng mặt nạ cho mỗi kênh màu
    segmented_flower[:, :, i] = flower_mask * image[:, :, i]

# Hiển thị kết quả
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Segmented Flower')
plt.imshow(segmented_flower)
plt.axis('off')
plt.show()
