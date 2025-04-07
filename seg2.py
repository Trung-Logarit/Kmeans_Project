import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_hsv(image, k=6, max_iterations=50, tolerance=1e-4, seed=42):
    """
    Apply K-Means clustering in HSV color space using k-means++ initialization.
    """
    np.random.seed(seed)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w, c = hsv_image.shape
    data = hsv_image.reshape((-1, 3)).astype(np.float32)

    # Initialize the first centroid (index 0 for reproducibility)
    first_centroid_index = 0
    centroids = [data[first_centroid_index]]

    for _ in range(k - 1):
        distances = np.sqrt(((data - np.array(centroids)[:, np.newaxis]) ** 2).sum(axis=2))
        min_distances = np.min(distances, axis=0)
        probabilities = min_distances / min_distances.sum()
        centroids.append(data[np.random.choice(data.shape[0], p=probabilities)])
    centroids = np.array(centroids)

    # K-Means iterative updates
    for _ in range(max_iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)
        new_centroids = np.array([
            data[cluster_assignments == i].mean(axis=0) if np.any(cluster_assignments == i)
            else centroids[i]
            for i in range(k)
        ])
        if np.abs(centroids - new_centroids).sum() < tolerance:
            break
        centroids = new_centroids

    cluster_assignments = cluster_assignments.reshape((h, w))
    centroids = centroids.astype(np.uint8)

    return cluster_assignments, centroids, hsv_image

def segment_red_hsv_kmeans(image, k=6):
    """
    Segment the 'red' regions using K-Means in HSV space.
    Then refine the red clusters by checking each pixel's actual Hue & Saturation.
    """
    cluster_assignments, centers, hsv_image = kmeans_hsv(image, k=k, seed=42)
    h, w = cluster_assignments.shape

    # We will build the final mask based on both cluster centroid and per-pixel check
    mask = np.zeros((h, w), dtype=np.uint8)

    # Tách kênh H, S, V để kiểm tra từng pixel
    H, S, V = cv2.split(hsv_image)

    # 1) Xác định cụm nào "có khả năng đỏ" dựa trên tâm cụm (centroid)
    #    (VD: Hue < 15 hoặc Hue > 165, Saturation >= 60)
    possible_red_clusters = []
    for i, (h_val, s_val, v_val) in enumerate(centers):
        if ((h_val <= 15 or h_val >= 165) and s_val >= 60):
            possible_red_clusters.append(i)

    # 2) Kiểm tra từng pixel trong những cụm này: Hue và Saturation thực tế
    #    Hue < 15 hoặc Hue > 165, S >= 60
    for cluster_idx in possible_red_clusters:
        # Pixel thuộc cụm này
        cluster_mask = (cluster_assignments == cluster_idx)
        # Kiểm tra Hue và Saturation thực sự của pixel
        pixel_level_check = ((H <= 15) | (H >= 165)) & (S >= 60)
        # Kết hợp: pixel vừa thuộc cụm, vừa thỏa Hue/Sat => là "đỏ"
        final_check = cluster_mask & pixel_level_check
        mask[final_check] = 255

    # 3) Morphological operations
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((9, 9), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # 4) Tạo overlay
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = rgb_image.copy()
    overlay[closed == 255] = (0, 0, 255) 

    blended = cv2.addWeighted(rgb_image, 0.5, overlay, 0.5, 0)

    return hsv_image, cluster_assignments, centers, mask, closed, rgb_image, blended

def main():
    image_path = r"D:\KY_8\TRUNG\T4_AI\CODE\K_MEAN\straw2.jpeg"  # Cập nhật đường dẫn
    image = cv2.imread(image_path)

    (hsv_image, cluster_assignments, centers, 
     mask, closed, rgb_image, blended) = segment_red_hsv_kmeans(image, k=6)

    # Chuyển một số kết quả sang RGB để hiển thị (nếu cần)
    # Ở đây hiển thị HSV gốc -> tách kênh Hue/Sat/Value trực tiếp
    # Thay vì convert sang RGB, ta sẽ hiển thị kênh H, S, V để quan sát.
    H, S, V = cv2.split(hsv_image)

    plt.figure(figsize=(20, 8))

    plt.subplot(2, 4, 1)
    plt.imshow(rgb_image)
    plt.title("Original Image (RGB)")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(H, cmap='gray')
    plt.title("Hue Channel")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(S, cmap='gray')
    plt.title("Saturation Channel")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(V, cmap='gray')
    plt.title("Value Channel")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    # Hiển thị phân cụm (nếu muốn xem). Ta có centers => segmented image
    segmented_hsv = centers[cluster_assignments.flatten()].reshape(hsv_image.shape)
    segmented_hsv_rgb = cv2.cvtColor(segmented_hsv, cv2.COLOR_HSV2RGB)
    plt.imshow(segmented_hsv_rgb)
    plt.title("Clustered Image (HSV→RGB)")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(mask, cmap='gray')
    plt.title("Initial Mask (Pixel-level check)")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(closed, cmap='gray')
    plt.title("Morphological Cleaned Mask")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.imshow(blended)
    plt.title("Final Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
