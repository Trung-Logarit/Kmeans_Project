import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Đọc ảnh
    image_path = r"D:\KY_8\TRUNG\T4_AI\CODE\K_MEAN\straw2.jpeg"  # Cập nhật đường dẫn
    image = cv2.imread(image_path)

    # Chuyển ảnh sang HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # In ra chi tiết về kích thước ảnh HSV
    h, w, c = hsv_image.shape
    print(f"Chiều cao (h): {h}, Chiều rộng (w): {w}, Số kênh (c): {c}")
    
    # Chuyển đổi ảnh HSV thành dạng dữ liệu cho K-Means (reshaping)
    data = hsv_image.reshape((-1, 3)).astype(np.float32)
    print(f"Dữ liệu sau khi reshaping có dạng: {data.shape}")

    # Hiển thị ảnh gốc và ảnh HSV
    plt.figure(figsize=(20, 10))

    # 1) Ảnh RGB gốc
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image (RGB)")
    plt.axis("off")

    # 2) Ảnh HSV
    plt.subplot(2, 4, 2)
    plt.imshow(hsv_image)
    plt.title("HSV Image")
    plt.axis("off")

    # Tách các kênh H, S, V từ ảnh HSV
    H, S, V = cv2.split(hsv_image)

    # 3) Kênh Hue (H)
    plt.subplot(2, 4, 3)
    plt.imshow(H, cmap='gray')
    plt.title("Hue Channel (HSV)")
    plt.axis("off")

    # 4) Kênh Saturation (S)
    plt.subplot(2, 4, 4)
    plt.imshow(S, cmap='gray')
    plt.title("Saturation Channel (HSV)")
    plt.axis("off")

    # 5) Kênh Value (V)
    plt.subplot(2, 4, 5)
    plt.imshow(V, cmap='gray')
    plt.title("Value Channel (HSV)")
    plt.axis("off")

    # In ra dữ liệu cho K-Means
    print("\nDữ liệu được chuẩn bị cho K-Means (mảng 2 chiều):")
    print(data[:7])  # In 7 dòng đầu tiên để kiểm tra

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
