import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    image_path = r"D:\KY_8\TRUNG\T4_AI\CODE\K_MEAN\straw2.jpeg"  # Cập nhật đường dẫn
    image = cv2.imread(image_path)

    # Chuyển ảnh sang HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tách các kênh màu RGB
    (B, G, R) = cv2.split(image)

    # Tách các kênh màu HSV
    (H, S, V) = cv2.split(hsv_image)

    # Hiển thị 7 ảnh trong 1 khung hình
    plt.figure(figsize=(20, 10))

    # 1) Ảnh gốc RGB
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image (RGB)")
    plt.axis("off")

    # 2) Kênh Red
    plt.subplot(2, 4, 2)
    plt.imshow(R, cmap='gray')
    plt.title("Red Channel")
    plt.axis("off")

    # 3) Kênh Green
    plt.subplot(2, 4, 3)
    plt.imshow(G, cmap='gray')
    plt.title("Green Channel")
    plt.axis("off")

    # 4) Kênh Blue
    plt.subplot(2, 4, 4)
    plt.imshow(B, cmap='gray')
    plt.title("Blue Channel")
    plt.axis("off")

    # 5) Kênh Hue
    plt.subplot(2, 4, 5)
    plt.imshow(H, cmap='gray')
    plt.title("Hue Channel (HSV)")
    plt.axis("off")

    # 6) Kênh Saturation
    plt.subplot(2, 4, 6)
    plt.imshow(S, cmap='gray')
    plt.title("Saturation Channel (HSV)")
    plt.axis("off")

    # 7) Kênh Value
    plt.subplot(2, 4, 7)
    plt.imshow(V, cmap='gray')
    plt.title("Value Channel (HSV)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
