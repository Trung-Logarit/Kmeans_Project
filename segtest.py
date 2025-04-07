import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_hsv_channels(image):
    """
    Chuyển ảnh từ BGR sang HSV và hiển thị từng kênh H, S, V.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(h, cmap='gray')
    plt.title("Hue Channel")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(s, cmap='gray')
    plt.title("Saturation Channel")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(v, cmap='gray')
    plt.title("Value Channel")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    image_path = r"D:\KY_8\TRUNG\T4_AI\CODE\K_MEAN\straw2.jpeg"  # Cập nhật đường dẫn ảnh của bạn
    image = cv2.imread(image_path)
    
    # Hiển thị các kênh HSV
    display_hsv_channels(image)

if __name__ == "__main__":
    main()
