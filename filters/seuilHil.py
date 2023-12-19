import cv2
import numpy as np
import matplotlib.pyplot as plt


def SeuilHys(image, low_threshold=50, high_threshold=100):
    image_data = image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply hysteresis thresholding manually
    high_mask = img > high_threshold
    low_mask = (img > low_threshold) & (img <= high_threshold)
    hyst_img = np.zeros_like(img, dtype=np.uint8)
    hyst_img[high_mask] = 255
    hyst_img[low_mask] = 128

    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    # Thresholded Image
    plt.subplot(1, 2, 2)
    plt.imshow(hyst_img, cmap='gray')
    plt.title("Image after Hysteresis Thresholding")

    plt.show()
