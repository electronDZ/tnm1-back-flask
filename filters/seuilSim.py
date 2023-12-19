import cv2
import numpy as np
import matplotlib.pyplot as plt

def SeuilSim(image, threshold=128):
    image_data = image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply simple thresholding manually
    thresh_img = np.where(img > threshold, 255, 0).astype(np.uint8)

    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    # Thresholded Image
    plt.subplot(1, 2, 2)
    plt.imshow(thresh_img, cmap='gray')
    plt.title("Image after Simple Thresholding")

    plt.show()
