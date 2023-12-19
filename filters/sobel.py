import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_sobel_filter_manual(image):
    image_data = image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image to float32 before applying Sobel filter
    img = np.float32(img)

    # Define the Sobel kernels
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Apply the kernels to the grayscale image
    img_sobel_x = cv2.filter2D(img, -1, kernel_x)
    img_sobel_y = cv2.filter2D(img, -1, kernel_y)

    # Combine the effects of the x and y kernels
    img_sobel = cv2.sqrt(cv2.addWeighted(cv2.pow(img_sobel_x, 2.0), 1.0, cv2.pow(img_sobel_y, 2.0), 1.0, 0))

    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    # Image after applying the Sobel filter
    plt.subplot(1, 2, 2)
    plt.imshow(img_sobel, cmap='gray')
    plt.title("Image after Sobel filter")

    plt.show()
