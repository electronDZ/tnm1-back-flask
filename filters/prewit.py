import cv2
import numpy as np
import matplotlib.pyplot as plt

def prewitt(image):
    image_data = image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the Prewitt kernels
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Apply the kernels to the grayscale image
    img_prewitt_x = cv2.filter2D(img, -1, kernel_x)
    img_prewitt_y = cv2.filter2D(img, -1, kernel_y)

    # Combine the effects of the x and y kernels
    img_prewitt = img_prewitt_x + img_prewitt_y

    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    # Image after applying the Prewitt filter
    plt.subplot(1, 2, 2)
    plt.imshow(img_prewitt, cmap='gray')
    plt.title("Image after Prewitt filter")

    plt.show()
