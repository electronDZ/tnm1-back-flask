import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_roberts_filter(image):
    image_data = image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define the Roberts Cross kernels
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    # Apply the kernels to the grayscale image
    img_roberts_x = cv2.filter2D(img, -1, kernel_x)
    img_roberts_y = cv2.filter2D(img, -1, kernel_y)

    # Combine the effects of the x and y kernels
    img_roberts = img_roberts_x + img_roberts_y

    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")

    # Image after applying the Roberts Cross filter
    plt.subplot(1, 2, 2)
    plt.imshow(img_roberts)
    plt.title("Image after Roberts Cross filter")

    plt.show()
