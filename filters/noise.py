import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.image_to_rgb import image_to_rgb_convertor


def apply_gaussian_noise(image, mean=7, sigma=50):
    """Apply Gaussian noise to an image."""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)  # Clip values to the valid range [0, 255]
    return noisy.astype(np.uint8)


def apply_papper_and_salt_noise(image, pepper_prob=0.2, salt_prob=0.2):
    """Apply salt-and-pepper noise to an image."""
    noisy = np.copy(image)

    # Salt noise
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    noisy[salt_mask] = 255

    # Pepper noise
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy[pepper_mask] = 0

    return noisy.astype(np.uint8)


def noise_function0(image, mean=0, sigma=50):
    """Load an image, display the original and noisy versions."""
    # Load an image
    image_data = image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # Apply Gaussian noise
    noisy_image = apply_gaussian_noise(original_image, mean, sigma)

    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), cmap="gray")
    plt.title("Original Image")

    # Noisy Image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB), cmap="gray")
    plt.title("Noisy Image")

    plt.show()


# # Example usage
# image = "path/to/your/image.jpg"
# noise_function(image)


# papper and salt
def noise_function5(image, pepper_prob=0.2, salt_prob=0.2):
    image_data = image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    noisy_image = apply_papper_and_salt_noise(original_image, pepper_prob, salt_prob)
    return noisy_image
    # Display the original and noisy grayscale images side by side using matplotlib
    # plt.figure(figsize=(10, 5))

    # Original Grayscale Image
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_image)
    # plt.title("Original Grayscale Image")

    # # Noisy Grayscale Image with Salt-and-Pepper Noise
    # plt.subplot(1, 2, 2)
    # plt.imshow(noisy_image)
    # plt.title("Noisy Grayscale Image with Salt-and-Pepper Noise")

    # plt.show()


# filter with karnel *(moyanner filtering )
def noise_function(image, kernel_size=3):
    img_cv2_rgb = image_to_rgb_convertor(image)

    noisy = noise_function5(image, pepper_prob=0.2, salt_prob=0.2)

    kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size * kernel_size

    filtered_image = cv2.filter2D(img_cv2_rgb, -1, kernel)

    plt.subplot(1, 2, 1)
    plt.imshow(noisy)
    plt.title("noisy image")

    # Noisy Grayscale Image with Salt-and-Pepper Noise
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image)
    plt.title("image after the filter")

    plt.show()
