import numpy as np

from utils.image_to_rgb import image_to_rgb_convertor
from helpers.compare_2_images import diff_2_images

import cv2
import matplotlib.pylab as plt


def RGB_to_YUV_convertor(image_path):
    rgb_to_yuz_matrix = [
        [0.2990000, 0.5870000, 0.1140000],
        [-0.168736, -0.331264, 0.5000000],
        [0.5000000, 0.0565000, -0.081312],
    ]
    
    img_cv2_rgb = image_to_rgb_convertor(image_path)
    # xyz_color = cv2.cvtColor(img_cv2_rgb, cv2.COLOR_RGB2YUV)
    # xyz_color = cv2.cvtColor(img_cv2_rgb, cv2.COLOR_RGB2YUV)

    # Define the transformation matrix
    transformation_matrix = np.array(rgb_to_yuz_matrix)

    flatten_image = img_cv2_rgb.reshape((-1, 3))

    yuv_image = np.dot(flatten_image, transformation_matrix.T)
    yuv_image = yuv_image.reshape(img_cv2_rgb.shape)
    yuv_image_to_show = yuv_image.astype(np.uint8)

    # Split the YUV image into its channels
    y_channel = yuv_image[:, :, 0]
    u_channel = yuv_image[:, :, 1]
    v_channel = yuv_image[:, :, 2]

    # Define the reverse transformation matrix (3x3 matrix)
    reverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Flatten the XYZ image
    flatten_xyz_image = yuv_image.reshape((-1, 3))

    # Use the reverse transformation matrix to convert back to RGB
    rgb_image = np.dot(flatten_xyz_image, reverse_transformation_matrix.T)

    # Reshape the RGB image to its original shape
    rgb_image = rgb_image.reshape(yuv_image.shape)

    # Convert the RGB image to uint8 data type (0-255 range)
    rgb_image_to_show = rgb_image.astype(np.uint8)


    # Get the diff percentage
    percent_difference = diff_2_images(img_cv2_rgb, rgb_image_to_show)

    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    axs[0, 0].imshow(img_cv2_rgb)
    axs[0, 0].axis("off")
    axs[0, 0].set_title("RGB")
    axs[0, 1].imshow(y_channel, cmap='gray')
    axs[0, 1].axis("off")
    axs[0, 1].set_title("Y")
    axs[0, 2].imshow(u_channel, cmap='gray')
    axs[0, 2].axis("off")
    axs[0, 2].set_title("U")
    axs[0, 3].imshow(v_channel, cmap='gray')
    axs[0, 3].axis("off")
    axs[0, 3].set_title("V")

    # axs[1, 0].imshow(rgb_image_to_show)
    axs[1, 0].axis("off")
    # axs[1, 0].set_title("RGB")
    axs[1, 1].imshow(rgb_image_to_show)
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Original")
    axs[1, 2].imshow(rgb_image_to_show)
    axs[1, 2].axis("off")
    axs[1, 2].set_title("YUV to RGB")
    # axs[1, 3].imshow(z)
    axs[1, 3].axis("off")
    # axs[1, 3].set_title("Z")
    axs[1, 3].text(
        0.5,
        0.5,
        f"difference: {percent_difference:.2f}%",
        fontsize=12,
        ha="center",
        va="center",
    )
    plt.show()
