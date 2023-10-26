# rgb hsl yuz xyz
import numpy as np
from PIL import Image

from utils.image_to_rgb import image_to_rgb_convertor
from helpers.compare_2_images import diff_2_images

import cv2
import matplotlib.pylab as plt


def display_signle_image(img_cv2, title):
    fig, axs = plt.subplots(figsize=(10, 5))
    axs.imshow(img_cv2)
    axs.axis("off")
    axs.set_title(title)
    plt.show()


def display_comparing_two_images(
    first_img_cv2, first_title, seconde_img_cv2, seconde_title
):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(first_img_cv2)
    axs[0].axis("off")
    axs[0].set_title(first_title)
    axs[1].imshow(seconde_img_cv2)
    axs[1].axis("off")
    axs[1].set_title(seconde_title)
    plt.show()


def RGB_to_XYZ_convertor(image_path):
    img_cv2_rgb = image_to_rgb_convertor(image_path)
    # xyz_color = cv2.cvtColor(img_cv2_rgb, cv2.COLOR_RGB2XYZ)

    # Define the transformation matrix
    transformation_matrix = np.array(
        [[2.7690, 1.7518, 1.1300], [1.0000, 4.5907, 0.0601], [0.0000, 0.0565, 5.5943]]
    )

    flatten_image = img_cv2_rgb.reshape((-1, 3))

    xyz_image = np.dot(flatten_image, transformation_matrix.T)
    xyz_image = xyz_image.reshape(img_cv2_rgb.shape)
    xyz_image_to_show = xyz_image.astype(np.uint8)
    

    # Split the XYZ image into its channels
    x_channel = xyz_image_to_show[:, :, 0]
    y_channel = xyz_image_to_show[:, :, 1]
    z_channel = xyz_image_to_show[:, :, 2]

    # Define the reverse transformation matrix (3x3 matrix)
    reverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Flatten the XYZ image
    flatten_xyz_image = xyz_image.reshape((-1, 3))

    # Use the reverse transformation matrix to convert back to RGB
    rgb_image = np.dot(flatten_xyz_image, reverse_transformation_matrix.T)

    # Reshape the RGB image to its original shape
    rgb_image = rgb_image.reshape(xyz_image.shape)

    # Convert the RGB image to uint8 data type (0-255 range)
    rgb_image_to_show = rgb_image.astype(np.uint8)


    # Get the diff percentage
    percent_difference = diff_2_images(img_cv2_rgb, rgb_image_to_show)

    # Split the XYZ image into its channels
    x_channel = rgb_image[:, :, 0]
    y_channel = rgb_image[:, :, 1]
    z_channel = rgb_image[:, :, 2]
   

    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    axs[0, 0].imshow(img_cv2_rgb)
    axs[0, 0].axis("off")
    axs[0, 0].set_title("RGB")
    axs[0, 1].imshow(x_channel, cmap='gray')
    axs[0, 1].axis("off")
    axs[0, 1].set_title("X")
    axs[0, 2].imshow(y_channel, cmap='gray')
    axs[0, 2].axis("off")
    axs[0, 2].set_title("Y")
    axs[0, 3].imshow(z_channel, cmap='gray')
    axs[0, 3].axis("off")
    axs[0, 3].set_title("Z")

    # axs[1, 0].imshow(rgb_image_to_show)
    axs[1, 0].axis("off")
    # axs[1, 0].set_title("RGB")
    axs[1, 1].imshow(img_cv2_rgb)
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Original")
    axs[1, 2].imshow(rgb_image_to_show)
    axs[1, 2].axis("off")
    axs[1, 2].set_title("XYZ to RGB")
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
