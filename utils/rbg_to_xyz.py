# rgb hsl yuz xyz
import numpy as np
from PIL import Image

from utils.image_to_rgb import image_to_rgb_convertor

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
    transformation_matrix = np.array([
        [2.7690 , 1.7518 , 1.1300],
        [1.0000 , 4.5907 , 0.0601],
        [0.0000 , 0.0565 , 5.5943]
    ])
    transformation_matrix_adobe = np.array([
        [	0.5767309 , 0.1855540 , 0.1881852],
        [  0.2973769 , 0.6273491 , 0.0752741],
        [ 0.0270343 , 0.0706872 , 0.9911085]
    ])

    flatten_image = img_cv2_rgb.reshape((-1, 3))


    xyz_image = np.dot(flatten_image, transformation_matrix.T)
    xyz_image = xyz_image.reshape(img_cv2_rgb.shape)
    xyz_image_to_show = xyz_image.astype(np.uint8)

    x = np.copy(xyz_image_to_show) 
    x[:,:,1] = 0
    x[:,:,2] = 0
    y = np.copy(xyz_image_to_show) 
    y[:,:,0] = 0
    y[:,:,2] = 0
    z = np.copy(xyz_image_to_show) 
    z[:,:,0] = 0
    z[:,:,1] = 0
    

    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    axs[0].imshow(img_cv2_rgb)
    axs[0].axis("off")
    axs[0].set_title("RGB")
    axs[1].imshow(x)
    axs[1].axis("off")
    axs[1].set_title("X")
    axs[2].imshow(y)
    axs[2].axis("off")
    axs[2].set_title("Y")
    axs[3].imshow(z)
    axs[3].axis("off")
    axs[3].set_title("Z")
    plt.show()


# image_to_test = "../assets/tiberli.jpg"
# RGB_to_XYZ_convertor(image_to_test)
