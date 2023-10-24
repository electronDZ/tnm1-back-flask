import numpy as np

from utils.image_to_rgb import image_to_rgb_convertor

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

    # Define the transformation matrix
    transformation_matrix = np.array(rgb_to_yuz_matrix)

    flatten_image = img_cv2_rgb.reshape((-1, 3))

    xyz_image = np.dot(flatten_image, transformation_matrix.T)
    xyz_image = xyz_image.reshape(img_cv2_rgb.shape)
    xyz_image_to_show = xyz_image.astype(np.uint8)

    x = np.copy(xyz_image_to_show)
    x[:, :, 1] = 0
    x[:, :, 2] = 0
    y = np.copy(xyz_image_to_show)
    y[:, :, 0] = 0
    y[:, :, 2] = 0
    z = np.copy(xyz_image_to_show)
    z[:, :, 0] = 0
    z[:, :, 1] = 0

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
