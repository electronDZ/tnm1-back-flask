import numpy as np

from utils.image_to_rgb import image_to_rgb_convertor
from helpers.HSL_helpers import calculate_v, calculate_h, calculate_s, calculate_rgb
from helpers.compare_2_images import diff_2_images

import cv2
import matplotlib.pylab as plt

def RGB_to_HSL_convertor(image_path):
    img_cv2_rgb = image_to_rgb_convertor(image_path)
    # xyz_color = cv2.cvtColor(img_cv2_rgb, cv2.COLOR_RGB2HLS_FULL)


    # Create an empty NumPy array for storing the HSL values
    hsl_image = np.empty_like(img_cv2_rgb, dtype=np.float64)

    normalized_img_cv2_rgb = img_cv2_rgb / 255 # np.arccos function returns NaN when its input is outside the range -1 to 1

    # Iterate through each pixel of the img_cv2_rgb
    for i in range(normalized_img_cv2_rgb.shape[0]):
        for j in range(normalized_img_cv2_rgb.shape[1]):
            r, g, b = normalized_img_cv2_rgb[i, j]

            h = calculate_h(r, g, b)
            s = calculate_s(r, g, b)
            v = calculate_v(r, g, b)

            hsl_image[i, j] = [h, s, v]


    hsl_to_rgb_image = np.empty_like(hsl_image, dtype=np.float64)
    
    # Iterate through each pixel of the hsl_image
    for i in range(hsl_to_rgb_image.shape[0]):
        for j in range(hsl_to_rgb_image.shape[1]):
            h, s, v = hsl_image[i, j]
            hsl_to_rgb_image[i, j] = calculate_rgb(h, s, v)

    # Get the diff percentage
    percent_difference = diff_2_images(img_cv2_rgb, hsl_to_rgb_image)

    # Split the XYZ image into its channels
    h_channel = hsl_image[:, :, 0]
    s_channel = hsl_image[:, :, 1]
    l_channel = hsl_image[:, :, 2]
   

    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    axs[0, 0].imshow(img_cv2_rgb)
    axs[0, 0].axis("off")
    axs[0, 0].set_title("RGB")
    axs[0, 1].imshow(h_channel, cmap='gray')
    axs[0, 1].axis("off")
    axs[0, 1].set_title("H")
    axs[0, 2].imshow(s_channel, cmap='gray')
    axs[0, 2].axis("off")
    axs[0, 2].set_title("S")
    axs[0, 3].imshow(l_channel, cmap='gray')
    axs[0, 3].axis("off")
    axs[0, 3].set_title("L")

    # axs[1, 0].imshow(rgb_image_to_show)
    axs[1, 0].axis("off")
    # axs[1, 0].set_title("RGB")
    axs[1, 1].imshow(img_cv2_rgb)
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Original")
    axs[1, 2].imshow(hsl_to_rgb_image)
    axs[1, 2].axis("off")
    axs[1, 2].set_title("HSL to RGB")
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