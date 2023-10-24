import numpy as np

from utils.image_to_rgb import image_to_rgb_convertor
from helpers.HSL_helpers import calculate_v, calculate_h, calculate_s

import cv2
import matplotlib.pylab as plt

def RGB_to_HSL_convertor(image_path):
    img_cv2_rgb = image_to_rgb_convertor(image_path)
    xyz_color = cv2.cvtColor(img_cv2_rgb, cv2.COLOR_RGB2HLS_FULL)


    # Create an empty NumPy array for storing the HSL values
    hsl_image = np.empty_like(img_cv2_rgb, dtype=np.float64)

    # Iterate through each pixel of the img_cv2_rgb
    for i in range(img_cv2_rgb.shape[0]):
        for j in range(img_cv2_rgb.shape[1]):
            # Extract R, G, and B values for the current pixel
            r, g, b = img_cv2_rgb[i, j]

            # Apply the functions to calculate HSL values
            v = calculate_v(r, g, b)
            s = calculate_s(r, g, b)
            h = calculate_h(r, g, b)

            # Store the HSL values in the new array
            hsl_image[i, j] = [h, s, v]

    hsl_image = hsl_image.astype(np.uint8)
    

    x = np.copy(xyz_color) 
    x[:,:,1] = 0
    x[:,:,2] = 0
    y = np.copy(xyz_color) 
    y[:,:,0] = 0
    y[:,:,2] = 0
    z = np.copy(xyz_color) 
    z[:,:,0] = 0
    z[:,:,1] = 0
    
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    axs[0].imshow(img_cv2_rgb)
    axs[0].axis("off")
    axs[0].set_title("RGB")
    axs[1].imshow(hsl_image)
    axs[1].axis("off")
    axs[1].set_title("X")
    axs[2].imshow(y)
    axs[2].axis("off")
    axs[2].set_title("Y")
    axs[3].imshow(z)
    axs[3].axis("off")
    axs[3].set_title("Z")
    plt.show()
