import cv2
import numpy as np
from utils.image_to_rgb import image_to_rgb_convertor


def image_to_bmp_converter(image):
    img_cv2_rgb = image_to_rgb_convertor(image)

    # Save the image in BMP format
    # cv2.imwrite("path/to/your/output_image.bmp", img_cv2_rgb)
