import numpy as np


def diff_2_images(image1, image2):
    # Check if the two images are exactly equal
    are_images_equal = np.array_equal(image1, image2)

    if are_images_equal:
        percent_difference = 0.0  # Images are identical
    else:
        # Calculate the number of differing pixels
        num_differing_pixels = np.count_nonzero(image1 != image2)

        # Calculate the total number of pixels in the images
        total_pixels = np.prod(image1.shape)

        # Calculate the percentage difference
        percent_difference = (num_differing_pixels / total_pixels) * 100.0
        
    return percent_difference
