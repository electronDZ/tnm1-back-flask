import cv2

def image_to_rgb_convertor(image):
  # get rgb array for the image.
  # img_cv2 = cv2.imread(image)
  # Converting from BGR to RGB
  img_cv2_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return img_cv2_rgb