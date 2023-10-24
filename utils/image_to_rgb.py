import cv2
import numpy as np

def image_to_rgb_convertor(image):
  # Read the image using OpenCV
  image_data = image.read()
  nparr = np.frombuffer(image_data, np.uint8)
  img_cv2_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  # Convert BGR to RGB
  img_cv2_rgb = cv2.cvtColor(img_cv2_bgr, cv2.COLOR_BGR2RGB)
  return img_cv2_rgb