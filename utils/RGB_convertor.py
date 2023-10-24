from utils.rbg_to_xyz import RGB_to_XYZ_convertor
from utils.rgb_to_hsl import RGB_to_HSL_convertor
from utils.rgb_to_yuv import RGB_to_YUV_convertor

def conidtional_converting(image, mode):
  if mode == 'XYZ':
    RGB_to_XYZ_convertor(image)
  if mode == 'HSL':
    RGB_to_HSL_convertor(image)
  if mode == 'YUV':
    RGB_to_YUV_convertor(image)

