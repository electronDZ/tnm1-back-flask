def conidtional_converting(image, mode, value):
    if mode == "noise":
        RGB_to_XYZ_convertor(image, mode, value)
    if mode == "filter":
        RGB_to_HSL_convertor(image)
