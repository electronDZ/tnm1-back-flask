import numpy as np


def calculate_v(r, g, b):
    v = (r + g + b) / 3
    return v


def calculate_s(r, g, b):
    s = 1 - (3 * min(r, g, b) / (r + g + b))
    return s


def calculate_h(r, g, b):
    theta = np.arccos(
        (r - g) + (r - b) / (2 * np.sqrt((r - g) ** 2 + (r - b) * (g - b)))
    )

    if b > g:
        h = 2 * np.pi - theta
    else:
        h = theta
    return h
