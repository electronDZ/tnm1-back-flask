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


def calculate_rgb(h, s, v):
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    elif 300 <= h < 360:
        r, g, b = c, 0, x
    else:
        r = g = b = 0

    r = r + m
    g = g + m
    b = b + m

    return [r, g, b]
