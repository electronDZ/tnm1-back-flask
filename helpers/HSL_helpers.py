import numpy as np
import math



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


def hsi_to_rgb(H, S, I):
    H = np.where(H < 2*np.pi, H, 0)
    H = np.where(H > 0, H, 0)

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    # RG sector (0 <= H < 2*pi/3).
    idx = (0 <= H) & (H < 2*np.pi/3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(np.pi/3 - H[idx]))
    G[idx] = 3*I[idx] - (R[idx] + B[idx])

    # BG sector (2*pi/3 <= H < 4*pi/3).
    idx = (2*np.pi/3 <= H) & (H < 4*np.pi/3)
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx] - 2*np.pi/3) / np.cos(np.pi - H[idx]))
    B[idx] = 3*I[idx] - (R[idx] + G[idx])

    # BR sector.
    idx = (4*np.pi/3 <= H) & (H < 2*np.pi)
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx] - 4*np.pi/3) / np.cos(5*np.pi/3 - H[idx]))
    R[idx] = 3*I[idx] - (G[idx] + B[idx])

    return [R, G, B]