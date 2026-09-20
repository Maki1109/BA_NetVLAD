
"""Synthetic motion-blur kernels for training / calibration."""
import cv2
import numpy as np

_CACHE = {}

def _kernel(length, angle_deg):
    key = (int(round(length)), int(round(angle_deg)) % 180)
    if key in _CACHE:
        return _CACHE[key]
    L = max(int(round(key[0])), 1)
    k = L + (1 - L % 2)                        # odd size
    kern = np.zeros((k, k), np.float32)
    c = k // 2
    a = np.deg2rad(key[1])
    dx, dy = int(round((L - 1) / 2 * np.cos(a))), int(round((L - 1) / 2 * np.sin(a)))
    cv2.line(kern, (c - dx, c - dy), (c + dx, c + dy), 1.0, 1)
    kern /= kern.sum()
    _CACHE[key] = kern
    return kern

def motion_blur(img_bgr, length, angle_deg=0):
    """Approximate linear motion PSF (yaw smear model). length = smear in pixels."""
    if length <= 1:
        return img_bgr.copy()
    kern = _kernel(length, angle_deg)
    return cv2.filter2D(img_bgr, -1, kern, borderType=cv2.BORDER_REPLICATE)
