
"""Blur gate: variance-of-Laplacian sharpness metric (BA-NetVLAD Step 1)."""
from dataclasses import dataclass
import cv2
import numpy as np

def laplacian_beta(img_bgr, gate_size=(120, 160), roi=None):
    """Return beta = Var(Laplacian(gray_downsampled(img_bgr))).

    Always computed at a FIXED gate resolution so that epsilon is portable
    across cameras/resolutions. Optional roi = (x0,y0,x1,y1) in ORIGINAL
    image coords, applied before resize (use a fixed center window if the
    environment has low-texture borders).
    """
    if roi is not None:
        x0, y0, x1, y1 = roi
        img_bgr = img_bgr[y0:y1, x0:x1]
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, gate_size, interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(g, cv2.CV_64F)          # 5-point Laplacian
    beta = float(lap.var())
    return beta

@dataclass
class BlurGate:
    epsilon: float = 120.0
    eps_high: float = None      # None -> hysteresis_ratio * epsilon (see __post_init__)
    hysteresis_ratio: float = 1.15
    gate_size: tuple = (120, 160)
    roi: tuple = None
    _state: bool = True         # True = currently sharp branch

    def __post_init__(self):
        # eps_high MUST track epsilon. A fixed literal (the old 138.0) is below
        # epsilon for any calibrated dataset (KITTI00: eps=1521.5), which inverts
        # the hysteresis: the gate drops to blur then snaps back the next frame,
        # chattering instead of latching.
        if self.eps_high is None or self.eps_high <= self.epsilon:
            self.eps_high = self.hysteresis_ratio * self.epsilon

    def __call__(self, img_bgr):
        beta = laplacian_beta(img_bgr, self.gate_size, self.roi)
        if self._state:                          # hysteresis
            if beta < self.epsilon:  self._state = False
        else:
            if beta >= self.eps_high: self._state = True
        return beta, self._state                 # beta, is_sharp
