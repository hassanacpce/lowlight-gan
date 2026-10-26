"""Classical enhancement baselines to compare the GAN against (uint8 RGB in / out)."""
import cv2
import numpy as np


def hist_equalization(img):
    """Global histogram equalisation on the luminance channel."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def clahe(img, clip=3.0, tile=8):
    """Contrast-Limited Adaptive Histogram Equalisation on the L channel (LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[..., 0] = c.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def auto_gamma(img, target=0.5):
    """Pick gamma so that the mean brightness maps to `target`, then apply it."""
    x = img.astype(np.float32) / 255.0
    mean = float(np.clip(x.mean(), 1e-3, 0.999))
    gamma = float(np.clip(np.log(target) / np.log(mean), 0.2, 1.0))
    return np.clip(np.power(x, gamma) * 255.0, 0, 255).astype(np.uint8)


BASELINES = {
    "Hist. Equalization": hist_equalization,
    "CLAHE": clahe,
    "Auto Gamma": auto_gamma,
}


def resize_max_side(img, max_side):
    """Downscale (never upscale) so the longest side is <= max_side. uint8 RGB in / out."""
    h, w = img.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return img
    k = max_side / max(h, w)
    return cv2.resize(img, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA)
