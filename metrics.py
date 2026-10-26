"""Image-quality metrics (uint8 RGB arrays, higher is better for both)."""
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(gt, pred):
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))


def ssim(gt, pred):
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
