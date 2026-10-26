"""Synthetic low-light degradation (numpy only).

Physically-motivated pipeline:
  1. gamma curve      -> crushes mid-tones/highlights (non-linear darkening)
  2. linear scale     -> simulates short exposure / low light level
  3. colour cast      -> slight per-channel gain (white-balance error in the dark)
  4. noise            -> signal-dependent shot noise + signal-independent read noise
  5. 8-bit quantise   -> like a real camera output
"""
import numpy as np


def darken(img, rng=None, gamma=None, scale=None, read_sigma=None, shot=None, color_cast=True):
    """img: float array HxWx3 in [0, 1]. Any argument left as None is sampled randomly.

    Returns a float32 array in [0, 1].
    """
    rng = rng if rng is not None else np.random.default_rng()
    gamma = rng.uniform(1.3, 3.5) if gamma is None else gamma
    scale = rng.uniform(0.35, 1.0) if scale is None else scale
    read_sigma = rng.uniform(0.004, 0.025) if read_sigma is None else read_sigma
    shot = rng.uniform(0.0005, 0.004) if shot is None else shot

    dark = np.power(np.clip(img, 0.0, 1.0), gamma) * scale
    if color_cast:
        dark = dark * rng.uniform(0.92, 1.08, size=(1, 1, 3))

    shot_noise = np.sqrt(np.maximum(dark, 0.0) * shot) * rng.normal(0.0, 1.0, dark.shape)
    read_noise = rng.normal(0.0, read_sigma, dark.shape) if read_sigma > 0 else 0.0
    dark = dark + shot_noise + read_noise

    dark = np.clip(dark, 0.0, 1.0)
    dark = np.round(dark * 255.0) / 255.0
    return dark.astype(np.float32)
