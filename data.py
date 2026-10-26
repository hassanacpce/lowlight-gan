"""Dataset: clean images in, (synthetically darkened, clean) pairs out. Darkening is done on the fly,
so every epoch sees new gamma / exposure / noise combinations (free data augmentation)."""
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from degrade import darken

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)


def split_files(files, val_frac=0.05, seed=0):
    files = list(files)
    random.Random(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    return files[n_val:], files[:n_val]


def _to_tensor(x):
    x = np.ascontiguousarray(x.transpose(2, 0, 1))
    return torch.from_numpy(x).float() * 2.0 - 1.0


class LowLightDataset(Dataset):
    def __init__(self, files, size=256, train=True, seed=0):
        self.files, self.size, self.train, self.seed = list(files), size, train, seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        s = self.size
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed + idx)

        # random rescale (train only) so the model sees detail at different scales; never below crop size
        k = rng.uniform(0.6, 1.2) if self.train else 1.0
        k = max(k, s / min(w, h))
        if abs(k - 1.0) > 1e-3:
            img = img.resize((math.ceil(w * k), math.ceil(h * k)), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        H, W = arr.shape[:2]

        if self.train:
            top, left = int(rng.integers(0, H - s + 1)), int(rng.integers(0, W - s + 1))
        else:  # deterministic for validation / evaluation
            top, left = (H - s) // 2, (W - s) // 2

        clean = arr[top:top + s, left:left + s]
        if self.train and rng.random() < 0.5:
            clean = clean[:, ::-1]
        dark = darken(clean, rng)
        return _to_tensor(dark), _to_tensor(clean)
