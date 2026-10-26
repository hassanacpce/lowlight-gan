"""Compare the GAN(s) against classical baselines with PSNR / SSIM.

Synthetic test set (held-out clean images, darkened with a fixed seed):
    python evaluate.py --data data/test_clean --ckpt GAN=checkpoints/gan/generator_best.pt \
                       --ckpt "U-Net (L1 only)=checkpoints/unet_l1/generator_best.pt"

Real paired test set (LOL dataset: folders low/ and high/):
    python evaluate.py --lol data/LOL/eval15 --ckpt GAN=checkpoints/gan/generator_best.pt
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from baselines import BASELINES
from data import EXTS, LowLightDataset, list_images
from baselines import resize_max_side
from inference import enhance, tensor_to_uint8
from metrics import psnr, ssim
from models import load_generator


def load_synthetic(folder, size, seed, max_images):
    files = list_images(folder)[:max_images]
    ds = LowLightDataset(files, size=size, train=False, seed=seed)
    return [(tensor_to_uint8(d), tensor_to_uint8(c), files[i].name) for i, (d, c) in enumerate(ds[i] for i in range(len(ds)))]


def load_lol(folder, max_images, max_side):
    low, high = Path(folder) / "low", Path(folder) / "high"
    names = sorted(p.name for p in low.iterdir() if p.suffix.lower() in EXTS)[:max_images]
    out = []
    for n in names:
        d = resize_max_side(np.array(Image.open(low / n).convert("RGB")), max_side)
        g = resize_max_side(np.array(Image.open(high / n).convert("RGB")), max_side)
        out.append((d, g, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="folder of clean images (synthetic darkening)")
    ap.add_argument("--lol", help="folder containing low/ and high/ (real paired data)")
    ap.add_argument("--ckpt", action="append", default=[], help="NAME=path/to/generator_best.pt (repeatable)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max_images", type=int, default=200)
    ap.add_argument("--max_side", type=int, default=0, help="downscale real images if > 0")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--grid", type=int, default=6, help="number of qualitative comparison rows to save")
    args = ap.parse_args()
    assert args.data or args.lol, "give --data or --lol"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pairs = load_lol(args.lol, args.max_images, args.max_side) if args.lol else \
        load_synthetic(args.data, args.size, args.seed, args.max_images)
    print(f"{len(pairs)} test pairs ({'LOL real' if args.lol else 'synthetic'}), device={device}")

    methods = {"Input (no enhancement)": lambda x: x}
    methods.update(BASELINES)
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        G = load_generator(path, device)
        methods[name] = (lambda g: (lambda x: enhance(g, x, device)))(G)

    scores = {m: {"psnr": [], "ssim": [], "ms": []} for m in methods}
    keep = {m: [] for m in methods}
    for i, (dark, gt, _) in enumerate(pairs):
        for m, fn in methods.items():
            t0 = time.time(); pred = fn(dark); dt = (time.time() - t0) * 1000
            scores[m]["psnr"].append(psnr(gt, pred)); scores[m]["ssim"].append(ssim(gt, pred)); scores[m]["ms"].append(dt)
            if i < args.grid:
                keep[m].append(pred)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)}")

    rows = []
    for m, s in scores.items():
        rows.append({"Method": m, "PSNR (dB)": np.mean(s["psnr"]), "PSNR std": np.std(s["psnr"]),
                     "SSIM": np.mean(s["ssim"]), "SSIM std": np.std(s["ssim"]), "ms / image": np.mean(s["ms"])})
    df = pd.DataFrame(rows)
    tag = "lol" if args.lol else "synthetic"
    df.to_csv(out / f"results_{tag}.csv", index=False)
    df.to_csv(out / "results.csv", index=False)  # the app reads this one
    (out / f"results_{tag}.md").write_text(df.round(3).to_markdown(index=False))
    print("\n" + df.round(3).to_string(index=False))

    # ---- bar chart
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, col, err in [(ax[0], "PSNR (dB)", "PSNR std"), (ax[1], "SSIM", "SSIM std")]:
        a.bar(df["Method"], df[col], yerr=df[err], capsize=3, color="#7C5CFF")
        a.set_title(col); a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(out / f"bar_{tag}.png", dpi=130); plt.close(fig)

    # ---- qualitative grid: one row per image, one column per method (+ ground truth)
    n = min(args.grid, len(pairs)); cols = list(methods) + ["Ground truth"]
    fig, ax = plt.subplots(n, len(cols), figsize=(2.6 * len(cols), 2.3 * n), squeeze=False)
    for r in range(n):
        for c, m in enumerate(cols):
            img = pairs[r][1] if m == "Ground truth" else keep[m][r]
            ax[r][c].imshow(img); ax[r][c].axis("off")
            if r == 0: ax[r][c].set_title(m, fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"comparison_{tag}.png", dpi=130); plt.close(fig)
    print(f"\nsaved tables + figures to {out}/")


if __name__ == "__main__":
    main()
