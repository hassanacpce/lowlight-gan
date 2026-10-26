"""Resize a folder of images (recursively) so training data loads fast.

    python prepare_data.py --src raw_images --dst data/clean --max_side 640
"""
import argparse
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--max_side", type=int, default=640)
ap.add_argument("--min_side", type=int, default=256, help="skip images smaller than this")
a = ap.parse_args()

dst = Path(a.dst)
dst.mkdir(parents=True, exist_ok=True)
kept = skipped = 0
for i, p in enumerate(sorted(Path(a.src).rglob("*"))):
    if p.suffix.lower() not in EXTS:
        continue
    try:
        im = Image.open(p).convert("RGB")
    except Exception:
        skipped += 1
        continue
    if min(im.size) < a.min_side:
        skipped += 1
        continue
    k = a.max_side / max(im.size)
    if k < 1:
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    im.save(dst / f"{i:06d}.jpg", quality=95)
    kept += 1
print(f"kept {kept}, skipped {skipped} -> {dst}")
