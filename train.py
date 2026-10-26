"""Train the low-light enhancement GAN (Pix2Pix-style, LSGAN + L1 [+ optional VGG perceptual loss]).

Main run  :  python train.py --data data/clean --name gan
Ablation  :  python train.py --data data/clean --name unet_l1 --lambda_gan 0     (no discriminator)
"""
import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data import LowLightDataset, list_images, split_files
from inference import tensor_to_uint8
from metrics import psnr, ssim
from models import PatchDiscriminator, UNetGenerator, init_weights


class VGGPerceptual(nn.Module):
    """L1 distance between VGG16 relu3_3 features (needs to download ImageNet weights once)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        self.f = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in self.f.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, a, b):
        a = ((a + 1) / 2 - self.mean) / self.std
        b = ((b + 1) / 2 - self.mean) / self.std
        return F.l1_loss(self.f(a), self.f(b))


@torch.no_grad()
def validate(G, loader, device, max_images):
    G.eval()
    P, S, Pi, Si = [], [], [], []
    for dark, clean in loader:
        fake = G(dark.to(device)).cpu()
        for i in range(dark.size(0)):
            gt, pr, inp = tensor_to_uint8(clean[i]), tensor_to_uint8(fake[i]), tensor_to_uint8(dark[i])
            P.append(psnr(gt, pr)); S.append(ssim(gt, pr))
            Pi.append(psnr(gt, inp)); Si.append(ssim(gt, inp))
            if len(P) >= max_images:
                break
        if len(P) >= max_images:
            break
    G.train()
    return float(np.mean(P)), float(np.mean(S)), float(np.mean(Pi)), float(np.mean(Si))


def plot_curves(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(ep, [h["loss_G"] for h in history], label="Generator")
    if history[0]["loss_D"] > 0:
        ax[0].plot(ep, [h["loss_D"] for h in history], label="Discriminator")
    ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["val_psnr"] for h in history], label="Enhanced")
    ax[1].plot(ep, [h["input_psnr"] for h in history], "--", label="Dark input")
    ax[1].set_title("Validation PSNR (dB)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    ax[2].plot(ep, [h["val_ssim"] for h in history], label="Enhanced")
    ax[2].plot(ep, [h["input_ssim"] for h in history], "--", label="Dark input")
    ax[2].set_title("Validation SSIM"); ax[2].set_xlabel("epoch"); ax[2].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of well-lit images")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--name", default="gan", help="run name -> checkpoints/<name>/")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ngf", type=int, default=64, help="generator width (32 -> ~4x smaller model)")
    ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--lambda_l1", type=float, default=100.0)
    ap.add_argument("--lambda_gan", type=float, default=1.0, help="0 disables the discriminator (ablation)")
    ap.add_argument("--lambda_perc", type=float, default=0.0, help=">0 adds VGG perceptual loss, e.g. 10")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--val_max", type=int, default=48)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default="", help="path to checkpoints/<name>/latest.pt")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(args.out) / args.name
    (run / "samples").mkdir(parents=True, exist_ok=True)
    print(f"device={device}  run dir={run}")

    files = list_images(args.data)
    assert len(files) > 10, f"found only {len(files)} images in {args.data}"
    train_f, val_f = split_files(files, args.val_frac, args.seed)
    print(f"{len(train_f)} train / {len(val_f)} val images")
    train_dl = DataLoader(LowLightDataset(train_f, args.size, True), batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True, pin_memory=(device == "cuda"))
    val_dl = DataLoader(LowLightDataset(val_f, args.size, False, seed=123), batch_size=8, shuffle=False,
                        num_workers=args.workers)
    fixed_dark, fixed_clean = next(iter(val_dl))
    fixed_dark, fixed_clean = fixed_dark[:4].to(device), fixed_clean[:4]

    use_gan = args.lambda_gan > 0
    G = UNetGenerator(ngf=args.ngf).to(device); G.apply(init_weights)
    D = PatchDiscriminator(ndf=args.ndf).to(device) if use_gan else None
    if D: D.apply(init_weights)
    perc = VGGPerceptual().to(device) if args.lambda_perc > 0 else None
    print(f"generator params: {sum(p.numel() for p in G.parameters()) / 1e6:.1f}M")

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999)) if D else None
    start_decay = args.epochs // 2  # constant LR for half the run, then linear decay to 0
    lam = lambda e: 1.0 if e < start_decay else max(0.0, 1.0 - (e - start_decay) / (args.epochs - start_decay + 1))
    schedG = torch.optim.lr_scheduler.LambdaLR(optG, lam)
    schedD = torch.optim.lr_scheduler.LambdaLR(optD, lam) if D else None

    history, start_epoch, best_psnr = [], 1, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        G.load_state_dict(ck["G"]); optG.load_state_dict(ck["optG"]); schedG.load_state_dict(ck["schedG"])
        if D and ck.get("D"):
            D.load_state_dict(ck["D"]); optD.load_state_dict(ck["optD"]); schedD.load_state_dict(ck["schedD"])
        history, start_epoch, best_psnr = ck["history"], ck["epoch"] + 1, ck["best_psnr"]
        print(f"resumed from epoch {ck['epoch']}")

    l1 = nn.L1Loss(); mse = nn.MSELoss()  # MSE on patch scores = LSGAN (more stable than BCE)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        agg = dict(G=0.0, D=0.0, L1=0.0, GAN=0.0); n = 0
        bar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for dark, clean in bar:
            dark, clean = dark.to(device), clean.to(device)
            fake = G(dark)

            if use_gan:  # ---- discriminator step
                optD.zero_grad()
                pr, pf = D(dark, clean), D(dark, fake.detach())
                loss_D = 0.5 * (mse(pr, torch.ones_like(pr)) + mse(pf, torch.zeros_like(pf)))
                loss_D.backward(); optD.step()
            else:
                loss_D = torch.zeros(())

            # ---- generator step
            optG.zero_grad()
            loss_l1 = l1(fake, clean)
            loss_G = args.lambda_l1 * loss_l1
            loss_gan = torch.zeros((), device=device)
            if use_gan:
                pf = D(dark, fake)
                loss_gan = mse(pf, torch.ones_like(pf))
                loss_G = loss_G + args.lambda_gan * loss_gan
            if perc is not None:
                loss_G = loss_G + args.lambda_perc * perc(fake, clean)
            loss_G.backward(); optG.step()

            agg["G"] += loss_G.item(); agg["D"] += loss_D.item()
            agg["L1"] += loss_l1.item(); agg["GAN"] += loss_gan.item(); n += 1
            bar.set_postfix(G=f"{loss_G.item():.2f}", D=f"{loss_D.item():.3f}")

        schedG.step()
        if schedD: schedD.step()

        v_psnr, v_ssim, i_psnr, i_ssim = validate(G, val_dl, device, args.val_max)
        row = dict(epoch=epoch, loss_G=agg["G"] / n, loss_D=agg["D"] / n, loss_L1=agg["L1"] / n,
                   loss_GAN=agg["GAN"] / n, val_psnr=v_psnr, val_ssim=v_ssim,
                   input_psnr=i_psnr, input_ssim=i_ssim, seconds=time.time() - t0)
        history.append(row)
        print(f"epoch {epoch:3d} | G {row['loss_G']:.3f} D {row['loss_D']:.3f} | "
              f"val PSNR {v_psnr:.2f} (input {i_psnr:.2f}) SSIM {v_ssim:.3f} (input {i_ssim:.3f}) | {row['seconds']:.0f}s")

        # ---- sample grid: dark | enhanced | ground truth
        G.eval()
        with torch.no_grad():
            fk = G(fixed_dark).cpu()
        G.train()
        grid = torch.cat([fixed_dark.cpu(), fk, fixed_clean], 0)
        save_image((grid + 1) / 2, run / "samples" / f"epoch_{epoch:03d}.png", nrow=4, padding=2)

        # ---- checkpoints
        state = dict(epoch=epoch, G=G.state_dict(), optG=optG.state_dict(), schedG=schedG.state_dict(),
                     D=D.state_dict() if D else None, optD=optD.state_dict() if D else None,
                     schedD=schedD.state_dict() if D else None, history=history,
                     best_psnr=max(best_psnr, v_psnr), ngf=args.ngf, args=vars(args))
        torch.save(state, run / "latest.pt")
        if v_psnr > best_psnr:
            best_psnr = v_psnr
            torch.save(dict(G=G.state_dict(), ngf=args.ngf, epoch=epoch,
                            val_psnr=float(v_psnr), val_ssim=float(v_ssim)), run / "generator_best.pt")
            print(f"           ^ new best (PSNR {v_psnr:.2f}) saved")

        with open(run / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
        plot_curves(history, run / "curves.png")

    print(f"done. best val PSNR {best_psnr:.2f}. Use {run / 'generator_best.pt'} in evaluate.py / app.py")


if __name__ == "__main__":
    main()
