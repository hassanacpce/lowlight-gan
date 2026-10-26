# 🌙 LowLight GAN Studio
**Low-light image enhancement with a conditional GAN (U-Net generator + PatchGAN discriminator)**, trained on
synthetically darkened images, benchmarked against Histogram Equalisation / CLAHE / Auto-Gamma with PSNR & SSIM,
and wrapped in a Streamlit web app you can demo live with your phone camera.

```
 dark image ──► U-Net Generator G ──► enhanced image ─┐
     │                                                 ├─► PatchGAN D  ──► real / fake score per patch
     └──────────────────────────────────  (dark, clean) ┘
 loss_G = 100·L1(enhanced, clean) + LSGAN adversarial   (+ optional VGG perceptual)
```

## 1. Project structure
```
lowlight-gan/
├── degrade.py        synthetic darkening: gamma → exposure → colour cast → shot+read noise → 8-bit
├── data.py           dataset (darkens on the fly = new degradation every epoch)
├── models.py         U-Net generator, PatchGAN discriminator
├── train.py          training loop, validation, checkpoints, sample grids, loss curves
├── evaluate.py       PSNR/SSIM vs baselines (+ ablation), tables, bar chart, qualitative grid
├── baselines.py      Histogram Equalisation, CLAHE, Auto-Gamma
├── inference.py      run the model on any-size image
├── app.py            Streamlit UI
├── prepare_data.py   resize a raw image folder for fast training
├── train_colab.ipynb one-click Colab training notebook
└── requirements.txt
```

## 2. Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Data
You only need **well-lit photos** — the darkening is simulated.

| Purpose | Dataset | Notes |
|---|---|---|
| Training | **COCO val2017** (~5,000 images) or **DIV2K train** (800 images) or any photo folder | 1,000+ varied images is enough |
| Real-world test (recommended) | **LOL dataset** (500 low/normal pairs; `eval15` = 15 test pairs) | Gives you *real* low-light numbers, not just synthetic |

```bash
python prepare_data.py --src raw_images --dst data/clean --max_side 640
# hold out ~100 images that the model never sees in training:
mkdir -p data/test_clean && ls data/clean | tail -n 100 | xargs -I{} mv data/clean/{} data/test_clean/
```

## 4. Train (use Colab T4 — open `train_colab.ipynb`, or run locally with a GPU)
```bash
# Main model
python train.py --data data/clean --name gan --epochs 60 --batch 16

# Ablation: identical U-Net but NO discriminator  → "does the GAN actually help?"
python train.py --data data/clean --name unet_l1 --lambda_gan 0 --epochs 60 --batch 16

# Optional: sharper textures with a VGG perceptual loss (downloads VGG16 weights once)
python train.py --data data/clean --name gan_perc --lambda_perc 10
```
Outputs in `checkpoints/<name>/`: `generator_best.pt` (best val PSNR — use this), `latest.pt` (resume with
`--resume`), `history.csv`, `curves.png`, `samples/epoch_XXX.png` (dark | enhanced | ground truth).
Use `--ngf 32` for a ~4× smaller model if you need a lightweight checkpoint (≈30 MB) for GitHub.

## 5. Evaluate
```bash
python evaluate.py --data data/test_clean \
    --ckpt GAN=checkpoints/gan/generator_best.pt \
    --ckpt "U-Net (L1 only)=checkpoints/unet_l1/generator_best.pt"

# real low-light images (LOL)
python evaluate.py --lol data/LOL/eval15 --ckpt GAN=checkpoints/gan/generator_best.pt
```
Writes `results/results.csv`, `.md` table, `bar_*.png`, and `comparison_*.png` (one row per image, one column per
method) — paste these straight into your report.

## 5b. Fill in your results (template)
| Method | PSNR (dB) ↑ | SSIM ↑ | ms/img |
|---|---|---|---|
| Input (no enhancement) | | | |
| Histogram Equalization | | | |
| CLAHE | | | |
| Auto Gamma | | | |
| U-Net (L1 only) | | | |
| **GAN (ours)** | | | |

## 6. Run the app
```bash
streamlit run app.py
```
* **Enhance** – upload or capture a photo, drag the before/after slider, brightness/contrast stats, histograms,
  strength slider, download PNG, side-by-side classical baselines.
* **Simulate & measure** – darken a good photo with adjustable gamma / exposure / noise and see exact PSNR/SSIM for every method.
* **Results & about** – evaluation tables, charts, training curves.

Works on CPU too (a 1024-px photo takes a few seconds). Deploy free on **Hugging Face Spaces (Streamlit)** — put
`generator_best.pt` in `checkpoints/gan/` (use `--ngf 32` or Git LFS for large files).

## 7. Design choices worth explaining in the viva
* **Why synthetic pairs?** Real paired low/normal-light data is expensive; physics-inspired degradation + on-the-fly
  randomisation gives unlimited data. The LOL test checks the sim-to-real gap honestly.
* **Why U-Net?** Skip connections carry fine detail and colour from input to output; a plain encoder-decoder blurs.
* **Why PatchGAN?** Judges local texture realism (70×70 patches) with far fewer parameters than a full-image critic.
* **Why L1 + GAN?** L1 anchors the output to the ground truth (good PSNR); the adversarial term removes blur and
  restores texture. The `unet_l1` ablation quantifies the GAN's contribution.
* **Why LSGAN loss?** Least-squares loss gives smoother gradients and more stable training than BCE.
* **Why InstanceNorm?** Works with batch size 1 at inference and arbitrary image sizes.
* **Metric caveat:** PSNR/SSIM reward fidelity; a GAN can look better yet score slightly lower. Show visual
  comparisons too, and mention it (optional extra metric: LPIPS).

## 8. Report outline (≈ 25–30 pages)
1. Introduction & problem statement · 2. Literature survey (HE/CLAHE, Retinex, Zero-DCE, EnlightenGAN, Pix2Pix) ·
3. Proposed system (block diagram, degradation model) · 4. Architecture & loss functions · 5. Dataset & preprocessing ·
6. Implementation (PyTorch, Colab T4, hyper-parameters) · 7. Results (table, bar chart, qualitative grid, training curves, ablation) ·
8. Real-world testing (phone photos, LOL) · 9. Limitations (very noisy inputs, colour shifts, sim-to-real gap) ·
10. Future scope (Zero-DCE-style unpaired training, video, mobile deployment via ONNX/TFLite) · 11. Conclusion.

## 9. Ideas to push it further
* Fine-tune on LOL's training pairs after pre-training on synthetic data, and report the improvement.
* Add LPIPS and NIQE to the evaluation.
* Add a denoising-strength control, or export to ONNX for an in-browser demo.
* Video mode: enhance webcam frames in real time with a smaller `--ngf 32` model.

## 10. Troubleshooting
* **CUDA out of memory** → `--batch 8` or `--size 192`.
* **Discriminator loss → 0, outputs look odd** → lower `--lambda_gan` (e.g. 0.5) or train longer.
* **Outputs too noisy / over-bright on real photos** → widen the noise range in `degrade.py`, or fine-tune on LOL.
* **App shows "No trained checkpoint"** → ensure `checkpoints/<name>/generator_best.pt` exists next to `app.py`.