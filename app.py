"""LowLight GAN Studio - Streamlit UI.   Run:  streamlit run app.py"""
import glob
import io
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from baselines import BASELINES, resize_max_side
from degrade import darken
from metrics import psnr, ssim

try:
    import torch
    from inference import enhance
    from smart_enhance import smart_enhance
    from models import load_generator
    TORCH_OK = True
except Exception:  # lets the UI start (baselines only) even if torch is missing
    TORCH_OK = False

try:
    from streamlit_image_comparison import image_comparison
    SLIDER_OK = True
except Exception:
    SLIDER_OK = False

st.set_page_config(page_title="LowLight GAN Studio", page_icon="🌙", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
h1 {background: linear-gradient(90deg,#7C5CFF,#22D3EE); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 800;}
div[data-testid="stMetric"] {background:#171B26; border:1px solid #262B3A; border-radius:12px; padding:12px 16px;}
.caption-pill {display:inline-block; padding:2px 10px; border-radius:999px; background:#262B3A; font-size:.8rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading generator...")
def get_model(path, device):
    return load_generator(path, device)


def read_image(file):
    """PIL -> uint8 RGB, honouring EXIF rotation (important for phone photos)."""
    return np.array(ImageOps.exif_transpose(Image.open(file)).convert("RGB"))


def to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def brightness(img):
    return float(img.mean() / 255.0)


def contrast(img):
    return float(img.astype(np.float32).mean(axis=2).std() / 255.0)


def histogram_fig(before, after):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    fig.patch.set_facecolor("#0E1117"); ax.set_facecolor("#0E1117")
    for img, label, color in [(before, "Input", "#F87171"), (after, "Enhanced", "#34D399")]:
        gray = img.astype(np.float32).mean(axis=2).ravel()
        ax.hist(gray, bins=64, range=(0, 255), alpha=0.6, label=label, color=color)
    ax.set_xlabel("brightness (0-255)", color="#aaa"); ax.tick_params(colors="#aaa")
    ax.legend(facecolor="#171B26", labelcolor="#ddd", edgecolor="none")
    for s in ax.spines.values(): s.set_color("#333")
    fig.tight_layout()
    return fig


def show_compare(left, right, l_label, r_label):
    if SLIDER_OK:
        image_comparison(img1=Image.fromarray(left), img2=Image.fromarray(right),
                         label1=l_label, label2=r_label, width=900, starting_position=50,
                         show_labels=True, make_responsive=True, in_memory=True)
    else:
        c1, c2 = st.columns(2)
        c1.image(left, caption=l_label, use_container_width=True)
        c2.image(right, caption=r_label, use_container_width=True)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🌙 LowLight GAN")
st.sidebar.caption("U-Net generator + PatchGAN discriminator")

device = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"
ckpts = sorted(glob.glob("checkpoints/*/generator_best.pt"))
if ckpts:
    ck_path = st.sidebar.selectbox("Model checkpoint", ckpts, format_func=lambda p: Path(p).parent.name)
else:
    ck_path = None
    st.sidebar.warning("No trained checkpoint found in `checkpoints/*/generator_best.pt`. "
                       "Only classical baselines are available until you train the GAN.")
G = get_model(ck_path, device) if (ck_path and TORCH_OK) else None

max_side = st.sidebar.slider("Max image side (px)", 384, 2048, 640, 128,
                             help="Large photos are downscaled for speed / memory.")
st.sidebar.markdown(f"<span class='caption-pill'>device: {device}</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Tip:** take a photo in a dim room with your phone, upload it in the *Enhance* tab, "
                    "and drag the slider to compare.")

st.title("LowLight GAN Studio")
st.caption("Brighten dark photos with a conditional GAN and compare against classical methods.")
tab1, tab2, tab3 = st.tabs(["✨ Enhance a photo", "🧪 Simulate & measure", "📊 Results & about"])

# ---------------------------------------------------------------- tab 1: enhance
with tab1:
    c_src, c_opt = st.columns([3, 2])
    with c_src:
        src = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
        file = (st.file_uploader("Drop a dark photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
                if src == "Upload" else st.camera_input("Take a low-light photo"))
    with c_opt:
        strength = st.slider("Enhancement strength", 0.0, 1.0, 1.0, 0.05,
                             help="Blend between the original (0) and the GAN output (1).")
        show_base = st.checkbox("Also run classical baselines", value=True)

    if file is None:
        st.info("Upload or capture a dark image to begin.")
    else:
        img = resize_max_side(read_image(file), max_side)
        if G is None:
            st.error("No generator loaded - showing classical baselines only.")
            enh, dt = None, 0.0
        else:
            with st.spinner("Enhancing..."):
                t0 = time.time(); gan_out, _ = smart_enhance(G, img, device); dt = time.time() - t0
            enh = np.clip(strength * gan_out.astype(np.float32) + (1 - strength) * img.astype(np.float32),
                          0, 255).astype(np.uint8)

        if enh is not None:
            show_compare(img, enh, "Original", "GAN enhanced")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean brightness", f"{brightness(enh):.2f}", f"{brightness(enh) - brightness(img):+.2f}")
            m2.metric("Contrast (std)", f"{contrast(enh):.2f}", f"{contrast(enh) - contrast(img):+.2f}")
            m3.metric("Resolution", f"{img.shape[1]}x{img.shape[0]}")
            m4.metric("Inference time", f"{dt * 1000:.0f} ms")
            h1, h2 = st.columns([3, 2])
            h1.pyplot(histogram_fig(img, enh))
            h2.download_button("⬇️ Download enhanced PNG", to_png_bytes(enh), "enhanced.png", "image/png",
                               use_container_width=True)

        if show_base:
            st.subheader("Classical baselines")
            cols = st.columns(len(BASELINES))
            for col, (name, fn) in zip(cols, BASELINES.items()):
                col.image(fn(img), caption=name, use_container_width=True)

# ---------------------------------------------------------------- tab 2: simulate
with tab2:
    st.write("Upload a **well-lit** photo, darken it with the same degradation model used for training, "
             "and see exactly how much each method recovers (ground truth is known, so PSNR/SSIM are exact).")
    ref_file = st.file_uploader("Well-lit reference photo", type=["jpg", "jpeg", "png", "webp"], key="ref")
    s1, s2, s3 = st.columns(3)
    gamma = s1.slider("Gamma (higher = darker)", 1.2, 4.0, 2.5, 0.1)
    scale = s2.slider("Exposure scale", 0.15, 1.0, 0.6, 0.05)
    noise = s3.slider("Noise level", 0.0, 0.06, 0.015, 0.005)

    if ref_file is not None:
        gt = resize_max_side(read_image(ref_file), min(max_side, 768))
        dark = (darken(gt.astype(np.float32) / 255.0, np.random.default_rng(0), gamma=gamma, scale=scale,
                       read_sigma=noise, shot=noise * 0.1, color_cast=False) * 255).round().astype(np.uint8)
        results = {"Input (dark)": dark}
        results.update({n: f(dark) for n, f in BASELINES.items()})
        if G is not None:
            results["GAN"], _ = smart_enhance(G, dark, device, allow_api=False)

        cols = st.columns(len(results) + 1)
        for col, (name, im) in zip(cols, list(results.items()) + [("Ground truth", gt)]):
            col.image(im, caption=name, use_container_width=True)

        table = pd.DataFrame([{"Method": n, "PSNR (dB)": round(psnr(gt, im), 2), "SSIM": round(ssim(gt, im), 3)}
                              for n, im in results.items()]).set_index("Method")
        t1, t2 = st.columns([2, 3])
        t1.dataframe(table, use_container_width=True)
        t2.bar_chart(table["PSNR (dB)"])
    else:
        st.info("Upload a normally-exposed image to run the experiment.")

# ---------------------------------------------------------------- tab 3: results & about
with tab3:
    st.subheader("Quantitative results")
    rcsv = Path("results/results.csv")
    if rcsv.exists():
        st.dataframe(pd.read_csv(rcsv).round(3), use_container_width=True, hide_index=True)
        for img_path in sorted(glob.glob("results/*.png")):
            st.image(img_path, caption=Path(img_path).name, use_container_width=True)
    else:
        st.info("Run `python evaluate.py ...` to generate `results/results.csv` and figures; they show up here.")

    curves = sorted(glob.glob("checkpoints/*/curves.png"))
    if curves:
        st.subheader("Training curves")
        for c in curves:
            st.image(c, caption=Path(c).parent.name, use_container_width=True)

    st.subheader("How it works")
    st.markdown("""
**Data** - no paired dataset is needed: well-lit photos are darkened on the fly
(`gamma curve -> exposure scale -> colour cast -> shot + read noise -> 8-bit quantisation`),
so the network sees a new degradation every epoch.

**Generator** - 6-level U-Net with skip connections (keeps fine detail and colour).
**Discriminator** - conditional 70x70 PatchGAN: judges *(dark input, candidate)* patch by patch.
**Loss** - `100 x L1 + LSGAN adversarial loss` (optionally + VGG perceptual loss).

**Evaluation** - PSNR / SSIM against Histogram Equalisation, CLAHE and Auto-Gamma on a held-out synthetic set
and on real paired images from the LOL dataset.
""")
