"""
smart_enhance.py  -  drop next to inference.py / app.py

Pipeline
  1. Local GAN with flip test-time-augmentation (TTA)      -> less artefacts
  2. Light post-processing (adaptive gamma, denoise, CLAHE, colour-cast fix)
  3. No-reference quality check (brightness, contrast, noise, cast, clipping)
  4. If still below the bar AND an API is configured -> try the remote model,
     keep whichever result scores higher.

The result always reports which path produced it (info["source"]), so the UI
can show it. Remote use is OFF unless you set LLGAN_API_URL.

Config (env vars or .streamlit/secrets.toml):
  LLGAN_API_URL   endpoint that accepts raw image bytes (POST) and returns image bytes
                  (Hugging Face Inference-style). Not tested against any specific provider.
  LLGAN_API_KEY   bearer token, if the endpoint needs one
"""

import os

import cv2
import numpy as np

from inference import enhance as gan_enhance

QUALITY_THRESHOLD = 0.70  # below this -> considered "not up to the mark"


# ----------------------------------------------------------------- quality
def quality_report(img):
    """No-reference quality metrics for a uint8 RGB image. Returns dict with 'score' in [0,1]."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = gray.mean() / 255.0
    contrast = gray.std() / 255.0
    # noise: robust MAD of the Laplacian response (Immerkaer-style)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    noise = np.median(np.abs(lap - np.median(lap))) / 0.6745 / 255.0
    ch = img.reshape(-1, 3).mean(0)
    cast = float(ch.max() - ch.min()) / 255.0
    clip = float(((gray < 3) | (gray > 252)).mean())

    s_bright = 1.0 - min(abs(mean - 0.45) / 0.30, 1.0)   # ideal mean ~0.45
    s_contrast = min(contrast / 0.20, 1.0)                # want std >= 0.20
    s_noise = 1.0 - min(noise / 0.04, 1.0)
    s_cast = 1.0 - min(cast / 0.20, 1.0)
    s_clip = 1.0 - min(clip / 0.15, 1.0)
    score = 0.35 * s_bright + 0.2 * s_contrast + 0.2 * s_noise + 0.15 * s_cast + 0.1 * s_clip
    return dict(score=float(score), mean=float(mean), contrast=float(contrast),
                noise=float(noise), cast=cast, clip=clip)


# ------------------------------------------------------------- local GAN
def gan_tta(G, img, device="cpu"):
    """Average the GAN output over original + horizontal flip (cheap 2x TTA)."""
    a = gan_enhance(G, img, device).astype(np.float32)
    b = gan_enhance(G, np.ascontiguousarray(img[:, ::-1]), device)[:, ::-1].astype(np.float32)
    return np.clip((a + b) / 2, 0, 255).astype(np.uint8)


def _adaptive_gamma(img, target=0.45):
    mean = max(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean() / 255.0, 1e-3)
    if abs(mean - target) < 0.07:
        return img
    gamma = np.clip(np.log(target) / np.log(mean), 0.5, 2.0)
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _fix_cast(img, strength=0.6):
    """Gentle gray-world correction (only partially, to keep natural colour)."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    gain = means.mean() / np.maximum(means, 1e-3)
    gain = 1.0 + (gain - 1.0) * strength
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def postprocess(img, rep):
    out = img
    if rep["noise"] > 0.015:
        h = int(np.clip(rep["noise"] * 400, 3, 10))
        out = cv2.fastNlMeansDenoisingColored(out, None, h, h, 5, 15)
    if rep["cast"] > 0.08:
        out = _fix_cast(out)
    out = _adaptive_gamma(out)
    if rep["contrast"] < 0.16:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out


# ---------------------------------------------------------------- remote
def _cfg(name):
    v = os.getenv(name)
    if v:
        return v
    try:  # streamlit secrets, if running inside streamlit
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def remote_enhance(img, timeout=60):
    """POST the image to the configured API. Returns uint8 RGB or None on any failure."""
    url = _cfg("LLGAN_API_URL")
    if not url:
        return None
    try:
        import requests
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        headers = {"Content-Type": "image/png"}
        key = _cfg("LLGAN_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = requests.post(url, data=buf.tobytes(), headers=headers, timeout=timeout)
        r.raise_for_status()
        out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if out is None:
            return None
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        if out.shape[:2] != img.shape[:2]:
            out = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        return out
    except Exception as e:  # network down, bad key, wrong format ...
        print(f"[smart_enhance] remote API failed: {e}")
        return None


# ------------------------------------------------------------------ main
def smart_enhance(G, img, device="cpu", threshold=QUALITY_THRESHOLD, allow_api=True):
    """
    Returns (result_uint8, info).  info = {source, score, report, tried}
    source is one of: "local", "local+post", "api"
    """
    tried = []
    raw = gan_tta(G, img, device)
    rep_raw = quality_report(raw)
    tried.append(("local", rep_raw["score"]))
    best, best_rep, source = raw, rep_raw, "local"

    if rep_raw["score"] < threshold:
        post = postprocess(raw, rep_raw)
        rep_post = quality_report(post)
        tried.append(("local+post", rep_post["score"]))
        if rep_post["score"] > best_rep["score"]:
            best, best_rep, source = post, rep_post, "local+post"

    if best_rep["score"] < threshold and allow_api:
        remote = remote_enhance(img)
        if remote is not None:
            rep_r = quality_report(remote)
            tried.append(("api", rep_r["score"]))
            if rep_r["score"] > best_rep["score"]:
                best, best_rep, source = remote, rep_r, "api"

    return best, dict(source=source, score=best_rep["score"], report=best_rep, tried=tried)
