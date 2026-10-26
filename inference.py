"""Helpers to run the generator on arbitrary-size uint8 RGB images."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from baselines import resize_max_side  # noqa: F401  (re-exported)


def tensor_to_uint8(t):
    """CHW tensor in [-1, 1] -> HWC uint8."""
    t = ((t.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).round().byte()
    return t.permute(1, 2, 0).numpy()


@torch.no_grad()
def enhance(G, img, device="cpu", multiple=64, min_size=128):
    """uint8 HxWx3 -> enhanced uint8 HxWx3 (same size). Pads internally to a valid U-Net size."""
    H, W = img.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(127.5).sub(1.0)
    x = x.unsqueeze(0).to(device)
    th = max(min_size, int(np.ceil(H / multiple)) * multiple)
    tw = max(min_size, int(np.ceil(W / multiple)) * multiple)
    ph, pw = th - H, tw - W
    if ph or pw:
        mode = "reflect" if (ph < H and pw < W) else "replicate"
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    y = G(x)[:, :, :H, :W]
    return tensor_to_uint8(y[0])
