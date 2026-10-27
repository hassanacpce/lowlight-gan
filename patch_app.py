import re
from pathlib import Path

p = Path("app.py")
s = p.read_text(encoding="utf-8")

if "smart_enhance" in s:
    raise SystemExit("app.py is already patched.")

Path("app.py.bak").write_text(s, encoding="utf-8")

# 1) add the import right under 'from inference import enhance' (keeps the same indentation)
s, n1 = re.subn(
    r"^([ \t]*)from inference import enhance[ \t]*$",
    r"\1from inference import enhance\n\1from smart_enhance import smart_enhance",
    s, count=1, flags=re.M,
)

# 2) main Enhance tab
s, n2 = re.subn(
    r"gan_out = enhance\(G, img, device\)",
    "gan_out, _ = smart_enhance(G, img, device)",
    s,
)

# 3) Simulate & measure tab (no API, so PSNR/SSIM measure your model only)
s, n3 = re.subn(
    r'results\["GAN"\] = enhance\(G, dark, device\)',
    'results["GAN"], _ = smart_enhance(G, dark, device, allow_api=False)',
    s,
)

p.write_text(s, encoding="utf-8")
print(f"import: {n1}  enhance tab: {n2}  simulate tab: {n3}  (each should be 1)")