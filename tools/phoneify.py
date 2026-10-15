"""렌더 이미지를 폰 사진처럼: 렌즈 방사왜곡 + 센서 노이즈 + JPEG.

python tools/phoneify.py RENDER_DIR OUT_DIR --k1 -0.06 --k2 0.015
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--k1", type=float, default=-0.06)
ap.add_argument("--k2", type=float, default=0.015)
ap.add_argument("--noise", type=float, default=2.0)
ap.add_argument("--quality", type=int, default=90)
a = ap.parse_args()
dst = Path(a.dst)
dst.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
files = sorted(Path(a.src).glob("*.png"))
img = cv2.imread(str(files[0]))
h, w = img.shape[:2]
f = w / 36.0 * 26.0
K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
D = np.array([a.k1, a.k2, 0, 0, 0])
# 왜곡된 출력 픽셀 → 원본(핀홀) 픽셀 좌표 맵
uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
xd = (uu - w / 2) / f
yd = (vv - h / 2) / f
x, y = xd.copy(), yd.copy()
for _ in range(20):  # 역왜곡 반복 해법
    r2 = x * x + y * y
    rad = 1 + a.k1 * r2 + a.k2 * r2 * r2
    x, y = xd / rad, yd / rad
mx = (x * f + w / 2).astype(np.float32)
my = (y * f + h / 2).astype(np.float32)
for p in files:
    im = cv2.imread(str(p))
    out = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = np.clip(out + rng.normal(0, a.noise, out.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(dst / (p.stem + ".jpg")), out, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
print(f"{len(files)}장, 왜곡 k1={a.k1} k2={a.k2}")
