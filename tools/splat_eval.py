"""스플랫 화질 비교용 간단 렌더러(CPU, numpy) — 학습에 안 쓴 사진 위치에서 그려 원본 사진과 PSNR·SSIM 비교.

python tools/splat_eval.py SPLAT.ply DATASET_ALL HOLDOUT_NAMES... [--save out_dir]
DATASET_ALL: images/ + sparse/0 (모든 사진 카메라 포함, 학습셋과 같은 해상도)
서로 다른 학습 프로그램(Brush·Spirula 등)을 같은 렌더러로 공정하게 비교하려는 도구(속도·정밀도는 실제 뷰어보다 낮음).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from photo3d.splatclean import read_splat  # noqa: E402

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435]


def sh_color(dc, rest, d):
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    c = C0 * dc
    n = rest.shape[1]
    if n >= 3:
        c = c - C1 * y[:, None] * rest[:, 0] + C1 * z[:, None] * rest[:, 1] - C1 * x[:, None] * rest[:, 2]
    if n >= 8:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        c = (c + C2[0] * xy[:, None] * rest[:, 3] + C2[1] * yz[:, None] * rest[:, 4]
             + C2[2] * (2 * zz - xx - yy)[:, None] * rest[:, 5] + C2[3] * xz[:, None] * rest[:, 6]
             + C2[4] * (xx - yy)[:, None] * rest[:, 7])
        if n >= 15:
            c = (c + C3[0] * (y * (3 * xx - yy))[:, None] * rest[:, 8] + C3[1] * (xy * z)[:, None] * rest[:, 9]
                 + C3[2] * (y * (4 * zz - xx - yy))[:, None] * rest[:, 10]
                 + C3[3] * (z * (2 * zz - 3 * xx - 3 * yy))[:, None] * rest[:, 11]
                 + C3[4] * (x * (4 * zz - xx - yy))[:, None] * rest[:, 12]
                 + C3[5] * (z * (xx - yy))[:, None] * rest[:, 13] + C3[6] * (x * (xx - 3 * yy))[:, None] * rest[:, 14])
    return np.clip(c + 0.5, 0, 1)


def load(path: Path):
    a, props, _ = read_splat(path)
    col = {q: i for i, q in enumerate(props)}
    g = lambda k: a[:, col[k]]  # noqa: E731
    xyz = np.stack([g("x"), g("y"), g("z")], 1).astype(np.float64)
    dc = np.stack([g("f_dc_0"), g("f_dc_1"), g("f_dc_2")], 1)
    nrest = len([q for q in props if q.startswith("f_rest_")]) // 3
    rest = np.zeros((len(a), nrest, 3), np.float32)
    for k in range(nrest):  # INRIA 순서: 채널별로 [R0..Rn, G0..Gn, B0..Bn]
        for ch in range(3):
            rest[:, k, ch] = g(f"f_rest_{ch * nrest + k}")
    op = 1 / (1 + np.exp(-g("opacity")))
    sc = np.exp(np.stack([g("scale_0"), g("scale_1"), g("scale_2")], 1)).astype(np.float64)
    q = np.stack([g("rot_0"), g("rot_1"), g("rot_2"), g("rot_3")], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return xyz, dc, rest, op, sc, q


def quat_mat(q):
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def render(splat, K, R, t, W, H, bg=0.0):
    xyz, dc, rest, op, sc, q = splat
    pc = xyz @ R.T + t
    z = pc[:, 2]
    keep = z > 0.05
    pc, z = pc[keep], z[keep]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = fx * pc[:, 0] / z + cx, fy * pc[:, 1] / z + cy
    Rq = quat_mat(q[keep])
    M = Rq * sc[keep][:, None, :]
    cov = M @ M.transpose(0, 2, 1)
    covc = R @ cov @ R.T
    J = np.zeros((len(z), 2, 3))
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * pc[:, 0] / z ** 2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * pc[:, 1] / z ** 2
    c2 = J @ covc @ J.transpose(0, 2, 1)
    c2[:, 0, 0] += 0.3
    c2[:, 1, 1] += 0.3
    det = c2[:, 0, 0] * c2[:, 1, 1] - c2[:, 0, 1] ** 2
    ia, ib, ic = c2[:, 1, 1] / det, -c2[:, 0, 1] / det, c2[:, 0, 0] / det
    rad = 3 * np.sqrt(np.maximum(c2[:, 0, 0], c2[:, 1, 1]))
    vis = (det > 0) & (u + rad > 0) & (u - rad < W) & (v + rad > 0) & (v - rad < H) & (rad < 4 * max(W, H))
    cam_c = -R.T @ t
    d = xyz[keep] - cam_c
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    colr = sh_color(dc[keep], rest[keep], d)
    o = op[keep]
    img = np.zeros((H, W, 3))
    T = np.ones((H, W))
    for i in np.where(vis)[0][np.argsort(z[vis])]:
        x0, x1 = int(max(0, u[i] - rad[i])), int(min(W, u[i] + rad[i] + 1))
        y0, y1 = int(max(0, v[i] - rad[i])), int(min(H, v[i] + rad[i] + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1) + 0.5 - u[i]
        ys = np.arange(y0, y1) + 0.5 - v[i]
        dx, dy = np.meshgrid(xs, ys)
        a = np.minimum(0.99, o[i] * np.exp(-0.5 * (ia[i] * dx * dx + 2 * ib[i] * dx * dy + ic[i] * dy * dy)))
        a[a < 1 / 255] = 0
        Tp = T[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += (Tp * a)[..., None] * colr[i]
        T[y0:y1, x0:x1] = Tp * (1 - a)
    return np.clip(img + T[..., None] * bg, 0, 1)


def psnr(a, b):
    return float(10 * np.log10(1 / max(np.mean((a - b) ** 2), 1e-10)))


def ssim(a, b):
    ga, gb = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0, \
        cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    mu_a, mu_b = cv2.GaussianBlur(ga, (11, 11), 1.5), cv2.GaussianBlur(gb, (11, 11), 1.5)
    va = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a ** 2
    vb = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b ** 2
    cab = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * mu_a * mu_b + c1) * (2 * cab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean())


def evaluate(splat_path: Path, ds: Path, names: list[str], save: Path | None = None) -> dict:
    sp = load(splat_path)
    rec = pycolmap.Reconstruction(ds / "sparse" / "0")
    res = {}
    for i in rec.reg_image_ids():
        im = rec.image(i)
        if im.name not in names:
            continue
        cam = rec.camera(im.camera_id)
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R, t = T.rotation.matrix(), T.translation
        out = render(sp, cam.calibration_matrix(), R, t, cam.width, cam.height)
        gt = cv2.imread(str(ds / "images" / im.name))[:, :, ::-1] / 255.0
        res[im.name] = {"psnr": round(psnr(out, gt), 2), "ssim": round(ssim(out, gt), 3)}
        if save:
            save.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save / f"{Path(im.name).stem}.jpg"), (np.concatenate([out, gt], 1)[:, :, ::-1] * 255).astype(np.uint8))
    res["mean"] = {k: round(float(np.mean([v[k] for v in res.values()])), 3) for k in ("psnr", "ssim")}
    res["splats"] = int(len(sp[0]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("splat")
    ap.add_argument("dataset")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--save")
    a = ap.parse_args()
    import json
    print(json.dumps(evaluate(Path(a.splat), Path(a.dataset), a.names, Path(a.save) if a.save else None)))
