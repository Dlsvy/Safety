"""ETH3D 장면(레이저 스캔 정답)으로 3D 모델의 정확도·완전도 평가 — 실내 성능 확인용.

python tools/eval_eth3d.py SCENE_DIR OUT_DIR [--mesh site_model_zup.obj|site_filled.glb] [--tol 0.02 0.05 0.1]
  SCENE_DIR: dslr_calibration_undistorted/images.txt, dslr_scan_eval/scan*.ply + scan_alignment.mlp
  OUT_DIR  : photo3d run 결과 폴더(work/sfm/georef 의 카메라로 정답 좌표에 7-파라미터 정합)
지표(ETH3D 방식 단순화):
  정확도 = 우리 표면 점 중 정답 스캔에서 τ 이내인 비율
  완전도 = 정답 스캔 점 중 우리 표면에서 τ 이내인 비율(우리 표면 0.5 m 이내 = 사진에 찍힌 범위 근처만)
  F1     = 둘의 조화평균
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pycolmap
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from photo3d.scale import umeyama  # noqa: E402


def gt_cams(images_txt: Path) -> dict:
    out = {}
    lines = [ln for ln in images_txt.read_text().splitlines() if ln and not ln.startswith("#")]
    for ln in lines[0::2]:
        t = ln.split()
        q = np.array(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        R = pycolmap.Rotation3d(np.array([q[1], q[2], q[3], q[0]])).matrix()
        out[Path(t[9]).stem.upper()] = -R.T @ tv
    return out


def read_mlp(p: Path) -> dict:
    res = {}
    for m in re.finditer(r'filename="([^"]+)".*?<MLMatrix44>(.*?)</MLMatrix44>', p.read_text(), re.S):
        res[m.group(1)] = np.array(m.group(2).split(), float).reshape(4, 4)
    return res


def load_gt(scene: Path, max_pts: int = 2_000_000) -> np.ndarray:
    ev = scene / "dslr_scan_eval"
    pts = []
    for f, M in read_mlp(ev / "scan_alignment.mlp").items():
        v = np.asarray(trimesh.load(ev / f, process=False).vertices, float)
        pts.append(v @ M[:3, :3].T + M[:3, 3])
    P = np.concatenate(pts)
    if len(P) > max_pts:
        P = P[np.random.default_rng(0).choice(len(P), max_pts, replace=False)]
    return P


def evaluate(scene: Path, out: Path, mesh: str = "site_model_zup.obj", tols=(0.02, 0.05, 0.10),
             near_ref: str | None = None) -> dict:
    G = gt_cams(scene / "dslr_calibration_undistorted" / "images.txt")
    rec = pycolmap.Reconstruction(out / "work" / "sfm" / "georef")
    src, dst = [], []
    for i in rec.reg_image_ids():
        im = rec.image(i)
        key = next((k for k in G if k in Path(im.name).stem.upper()), None)
        if key:
            src.append(im.projection_center())
            dst.append(G[key])
    src, dst = np.array(src), np.array(dst)
    s, R, t = umeyama(src, dst)
    cam_err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    mp = out / mesh
    m = trimesh.load(mp, force="mesh", process=False)
    V = np.asarray(m.vertices, float)
    if mp.suffix == ".glb":  # GLB 는 Y-up → Z-up
        V = V @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float).T
    m = trimesh.Trimesh(V, m.faces, process=False)
    P = trimesh.sample.sample_surface_even(m, 1_000_000)[0]
    P = (s * (R @ P.T)).T + t
    gt = load_gt(scene)
    d_gt, _ = cKDTree(P).query(gt, workers=-1)
    d_our, _ = cKDTree(gt).query(P, workers=-1)
    near = d_gt < 0.5
    if near_ref:  # 완전도 분모를 고정(여러 결과를 같은 정답 범위로 비교)
        mr = trimesh.load(out / near_ref, force="mesh", process=False)
        Pr = trimesh.sample.sample_surface_even(trimesh.Trimesh(mr.vertices, mr.faces, process=False), 1_000_000)[0]
        near = cKDTree((s * (R @ Pr.T)).T + t).query(gt, workers=-1)[0] < 0.5
    res = {"mesh": mesh, "cameras_matched": int(len(src)), "camera_align_err_mm_median": round(float(np.median(cam_err) * 1000), 1),
           "gt_near_ratio": round(float(near.mean()) * 100, 1)}
    for tol in tols:
        acc = float((d_our < tol).mean())
        comp = float((d_gt[near] < tol).mean())
        res[f"{round(tol * 100)}cm"] = {"정확도": round(acc * 100, 1), "완전도": round(comp * 100, 1),
                                        "F1": round(200 * acc * comp / max(acc + comp, 1e-9), 1)}
    res["median_dist_mm"] = round(float(np.median(d_our) * 1000), 1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("out")
    ap.add_argument("--mesh", default="site_model_zup.obj")
    ap.add_argument("--tol", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    a = ap.parse_args()
    print(json.dumps(evaluate(Path(a.scene), Path(a.out), a.mesh, a.tol), ensure_ascii=False, indent=1))
