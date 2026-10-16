"""SfM 설정 비교 실험 — 같은 사진에 특징점·검증 기준·매퍼를 바꿔 돌려 등록률·모델 수·카메라 위치 오차(ETH3D 정답)를 비교.

python tools/sfm_bench.py SCENE_DIR [SCENE_DIR...] --side 1600 --out bench.json
SCENE_DIR: images/(사진) + dslr_calibration_undistorted/images.txt(정답 카메라, 있으면 오차 계산)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from photo3d.scale import umeyama  # noqa: E402

FEATS = {
    "sift8k": dict(n=8000),
    "sift8k_low": dict(n=8000, peak=0.002),           # 대비 낮은 흰 벽에서도 특징점
    "sifthq12k": dict(n=12000, affine=True, dsp=True),  # 어파인+DSP
    "sifthq12k_low": dict(n=12000, affine=True, dsp=True, peak=0.002),
}
VERIFY = {"strict40": 40, "std15": 15}
MAPPERS = ["global", "incr_strict", "incr_relaxed"]


def prep(scene: Path, side: int, work: Path) -> Path:
    out = work / "img"
    if out.exists():
        return out
    out.mkdir(parents=True)
    for p in sorted(scene.rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "thumb" not in p.name:
            im = cv2.imread(str(p))
            s = side / max(im.shape[:2])
            if s < 1:
                im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / (p.stem + ".jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def gt_centers(scene: Path) -> dict:
    f = next(scene.rglob("images.txt"), None)
    if f is None:
        return {}
    out = {}
    lines = [ln for ln in f.read_text().splitlines() if ln and not ln.startswith("#")]
    for ln in lines[0::2]:
        t = ln.split()
        q = np.array(list(map(float, t[1:5])))
        R = pycolmap.Rotation3d(np.array([q[1], q[2], q[3], q[0]])).matrix()
        out[Path(t[9]).stem] = -R.T @ np.array(list(map(float, t[5:8])))
    return out


def extract(img: Path, db: Path, fk: str):
    c = FEATS[fk]
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = c["n"]
    ext.max_image_size = 4000
    if c.get("affine"):
        ext.sift.estimate_affine_shape = True
        ext.sift.domain_size_pooling = True
    if c.get("peak"):
        ext.sift.peak_threshold = c["peak"]
    rd = pycolmap.ImageReaderOptions()
    rd.camera_model = "OPENCV"
    pycolmap.extract_features(db, img, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=rd, extraction_options=ext)


def match(db: Path, fk: str, inl: int):
    m = pycolmap.FeatureMatchingOptions()
    if FEATS[fk].get("affine"):
        m.guided_matching = True
    tv = pycolmap.TwoViewGeometryOptions()
    tv.min_num_inliers = inl
    tv.use_degensac = True
    pycolmap.match_exhaustive(db, matching_options=m, verification_options=tv)


def mapping(db: Path, img: Path, sp: Path, mk: str):
    if mk == "global":
        return pycolmap.global_mapping(db, img, sp)
    o = pycolmap.IncrementalPipelineOptions()
    o.ba_refine_principal_point = False
    if mk == "incr_strict":
        o.min_num_matches, o.mapper.abs_pose_min_num_inliers = 30, 60
    else:
        o.min_num_matches, o.mapper.abs_pose_min_num_inliers = 15, 20
        o.structure_less_registration_fallback = True
    return pycolmap.incremental_mapping(db, img, sp, options=o)


def score(recs: dict, n: int, gt: dict) -> dict:
    recs = list(recs.values())
    if not recs:
        return {"reg": 0, "models": 0}
    best = max(recs, key=lambda r: r.num_reg_images())
    res = {"reg": round(100 * best.num_reg_images() / n, 1), "models": len(recs),
           "pts_per_img": round(float(np.mean([best.image(i).num_points3D for i in best.reg_image_ids()])), 1),
           "reproj": round(best.compute_mean_reprojection_error(), 3)}
    if gt:
        S, D = [], []
        for i in best.reg_image_ids():
            k = Path(best.image(i).name).stem
            if k in gt:
                S.append(best.image(i).projection_center())
                D.append(gt[k])
        if len(S) >= 3:
            S, D = np.array(S), np.array(D)
            s, R, t = umeyama(S, D)
            e = np.linalg.norm((s * (R @ S.T)).T + t - D, axis=1)
            res["pos_err_mm_med"] = round(float(np.median(e) * 1000), 1)
            res["pos_err_mm_max"] = round(float(e.max() * 1000), 1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--side", type=int, default=1600)
    ap.add_argument("--feats", nargs="+", default=list(FEATS))
    ap.add_argument("--verify", nargs="+", default=list(VERIFY))
    ap.add_argument("--mappers", nargs="+", default=MAPPERS)
    ap.add_argument("--work", default="bench_work")
    ap.add_argument("--out", default="bench.json")
    a = ap.parse_args()
    pycolmap.logging.minloglevel = 2
    allres = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for sc in map(Path, a.scenes):
        W = Path(a.work) / sc.name
        img = prep(sc / "images", a.side, W)
        n = len(list(img.iterdir()))
        gt = gt_centers(sc)
        for fk in a.feats:
            base = W / f"{fk}.db"
            if not base.exists():
                t0 = time.time(); extract(img, base, fk); tf = time.time() - t0
            else:
                tf = 0
            for vk in a.verify:
                db = W / f"{fk}_{vk}.db"
                if not db.exists():
                    shutil.copy(base, db)
                    t0 = time.time(); match(db, fk, VERIFY[vk]); tm = time.time() - t0
                for mk in a.mappers:
                    key = f"{sc.name}|{fk}|{vk}|{mk}"
                    if key in allres:
                        continue
                    sp = W / f"sp_{fk}_{vk}_{mk}"
                    shutil.rmtree(sp, ignore_errors=True); sp.mkdir(parents=True)
                    t0 = time.time()
                    try:
                        r = score(mapping(db, img, sp, mk), n, gt)
                    except Exception as e:
                        r = {"error": str(e)[:120]}
                    r["map_s"] = round(time.time() - t0, 1)
                    allres[key] = r
                    print(key, r, flush=True)
                    Path(a.out).write_text(json.dumps(allres, ensure_ascii=False, indent=1))
