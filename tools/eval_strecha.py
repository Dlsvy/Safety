"""실사 정확도 평가 — Strecha 벤치마크(실제 건물 DSLR 사진 + 레이저스캔 정합 정답 카메라).

python tools/eval_strecha.py SCENE_DIR OUT_DIR [--mapper incremental|global] [--features sift|aliked]
  SCENE_DIR: .../Benchmarking_Camera_Calibration_2008/castle-P30 (images/, gt_dense_cameras/)

지표
 1) 형상: 복원 카메라 중심 ↔ 정답 중심 7-파라미터 정합 후 잔차 (mm)
 2) 줄자 1개 축척: 가장 먼 두 카메라 간 실거리 1개로만 축척 → 강체 정합 후 잔차 (현장 줄자 방식 모의)
 3) 렌즈 보정: 추정 초점거리 vs 정답 (%)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from photo3d import frames, sfm  # noqa: E402
from photo3d.scale import umeyama  # noqa: E402


def read_gt(scene: Path):
    gt = {}
    for f in sorted((scene / "gt_dense_cameras").glob("*.camera")):
        v = np.loadtxt(f, max_rows=8)
        K, C = v[0:3], v[7]
        gt[f.name.replace(".camera", "")] = {"K": K, "C": C}
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("out")
    ap.add_argument("--mapper", default="incremental")
    ap.add_argument("--camera-model", default="OPENCV")
    ap.add_argument("--refine-pp", action="store_true")
    ap.add_argument("--no-robust", action="store_true")
    ap.add_argument("--keep", type=int, default=0, help="사진을 고르게 N장만 사용(적은 장수 시험)")
    ap.add_argument("--features", default="sift", choices=["sift", "sift_hq", "sift_lg", "aliked"])
    a = ap.parse_args()
    scene, out = Path(a.scene), Path(a.out)
    gt = read_gt(scene)
    t0 = time.time()
    img_dir = out / "images"
    if img_dir.exists():
        import shutil
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    # 원본 이름 유지(정답 매칭용)
    import shutil
    allp = sorted((scene / "images").glob("*.jpg"))
    if a.keep and a.keep < len(allp):
        allp = [allp[int(round(i))] for i in np.linspace(0, len(allp) - 1, a.keep)]
    for p in allp:
        shutil.copy2(p, img_dir / p.name)
    rec, st = sfm.run_sfm(img_dir, out / "sfm", sequential=False, mapper=a.mapper,
                           features=a.features, camera_model=a.camera_model,
                           refine_pp=a.refine_pp, robust=not a.no_robust)
    names, src, dst = [], [], []
    for iid in rec.reg_image_ids():
        im = rec.image(iid)
        if im.name in gt:
            names.append(im.name)
            src.append(im.projection_center())
            dst.append(gt[im.name]["C"])
    src, dst = np.array(src), np.array(dst)
    s, R, t = umeyama(src, dst)
    res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    # 줄자 1개 모의: 가장 먼 정답 카메라 쌍의 거리로 축척
    D = np.linalg.norm(dst[:, None] - dst[None], axis=2)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    s1 = D[i, j] / np.linalg.norm(src[i] - src[j])
    _, R1, t1 = umeyama(src * s1, dst, with_scale=False)
    res1 = np.linalg.norm((R1 @ (src * s1).T).T + t1 - dst, axis=1)
    extent = float(D.max())
    cam = rec.camera(rec.image(rec.reg_image_ids()[0]).camera_id)
    f_est = float(cam.mean_focal_length())
    f_gt = float(np.mean([g["K"][0, 0] for g in gt.values()]))  # 원본 3072px 기준
    f_gt *= cam.width / 3072.0
    r = {
        "scene": scene.name, "mapper": a.mapper, "features": a.features, "camera_model": a.camera_model, "refine_pp": a.refine_pp,
        "pp_est": [round(float(v), 1) for v in (cam.principal_point_x, cam.principal_point_y)],
        "registered": f"{len(names)}/{len(allp)}",
        "reproj_px": round(st["mean_reprojection_error_px"], 3),
        "scene_extent_m": round(extent, 2),
        "shape_sim3": {"rmse_mm": round(float(np.sqrt((res ** 2).mean()) * 1000), 1),
                       "max_mm": round(float(res.max() * 1000), 1)},
        "tape_scale_1dist": {"rmse_mm": round(float(np.sqrt((res1 ** 2).mean()) * 1000), 1),
                             "scale_err_pct": round(float((s1 / s - 1) * 100), 3)},
        "relative_rmse_ppm_of_extent": round(float(np.sqrt((res ** 2).mean()) / extent * 1e6), 0),
        "focal": {"est_px": round(f_est, 1), "gt_px": round(f_gt, 1),
                  "err_pct": round((f_est / f_gt - 1) * 100, 3)},
        "distortion_est": [round(float(v), 5) for v in cam.params[4:]],
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(r, open(out / "eval.json", "w"), indent=2)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
