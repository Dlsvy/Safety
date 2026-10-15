"""마커 가상현장에서 축척·정합 방식별 정확도 비교 (SfM 1회 + 정합 4방식).

python tools/eval_synth_georef.py GT_DIR IMAGES_DIR OUT_DIR
  A marker   : 마커 크기(0.5m)만
  B tape     : 마커 + 줄자 긴 거리 2개(--scale-dist)
  C gcp      : GCP 4점 유사변환 (2점은 검사점)
  D gcp_ba   : GCP 4점을 번들조정에 고정 (2점은 검사점)
지표: 카메라 중심 오차(정답 포즈 대비, 강체/좌표계 그대로), 검사점 오차, 축척 오차
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from photo3d import scale, sfm  # noqa: E402
from photo3d.scale import umeyama  # noqa: E402

gtd, imgs, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
gt = json.load(open(gtd / "gt.json"))
M = {int(k): np.array(v) for k, v in gt["markers"].items()}
gt_c = {p["name"].replace(".png", ".jpg"): np.array(p["location"]) for p in gt["poses"]}

CHECK = [2, 4]
CTRL = [m for m in M if m not in CHECK]
with open(out / "gcp.csv", "w") as f:
    for m, X in M.items():
        f.write(f"{m},{X[0]},{X[1]},{X[2]}" + (",check\n" if m in CHECK else "\n"))
with open(out / "scale.csv", "w") as f:  # 줄자 긴 거리 2개
    for a, b in [(1, 2), (0, 3)]:
        f.write(f"{a},{b},{np.linalg.norm(M[a] - M[b]):.4f}\n")
with open(out / "check.csv", "w") as f:  # 독립 검증 거리
    for a, b in [(2, 4), (4, 5), (1, 3)]:
        f.write(f"{a},{b},{np.linalg.norm(M[a] - M[b]):.4f}\n")

rec0, st = sfm.run_sfm(imgs, out / "sfm", sequential=False)
base = out / "sfm" / "base"
base.mkdir(parents=True, exist_ok=True)
rec0.write(base)
print("SfM", st["images_registered"], "/", st["images_input"], "reproj", round(st["mean_reprojection_error_px"], 3))


def cam_err(rec, rigid_align, offset=0.0):
    names = [rec.image(i).name for i in rec.reg_image_ids()]
    src = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()]) + offset
    dst = np.array([gt_c[n] for n in names])
    if rigid_align:  # 마커 모드는 좌표계가 0번 마커 기준이라 회전·이동만 맞춤(축척은 그대로 평가)
        _, R, t = umeyama(src, dst, with_scale=False)
        src = (R @ src.T).T + t
    e = np.linalg.norm(src - dst, axis=1)
    return {"rmse_mm": round(float(np.sqrt((e ** 2).mean()) * 1000), 1), "max_mm": round(float(e.max() * 1000), 1)}


results = {"sfm": {k: st[k] for k in ("images_registered", "images_input", "mean_reprojection_error_px")}}
variants = {
    "A_marker": dict(marker_size=0.5),
    "B_marker_tape": dict(marker_size=0.5, scale_csv=out / "scale.csv"),
    "C_gcp_sim3": dict(marker_size=0.5, gcp_csv=out / "gcp.csv", gcp_ba=False),
    "D_gcp_ba": dict(marker_size=0.5, gcp_csv=out / "gcp.csv", gcp_ba=True),
}
for name, kw in variants.items():
    rec = pycolmap.Reconstruction(base)
    g = scale.georeference(rec, imgs, kw.pop("marker_size"), "4x4_50", check_csv=out / "check.csv",
                           **kw, log=lambda m: print(f"[{name}]{m}"))
    r = {"method": g["method"], "markers": g.get("markers_triangulated"),
         "camera_center": cam_err(rec, rigid_align=g["method"].startswith("marker"),
                                  offset=np.array(g.get("coordinate_offset", [0, 0, 0]))),
         "check_distances_mm": {c["pair"]: c["error_mm"] for c in g.get("check_distances", [])},
         "checkpoint_mm": {k: round(v * 1000, 1) for k, v in g.get("checkpoint_error_m", {}).items()},
         "scale": g.get("sim3", {}).get("scale")}
    results[name] = r
json.dump(results, open(out / "georef_eval.json", "w"), indent=2, ensure_ascii=False, default=str)
print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
