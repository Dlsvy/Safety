"""가상현장 정답과 복원 결과 비교.

python tools/eval_accuracy.py OUT_DIR GT_DIR
 - OUT_DIR: photo3d 출력(site_model_zup.obj, report.json)
 - GT_DIR : synth_site_blender 출력(gt_mesh.ply, gt.json)

리포트:
 1) 표면 정확도: 복원 표면 → 정답 표면 거리 (점-평면), 대상별
 2) 완전도: 정답 표면 중 복원이 5/10cm 이내에 있는 비율, 대상별
 3) 치수: 건물 18x12, 슬래브 상단 9.25, 컨테이너 높이 2.59, 기둥 높이 6.0
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

out, gtd = Path(sys.argv[1]), Path(sys.argv[2])
rep = json.load(open(out / "report.json"))
gt = json.load(open(gtd / "gt.json"))
gt_mesh = trimesh.load(gtd / "gt_mesh.ply", process=False)
rec = trimesh.load(out / "site_model_zup.obj", process=False, force="mesh")

# ── 복원 → 정답 좌표 정렬 ──────────────────────────────
# 마커 모드는 0번 마커 중심을 원점으로 두므로, 마커 중심들로 '회전+이동'만(축척 X) 맞춘다.
# 축척을 추정하지 않으므로 축척 오차는 그대로 남아 평가된다.
mc = rep["georef"].get("marker_centers", {})
ids = sorted(set(map(int, mc)) & set(map(int, gt["markers"])))
src = np.array([mc[str(i)] if str(i) in mc else mc[i] for i in ids])
dst = np.array([gt["markers"][str(i)] for i in ids])
mu_s, mu_d = src.mean(0), dst.mean(0)
U, S, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
R = U @ D @ Vt
t = mu_d - R @ mu_s
marker_res = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
rot_deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
V = (R @ rec.vertices.T).T + t
rec = trimesh.Trimesh(V, rec.faces, process=False)

# ── 정답 표면 조밀 샘플 + 라벨 ───────────────────────────
BOXES = {  # (min, max)
    "building": ((-5.31, 7.69, 0.0), (13.31, 20.31, 12.76)),
    "container": ((-15.04, 0.78, 0.0), (-8.96, 3.22, 2.6)),
    "pallets": ((8.4, -4.9, 0.0), (11.0, -2.5, 1.41)),
    "pole": ((-6.16, -6.16, 0.0), (-5.84, -5.84, 6.01)),
}


def label(p):
    lab = np.full(len(p), "ground", dtype=object)
    for name, (lo, hi) in BOXES.items():
        m = np.all((p >= np.array(lo) - 0.02) & (p <= np.array(hi) + 0.02), axis=1) & (p[:, 2] > 0.02)
        lab[m] = name
    return lab


ROI = 32.0  # 촬영 궤적 안쪽 영역만 평가
gpts, gfi = trimesh.sample.sample_surface(gt_mesh, 2_000_000, seed=0)
gn = gt_mesh.face_normals[gfi]
keep = np.linalg.norm(gpts[:, :2] - np.array([1.0, 8.0]), axis=1) < ROI
gpts, gn = gpts[keep], gn[keep]
glab = label(gpts)
gtree = cKDTree(gpts)

rpts, _ = trimesh.sample.sample_surface(rec, 1_500_000, seed=0)
rpts = rpts[np.linalg.norm(rpts[:, :2] - np.array([1.0, 8.0]), axis=1) < ROI]
d, idx = gtree.query(rpts, workers=-1)
pp = np.abs(np.einsum("ij,ij->i", rpts - gpts[idx], gn[idx]))  # 점-평면
dist = np.where(d < 0.3, pp, d)  # 모서리 근처/엇나간 점은 점-점 거리
rlab = glab[idx]

rtree = cKDTree(rpts)
dc, _ = rtree.query(gpts, workers=-1)


def stats(x):
    return {"n": int(len(x)), "mean_mm": round(float(x.mean() * 1000), 1),
            "median_mm": round(float(np.median(x) * 1000), 1),
            "p90_mm": round(float(np.percentile(x, 90) * 1000), 1),
            "p95_mm": round(float(np.percentile(x, 95) * 1000), 1)}


res = {"alignment": {"markers_used": ids, "rotation_deg": round(float(rot_deg), 3),
                     "marker_center_residual_mm": [round(v * 1000, 1) for v in marker_res]},
       "accuracy_all": stats(dist), "accuracy_by_object": {}, "completeness": {}}
for name in ["ground", "building", "container", "pallets", "pole"]:
    m = rlab == name
    if m.sum() > 100:
        res["accuracy_by_object"][name] = stats(dist[m])
    g = glab == name
    if g.sum():
        res["completeness"][name] = {"within_5cm": round(float((dc[g] < 0.05).mean()), 3),
                                     "within_10cm": round(float((dc[g] < 0.10).mean()), 3)}
res["completeness"]["all"] = {"within_5cm": round(float((dc < 0.05).mean()), 3),
                              "within_10cm": round(float((dc < 0.10).mean()), 3)}


# ── 치수 측정 (복원 점만 사용) ────────────────────────────
def face_coord(axis, lo, hi, region, normal_axis_val):
    """region 박스 안 점 중 axis 좌표의 중앙값 (벽면 위치)"""
    m = np.all((rpts >= lo) & (rpts <= hi), axis=1)
    return float(np.median(rpts[m, axis])) if m.sum() > 50 else float("nan")


bw = 0.15  # 면 주변 탐색 두께
dims = {}
east = face_coord(0, (13 - bw, 9, 0.5), (13 + bw, 19, 2.8), None, None)
west = face_coord(0, (-5 - bw, 9, 0.5), (-5 + bw, 19, 2.8), None, None)
south = face_coord(1, (-4, 8 - bw, 0.5), (12, 8 + bw, 2.8), None, None)
north = face_coord(1, (-4, 20 - bw, 0.5), (12, 20 + bw, 2.8), None, None)
roof = face_coord(2, (-5.2, 7.8, 9.25 - bw), (13.2, 20.2, 9.25 + bw), None, None)
ctop = face_coord(2, (-14.5, 1.2, 2.59 - bw), (-9.5, 2.8, 2.59 + bw), None, None)
ptop = float(np.max(rpts[np.linalg.norm(rpts[:, :2] - np.array([-6, -6]), axis=1) < 0.25, 2],
                    initial=np.nan))
gz = face_coord(2, (-3, -3, -bw), (3, 3, bw), None, None)


def dim(name, meas, true):
    dims[name] = {"true_m": true, "measured_m": round(meas, 4),
                  "error_mm": round((meas - true) * 1000, 1),
                  "error_pct": round((meas - true) / true * 100, 3)}


dim("building_length_EW", east - west, 18.0)
dim("building_width_NS", north - south, 12.0)
dim("roof_slab_top_height", roof - gz, 9.25)
dim("container_height", ctop - gz, 2.59)
dim("pole_height", ptop - gz, 6.0)
res["dimensions"] = dims
res["pipeline_report"] = {k: rep.get(k) for k in ("grade", "elapsed_min")}
res["pipeline_report"]["sfm"] = {k: rep["sfm"][k] for k in (
    "images_input", "images_registered", "mean_reprojection_error_px", "points3D")}
res["pipeline_report"]["cameras"] = rep["sfm"]["cameras"]
json.dump(res, open(out / "accuracy_eval.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))
