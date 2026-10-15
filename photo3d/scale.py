"""축척·좌표 정합 — 이 프로그램 정확도의 핵심.

사진만으로 만든 모델은 크기(축척)와 방향을 모른다. 현장에 크기를 아는 ArUco 마커를
깔고 찍으면 마커 모서리 4점을 여러 사진에서 삼각측량해 다음을 얻는다.

1) 마커 크기만 아는 경우 (기본)  → m 단위 축척 + 수평(바닥면) + 원점(가장 작은 ID 마커)
2) 마커 중심의 현장 좌표(GCP)를 3개 이상 아는 경우 → 현장 좌표계(측량/도면 좌표)로 7-파라미터 정합,
   잔차와 leave-one-out 검사점 오차를 보고
3) 검증 거리(check.csv: id1,id2,거리m) → 줄자로 잰 실거리와 모델 거리 비교

좌표계 출력: 오른손, Z-up, 단위 m (라이노·도면과 같다). 유니티 반입 시 GLB 단계에서 Y-up 으로 바꾼다.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycolmap

ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def _pose(img: pycolmap.Image) -> np.ndarray:
    cfw = img.cam_from_world
    cfw = cfw() if callable(cfw) else cfw
    return np.asarray(cfw.matrix())  # 3x4


def detect_markers(rec: pycolmap.Reconstruction, image_dir: Path, dict_name: str, log=print):
    """반환: {(marker_id, corner_k): [(image_id, xy_px), ...]}"""
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name]),
        _det_params())
    obs = defaultdict(list)
    seen = defaultdict(int)
    for iid in rec.reg_image_ids():
        im = rec.image(iid)
        gray = cv2.imread(str(image_dir / im.name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            continue
        for c, mid in zip(corners, ids.ravel()):
            c = c.reshape(4, 2).astype(np.float32)
            cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
            seen[int(mid)] += 1
            for k in range(4):
                obs[(int(mid), k)].append((iid, c[k].astype(np.float64)))
    log(f"  마커 검출: {dict(sorted(seen.items()))} (ID: 검출된 사진 수)")
    return obs


def _det_params():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.adaptiveThreshWinSizeMax = 53
    p.minMarkerPerimeterRate = 0.01  # 멀리 있는 작은 마커도 잡는다
    return p


def triangulate(rec, observations, max_err_px=2.0):
    """다시점 DLT + 재투영오차 기반 이상치 제거. 반환 (X, 평균오차px, 사용 시점수) 또는 None."""
    Ps, xs, cams = [], [], []
    for iid, xy in observations:
        im = rec.image(iid)
        cam = rec.camera(im.camera_id)
        n = np.asarray(cam.cam_from_img(xy.reshape(1, 2))).reshape(-1)[:2]
        Ps.append(_pose(im))
        xs.append(n)
        cams.append((cam, xy))
    if len(Ps) < 2:
        return None
    use = np.ones(len(Ps), bool)
    for _ in range(3):
        if use.sum() < 2:
            return None
        A = []
        for P, x, u in zip(Ps, xs, use):
            if u:
                A.append(x[0] * P[2] - P[0])
                A.append(x[1] * P[2] - P[1])
        _, _, vt = np.linalg.svd(np.asarray(A))
        X = vt[-1][:3] / vt[-1][3]
        errs = []
        for P, (cam, xy) in zip(Ps, cams):
            pc = P[:, :3] @ X + P[:, 3]
            if pc[2] <= 0:
                errs.append(1e9)
                continue
            proj = np.asarray(cam.img_from_cam(pc.reshape(1, 3))).reshape(-1)[:2]
            errs.append(float(np.linalg.norm(proj - xy)))
        errs = np.asarray(errs)
        new = errs < max(max_err_px, 3 * np.median(errs[use]))
        if (new == use).all():
            break
        use = new
    if use.sum() < 2:
        return None
    return X, float(errs[use].mean()), int(use.sum())


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale=True):
    """dst ≈ s R src + t"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum() * len(src) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def read_gcp(path: Path) -> tuple[dict[int, np.ndarray], set[int]]:
    """gcp.csv: marker_id,x,y,z[,check]  (마커 중심의 현장 좌표, m, Z-up)
    5번째 열이 'check' 인 점은 정합에 쓰지 않고 독립 검사점으로만 쓴다."""
    out, checks = {}, set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#") or not row[0].strip().lstrip("-").isdigit():
                continue
            out[int(row[0])] = np.array([float(v) for v in row[1:4]])
            if len(row) > 4 and row[4].strip().lower() == "check":
                checks.add(int(row[0]))
    return out, checks


def _center_px(corner_obs: dict[int, dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """이미지별 마커 중심 픽셀 = 네 모서리 대각선 교점(원근 투영에서도 정확한 사각형 중심)."""
    out = {}
    for iid, c in corner_obs.items():
        if len(c) != 4:
            continue
        h = lambda p: np.array([p[0], p[1], 1.0])
        l1 = np.cross(h(c[0]), h(c[2]))
        l2 = np.cross(h(c[1]), h(c[3]))
        x = np.cross(l1, l2)
        if abs(x[2]) > 1e-9:
            out[iid] = x[:2] / x[2]
    return out


def gcp_bundle_adjust(rec: pycolmap.Reconstruction, obs, control: dict[int, np.ndarray], log=print) -> int:
    """GCP 를 고정점으로 넣고 전체 BA — 유사변환으로 못 잡는 휘어짐까지 보정 (ODM/메타샤 방식).

    obs: detect_markers 결과, control: {marker_id: 현장좌표(오프셋 뺀 값)}. 추가한 GCP 점 수를 반환.
    """
    per_marker: dict[int, dict[int, dict[int, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    for (mid, k), lst in obs.items():
        for iid, xy in lst:
            per_marker[mid][iid][k] = xy
    cfg = pycolmap.BundleAdjustmentConfig()
    for iid in rec.reg_image_ids():
        cfg.add_image(iid)
    added = []
    for mid, X in control.items():
        cps = _center_px(per_marker.get(mid, {}))
        if len(cps) < 2:
            continue
        track = pycolmap.Track()
        for iid, xy in cps.items():
            im = rec.image(iid)
            im.points2D.append(pycolmap.Point2D(np.asarray(xy, float)))
            track.add_element(iid, im.num_points2D() - 1)
        pid = rec.add_point3D(np.asarray(X, float), track, np.array([255, 0, 0], np.uint8))
        added.append(pid)
    if len(added) < 3:
        log("  GCP-BA 건너뜀: 2장 이상에서 보인 GCP 가 3개 미만")
        return 0
    for pid in rec.point3D_ids():
        if pid in added:
            cfg.add_constant_point(pid)
        else:
            cfg.add_variable_point(pid)
    opt = pycolmap.BundleAdjustmentOptions()
    opt.refine_principal_point = False
    pycolmap.create_default_bundle_adjuster(opt, cfg, rec).solve()
    for pid in added:  # 이후 MVS 에 영향 없도록 GCP 점 제거
        rec.delete_point3D(pid)
    log(f"  GCP-BA: 제어점 {len(added)}개를 고정해 전체 번들조정 완료")
    return len(added)


def read_checks(path: Path) -> list[tuple[int, int, float]]:
    """check.csv: id1,id2,distance_m  (줄자/레이저거리계로 잰 마커 중심 간 거리)"""
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip().isdigit():
                continue
            out.append((int(row[0]), int(row[1]), float(row[2])))
    return out


def georeference(rec: pycolmap.Reconstruction, image_dir: Path, marker_size: float | None,
                 dict_name: str = "4x4_50", gcp_csv: Path | None = None,
                 check_csv: Path | None = None, markers_on_ground: bool = True,
                 scale_csv: Path | None = None, gcp_ba: bool = True, log=print) -> dict:
    """rec 를 제자리에서 미터·Z-up 좌표로 변환하고 정확도 지표를 반환한다."""
    report: dict = {"method": "none", "warnings": []}
    obs = detect_markers(rec, image_dir, dict_name, log=log)
    corners: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    tri_err = []
    for (mid, k), o in obs.items():
        r = triangulate(rec, o)
        if r is None:
            continue
        corners[mid][k] = r[0]
        tri_err.append(r[1])
    markers = {m: np.array([c[k] for k in range(4)]) for m, c in corners.items() if len(c) == 4}
    report["markers_triangulated"] = sorted(markers)
    report["marker_corner_reproj_px"] = float(np.mean(tri_err)) if tri_err else None

    gcp, gcp_checks = read_gcp(gcp_csv) if gcp_csv else ({}, set())
    gcp_common = sorted((set(gcp) - gcp_checks) & set(markers))
    gcp_checkpoints = sorted(gcp_checks & set(markers))

    if len(gcp_common) >= 3:
        # ── 2) GCP 정합: 현장 좌표계 ───────────────────────────
        src = np.array([markers[m].mean(0) for m in gcp_common])
        dst = np.array([gcp[m] for m in gcp_common])
        offset = dst.mean(0).round(0)  # TM 좌표처럼 큰 값의 부동소수 손실 방지
        s, R, t = umeyama(src, dst - offset)
        res = {m: float(np.linalg.norm(s * R @ src[i] + t - (dst[i] - offset)))
               for i, m in enumerate(gcp_common)}
        loo = {}
        if len(gcp_common) >= 4:
            for i, m in enumerate(gcp_common):
                keep = [j for j in range(len(src)) if j != i]
                s2, R2, t2 = umeyama(src[keep], dst[keep] - offset)
                loo[m] = float(np.linalg.norm(s2 * R2 @ src[i] + t2 - (dst[i] - offset)))
        report.update(method="gcp", gcp_used=gcp_common, gcp_residual_m=res,
                      gcp_rmse_m=float(np.sqrt(np.mean(np.square(list(res.values()))))),
                      checkpoint_loo_error_m=loo, coordinate_offset=offset.tolist())
        if marker_size:
            edges = _edge_lengths(markers, s)
            report["marker_edge_check"] = _edge_stats(edges, marker_size)
        log(f"  GCP {len(gcp_common)}점 정합: RMSE {report['gcp_rmse_m']*1000:.1f}mm"
            + (f", 검사점(LOO) 평균 {np.mean(list(loo.values()))*1000:.1f}mm" if loo else ""))
    elif markers and (marker_size or scale_csv):
        # ── 1) 마커 크기만으로 축척 + 바닥 수평 ─────────────────
        edges = np.concatenate([[np.linalg.norm(m[(k + 1) % 4] - m[k]) for k in range(4)]
                                for m in markers.values()])
        s = marker_size / np.median(edges) if marker_size else 1.0
        report["scale_source"] = "marker_edges"
        if scale_csv:
            # 줄자로 잰 긴 거리로 축척 결정(최소제곱). 기준 길이가 길수록 축척 오차가 작다.
            pairs = [(a, b, d) for a, b, d in read_checks(scale_csv) if a in markers and b in markers]
            if pairs:
                dm = np.array([np.linalg.norm(markers[a].mean(0) - markers[b].mean(0)) for a, b, _ in pairs])
                dt = np.array([d for *_, d in pairs])
                s_d = float((dt * dm).sum() / (dm * dm).sum())
                report["scale_source"] = "tape_distances"
                if marker_size:
                    report["scale_marker_vs_tape_pct"] = round((s / s_d - 1) * 100, 3)
                log(f"  축척: 줄자거리 {len(pairs)}개 사용"
                    + (f" (마커 변 기준과 차이 {report['scale_marker_vs_tape_pct']:+.2f}%)" if marker_size else ""))
                s = s_d
            else:
                report["warnings"].append("scale.csv 의 마커 쌍이 검출되지 않아 마커 변 길이로 축척")
        origin_id = min(markers)
        m0 = markers[origin_id]
        if markers_on_ground and len(markers) >= 3:
            # 여러 마커 모서리 전체로 바닥 평면을 맞춰 기울기 오차를 줄인다
            pts = np.concatenate(list(markers.values()))
            ctr = pts.mean(0)
            normal = np.linalg.svd(pts - ctr)[2][2]
        else:
            normal = np.cross(m0[1] - m0[0], m0[3] - m0[0])
        normal /= np.linalg.norm(normal)
        # 카메라가 있는 쪽이 위(+Z)
        cam_ctr = np.mean([rec.image(i).projection_center() for i in rec.reg_image_ids()], 0)
        if np.dot(cam_ctr - m0.mean(0), normal) < 0:
            normal = -normal
        x = m0[1] - m0[0]  # 마커 윗변 방향 = X축
        x -= normal * np.dot(x, normal)
        x /= np.linalg.norm(x)
        y = np.cross(normal, x)
        R = np.stack([x, y, normal])  # 행이 새 축
        t = -s * R @ m0.mean(0)
        report.update(method="marker_scale", origin_marker=origin_id)
        if marker_size:
            report["marker_edge_check"] = _edge_stats(_edge_lengths(markers, s), marker_size)
        if len(markers) < 2:
            report["warnings"].append("마커가 1개만 잡혔습니다 — 축척 오차가 커질 수 있으니 3개 이상을 넓게 배치하세요")
        log(f"  마커 축척: ID{origin_id} 원점, 마커 {len(markers)}개"
            + (f", 변 길이 편차 {report['marker_edge_check']['std_mm']:.1f}mm" if marker_size else ""))
    else:
        # ── 0) 마커 없음: 축척 미확정으로 표시 (폰 GPS 는 수 m 오차라 축척 기준으로 부적합) ──
        report["warnings"].append(
            "마커/GCP가 없어 실제 축척을 확정할 수 없습니다. 결과의 단위는 임의값입니다. "
            "시뮬레이션용으로는 반드시 마커(권장) 또는 줄자 검증거리를 넣으세요.")
        log("  경고: 마커 미검출 — 축척 미확정(임의 단위)")
        # 축척은 모르지만 수평은 잡는다: 폰을 대체로 똑바로 들고 찍으므로 카메라 '위'(-Y) 평균 = 하늘 방향
        ups = []
        for i in rec.reg_image_ids():
            Rcw = _pose(rec.image(i))[:, :3]
            ups.append(-Rcw[1])  # 카메라 -Y 축의 월드 방향
        up = np.mean(ups, 0)
        up /= np.linalg.norm(up)
        x = np.cross([0.0, 1.0, 0.0], up) if abs(up[1]) < 0.9 else np.cross([1.0, 0.0, 0.0], up)
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(up, x), up])
        ctr = np.mean([rec.image(i).projection_center() for i in rec.reg_image_ids()], 0)
        rec.transform(pycolmap.Sim3d(1.0, pycolmap.Rotation3d(R), (-R @ ctr).reshape(3, 1)))
        report["method"] = "none_leveled"
        report["warnings"].append("수평만 카메라 방향으로 추정(축척·방위 없음)")
        return report

    sim = pycolmap.Sim3d(float(s), pycolmap.Rotation3d(R), np.asarray(t, float).reshape(3, 1))
    rec.transform(sim)
    report["sim3"] = {"scale": float(s), "R": R.tolist(), "t": np.asarray(t).tolist()}

    # 변환 후 마커 중심 좌표 기록
    centers = {m: (s * R @ c.mean(0) + t) for m, c in markers.items()}

    if report["method"] == "gcp" and gcp_ba:
        off = np.array(report["coordinate_offset"])
        if gcp_bundle_adjust(rec, obs, {m: gcp[m] - off for m in gcp_common}, log=log):
            report["method"] = "gcp_ba"
            centers = _retriangulate_centers(rec, obs)
            report["gcp_residual_after_ba_m"] = {
                m: float(np.linalg.norm(centers[m] - (gcp[m] - off))) for m in gcp_common if m in centers}
    if gcp_checkpoints:
        off = np.array(report["coordinate_offset"]) if "coordinate_offset" in report else 0
        ce = {m: float(np.linalg.norm(centers[m] - (gcp[m] - off))) for m in gcp_checkpoints if m in centers}
        report["checkpoint_error_m"] = ce
        if ce:
            log("  검사점(독립): " + ", ".join(f"ID{m} {v*1000:.1f}mm" for m, v in ce.items()))
    report["marker_centers"] = {m: v.round(4).tolist() for m, v in centers.items()}

    if check_csv:
        rows = []
        for a, b, d in read_checks(check_csv):
            if a in centers and b in centers:
                md = float(np.linalg.norm(centers[a] - centers[b]))
                rows.append({"pair": f"{a}-{b}", "measured_m": d, "model_m": round(md, 4),
                             "error_mm": round((md - d) * 1000, 1),
                             "error_pct": round((md - d) / d * 100, 3)})
        report["check_distances"] = rows
        if rows:
            log("  검증거리: " + ", ".join(f"{r['pair']} {r['error_mm']:+.1f}mm" for r in rows))
    return report


def _retriangulate_centers(rec, obs) -> dict[int, np.ndarray]:
    """현재 카메라 자세로 마커 모서리를 다시 삼각측량해 중심을 구한다."""
    corners = defaultdict(dict)
    for (mid, k), o in obs.items():
        r = triangulate(rec, o)
        if r is not None:
            corners[mid][k] = r[0]
    return {m: np.mean([c[k] for k in range(4)], 0) for m, c in corners.items() if len(c) == 4}


def _edge_lengths(markers, s):
    return np.concatenate([[s * np.linalg.norm(m[(k + 1) % 4] - m[k]) for k in range(4)]
                           for m in markers.values()])


def _edge_stats(edges, size):
    return {"nominal_m": size, "mean_m": float(edges.mean()),
            "std_mm": float(edges.std() * 1000),
            "max_abs_err_mm": float(np.abs(edges - size).max() * 1000)}
