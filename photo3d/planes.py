"""평면 찾기 · 바닥 정사영상(사진으로 만든 바닥 텍스처) · 모서리 정리.

1) find_planes: 조밀 점군에서 큰 평면(바닥·벽·천장)을 RANSAC 으로 차례로 찾는다.
2) ground_ortho: 바닥 평면 위 격자의 각 지점을 사진들에 투영 → 바닥을 가장 정면에 가깝게·가깝게 본 사진 2~3장을
   가중 평균해 정사영상(위에서 내려다본 사진)을 만든다. 다른 물체에 가려진 사진은 점군 깊이로 걸러낸다.
   (Pix4D·Metashape 의 정사영상/평면 텍스처 방식 — 비스듬히 찍힌 바닥은 MVS·스플랫 모두 약해 지저분해지는 곳)
3) ground_splats: 바닥 근처의 지저분한 가우시안을 지우고, 정사영상 색을 가진 납작한 가우시안으로 바닥을 깐다.
4) snap_vertices: 메쉬 정점을 가까운 평면(두 평면 근처면 교선)으로 옮겨 벽·바닥을 평평하게, 모서리를 곧게.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _fit_plane(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    return n / np.linalg.norm(n), -float(n @ c)


def find_planes(xyz: np.ndarray, thr: float, max_planes: int = 8, min_frac: float = 0.03,
                iters: int = 400, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    pts = xyz if len(xyz) <= 400_000 else xyz[rng.choice(len(xyz), 400_000, replace=False)]
    left = np.arange(len(pts))
    out = []
    for _ in range(max_planes):
        if len(left) < max(500, min_frac * len(pts)):
            break
        P = pts[left]
        best, best_n = None, 0
        for _ in range(iters):
            s = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            d = -n @ s[0]
            cnt = int((np.abs(P @ n + d) < thr).sum())
            if cnt > best_n:
                best, best_n = (n, d), cnt
        if best is None or best_n < min_frac * len(pts):
            break
        n, d = best
        inl = np.abs(P @ n + d) < thr
        n, d = _fit_plane(P[inl])  # 최소제곱으로 다듬기
        inl = np.abs(P @ n + d) < thr
        out.append({"n": n, "d": d, "count": int(inl.sum()), "frac": float(inl.mean() * len(P) / len(pts))})
        left = left[~inl]
    return out


def up_vector(rec) -> np.ndarray:
    """중력 위쪽 추정: 사진의 '위' 방향(카메라 -y) 평균. 사람은 대개 폰을 세워(수평으로) 찍는다."""
    ups = []
    for i in rec.reg_image_ids():
        im = rec.image(i)
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        ups.append(-T.rotation.matrix()[1])
    u = np.mean(ups, 0)
    return u / np.linalg.norm(u)


def pick_ground(planes: list[dict], up: np.ndarray, cams: np.ndarray, cos_min: float = 0.9) -> dict | None:
    """위쪽과 나란한 법선을 가진 평면 중 카메라보다 아래에 있는 가장 큰 것 = 바닥(법선이 위를 향하게 정리)."""
    cands = []
    for p in planes:
        n, d = p["n"].copy(), p["d"]
        c = float(n @ up)
        if abs(c) < cos_min:
            continue
        if c < 0:
            n, d = -n, -d
        h = cams @ n + d  # 카메라 높이(바닥 위가 +)
        if np.median(h) > 0.2:
            cands.append({**p, "n": n, "d": d, "cam_h": float(np.median(h))})
    return max(cands, key=lambda p: p["count"]) if cands else None


def find_ground(xyz: np.ndarray, up: np.ndarray, cams: np.ndarray, thr: float, iters: int = 800,
                seed: int = 0) -> dict | None:
    """바닥 전용 찾기. 비스듬히 찍힌 바닥은 조밀 점이 거의 안 생겨(castle: 전체의 1~2%) 일반 평면 찾기로는
    벽이나 물체 윗면을 바닥으로 잘못 잡는다. 대신 '칸마다 가장 낮은 점'(벽 밑동·물체 밑·남은 바닥 점)을 모아
    법선이 위쪽 25° 안인 평면을 맞춘다 — 벽은 바닥에서 시작하므로 벽 밑동들이 바닥 높이를 알려준다."""
    rng = np.random.default_rng(seed)
    u, v = _plane_basis(up)
    h = xyz @ up
    cu, cv_ = xyz @ u, xyz @ v
    ext = max(float(np.ptp(cu)), float(np.ptp(cv_)), 1e-6)
    cell = ext / 120
    key = ((cu - cu.min()) / cell).astype(np.int64) * 100_000 + ((cv_ - cv_.min()) / cell).astype(np.int64)
    order = np.lexsort((h, key))
    k_s = key[order]
    starts = np.r_[0, np.nonzero(np.diff(k_s))[0] + 1]
    cnt = np.diff(np.r_[starts, len(k_s)])
    keep = cnt >= 5
    # 칸에서 아래쪽 5% 위치의 점(맨 아래 1개는 잡음일 수 있음)
    pick = order[starts[keep] + (cnt[keep] * 0.05).astype(int)]
    S = xyz[pick]
    ch = float(np.median(cams @ up))
    S = S[S @ up < ch]
    if len(S) < 30:
        return None
    t2 = max(2 * thr, 0.01 * float(np.ptp(h)))
    best, best_n = None, 0
    for _ in range(iters):
        s3 = S[rng.choice(len(S), 3, replace=False)]
        n = np.cross(s3[1] - s3[0], s3[2] - s3[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if n @ up < 0:
            n = -n
        if n @ up < 0.9:
            continue
        d = -n @ s3[0]
        if np.median(cams @ n + d) <= 0:
            continue
        dist = S @ n + d
        score = int((np.abs(dist) < t2).sum()) - 2 * int((dist < -3 * t2).sum())  # 바닥 아래 점이 많으면 감점
        if score > best_n:
            best, best_n = (n, d), score
    if best is None or best_n < 20:
        return None
    n, d = best
    inl = np.abs(S @ n + d) < t2
    n, d = _fit_plane(S[inl])
    if n @ up < 0:
        n, d = -n, -d
    c = int((np.abs(xyz @ n + d) < thr).sum())
    return {"n": n, "d": d, "count": c, "frac": c / len(xyz), "cam_h": float(np.median(cams @ n + d)),
            "support_cells": int(inl.sum()), "support_frac": float(inl.mean())}


def _plane_basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def ground_ortho(ground: dict, xyz: np.ndarray, sparse: Path, img_dir: Path, thr: float,
                 res: float | None = None, max_px: int = 4096, log=print) -> dict | None:
    """바닥 정사영상. 반환: {"img": HxWx3, "origin", "u", "v", "res", "mask"} (바닥 범위만 mask=True)"""
    import cv2
    from .texture import _cameras, _project, _sample
    n, d = ground["n"], ground["d"]
    on = np.abs(xyz @ n + d) < 2 * thr
    G = xyz[on]
    if len(G) < 200:
        return None
    u, v = _plane_basis(n)
    o = -d * n  # 평면 위 한 점
    gu, gv = (G - o) @ u, (G - o) @ v
    lo = np.percentile(np.c_[gu, gv], 0.5, 0)
    hi = np.percentile(np.c_[gu, gv], 99.5, 0)
    pad = 0.1 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ext = hi - lo
    res = res or float(ext.max()) / max_px  # 기본: 긴 변 4096px
    W, H = int(np.ceil(ext[0] / res)), int(np.ceil(ext[1] / res))
    # 바닥 범위: 바닥 점 + 바닥 가까이(카메라 높이의 절반 아래) 있는 벽 밑동·물체 점들의 볼록 껍질.
    # 비스듬한 바닥은 조밀 점이 군데군데만 생기므로 점 있는 칸만 쓰면 바닥이 구멍투성이가 된다.
    hh = xyz @ n + d
    cam_h = float(ground.get("cam_h") or np.percentile(hh, 50))
    base = xyz[(hh > -2 * thr) & (hh < 0.5 * cam_h)]
    bu, bv = (base - o) @ u, (base - o) @ v
    pix = np.c_[(bu - lo[0]) / res, (bv - lo[1]) / res]
    pix = pix[(pix[:, 0] > -0.05 * W) & (pix[:, 1] > -0.05 * H) & (pix[:, 0] < 1.05 * W) & (pix[:, 1] < 1.05 * H)]
    mask8 = np.zeros((H, W), np.uint8)
    if len(pix) >= 3:
        cv2.fillConvexPoly(mask8, cv2.convexHull(pix.astype(np.float32)).astype(np.int32), 255)
    mask = mask8 > 0
    yy, xx = np.nonzero(mask)
    X = o + np.outer(lo[0] + (xx + 0.5) * res, u) + np.outer(lo[1] + (yy + 0.5) * res, v)
    acc = np.zeros((len(X), 3), np.float64)
    wsum = np.zeros(len(X))
    cams = _cameras(sparse, img_dir)
    zcache: dict[str, np.ndarray] = {}
    xyz_z = xyz[:: max(1, len(xyz) // 2_000_000)]  # 가림 검사용 점(사진 수백 장·점 수천만이어도 몇 분 안)
    s = 4  # 가림 검사 칸 크기(px)

    def view(cam, im, Xq):
        """이 사진에서 Xq 중 보이는 점의 (번호, 가중치, 색)"""
        uv, z = _project(cam, Xq)
        ok = (z > 0) & (uv[:, 0] >= 1) & (uv[:, 1] >= 1) & (uv[:, 0] < cam["w"] - 2) & (uv[:, 1] < cam["h"] - 2)
        if not ok.any():
            return None
        Dz = zcache.get(cam["name"])
        if Dz is None:  # 가림 검사: 조밀 점을 이 사진에 투영한 깊이(앞에 다른 물체가 있으면 제외)
            uvp, zp = _project(cam, xyz_z)
            okp = (zp > 0) & (uvp[:, 0] >= 0) & (uvp[:, 1] >= 0) & (uvp[:, 0] < cam["w"]) & (uvp[:, 1] < cam["h"])
            Dz = np.full((cam["h"] // s + 1, cam["w"] // s + 1), 1e9, np.float32)
            np.minimum.at(Dz, ((uvp[okp, 1] // s).astype(int), (uvp[okp, 0] // s).astype(int)),
                          zp[okp].astype(np.float32))
            # 점 사이 빈틈으로 벽 뒤 바닥이 '보이는' 것 막기: 가까운 깊이를 주변 칸으로 번지게(최솟값 필터)
            Dz = cv2.erode(Dz, np.ones((5, 5), np.uint8))
            zcache[cam["name"]] = Dz
        idx = np.where(ok)[0]
        zi = Dz[(uv[idx, 1] // s).astype(int), (uv[idx, 0] // s).astype(int)]
        idx = idx[~(zi < z[idx] * 0.97 - 2 * thr)]
        if not len(idx):
            return None
        # 좋은 시점: 바닥을 정면으로(광선과 법선 사이 각이 작게), 가깝게
        ray = Xq[idx] - cam["C"]
        dist = np.linalg.norm(ray, axis=1)
        cosang = np.abs(ray @ n) / np.maximum(dist, 1e-9)
        w = (cosang ** 4) / np.maximum(dist, 1e-6) ** 2 * (cosang > 0.03)
        # 사진 가장자리로 갈수록 가중치를 부드럽게 낮춤 → 사진 경계의 계단 같은 이음매 줄이기
        uvv = uv[idx]
        edge = np.minimum.reduce([uvv[:, 0], cam["w"] - uvv[:, 0], uvv[:, 1], cam["h"] - uvv[:, 1]])
        fe = np.clip(edge / (0.2 * min(cam["w"], cam["h"])), 0.0, 1.0)
        w = w ** 3 * (fe * fe * (3 - 2 * fe) + 1e-6)  # 세제곱: 가장 좋은 사진 위주(흐림 줄이기)
        return idx, w, _sample(im, uvv[:, 0], uvv[:, 1]).astype(np.float64)

    # 1단계(성긴 격자): 사진마다 밝기·색 차이(노출·화이트밸런스) 보정값 — 겹치는 곳에서 합의 색과 비교
    sub = np.arange(0, len(X), max(1, len(X) // 60_000))
    Xs = X[sub]
    per = []
    acc_s = np.zeros((len(Xs), 3))
    w_s = np.zeros(len(Xs))
    for cam in cams:
        im = cv2.imread(str(cam["path"]), cv2.IMREAD_COLOR)
        r = view(cam, im, Xs) if im is not None else None
        per.append(r)
        if r:
            acc_s[r[0]] += r[2] * r[1][:, None]
            w_s[r[0]] += r[1]
    cons = acc_s / np.maximum(w_s, 1e-30)[:, None]
    gains = {}
    for cam, r in zip(cams, per):
        if r and len(r[0]) > 50:
            g = np.median(cons[r[0]] + 1, 0) / np.median(r[2] + 1, 0)
            gains[cam["name"]] = np.clip(g, 0.6, 1.6)
    # 2단계: 전체 격자를 보정된 색으로
    for cam in cams:
        if cam["name"] not in gains:
            continue
        im = cv2.imread(str(cam["path"]), cv2.IMREAD_COLOR)
        r = view(cam, im, X)
        if r:
            acc[r[0]] += r[2] * gains[cam["name"]] * r[1][:, None]
            wsum[r[0]] += r[1]
    ok = wsum > 0
    img = np.zeros((H, W, 3), np.uint8)
    img[yy[ok], xx[ok]] = np.clip(acc[ok] / wsum[ok, None], 0, 255).astype(np.uint8)
    m2 = np.zeros((H, W), bool)
    m2[yy[ok], xx[ok]] = True
    log(f"  바닥 정사영상: {W}×{H}px, 바닥 범위 중 사진에 보인 곳 {m2.sum() / max(mask.sum(), 1) * 100:.0f}%")
    return {"img": img, "mask": m2, "origin": o + lo[0] * u + lo[1] * v, "u": u, "v": v, "n": n, "res": res}


def ground_splats(a: np.ndarray, props: list[str], ortho: dict, ground: dict, thr: float,
                  step: int = 1) -> tuple[np.ndarray, dict]:
    """바닥 근처 가우시안 정리 + 정사영상 색의 납작한 가우시안 바닥 추가."""
    col = {q: i for i, q in enumerate(props)}
    xyz = a[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    n, d = ground["n"], ground["d"]
    h = xyz @ n + d
    o, u, v, res = ortho["origin"], ortho["u"], ortho["v"], ortho["res"]
    pu, pv = (xyz - o) @ u, (xyz - o) @ v
    ix, iy = (pu / res).astype(int), (pv / res).astype(int)
    Hh, Ww = ortho["mask"].shape
    inside = (ix >= 0) & (iy >= 0) & (ix < Ww) & (iy < Hh)
    inside[inside] = ortho["mask"][iy[inside], ix[inside]]
    # 바닥 범위 안에서 바닥 바로 위(2 thr 이내)와 바닥 아래(있을 수 없는 곳)의 가우시안 = 지저분한 바닥 → 교체
    band = 2 * thr
    drop = inside & (h < band)
    keep = a[~drop]
    import cv2
    if step > 1:  # 칸을 키울 때는 평균 색(면적 평균)으로
        Hs, Ws = Hh // step, Ww // step
        img = cv2.resize(ortho["img"][:Hs * step, :Ws * step], (Ws, Hs), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(ortho["mask"][:Hs * step, :Ws * step].astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_AREA) > 0
    else:
        img, msk = ortho["img"], ortho["mask"]
    yy, xx = np.nonzero(msk)
    rgb = img[yy, xx].astype(np.float64) / 255.0
    yy, xx = yy * step, xx * step
    res = res * step
    P = o + np.outer(xx * ortho["res"] + 0.5 * res, u) + np.outer(yy * ortho["res"] + 0.5 * res, v)
    m = len(P)
    new = np.zeros((m, len(props)), np.float32)
    new[:, col["x"]], new[:, col["y"]], new[:, col["z"]] = P[:, 0], P[:, 1], P[:, 2]
    C0 = 0.28209479177387814
    for k in range(3):
        new[:, col[f"f_dc_{k}"]] = (rgb[:, k] - 0.5) / C0
    new[:, col["opacity"]] = 4.0  # 불투명(시그모이드 0.98)
    sxy = np.log(res * 0.6)  # 이웃 원판과 살짝 겹치게(빈틈 없이)
    new[:, col["scale_0"]], new[:, col["scale_1"]], new[:, col["scale_2"]] = sxy, sxy, np.log(res * 0.05)
    # 회전: 가우시안의 z축(세 번째 축)을 바닥 법선에 맞춤 → 납작한 원판
    R = np.stack([u, v, n], 1)  # 열 = 새 축
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    else:
        import cv2
        rv, _ = cv2.Rodrigues(R)
        ang = np.linalg.norm(rv)
        ax = rv.ravel() / max(ang, 1e-12)
        q = np.r_[np.cos(ang / 2), ax * np.sin(ang / 2)]
    for k in range(4):
        new[:, col[f"rot_{k}"]] = q[k]
    return np.vstack([keep, new]), {"removed": int(drop.sum()), "added": int(m)}


def snap_vertices(V: np.ndarray, planes: list[dict], thr: float) -> tuple[np.ndarray, dict]:
    """평면에서 thr 안의 정점은 평면 위로, 두 평면 모두 가까우면 교선 위로(모서리 곧게)."""
    V = V.copy()
    if not planes:
        return V, {"snapped": 0, "edges": 0}
    N = np.array([p["n"] for p in planes])
    D = np.array([p["d"] for p in planes])
    dist = V @ N.T + D  # (nv, np)
    ad = np.abs(dist)
    order = np.argsort(ad, 1)
    i1, i2 = order[:, 0], order[:, 1] if len(planes) > 1 else order[:, 0]
    a1 = ad[np.arange(len(V)), i1]
    a2 = ad[np.arange(len(V)), i2]
    one = a1 < thr
    two = one & (a2 < thr) & (len(planes) > 1)
    # 두 평면 교선: 법선이 충분히 다를 때만(같은 방향 평면 둘 사이는 제외)
    if two.any():
        n1, n2 = N[i1[two]], N[i2[two]]
        ok = np.abs((n1 * n2).sum(1)) < 0.9
        idx = np.where(two)[0][ok]
        n1, n2 = n1[ok], n2[ok]
        d1, d2 = D[i1[idx]], D[i2[idx]]
        p = V[idx]
        # 점 p 에서 두 평면 모두 만족하는 가장 가까운 점: p - [n1 n2] (G^-1) [n1·p+d1, n2·p+d2]
        A = np.stack([n1, n2], 1)  # (k,2,3)
        G = A @ A.transpose(0, 2, 1)
        r = np.stack([(n1 * p).sum(1) + d1, (n2 * p).sum(1) + d2], 1)
        lam = np.linalg.solve(G, r[..., None])[..., 0]
        V[idx] = p - (A.transpose(0, 2, 1) @ lam[..., None])[..., 0]
        two_mask = np.zeros(len(V), bool)
        two_mask[idx] = True
    else:
        two_mask = np.zeros(len(V), bool)
    sing = one & ~two_mask
    V[sing] -= dist[np.where(sing)[0], i1[sing]][:, None] * N[i1[sing]]
    return V, {"snapped": int(one.sum()), "edges": int(two_mask.sum())}


def improve_ground_splat(splat: Path, dense_dir: Path, out_dir: Path, max_new: int = 1_500_000,
                         log=print) -> dict | None:
    """스플랫 바닥을 사진 정사영상으로 교체. dense_dir: COLMAP 조밀 작업 폴더(fused.ply, sparse/, images/).
    바닥이 평평하지 않거나(경사·계단·지형) 찾지 못하면 아무것도 바꾸지 않는다."""
    import json
    import cv2
    import pycolmap
    from scipy.spatial import cKDTree
    from .fill import read_points
    from .splatclean import read_splat, write_splat
    xyz = np.asarray(read_points(dense_dir / "fused.ply")[0], np.float64)
    if len(xyz) < 5000:
        return None
    rng = np.random.default_rng(0)
    smp = xyz[rng.choice(len(xyz), min(20000, len(xyz)), replace=False)]
    spacing = float(np.median(cKDTree(xyz).query(smp, k=2)[0][:, 1]))
    thr = 2 * spacing
    rec = pycolmap.Reconstruction(dense_dir / "sparse")
    cams = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])
    up = up_vector(rec)
    g = find_ground(xyz, up, cams, thr)
    if g is None:
        log("  바닥 평면을 찾지 못함 → 스플랫 바닥 그대로")
        return None
    if g["support_cells"] < 30:
        log("  바닥이 평평하지 않음(경사·지형) → 스플랫 바닥 그대로")
        return None
    o = ground_ortho(g, xyz, dense_dir / "sparse", dense_dir / "images", thr, max_px=3072, log=log)
    if o is None or o["mask"].sum() < 1000:
        return None
    cv2.imwrite(str(out_dir / "site_ground_ortho.jpg"), o["img"][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    meta = {"origin": o["origin"].tolist(), "u_axis": o["u"].tolist(), "v_axis": o["v"].tolist(),
            "normal": o["n"].tolist(), "m_per_px": o["res"], "size_px": list(o["img"].shape[1::-1]),
            "note": "바닥 정사영상: 픽셀(x,y) 의 3D 위치 = origin + (x+0.5)*m_per_px*u_axis + (y+0.5)*m_per_px*v_axis"}
    (out_dir / "site_ground_ortho.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    a, props, _ = read_splat(splat)
    step = int(np.ceil(np.sqrt(o["mask"].sum() / max_new)))
    b, info = ground_splats(a, props, o, g, thr, step=max(1, step))
    write_splat(splat, b, props, "photo3d: cleaned splat + photo ground, Z-up")
    log(f"  바닥 교체: 지저분한 바닥 가우시안 {info['removed']:,}개 → 사진 바닥 {info['added']:,}개 "
        f"(정사영상 site_ground_ortho.jpg)")
    return {**info, "ortho": "site_ground_ortho.jpg", "m_per_px": o["res"]}


def planarize_mesh(mesh_path: Path, points_path: Path, sparse: Path, dst: Path, log=print) -> Path:
    """메쉬의 바닥·벽·천장을 측정 점군에서 찾은 평면에 붙여 평평하게, 두 평면이 만나는 곳은 곧은 모서리로.
    평면에서 점 간격의 1.5배 안의 정점만 옮긴다(측정 오차 범위 안) → 정확도는 유지(ETH3D office F1 그대로)."""
    import pycolmap
    import trimesh
    from scipy.spatial import cKDTree
    try:
        m = trimesh.load(mesh_path, process=False, force="mesh")
        from .fill import read_points
        pts = np.asarray(read_points(points_path)[0], float)
        sub = pts[:: max(1, len(pts) // 20000)]
        sp = float(np.median(cKDTree(pts).query(sub, k=2)[0][:, 1]))
        thr = 1.5 * sp
        pl = find_planes(pts, thr, max_planes=10, min_frac=0.02)
        try:
            rec = pycolmap.Reconstruction(sparse)
            cams = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])
            g = find_ground(pts, up_vector(rec), cams, thr)
            if g is not None and not any(abs(float(p["n"] @ g["n"])) > 0.99 and abs(abs(p["d"]) - abs(g["d"])) < thr
                                         for p in pl):
                pl.append(g)
        except Exception:
            pass
        if not pl:
            return mesh_path
        V, st = snap_vertices(np.asarray(m.vertices, float), pl, thr)
        m.vertices = V
        m.export(dst)
        log(f"  평면 정리: 평면 {len(pl)}개(바닥·벽 등)에 정점 {st['snapped']:,}개 맞춤, 모서리 정점 {st['edges']:,}개 곧게")
        return dst
    except Exception as e:
        log(f"  평면 정리 생략({e})")
        return mesh_path


def floor_patch(points_path: Path, sparse: Path, img_dir: Path, mesh, out_dir: Path | None = None,
                max_faces: int = 400_000, log=print):
    """메쉬에 바닥이 비어 있는 곳(비스듬히 찍혀 측정점이 없는 바닥)을 사진 정사영상을 입힌 평면 조각으로 메운다.
    반환: (바닥 조각 trimesh | None, 정보). 기존 메쉬가 바닥을 덮고 있는 칸은 건드리지 않는다."""
    import cv2
    import pycolmap
    import trimesh
    from PIL import Image
    from scipy.spatial import cKDTree
    from .fill import read_points
    xyz = np.asarray(read_points(points_path)[0], np.float64)
    if len(xyz) < 5000:
        return None, {}
    sub = xyz[:: max(1, len(xyz) // 20000)]
    spacing = float(np.median(cKDTree(xyz).query(sub, k=2)[0][:, 1]))
    thr = 2 * spacing
    rec = pycolmap.Reconstruction(sparse)
    cams = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])
    g = find_ground(xyz, up_vector(rec), cams, thr)
    if g is None or g["support_cells"] < 30:
        log("  바닥 조각: 평평한 바닥을 찾지 못함 → 생략")
        return None, {}
    o = ground_ortho(g, xyz, sparse, img_dir, thr, max_px=4096, log=log)
    if o is None:
        return None, {}
    if out_dir is not None:
        cv2.imwrite(str(Path(out_dir) / "site_ground_ortho.jpg"), o["img"][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    n, d, res = o["n"], g["d"], o["res"]
    H, W = o["mask"].shape
    # 칸 크기: 면 수 상한 안에서(칸 하나 = 삼각형 2개)
    step = max(1, int(np.ceil(np.sqrt(2 * o["mask"].sum() / max_faces))))
    Hs, Ws = H // step, W // step
    cell_ok = cv2.resize(o["mask"][:Hs * step, :Ws * step].astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_AREA) > 0
    # 기존 메쉬가 바닥 근처(±3 thr)에 있는 칸은 제외 → 측정된 바닥은 그대로
    V = np.asarray(mesh.vertices, np.float64)
    near = np.abs(V @ n + d) < 3 * thr
    pv = V[near] - o["origin"]
    cx = (pv @ o["u"] / (res * step)).astype(int)
    cy = (pv @ o["v"] / (res * step)).astype(int)
    k = (cx >= 0) & (cy >= 0) & (cx < Ws) & (cy < Hs)
    covered = np.zeros((Hs, Ws), np.uint8)
    covered[cy[k], cx[k]] = 1
    covered = covered > 0  # 넓히지 않음(넓히면 측정 바닥과 사진 바닥 사이에 틈)
    use = cell_ok & ~covered
    if use.sum() < 10:
        return None, {"cells": 0}
    yy, xx = np.nonzero(use)
    # 칸마다 네 꼭짓점(공유 정점)
    vid = -np.ones((Hs + 1, Ws + 1), np.int64)
    corners = np.unique(np.concatenate([np.c_[yy, xx], np.c_[yy, xx + 1], np.c_[yy + 1, xx], np.c_[yy + 1, xx + 1]]), axis=0)
    vid[corners[:, 0], corners[:, 1]] = np.arange(len(corners))
    P = o["origin"] + np.outer(corners[:, 1] * res * step, o["u"]) + np.outer(corners[:, 0] * res * step, o["v"])
    P -= n * 0.25 * thr  # 측정 바닥보다 아주 살짝 아래(겹치는 가장자리 깜빡임 방지)
    a, b, c, e = vid[yy, xx], vid[yy, xx + 1], vid[yy + 1, xx], vid[yy + 1, xx + 1]
    F = np.concatenate([np.c_[a, c, b], np.c_[b, c, e]])
    if (np.cross(P[F[0, 1]] - P[F[0, 0]], P[F[0, 2]] - P[F[0, 0]]) @ n) < 0:
        F = F[:, ::-1]  # 앞면이 위를 보게
    uv = np.c_[corners[:, 1] * step / W, 1 - corners[:, 0] * step / H]
    img = Image.fromarray(o["img"])
    m = trimesh.Trimesh(P, F, visual=trimesh.visual.TextureVisuals(uv=uv, image=img), process=False)
    info = {"cells": int(use.sum()), "faces": int(len(F)), "m_per_px": round(res, 4),
            "area_m2": round(float(use.sum()) * (res * step) ** 2, 1)}
    log(f"  바닥 조각: 사진 바닥 {info['area_m2']} m² 추가(측정이 빈 바닥만, 면 {len(F):,}개)")
    return m, info
