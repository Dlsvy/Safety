"""사진 텍스처 굽기 — 메쉬에 원본(왜곡보정) 사진을 직접 투영해 DJI Terra 처럼 선명한 표면을 만든다.

OpenMVS TextureMesh 가 Windows 에서 검은 텍스처를 내는 문제(PC 3회)를 피하려고 파이썬으로 직접 구현.
방식은 Waechter et al. "Let There Be Color!"(ECCV14) 의 핵심을 단순화했다.
 1) 면마다 '가장 잘 보이는 사진' 선택: 정면성(cos) × 화면상 크기(해상도), Embree 광선으로 가림 검사
 2) 이웃 면과 같은 사진을 쓰도록 라벨 다듬기(이음새 감소)
 3) 사진 간 밝기·색 차이를 사진별 RGB 이득으로 전역 보정(겹치는 면의 색을 맞춤)
 4) 같은 사진을 쓰는 이웃 면 묶음(패치)마다 사진 투영 좌표를 UV 로 쓰고 사진 영역을 잘라 아틀라스에 포장
    (mvs-texturing 방식 — 사진 해상도 그대로, 패치 여백은 사진 자체로 채워져 밉맵 경계선 없음)
라이선스: embreex/Embree(MIT/Apache-2.0)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def _ensure_deps(log=print):
    try:
        import embreex  # noqa: F401
    except ImportError:
        log("  텍스처 도구(xatlas, embreex) 설치 중…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "embreex"], check=True)


def _cameras(sparse: Path, img_dir: Path):
    import pycolmap
    rec = pycolmap.Reconstruction(sparse)
    cams = []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        c = rec.cameras[im.camera_id]
        p = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        T = p.matrix()
        cams.append({"name": im.name, "K": c.calibration_matrix(), "R": T[:, :3], "t": T[:, 3],
                     "C": -T[:, :3].T @ T[:, 3], "w": c.width, "h": c.height, "path": img_dir / im.name})
    return cams


def _project(cam, X):
    Xc = X @ cam["R"].T + cam["t"]
    z = Xc[:, 2]
    uv = (Xc @ cam["K"].T)[:, :2] / np.maximum(z, 1e-9)[:, None]
    return uv, z


def _sample(im, x, y):
    """이미지에서 (x,y) 위치 색을 쌍선형으로 읽는다 → (N,3) RGB float. remap 의 크기 제한 회피용 분할."""
    import cv2
    n = len(x)
    cols = 1024
    rows = (n + cols - 1) // cols
    pad = rows * cols - n
    mx = np.r_[x.astype(np.float32), np.zeros(pad, np.float32)].reshape(rows, cols)
    my = np.r_[y.astype(np.float32), np.zeros(pad, np.float32)].reshape(rows, cols)
    out = np.empty((rows, cols, 3), np.uint8)
    for r0 in range(0, rows, 16384):
        out[r0:r0 + 16384] = cv2.remap(im, mx[r0:r0 + 16384], my[r0:r0 + 16384], cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
    return out.reshape(-1, 3)[:n, ::-1].astype(np.float32)


def _fill_empty(tex: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """칠하지 않은 텍셀(패치·셀 사이 틈, 아틀라스 남는 곳)을 주변 색으로 채운다(push-pull).
    빈 곳이 검정이면 멀리서 볼 때(mipmap) 검정이 섞여 창·벽 경계마다 검은 띠가 생긴다."""
    import cv2
    if filled.all() or not filled.any():
        return tex
    img =tex.astype(np.float32) * filled[..., None]
    w = filled.astype(np.float32)
    levels = [(img, w)]
    while max(levels[-1][1].shape) > 1:  # push: 칠한 텍셀만으로 평균(가중치 = 칠한 비율)
        i, ww = levels[-1]
        size = (max(1, (ww.shape[1] + 1) // 2), max(1, (ww.shape[0] + 1) // 2))
        levels.append((cv2.resize(i, size, interpolation=cv2.INTER_AREA),
                       cv2.resize(ww, size, interpolation=cv2.INTER_AREA)))
    col = None
    for i, ww in reversed(levels):  # pull: 칠한 비율만큼 자기 색, 나머지는 한 단계 거친 색
        if col is None:
            col = i / max(float(ww.max()), 1e-8)
            continue
        up = cv2.resize(col, (ww.shape[1], ww.shape[0]), interpolation=cv2.INTER_LINEAR)
        col = i + up * (1 - ww)[..., None]
    out = tex.copy()
    out[~filled] = np.clip(col[~filled], 0, 255).astype(np.uint8)
    return out


def bake(mesh, sparse: Path, img_dir: Path, atlas: int = 8192, max_faces: int = 600_000,
         vertex_only: np.ndarray | None = None, log=print):
    """vertex-color 메쉬 → TextureVisuals 메쉬(trimesh). 사진에 안 보인 면은 정점 색을 그대로 굽는다."""
    _ensure_deps(log)
    import cv2
    import trimesh
    from PIL import Image
    from scipy.spatial import cKDTree

    vcol = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32) \
        if mesh.visual.kind == "vertex" else np.full((len(mesh.vertices), 3), 160, np.float32)
    if len(mesh.faces) > max_faces:
        vertex_only = None  # 단순화하면 면 번호가 바뀐다
        from .export import decimate
        src_v = np.asarray(mesh.vertices)
        mesh = decimate(mesh, max_faces)
        _, idx = cKDTree(src_v).query(mesh.vertices, workers=-1)
        vcol = vcol[idx]
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces, np.int64)
    nF = len(F)
    fc = V[F].mean(1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    farea = np.linalg.norm(fn, axis=1) / 2
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-18)
    cams = _cameras(sparse, img_dir)
    log(f"  사진 텍스처: 면 {nF:,} · 사진 {len(cams)}장 · 아틀라스 {atlas}px")

    # ── 1) 면별 사진 점수 (가림 검사 포함), 상위 4개 후보 유지
    tri = trimesh.Trimesh(V, F, process=False)
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        rmi = RayMeshIntersector(tri)
    except Exception:
        rmi = tri.ray
    K_TOP = 4
    horiz = np.abs(fn[:, 2]) > 0.8  # 수평 면(좌표는 Z-up)
    # 사진별 선명도(라플라시안 분산, 중앙값 대비 0.5~1.2) — 흔들린 사진의 점수를 낮춘다
    lv = []
    for cam in cams:
        g = cv2.imread(str(cam["path"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            lv.append(np.nan)
            continue
        g = cv2.resize(g, None, fx=min(1.0, 1024 / max(g.shape)), fy=min(1.0, 1024 / max(g.shape)), interpolation=cv2.INTER_AREA)
        lv.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    lv = np.array(lv)
    med = np.nanmedian(lv) if np.isfinite(lv).any() else 1.0
    sharp = np.clip(np.sqrt(np.nan_to_num(lv, nan=med) / max(med, 1e-9)), 0.5, 1.2)
    top_s = np.zeros((nF, K_TOP), np.float32)
    top_c = np.full((nF, K_TOP), -1, np.int32)
    for ci, cam in enumerate(cams):
        uv, z = _project(cam, fc)
        m = (z > 0) & (uv[:, 0] > 2) & (uv[:, 1] > 2) & (uv[:, 0] < cam["w"] - 3) & (uv[:, 1] < cam["h"] - 3)
        view = cam["C"] - fc
        dist = np.linalg.norm(view, axis=1)
        cos = (fn * view).sum(1) / np.maximum(dist, 1e-12)
        m &= cos > 0.15
        cand = np.where(m)[0]
        if len(cand) == 0:
            continue
        # 사진 중심에서 면 중심으로 광선 → 처음 맞는 면이 자기 자신(또는 아주 가까운 곳)이면 보임
        d = fc[cand] - cam["C"]
        dn = np.linalg.norm(d, axis=1)
        d /= dn[:, None]
        locs, ri, ti = rmi.intersects_location(np.repeat(cam["C"][None], len(cand), 0), d, multiple_hits=False)
        hit = np.full(len(cand), np.inf)
        hit[ri] = np.linalg.norm(locs - cam["C"], axis=1)
        vis = hit >= dn * (1 - 0.01) - 1e-6
        cand = cand[vis]
        f = cam["K"][0, 0]
        # 정면성 cos³(예전 cos²) × 화면상 크기 × 사진 선명도 — 정면에 가깝고 가깝고 선명한 사진이 이긴다
        s = (cos[cand] ** 3 * farea[cand] * (f / np.maximum(z[cand], 1e-9)) ** 2 * sharp[ci]).astype(np.float32)
        # 바닥·지붕 평부처럼 수평인 면을 70° 넘게 비스듬히 본 사진은 다른 사진이 없을 때만(재질이 늘어지고 흐려짐)
        s[horiz[cand] & (cos[cand] < 0.34)] *= 0.02
        # 사진 가장자리는 왜곡·비네팅이 커서 약간 감점
        cu = np.abs(uv[cand, 0] / cam["w"] - 0.5) * 2
        cv_ = np.abs(uv[cand, 1] / cam["h"] - 0.5) * 2
        s *= (1 - 0.3 * np.maximum(cu, cv_) ** 2).astype(np.float32)
        # 상위 K 갱신
        worst = top_s[cand].argmin(1)
        better = s > top_s[cand, worst]
        cb, wb = cand[better], worst[better]
        top_s[cb, wb] = s[better]
        top_c[cb, wb] = ci
    order = np.argsort(-top_s, 1)
    top_s = np.take_along_axis(top_s, order, 1)
    top_c = np.take_along_axis(top_c, order, 1)
    label = top_c[:, 0].copy()
    if vertex_only is not None and len(vertex_only) == nF:
        label[vertex_only] = -1  # 메운(추정) 면은 사진 대신 주변 실제 색(정점 색) 유지
    seen = label >= 0
    log(f"  사진에 보이는 면 {seen.mean():.1%}")

    # ── 2) 라벨 다듬기: 이웃 다수가 쓰는 사진이 이 면의 후보(점수 60% 이상)면 그쪽으로
    adj = tri.face_adjacency
    a_, b_ = adj[:, 0], adj[:, 1]
    for _ in range(3):
        changed = 0
        for dst, ls in ((b_, label[a_]), (a_, label[b_])):
            ok = (ls >= 0) & (label[dst] != ls)
            dd, lv = dst[ok], ls[ok]
            in_top = (top_c[dd] == lv[:, None]) & (top_s[dd] >= 0.6 * top_s[dd, :1])
            sw = in_top.any(1)
            label[dd[sw]] = lv[sw]
            changed += int(sw.sum())
        if changed == 0:
            break

    # ── 3) 사진별 RGB 이득(전역 색 보정): 면 중심에서 후보 사진들의 색 차이를 최소화
    imgs = {}

    def img(ci):
        if ci not in imgs:
            im = cv2.imread(str(cams[ci]["path"]), cv2.IMREAD_COLOR)
            if im is None:
                im = np.asarray(Image.open(cams[ci]["path"]).convert("RGB"))[:, :, ::-1].copy()
            imgs[ci] = im
        return imgs[ci]

    nC = len(cams)
    samp = np.full((nF, K_TOP, 3), np.nan, np.float32)
    for ci in range(nC):
        sel = np.argwhere(top_c == ci)
        if len(sel) == 0:
            continue
        uv, _ = _project(cams[ci], fc[sel[:, 0]])
        im = img(ci)
        x = np.clip(uv[:, 0].astype(np.float32), 0, im.shape[1] - 1)
        y = np.clip(uv[:, 1].astype(np.float32), 0, im.shape[0] - 1)
        c = _sample(im, x, y)
        samp[sel[:, 0], sel[:, 1]] = c
    gain = np.ones((nC, 3), np.float32)
    rows_i, rows_j, diffs = [], [], []
    for k in range(1, K_TOP):
        ok = (top_c[:, 0] >= 0) & (top_c[:, k] >= 0) & ~np.isnan(samp[:, 0, 0]) & ~np.isnan(samp[:, k, 0])
        ok &= (samp[:, 0] > 8).all(1) & (samp[:, k] > 8).all(1) & (samp[:, 0] < 250).all(1) & (samp[:, k] < 250).all(1)
        rows_i.append(top_c[ok, 0]); rows_j.append(top_c[ok, k])
        diffs.append(np.log(samp[ok, 0]) - np.log(samp[ok, k]))
    if rows_i and sum(len(r) for r in rows_i) > 100:
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import lsqr
        I = np.concatenate(rows_i); J = np.concatenate(rows_j); D = np.concatenate(diffs)
        n = len(I)
        # log g_i - log g_j = -(log c_i - log c_j)  + 정규화(log g ≈ 0)
        A = coo_matrix((np.r_[np.ones(n), -np.ones(n), np.full(nC, 0.3)],
                        (np.r_[np.arange(n), np.arange(n), n + np.arange(nC)],
                         np.r_[I, J, np.arange(nC)])), shape=(n + nC, nC)).tocsr()
        for ch in range(3):
            lg = lsqr(A, np.r_[-D[:, ch], np.zeros(nC)])[0]
            gain[:, ch] = np.exp(np.clip(lg, -0.7, 0.7))
        log(f"  사진별 색 보정 이득 범위 {gain.min():.2f}~{gain.max():.2f} ({n:,}쌍)")

    # ── 4) 패치 아틀라스: 같은 사진을 쓰는 이웃 면 묶음(패치)마다 그 사진의 투영 좌표를 그대로 UV 로 쓰고,
    #       패치의 사진 영역을 잘라 아틀라스에 붙인다(mvs-texturing 방식, xatlas 보다 수십 배 빠르고 사진 해상도 그대로)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    same = label[a_] == label[b_]
    g = coo_matrix((np.ones(int(same.sum())), (a_[same], b_[same])), shape=(nF, nF))
    ncomp, comp = connected_components(g, directed=False)
    # 면 꼭짓점의 사진 좌표 (보이는 면), 안 보인 면은 정점 색 셀로
    corner_uv = np.zeros((nF, 3, 2), np.float64)
    for ci in np.unique(label[label >= 0]):
        fs = np.where(label == ci)[0]
        uv, _ = _project(cams[ci], V[F[fs]].reshape(-1, 3))
        corner_uv[fs] = uv.reshape(-1, 3, 2)
    seen_f = label >= 0
    patches = []  # (comp id, label, x0, y0, w, h)
    comp_s = comp[seen_f]
    order = np.argsort(comp_s, kind="stable")
    fs_sorted = np.where(seen_f)[0][order]
    cs = comp_s[order]
    starts = np.r_[0, np.nonzero(np.diff(cs))[0] + 1]
    ends = np.r_[starts[1:], len(cs)]
    pad = 4  # 축소 보기(mipmap)에서 이웃 패치 색이 섞이지 않게 여유(예전 2px → 창·벽 경계에 띠)
    for st, en in zip(starts, ends):
        fs = fs_sorted[st:en]
        uv = corner_uv[fs].reshape(-1, 2)
        lo = np.floor(uv.min(0)) - pad
        hi = np.ceil(uv.max(0)) + pad
        patches.append((fs, int(label[fs[0]]), lo, hi))
    # ── 4-1) 이음새 색 맞춤(seam leveling): 패치 경계의 같은 꼭짓점이 두 패치에서 같은 색이 되도록
    #        패치마다 RGB 오프셋을 최소제곱으로 구한다(Waechter 의 전역 보정을 패치 단위로 단순화)
    p_off = np.zeros((len(patches), 3), np.float32)
    try:
        pid_of_comp = np.full(ncomp, -1, np.int64)
        for i, (fs, ci, lo, hi) in enumerate(patches):
            pid_of_comp[comp[fs[0]]] = i
        cv_ = F.ravel()
        cp_ = pid_of_comp[np.repeat(comp, 3)]
        ok = cp_ >= 0
        key = np.unique(np.stack([cv_[ok], cp_[ok]], 1), axis=0)          # (꼭짓점, 패치) 고유 쌍
        order_ = np.argsort(key[:, 0], kind="stable")
        key = key[order_]
        same_v = key[1:, 0] == key[:-1, 0]
        ia, ib = np.where(same_v)[0], np.where(same_v)[0] + 1              # 같은 꼭짓점을 공유하는 패치 쌍
        if len(ia) > 50:
            need = np.unique(np.r_[ia, ib])
            col = np.zeros((len(key), 3), np.float32)
            labs = np.array([patches[q][1] for q in key[need, 1]])
            for ci in np.unique(labs):
                rows = need[labs == ci]
                uv, _ = _project(cams[ci], V[key[rows, 0]])
                col[rows] = _sample(img(ci), uv[:, 0], uv[:, 1]) * gain[ci]
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import lsqr
            n, P = len(ia), len(patches)
            A = coo_matrix((np.r_[np.ones(n), -np.ones(n), np.full(P, 0.1)],
                            (np.r_[np.arange(n), np.arange(n), n + np.arange(P)],
                             np.r_[key[ia, 1], key[ib, 1], np.arange(P)])), shape=(n + P, P)).tocsr()
            before = float(np.abs(col[ia] - col[ib]).mean())
            for ch in range(3):
                p_off[:, ch] = lsqr(A, np.r_[col[ib, ch] - col[ia, ch], np.zeros(P)], iter_lim=300)[0]
            p_off = np.clip(p_off, -40, 40)
            after = float(np.abs((col[ia] + p_off[key[ia, 1]]) - (col[ib] + p_off[key[ib, 1]])).mean())
            log(f"  이음새 색 차이 {before:.1f} → {after:.1f} (경계 {n:,}쌍)")
    except Exception as e:
        log(f"  이음새 색 맞춤 생략: {e}")
    # 비정상적으로 큰 패치(카메라 가까이·뒤에 걸친 면이 사진 밖으로 수만 px 뻗음)는 사진 대신 정점 색으로.
    # 예전: 높이가 수십만 px 인 패치 하나를 그대로 잘라 붙이다 수십 분 멈춘 듯 보였음(PC castle-P19, 빠르게 모드)
    keep_i, n_big = [], 0
    for i, (fs, ci, lo, hi) in enumerate(patches):
        lim = 1.05 * max(cams[ci]["w"], cams[ci]["h"])
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or float(np.max(hi - lo)) > lim:
            seen_f[fs] = False
            n_big += 1
        else:
            keep_i.append(i)
    if n_big:
        p_off = p_off[keep_i]
        patches = [patches[i] for i in keep_i]
        log(f"  사진 밖으로 뻗은 패치 {n_big:,}개는 정점 색으로")
    unseen = np.where(~seen_f)[0]
    cell = 6  # 안 보인 면: 6×6 텍셀 셀(정점 색을 섞어 칠함)
    # 전체 면적에 맞춰 축척 s 결정 → 선반(shelf) 방식으로 포장, 넘치면 s 를 줄여 다시
    areas = np.array([float(np.prod(hi - lo)) for _, _, lo, hi in patches]) if patches else np.zeros(0)
    extra = len(unseen) * (cell + 1) ** 2
    s_scale = min(1.0, np.sqrt(0.80 * atlas * atlas / max(areas.sum() + extra, 1.0)))
    for _ in range(12):
        sizes = [(max(1, int(np.ceil((hi[0] - lo[0]) * s_scale))) + 1,
                  max(1, int(np.ceil((hi[1] - lo[1]) * s_scale))) + 1) for _, _, lo, hi in patches]
        idx_sorted = sorted(range(len(patches)), key=lambda i: -sizes[i][1])
        pos = [None] * len(patches)
        x = y = rowh = 0
        ok = True
        for i in idx_sorted:
            w, h = sizes[i]
            if w > atlas or h > atlas:
                ok = False
                break
            if x + w > atlas:
                x, y, rowh = 0, y + rowh, 0
            pos[i] = (x, y)
            x += w
            rowh = max(rowh, h)
        cells_per_row = atlas // (cell + 1)
        need_rows = (len(unseen) + cells_per_row - 1) // cells_per_row if len(unseen) else 0
        total_h = y + rowh + need_rows * (cell + 1)
        if ok and total_h <= atlas:
            break
        s_scale *= 0.9
    else:  # 12번 줄여도 안 들어가면 → 안 들어가는 패치는 정점 색으로(멈추거나 실패하지 않게)
        drop = [i for i in range(len(patches)) if pos[i] is None or sizes[i][0] > atlas or sizes[i][1] > atlas
                or pos[i][1] + sizes[i][1] > atlas]
        if drop:
            for i in drop:
                seen_f[patches[i][0]] = False
            dset = set(drop)
            keep_i = [i for i in range(len(patches)) if i not in dset]
            patches = [patches[i] for i in keep_i]
            pos = [pos[i] for i in keep_i]
            sizes = [sizes[i] for i in keep_i]
            p_off = p_off[keep_i]
            unseen = np.where(~seen_f)[0]
            need_rows = (len(unseen) + cells_per_row - 1) // cells_per_row if len(unseen) else 0
            y = max((p[1] for p in pos), default=0)
            rowh = max((sizes[i][1] for i in range(len(pos)) if pos[i][1] == y), default=0)
            log(f"  아틀라스에 안 들어간 패치 {len(drop):,}개는 정점 색으로")
    H = W = atlas
    tex = np.zeros((H, W, 3), np.uint8)
    filled = np.zeros((H, W), bool)  # 색을 칠한 텍셀 — 나머지는 끝에 주변 색으로 채운다
    out_uv = np.zeros((nF, 3, 2), np.float64)  # 픽셀 좌표(아틀라스)
    for i, (fs, ci, lo, hi) in enumerate(patches):
        px, py = pos[i]
        w, h = sizes[i]
        im = img(ci)
        # 사진의 패치 영역을 축척 s 로 잘라 붙임 (사진 밖은 가장자리 반사로 채움)
        M2 = np.array([[s_scale, 0, -lo[0] * s_scale], [0, s_scale, -lo[1] * s_scale]], np.float32)
        crop = cv2.warpAffine(im, M2, (w, h), flags=cv2.INTER_AREA if s_scale < 0.7 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
        crop = np.clip(crop[:, :, ::-1].astype(np.float32) * gain[ci] + p_off[i], 0, 255).astype(np.uint8)
        hh, ww = min(h, H - py), min(w, W - px)
        tex[py:py + hh, px:px + ww] = crop[:hh, :ww]
        filled[py:py + hh, px:px + ww] = True
        out_uv[fs] = (corner_uv[fs] - lo) * s_scale + np.array([px, py])
    if len(unseen):
        y0 = y + rowh
        k = np.arange(len(unseen))
        cx = (k % cells_per_row) * (cell + 1)
        cy = y0 + (k // cells_per_row) * (cell + 1)
        tri_cell = np.array([[0.5, 0.5], [cell - 0.5, 0.5], [0.5, cell - 0.5]])
        # 셀 안 텍셀마다 삼각형 무게중심 좌표 → 세 꼭짓점 색을 섞어 칠함(면 하나가 단색 덩어리가 되지 않게)
        gx, gy = np.meshgrid(np.arange(cell) + 0.5, np.arange(cell) + 0.5)
        A = np.array([[tri_cell[1, 0] - tri_cell[0, 0], tri_cell[2, 0] - tri_cell[0, 0]],
                      [tri_cell[1, 1] - tri_cell[0, 1], tri_cell[2, 1] - tri_cell[0, 1]]])
        l12 = np.linalg.solve(A, np.stack([gx.ravel() - tri_cell[0, 0], gy.ravel() - tri_cell[0, 1]]))
        bw = np.clip(np.stack([1 - l12.sum(0), l12[0], l12[1]], 1), 0, 1)
        bw /= bw.sum(1, keepdims=True)                                   # (cell², 3)
        vc = vcol[F[unseen]]                                             # (n,3 꼭짓점,3 RGB)
        cols = np.einsum("tk,nkc->ntc", bw, vc)                          # (n, cell², 3)
        ty = cy[:, None] + gy.ravel().astype(int)[None]
        tx = cx[:, None] + gx.ravel().astype(int)[None]
        ok_ = ty < H
        tex[ty[ok_], tx[ok_]] = np.clip(cols[ok_], 0, 255).astype(np.uint8)
        filled[ty[ok_], tx[ok_]] = True
        out_uv[unseen] = tri_cell[None] + np.stack([cx, cy], 1)[:, None, :]
    tex = _fill_empty(tex, filled)
    log(f"  패치 {len(patches):,}개 · 축척 {s_scale:.2f}(사진 1px → 아틀라스 {s_scale:.2f}px) · 안 보인 면 {len(unseen):,}")
    # 면 꼭짓점별 UV 를 가진 메쉬 (패치 경계에서 정점이 갈라짐) — 같은 패치 안 공유 정점은 합친다
    key = np.repeat(comp, 3).astype(np.int64) * np.int64(len(V)) + F.ravel()
    key = np.where(np.repeat(seen_f, 3), key, -1 - np.arange(nF * 3))
    uk, inv = np.unique(key, return_inverse=True)
    first = np.zeros(len(uk), np.int64)
    first[inv[::-1]] = np.arange(nF * 3)[::-1]
    newV = V[F.ravel()[first]]
    newUV = out_uv.reshape(-1, 2)[first] / np.array([W, H])
    newUV[:, 1] = 1 - newUV[:, 1]
    newF = inv.reshape(nF, 3)
    import io
    buf = io.BytesIO()
    Image.fromarray(tex).save(buf, format="JPEG", quality=90)  # GLB 안에 JPEG(PNG 대비 용량 1/5, 저장 빠름)
    buf.seek(0)
    pil = Image.open(buf)
    pil.load()
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=pil, metallicFactor=0.0, roughnessFactor=1.0)
    out_mesh = trimesh.Trimesh(newV, newF, process=False,
                               visual=trimesh.visual.TextureVisuals(uv=newUV, material=mat))
    out_mesh.metadata["bake"] = {"seen": float(seen_f.mean()), "patches": len(patches), "scale": float(s_scale)}
    log(f"  텍스처 굽기 완료: {W}×{H}px")
    return out_mesh, label
