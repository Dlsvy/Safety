"""측량형 결과물(포토그래메트리 탭) — Pix4D·Metashape·ODM 이 건설현장에 주는 기본 산출물.

- 정사영상(site_ortho.jpg + .jgw 월드파일): 텍스처 메쉬를 위에서 수직으로 본 사진. 1px = 몇 cm 가 일정해
  거리·면적을 바로 잴 수 있다(현장 배치도·공정 사진 대장용). ODM 의 odm_orthophoto 도 메쉬를 위에서 그린다.
- 수치표면모델 DSM(site_dsm.tif, float32 m + 보기용 site_dsm_color.png): 칸마다 가장 높은 표면 높이.
  토공량·적치물 부피·높이 확인용.
- 점군(site_points.las, LAS 1.2 점형식 2 = 좌표+색): CloudCompare·Civil3D·Revit·ReCap·Rhino 에서 바로 열림.
좌표는 모델 좌표(Z-up, m). GPS·기준점으로 맞췄으면 현장 좌표(ENU) 그대로.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def _tex_colors(mesh, fidx, bary):
    """면 번호·무게중심좌표 → 색(uint8 RGB). 텍스처가 있으면 UV 로, 없으면 정점 색으로."""
    vis = mesh.visual
    f = mesh.faces[fidx]
    if getattr(vis, "kind", None) == "texture" and getattr(vis, "uv", None) is not None:
        img = vis.material.baseColorTexture if hasattr(vis.material, "baseColorTexture") else None
        if img is None:
            img = getattr(vis.material, "image", None)
        if img is not None:
            im = np.asarray(img.convert("RGB"))
            uv = (vis.uv[f] * bary[:, :, None]).sum(1)
            h, w = im.shape[:2]
            x = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(np.int64)
            y = np.clip((1 - uv[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(np.int64)
            return im[y, x]
    try:
        vc = np.asarray(mesh.visual.to_color().vertex_colors)[:, :3].astype(np.float64)
        return np.clip((vc[f] * bary[:, :, None]).sum(1), 0, 255).astype(np.uint8)
    except Exception:
        return np.full((len(fidx), 3), 160, np.uint8)


def photo_gsd(sparse: Path) -> float | None:
    """사진 1px 이 땅에서 몇 m 인지(GSD): 사진마다 보이는 3D 점까지 깊이 중앙값 / 초점거리(px) → 사진들의 중앙값."""
    import pycolmap
    try:
        rec = pycolmap.Reconstruction(sparse)
        xyz = {pid: p.xyz for pid, p in rec.points3D.items()}
        g = []
        for iid in rec.reg_image_ids():
            im = rec.images[iid]
            cam = rec.cameras[im.camera_id]
            T = np.asarray((im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix())
            ids = [p.point3D_id for p in im.points2D if p.has_point3D()]
            if len(ids) < 20:
                continue
            P = np.array([xyz[i] for i in ids[:2000]])
            z = P @ T[:, :3].T[:, 2] + T[2, 3]
            g.append(float(np.median(z[z > 0])) / float(cam.calibration_matrix()[0, 0]))
        return float(np.median(g)) if g else None
    except Exception:
        return None


def _true_ortho(dsm: np.ndarray, valid: np.ndarray, lo, hi, res: float, sparse: Path, img_dir: Path, log=print):
    """참 정사영상: 정사 화소마다 DSM 높이의 3D 점을 원본(왜곡보정) 사진에 투영해 색을 가져온다.
    가림은 사진별 깊이 버퍼(DSM 을 사진에 투영)로, 사진 선택은 수직에 가까운 시선·가까운 거리·사진 가장자리 감점.
    반환: (색 H×W×3, 채운 칸 mask) — 메쉬 텍스처(축소·여러 번 재표본)보다 사진 해상도 그대로라 선명하다."""
    import cv2
    import pycolmap
    rec = pycolmap.Reconstruction(sparse)
    H, W = dsm.shape
    rr, cc = np.nonzero(valid)
    X = np.stack([lo[0] + (cc + 0.5) * res, hi[1] - (rr + 0.5) * res, dsm[rr, cc].astype(np.float64)], 1)
    n = len(X)
    best = np.zeros(n, np.float32)
    bcam = np.full(n, -1, np.int32)
    bu = np.zeros(n, np.float32)
    bv = np.zeros(n, np.float32)
    cams = []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        cam = rec.cameras[im.camera_id]
        T = np.asarray((im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix())
        K = np.asarray(cam.calibration_matrix())
        cams.append((im.name, K, T, int(cam.width), int(cam.height)))
    coarse = np.arange(0, n, max(1, n // 200_000))
    def project(ci):
        name, K, T, w, h = cams[ci]
        # 먼저 성긴 점으로 이 사진에 보이는 범위를 잡고 그 안의 화소만 계산(속도)
        Xc = X[coarse] @ T[:, :3].T + T[:, 3]
        u = K[0, 0] * Xc[:, 0] / np.maximum(Xc[:, 2], 1e-9) + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / np.maximum(Xc[:, 2], 1e-9) + K[1, 2]
        ins = (Xc[:, 2] > 0) & (u >= 0) & (v >= 0) & (u < w) & (v < h)
        if not ins.any():
            return None
        r0, r1 = rr[coarse[ins]].min() - 64, rr[coarse[ins]].max() + 64
        c0, c1 = cc[coarse[ins]].min() - 64, cc[coarse[ins]].max() + 64
        sel = np.flatnonzero((rr >= r0) & (rr <= r1) & (cc >= c0) & (cc <= c1))
        Xc = X[sel] @ T[:, :3].T + T[:, 3]
        z = Xc[:, 2]
        u = K[0, 0] * Xc[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
        ok = (z > 0) & (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
        sel, z, u, v = sel[ok], z[ok], u[ok], v[ok]
        if not len(sel):
            return None
        # 가림: 사진 화면 4px 칸마다 가장 가까운 깊이 — 그보다 (허용오차) 더 먼 점은 가려진 것
        g = 4
        gw = w // g + 1
        key = (v.astype(np.int64) // g) * gw + (u.astype(np.int64) // g)
        zb = np.full((h // g + 1) * gw, np.inf)
        np.minimum.at(zb, key, z)
        # 허용오차: 깊이 버퍼 한 칸(4px)이 땅에서 덮는 크기의 8배 — 너무 작으면 비스듬한 면이 '가려짐'으로 잘못 판정돼
        # 삼각형 모양으로 다른(흐린) 사진이 끼어든다(도로에서 확인)
        vis = z <= zb[key] + np.maximum(3 * res, 8 * g * z / K[0, 0])
        sel, z, u, v = sel[vis], z[vis], u[vis], v[vis]
        # 점수: 그 지점을 가장 바로 위에서 본 사진(정사영상 표준 — 건물 쓰러짐 최소). 각도를 강하게(cos⁸) 두면
        # 사진마다 '자기 바로 아래' 넓은 덩어리를 맡아 이음새가 적다(화소마다 들쭉날쭉 바뀌면 기와 음영 얼룩)
        C = -T[:, :3].T @ T[:, 3]
        d = X[sel] - C
        cosv = np.abs(d[:, 2]) / np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        edge = np.maximum(np.abs(u / w - 0.5), np.abs(v / h - 0.5)) * 2
        sc = (cosv ** 8 * (K[0, 0] / z) ** 0.5 * (1 - 0.3 * edge ** 2)).astype(np.float32)
        return sel, u, v, sc

    for ci in range(len(cams)):
        r = project(ci)
        if r is None:
            continue
        sel, u, v, sc = r
        b = sc > best[sel]
        best[sel[b]] = sc[b]
        bcam[sel[b]] = ci
        bu[sel[b]] = u[b]
        bv[sel[b]] = v[b]
    out = np.zeros((n, 3), np.float32)
    gains = []
    for ci, (name, K, T, w, h) in enumerate(cams):
        idx = np.flatnonzero(bcam == ci)
        if not len(idx):
            continue
        im = cv2.imread(str(Path(img_dir) / name), cv2.IMREAD_COLOR)
        if im is None:
            bcam[idx] = -1
            continue
        m = len(idx)
        cols = 1024
        rows = (m + cols - 1) // cols
        mx = np.zeros(rows * cols, np.float32)
        my = np.zeros(rows * cols, np.float32)
        mx[:m], my[:m] = bu[idx], bv[idx]
        smp = cv2.remap(im, mx.reshape(rows, cols), my.reshape(rows, cols), cv2.INTER_LINEAR).reshape(-1, 3)[:m]
        out[idx] = smp[:, ::-1]
        gains.append(ci)
    got = bcam >= 0
    # 사진마다 밝기가 달라 생기는 경계: 이웃한 다른 사진 화소와의 평균 밝기 차로 사진별 이득을 맞춘다
    lab = np.full((H, W), -1, np.int32)
    lab[rr, cc] = bcam
    img = np.zeros((H, W, 3), np.float32)
    img[rr, cc] = out
    a, b_ = lab[:, :-1], lab[:, 1:]
    m_ = (a >= 0) & (b_ >= 0) & (a != b_)
    if m_.any() and len(cams) > 1:
        ia, ib = a[m_], b_[m_]
        la = img[:, :-1][m_].mean(1) + 1
        lb = img[:, 1:][m_].mean(1) + 1
        lg = np.zeros(len(cams))
        for _ in range(50):  # log 이득: 경계 양쪽 밝기 차를 줄이는 방향으로 반복(약한 0 고정)
            r = (np.log(la) + lg[ia]) - (np.log(lb) + lg[ib])
            gsum = np.zeros(len(cams))
            cnt = np.zeros(len(cams)) + 1e-9
            np.add.at(gsum, ia, -r)
            np.add.at(gsum, ib, r)
            np.add.at(cnt, ia, 1)
            np.add.at(cnt, ib, 1)
            lg += 0.5 * gsum / (cnt + 50)
            lg -= lg.mean()
        gain = np.exp(np.clip(lg, -0.5, 0.5))
        img[rr, cc] *= gain[np.maximum(bcam, 0)][:, None]
    mask = np.zeros((H, W), bool)
    mask[rr[got], cc[got]] = True
    log(f"  참 정사영상: 사진 {len(gains)}장에서 직접 색(정사 화소의 {got.mean():.0%})")
    return np.clip(img, 0, 255).astype(np.uint8), mask


def ortho_dsm(meshes, out: Path, res: float | None = None, max_px: int = 8192, max_samples: int = 60_000_000,
              z_cut: float | None = None, photos: tuple | None = None, log=print) -> dict | None:
    """텍스처 메쉬(Z-up, m; 여러 개면 목록)를 위에서 내려다본 정사영상 + DSM.
    면을 화소 크기에 맞춰 촘촘히 찍고 칸마다 가장 높은 점을 남긴다.
    z_cut: 이 높이보다 위는 빼고 그린다(실내·지붕 아래 현장: 천장 대신 바닥이 보이는 평면도)."""
    import cv2
    meshes = [m for m in (meshes if isinstance(meshes, (list, tuple)) else [meshes]) if m is not None and len(m.faces)]
    if not meshes:
        return None
    allV = np.vstack([np.asarray(m.vertices, np.float64) for m in meshes])
    lo, hi = allV[:, :2].min(0), allV[:, :2].max(0)
    ext = hi - lo

    def area_xy(m):
        T = np.asarray(m.vertices, np.float64)[np.asarray(m.faces)]
        return T, 0.5 * np.abs((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1])
                               - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))
    tris = [area_xy(m) for m in meshes]
    if photos:
        # 참 정사영상은 위에서 내려다본 사진(드론 격자 비행)이 많을 때만 — 비스듬히 선회한 사진만 있으면 사진마다
        # 대기·대비가 달라 조각조각 얼룩진다(Pix4D 건물 36장 확인) → 그때는 이음새가 정리된 메쉬 텍스처로
        try:
            import pycolmap
            from .planes import is_nadir, up_vector
            r_ = pycolmap.Reconstruction(photos[0])
            if not is_nadir(r_, up_vector(r_)):
                log("  정사영상: 비스듬한 사진 위주 → 메쉬 텍스처로 그림")
                photos = None
        except Exception:
            photos = None
    gsd = photo_gsd(photos[0]) if photos else None
    if res is None and gsd:
        # 사진에서 직접 색을 가져오므로 메쉬 면 크기가 아니라 사진 해상도(GSD)까지 곱게
        res = max(gsd, float(ext.max()) / 16384, 0.002)
        max_samples = max(max_samples, 120_000_000)
    elif res is None:
        res = float(ext.max()) / max_px
        a0 = tris[0][1]
        # 메쉬보다 훨씬 고운 화소는 의미 없음 → 위에서 본 면 크기의 중앙값 근처까지만
        med = float(np.sqrt(np.median(a0[a0 > 0]))) if (a0 > 0).any() else res
        res = max(res, 0.5 * med, 0.002)
    for _ in range(8):  # 점 수가 너무 많으면 화소를 키운다
        tot = sum(int(np.clip(np.ceil(2.0 * a / res ** 2), 1, 4096).sum()) for _, a in tris)
        if tot <= max_samples:
            break
        res *= 1.4
    W, H = int(np.ceil(ext[0] / res)) + 1, int(np.ceil(ext[1] / res)) + 1
    rng = np.random.default_rng(0)
    zbuf = np.full(W * H, -np.inf, np.float64)
    rgb = np.zeros((W * H, 3), np.uint8)
    for mesh, (T, a_xy) in zip(meshes, tris):
        n_f = np.clip(np.ceil(2.0 * a_xy / res ** 2), 1, 4096).astype(np.int64)
        fid_all = np.repeat(np.arange(len(T)), n_f)
        for s in range(0, len(fid_all), 8_000_000):  # 나눠서(메모리)
            fid = fid_all[s:s + 8_000_000]
            r1, r2 = rng.random(len(fid)), rng.random(len(fid))
            sq = np.sqrt(r1)
            bary = np.stack([1 - sq, sq * (1 - r2), sq * r2], 1)
            P = (T[fid] * bary[:, :, None]).sum(1)
            if z_cut is not None:
                below = P[:, 2] <= z_cut
                P, fid, bary = P[below], fid[below], bary[below]
            ix = np.clip(((P[:, 0] - lo[0]) / res).astype(np.int64), 0, W - 1)
            iy = np.clip(((hi[1] - P[:, 1]) / res).astype(np.int64), 0, H - 1)  # 위쪽 = 북(+Y)
            cell = iy * W + ix
            order = np.lexsort((P[:, 2], cell))  # 칸별 가장 높은 점이 마지막
            c_s = cell[order]
            last = np.r_[c_s[1:] != c_s[:-1], True]
            sel = order[last]
            sel = sel[P[sel, 2] > zbuf[cell[sel]]]
            zbuf[cell[sel]] = P[sel, 2]
            rgb[cell[sel]] = _tex_colors(mesh, fid[sel], bary[sel])
    valid = np.isfinite(zbuf).reshape(H, W)
    img = rgb.reshape(H, W, 3)
    dsm = np.where(valid, zbuf.reshape(H, W), np.nan).astype(np.float32)
    true_ortho = False
    if photos and valid.any():
        try:
            pim, pm = _true_ortho(dsm, valid, lo, hi, res, photos[0], photos[1], log=log)
            # 사진마다 빛·반사가 달라 생기는 밝기 얼룩: 세밀한 무늬는 사진에서, 넓은 밝기는 이음새가 정리된
            # 메쉬 텍스처에서 가져온다(두 영상의 흐린 판 비율로 보정, 0.6~1.6배)
            sig = max(3.0, 0.08 / res)  # 약 8cm
            msk = (pm & valid).astype(np.float32)
            bm = cv2.GaussianBlur(msk, (0, 0), sig) + 1e-6
            lo_ref = cv2.GaussianBlur(img.astype(np.float32) * msk[..., None], (0, 0), sig) / bm[..., None]
            lo_pho = cv2.GaussianBlur(pim.astype(np.float32) * msk[..., None], (0, 0), sig) / bm[..., None]
            ratio = np.clip((lo_ref + 4) / (lo_pho + 4), 0.6, 1.6)
            pim = np.clip(pim * ratio, 0, 255).astype(np.uint8)
            img = np.where(pm[..., None], pim, img)
            true_ortho = True
        except Exception as e:
            log(f"  참 정사영상 생략(메쉬 텍스처로): {e}")
    # 작은 빈칸(점 사이 틈)만 메우기 — 큰 빈 곳은 실제로 찍히지 않은 곳이라 그대로 둔다
    hole = (~valid).astype(np.uint8)
    small = cv2.morphologyEx(hole, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # 큰 빈 곳만 남음
    fillm = (hole & (1 - small)).astype(np.uint8)
    if fillm.any():
        img = cv2.inpaint(img, fillm, 3, cv2.INPAINT_TELEA)
        zf = np.nan_to_num(dsm, nan=float(np.nanmin(dsm)) if valid.any() else 0.0)
        zf = np.where(fillm > 0, cv2.dilate(zf, np.ones((3, 3), np.uint8)), zf)
        dsm = np.where(fillm > 0, zf, dsm)
        valid = valid | (fillm > 0)
    out = Path(out)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    cv2.imwrite(str(out / "site_ortho.jpg"), np.where(valid[..., None], img, 255)[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out / "site_ortho.png"), np.dstack([img[:, :, ::-1], alpha]))  # 투명 배경(CAD·GIS 겹치기용)
    # 월드파일: 화소 크기, 회전 0, 0, -화소 크기, 왼쪽 위 화소 중심 좌표
    wf = f"{res:.6f}\n0\n0\n{-res:.6f}\n{lo[0] + res / 2:.4f}\n{hi[1] - res / 2:.4f}\n"
    for ext_ in ("jgw", "pgw", "tfw"):
        (out / f"site_ortho.{ext_}" if ext_ != "tfw" else out / "site_dsm.tfw").write_text(wf)
    from PIL import Image
    Image.fromarray(np.nan_to_num(dsm, nan=-9999.0).astype(np.float32), mode="F").save(out / "site_dsm.tif")
    # 보기용 DSM: 높이 색 + 음영
    z = dsm.copy()
    zl, zh = (np.nanpercentile(z, 1), np.nanpercentile(z, 99)) if valid.any() else (0, 1)
    t = np.clip((np.nan_to_num(z, nan=zl) - zl) / max(zh - zl, 1e-6), 0, 1)
    col = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    gy, gx = np.gradient(np.nan_to_num(z, nan=zl) / res)
    shade = np.clip(0.75 + 0.25 * (-gx - gy) / np.sqrt(1 + gx ** 2 + gy ** 2), 0.4, 1.0)
    col = (col * shade[..., None]).astype(np.uint8)
    col[~valid] = 255
    cv2.imwrite(str(out / "site_dsm_color.png"), col)
    meta = {"m_per_px": round(res, 5), "size_px": [W, H], "x_min": float(lo[0]), "y_max": float(hi[1]),
            "z_cut_m": None if z_cut is None else round(float(z_cut), 3),
            "true_ortho": true_ortho, "photo_gsd_m": None if gsd is None else round(gsd, 4),
            "z_range_m": [round(float(zl), 3), round(float(zh), 3)], "coverage": round(float(valid.mean()), 3),
            "note": "화소 (c,r) 중심 = (x_min + (c+0.5)*m_per_px, y_max - (r+0.5)*m_per_px). site_dsm.tif = 높이(m), 빈 곳 -9999"}
    (out / "site_ortho.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  정사영상·DSM: {W}×{H}px ({res * 100:.1f} cm/px), 높이 {zl:.2f}~{zh:.2f} m"
        + (f" · {z_cut:.2f} m 위(천장·지붕)는 잘라 평면도처럼" if z_cut is not None else ""))
    return meta


def plan_cut_height(meshes, cams: np.ndarray | None) -> float | None:
    """카메라보다 위에 큰 수평면(천장·지붕·캐노피)이 있으면 평면도 절단 높이(카메라 높이 + 여유)를 돌려준다.
    드론처럼 카메라가 모든 것 위에 있으면 None."""
    if cams is None or len(cams) < 3:
        return None
    zc = float(np.percentile(cams[:, 2], 90))
    area_hi = area_all = 0.0
    for m in meshes:
        n = np.asarray(m.face_normals)
        a = np.asarray(m.area_faces)
        hz = np.abs(n[:, 2]) > 0.8
        z = np.asarray(m.triangles_center)[:, 2]
        area_all += float(a[hz].sum())
        area_hi += float(a[hz & (z > zc + 0.3)].sum())
    if area_all > 0 and area_hi / area_all > 0.1:
        spread = float(np.ptp(cams[:, 2]))
        return zc + max(0.5, 0.2 * spread)
    return None


def write_las(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None, scale: float = 0.001) -> int:
    """LAS 1.2, 점형식 2(좌표 + RGB). 별도 라이브러리 없이 작성."""
    xyz = np.asarray(xyz, np.float64)
    n = len(xyz)
    mn, mx = xyz.min(0), xyz.max(0)
    off = np.floor(mn)
    q = np.round((xyz - off) / scale).astype(np.int32)
    rec = np.zeros(n, dtype=[("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("i", "<u2"), ("b", "u1"), ("c", "u1"),
                             ("a", "i1"), ("u", "u1"), ("p", "<u2"), ("r", "<u2"), ("g", "<u2"), ("bl", "<u2")])
    rec["x"], rec["y"], rec["z"] = q[:, 0], q[:, 1], q[:, 2]
    rec["b"] = 0b00001001  # 반사 1번째/총 1개
    rec["c"] = 1  # 분류 안 함
    if rgb is not None:
        c = np.asarray(rgb)[:, :3].astype(np.uint16) * 257
        rec["r"], rec["g"], rec["bl"] = c[:, 0], c[:, 1], c[:, 2]
    hdr = bytearray(227)
    struct.pack_into("<4sHHI2s2s8sBB32s32sHHHIIBHI5I3d3d6d", hdr, 0, b"LASF", 0, 0, 0, b"\0\0", b"\0\0", b"\0" * 8,
                     1, 2, b"photo3d".ljust(32, b"\0"), b"photo3d survey export".ljust(32, b"\0"),
                     1, 2026, 227, 227, 0, 2, 26, min(n, 0xFFFFFFFF), min(n, 0xFFFFFFFF), 0, 0, 0, 0,
                     scale, scale, scale, *off, mx[0], mn[0], mx[1], mn[1], mx[2], mn[2])
    with open(path, "wb") as f:
        f.write(bytes(hdr))
        f.write(rec.tobytes())
    return n


def export_points(points_ply: Path, out: Path, bounds: np.ndarray | None = None, max_points: int = 30_000_000,
                  log=print) -> dict | None:
    """조밀 점군 → site_points.las (메쉬 범위로 잘라 촬영 범위 밖 잡음 제외)."""
    from .fill import read_points
    xyz, _, rgb = read_points(points_ply)
    xyz = np.asarray(xyz, np.float64)
    rgb = np.asarray(rgb) if rgb is not None and len(rgb) == len(xyz) else None
    keep = np.isfinite(xyz).all(1)
    if bounds is not None:
        b = np.asarray(bounds, float)
        pad = 0.02 * (b[1] - b[0])
        keep &= ((xyz >= b[0] - pad) & (xyz <= b[1] + pad)).all(1)
    idx = np.where(keep)[0]
    if len(idx) > max_points:
        idx = idx[:: int(np.ceil(len(idx) / max_points))]
    n = write_las(out / "site_points.las", xyz[idx], rgb[idx] if rgb is not None else None)
    log(f"  점군 내보내기: site_points.las ({n:,}점, {(out / 'site_points.las').stat().st_size / 1e6:.0f} MB)")
    return {"file": "site_points.las", "points": int(n)}
