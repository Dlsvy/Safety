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


def ortho_dsm(meshes, out: Path, res: float | None = None, max_px: int = 8192, max_samples: int = 60_000_000,
              z_cut: float | None = None, log=print) -> dict | None:
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
    if res is None:
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
