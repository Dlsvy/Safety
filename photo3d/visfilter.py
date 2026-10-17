"""빈 공간 검사 — 여러 사진이 '여기는 비어 있다(그 뒤의 땅·벽이 보인다)'고 하는 곳의 점을 지운다.

삐죽 솟은 가시(사용자 Caliterra 결과)는 얇은 말뚝·전봇대가 조밀복원에서 시선 방향으로 길게 늘어난 점이다.
그 점은 다른 사진의 깊이 지도에서 보면 '보이는 표면보다 앞(카메라 쪽) 허공'에 있다 → 여러 장이 그렇게 말하면 가짜.
진짜 기둥은 그것을 본 사진의 깊이와 같은 깊이에 있어 지지되고 남는다(OpenMVS 융합·공간 깎기와 같은 원리).

깊이 지도: COLMAP(dense/stereo/depth_maps/*.geometric.bin, 카메라는 dense/sparse) 또는 OpenMVS(mvs/depth*.dmap).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _read_dmap(p: Path):
    """OpenMVS .dmap → (K, R, C, depth) — 세계→카메라: Xc = R (X - C), 깊이 = Xc.z"""
    b = p.read_bytes()
    name, typ, _ = struct.unpack_from("<HBB", b, 0)
    if name != 0x5244:
        return None
    _, _, dw, dh = struct.unpack_from("<4I", b, 4)
    o = 28
    ln, = struct.unpack_from("<H", b, o)
    o += 2 + ln
    nid, = struct.unpack_from("<I", b, o)
    o += 4 + 4 * nid
    K = np.frombuffer(b, "<f8", 9, o).reshape(3, 3).copy()
    o += 72
    R = np.frombuffer(b, "<f8", 9, o).reshape(3, 3).copy()
    o += 72
    C = np.frombuffer(b, "<f8", 3, o).copy()
    o += 24
    D = np.frombuffer(b, "<f4", dw * dh, o).reshape(dh, dw)
    return K, R, C, D


def _read_colmap_depth(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        head = b""
        while head.count(b"&") < 3:
            head += f.read(1)
        w, h, c = (int(x) for x in head.decode().split("&")[:3])
        a = np.fromfile(f, np.float32, w * h * c)
    return a.reshape(c, h, w)[0] if c > 1 else a.reshape(h, w)


def depth_views(work: Path):
    """(K, R, C, depth) 를 하나씩 — OpenMVS dmap 우선, 없으면 COLMAP 깊이 지도(+dense/sparse 카메라)"""
    dm = sorted((work / "mvs").glob("depth*.dmap"))
    if dm:
        for p in dm:
            r = _read_dmap(p)
            if r is not None:
                yield r
        return
    dd = work / "dense" / "stereo" / "depth_maps"
    if not dd.exists():
        return
    import pycolmap
    rec = pycolmap.Reconstruction(work / "dense" / "sparse")
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        f = dd / f"{im.name}.geometric.bin"
        if not f.exists():
            continue
        D = _read_colmap_depth(f)
        cam = rec.cameras[im.camera_id]
        K = np.asarray(cam.calibration_matrix(), float).copy()
        sx, sy = D.shape[1] / cam.width, D.shape[0] / cam.height
        K[0] *= sx
        K[1] *= sy
        T = np.asarray((im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix())
        R = T[:, :3]
        yield K, R, -R.T @ T[:, 3], D


def free_space_mask(xyz: np.ndarray, work: Path, rel: float = 0.03, min_viol: int = 2, log=print) -> np.ndarray | None:
    """지울 점 = True. 사진 깊이보다 rel(3%) 넘게 카메라 쪽에 있는 경우를 '빈 공간 위반'으로 세고,
    위반이 min_viol 장 이상이면서 같은 깊이로 '보인' 장수보다 많으면 지운다."""
    viol = np.zeros(len(xyz), np.int16)
    supp = np.zeros(len(xyz), np.int16)
    n = 0
    for K, R, C, D in depth_views(work):
        n += 1
        Xc = (xyz - C) @ R.T
        z = Xc[:, 2]
        h, w = D.shape
        u = K[0, 0] * Xc[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
        ok = (z > 0) & (u >= 0) & (v >= 0) & (u < w) & (v < h)
        idx = np.flatnonzero(ok)
        Dv = D[v[idx].astype(np.int64), u[idx].astype(np.int64)]
        has = Dv > 0
        idx, Dv, zz = idx[has], Dv[has], z[idx][has]
        viol[idx[zz < Dv * (1 - rel)]] += 1
        supp[idx[np.abs(zz - Dv) <= Dv * rel]] += 1
    if n == 0:
        return None
    bad = (viol >= min_viol) & (viol > supp)
    log(f"  빈 공간 검사: 깊이 지도 {n}장 · 여러 사진에서 '허공'으로 보인 점 {int(bad.sum()):,}개 제거"
        f"({bad.mean():.2%})")
    return bad
