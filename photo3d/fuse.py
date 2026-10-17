"""통합(포토그래메트리 + 가우시안 스플래팅) — 같은 사진·같은 카메라 위치로 두 결과를 만들고 서로 맞춘다.

조사 결과(2026-10): Bentley iTwin Capture·DJI Terra·Pix4D·Esri·Cesium 모두 '메쉬=측정, 스플랫=보기'로
같은 사진 블록에서 둘 다 만들어 겹쳐 보여준다. 학습에 측정 형상을 넣는 연구(MVPGS ECCV24, DN-Splatter WACV25,
Multiview Geometric Regularization CGF25, MeshGS ACCV24)는 형상 정확도를 올렸지만, 큰 야외 현장에서 화질(PSNR)이
오른다는 증거는 아직 없다 → 여기서는 효과가 확실한 것만 기본으로 켜고 나머지는 선택:

1) 시작점: 조밀 점군(이미 splat._seed_from_dense)
2) 표면 방향 지도: 사진 한 장 추정(MoGe) 대신 측정 메쉬를 각 사진 위치에서 그려 만든다(여러 사진으로 잰 값,
   메쉬가 없는 곳은 0 = 지도 없음). 깊이 지도는 PHOTO3D_FUSE_DEPTH=1 일 때만(MoGe 깊이는 화질을 떨어뜨렸다)
3) 정리: 사진에서 볼 때 측정 표면 '뒤'에 숨은 스플랫(벽 속·바닥 밑 잡티)을 지운다 — 표면에서 떨어져 있고
   그것을 보는 사진 대부분에서 표면 뒤에 있을 때만(MeshGS 의 거리 분류를 보수적으로)
4) 일치도 보고: 스플랫 중심 ~ 메쉬 거리(불투명도 가중) 중앙값·90%·허용오차 안 비율
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _cams(sparse: Path):
    import pycolmap
    rec = pycolmap.Reconstruction(sparse)
    out = []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        c = rec.cameras[im.camera_id]
        p = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        T = np.asarray(p.matrix())
        out.append({"name": im.name, "K": np.asarray(c.calibration_matrix()), "R": T[:, :3], "t": T[:, 3],
                    "w": int(c.width), "h": int(c.height)})
    return out


def _intersector(mesh):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        return RayMeshIntersector(mesh)
    except Exception:
        return mesh.ray


def _render(rmi, mesh, cam, max_side: int):
    """카메라 한 대에서 메쉬를 광선으로 그린다 → (z 깊이, 카메라 좌표 법선, 크기 s). 안 맞은 곳 z=0."""
    s = min(1.0, max_side / max(cam["w"], cam["h"]))
    W, H = max(1, int(round(cam["w"] * s))), max(1, int(round(cam["h"] * s)))
    K = cam["K"].copy()
    K[:2] *= s
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    dc /= np.linalg.norm(dc, axis=1, keepdims=True)
    R, t = cam["R"], cam["t"]
    C = -R.T @ t
    dw = dc @ R  # = (R^T dc^T)^T
    locs, ri, ti = rmi.intersects_location(np.repeat(C[None], len(dw), 0), dw, multiple_hits=False)
    z = np.zeros(len(dw))
    n = np.zeros((len(dw), 3))
    if len(ri):
        dist = np.linalg.norm(locs - C, axis=1)
        z[ri] = dist * dc[ri, 2]
        nc = mesh.face_normals[ti] @ R.T
        flip = (nc * dc[ri]).sum(1) > 0  # 카메라를 향하게
        nc[flip] *= -1
        cosv = -(nc * dc[ri]).sum(1)
        keep = cosv > 0.15  # 스치듯 보이는 면은 법선을 믿기 어렵다
        n[ri[keep]] = nc[keep]
    return z.reshape(H, W), n.reshape(H, W, 3), s


def write_prior_maps(mesh_path: Path, sparse: Path, data: Path, max_side: int = 1064, depth: bool = False,
                     log=print) -> dict:
    """Spirula 형식: normals/<이름>.png (카메라 OpenCV 좌표, 127.5+127.5n, 없음=검정), depths/<이름>.png (16bit mm, 없음=0)."""
    import cv2
    import trimesh
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    rmi = _intersector(mesh)
    cams = _cams(sparse)
    (data / "normals").mkdir(parents=True, exist_ok=True)
    if depth:
        (data / "depths").mkdir(parents=True, exist_ok=True)
    cover = []
    for cam in cams:
        z, n, _ = _render(rmi, mesh, cam, max_side)
        stem = str(Path(cam["name"]).with_suffix(""))
        (data / "normals" / stem).parent.mkdir(parents=True, exist_ok=True)
        img = np.where((np.abs(n).sum(-1) > 0)[..., None], np.clip(127.5 + 127.5 * n, 0, 255), 0).astype(np.uint8)
        cv2.imwrite(str(data / "normals" / f"{stem}.png"), img[..., ::-1])
        cover.append(float((z > 0).mean()))
        if depth:
            # 깊이가 크게 끊기는 경계(앞뒤 물체 사이)는 지도에서 뺀다
            zz = z.copy()
            gy, gx = np.gradient(zz)
            edge = (np.hypot(gx, gy) > 0.05 * np.maximum(zz, 1e-6)) & (zz > 0)
            zz[edge] = 0
            (data / "depths" / stem).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(data / "depths" / f"{stem}.png"), np.clip(zz * 1000, 0, 65535).astype(np.uint16))
    info = {"images": len(cams), "mesh_coverage": round(float(np.mean(cover)) if cover else 0.0, 3), "depth": depth}
    log(f"  통합: 측정 메쉬로 사진 {len(cams)}장의 표면 방향 지도{'·깊이 지도' if depth else ''} 생성"
        f"(사진 면적의 {info['mesh_coverage']:.0%}에 측정 표면)")
    return info


def mesh_prune_and_agree(splat_ply: Path, mesh_path: Path, sparse: Path, dense_ply: Path | None, out: Path,
                         max_side: int = 400, log=print) -> dict:
    """측정 표면 뒤에 숨은 스플랫 제거 + 메쉬와의 일치도. splat_ply 와 메쉬는 같은 좌표(조밀복원 좌표, Z-up)."""
    import trimesh
    from scipy.spatial import cKDTree
    from .splatclean import read_splat, write_splat
    a, props, _ = read_splat(splat_ply)
    col = {q: i for i, q in enumerate(props)}
    xyz = a[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    op = 1 / (1 + np.exp(-a[:, col["opacity"]].astype(np.float64)))
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    # 메쉬까지 거리(표면을 촘촘히 뽑은 점으로 근사)
    surf = np.asarray(trimesh.sample.sample_surface(mesh, 2_000_000, seed=0)[0])
    stree = cKDTree(surf)
    spacing = float(np.median(stree.query(surf[np.random.default_rng(0).choice(len(surf), min(20000, len(surf)), replace=False)], k=2)[0][:, 1]))
    tau = 3 * spacing
    if dense_ply is not None and Path(dense_ply).exists():
        from .fill import read_points
        dp = np.asarray(read_points(dense_ply)[0], float)
        dtree = cKDTree(dp)
        sp_d = float(np.median(dtree.query(dp[np.random.default_rng(1).choice(len(dp), min(20000, len(dp)), replace=False)], k=2)[0][:, 1]))
        tau = max(tau, 3 * sp_d)
        d_dense = dtree.query(xyz, workers=-1)[0]
    else:
        d_dense = np.full(len(xyz), np.inf)
    d_mesh = stree.query(xyz, workers=-1)[0]
    # 사진마다 메쉬 깊이를 낮은 해상도로 그려 스플랫이 표면 앞/뒤 어디인지 센다
    rmi = _intersector(mesh)
    cams = _cams(sparse)
    seen = np.zeros(len(xyz), np.int32)
    behind = np.zeros(len(xyz), np.int32)
    for cam in cams:
        z, _, s = _render(rmi, mesh, cam, max_side)
        H, W = z.shape
        Xc = xyz @ cam["R"].T + cam["t"]
        zc = Xc[:, 2]
        K = cam["K"]
        u = (K[0, 0] * Xc[:, 0] / np.maximum(zc, 1e-9) + K[0, 2]) * s
        v = (K[1, 1] * Xc[:, 1] / np.maximum(zc, 1e-9) + K[1, 2]) * s
        ok = (zc > 0) & (u >= 0) & (v >= 0) & (u < W) & (v < H)
        idx = np.where(ok)[0]
        zm = z[v[idx].astype(int), u[idx].astype(int)]
        has = zm > 0
        seen[idx[has]] += 1
        behind[idx[has & (zc[idx] > zm + tau)]] += 1
    hidden = (seen >= 2) & (behind >= 0.8 * seen) & (d_mesh > tau) & (d_dense > tau)
    keep = ~hidden
    write_splat(out, a[keep], props, comment="photo3d: fused (mesh-checked) splat, Z-up")
    w = op[keep]
    dk = d_mesh[keep]
    order = np.argsort(dk)
    cw = np.cumsum(w[order]) / max(w.sum(), 1e-12)

    def wq(q):
        return float(dk[order][min(len(order) - 1, np.searchsorted(cw, q))]) if len(order) else 0.0
    info = {"before": int(len(a)), "hidden_removed": int(hidden.sum()), "after": int(keep.sum()),
            "tau_m": round(tau, 4), "dist_p50_m": round(wq(0.5), 4), "dist_p90_m": round(wq(0.9), 4),
            "within_tau": round(float(w[dk <= tau].sum() / max(w.sum(), 1e-12)), 3)}
    log(f"  통합 정리: 측정 표면 뒤에 숨은 스플랫 {info['hidden_removed']:,}개 제거 → {info['after']:,}개 · "
        f"메쉬와 거리 중앙 {info['dist_p50_m'] * 1000:.0f}mm · 90% {info['dist_p90_m'] * 1000:.0f}mm · "
        f"허용({tau * 1000:.0f}mm) 안 {info['within_tau']:.0%}")
    return info


def pick_mesh(mvs_dir: Path) -> Path | None:
    """통합에 쓸 측정 메쉬: 평면 정리본 > 엄격 정리본 > 원본"""
    for n in ("scene_mesh_planar.ply", "scene_mesh_pruned.ply", "scene_mesh_poisson.ply", "scene_mesh.ply"):
        if (mvs_dir / n).exists():
            return mvs_dir / n
    return None
