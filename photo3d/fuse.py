"""통합(포토그래메트리 + 가우시안 스플래팅) — 같은 사진·같은 카메라 위치로 두 결과를 만들고 서로 맞춘다.

조사 결과(2026-10): Bentley iTwin Capture·DJI Terra·Pix4D·Esri·Cesium 모두 '메쉬=측정, 스플랫=보기'로
같은 사진 블록에서 둘 다 만들어 겹쳐 보여준다. 학습에 측정 형상을 넣는 연구(MVPGS ECCV24, DN-Splatter WACV25,
Multiview Geometric Regularization CGF25, MeshGS ACCV24)는 형상 정확도를 올렸지만, 큰 야외 현장에서 화질(PSNR)이
오른다는 증거는 아직 없다 → 여기서는 효과가 확실한 것만 기본으로 켜고 나머지는 선택:

1) 시작점: 조밀 점군(이미 splat._seed_from_dense)
2) 표면 방향 지도: 사진 한 장 추정(MoGe) 대신 측정 메쉬를 각 사진 위치에서 그려 만든다(여러 사진으로 잰 값,
   메쉬가 없는 곳은 0 = 지도 없음). 깊이 지도는 PHOTO3D_FUSE_DEPTH=1 일 때만(MoGe 깊이는 화질을 떨어뜨렸다)
3) 정리: 사진에서 볼 때 측정 표면 '뒤'에 숨은 스플랫(벽 속·바닥 밑 잡티)을 지운다 — 표면에서 떨어져 있고
   그것을 보는 사진 대부분에서 표면 뒤에 있을 때만(MeshGS 의 거리 분류를 보수적으로).
   표면 '앞' 허공에 뜬 스플랫(벽 앞 검은 얼룩)도 지운다 — 사진들이 그 자리를 지나 표면을 봤으니 빈 공간이다.
   앞쪽은 실제 물체(전선·나뭇가지)일 수 있어 더 엄격하게: 사진 3장 이상, 조밀점·메쉬에서 허용오차 3배 이상 떨어진 것,
   또는 어둡고 반투명한 것(로그 대비 어두운 반투명 덩어리가 벽 앞에 뜨는 전형적인 잡티)
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
    front = np.zeros(len(xyz), np.int32)
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
        # 그린 깊이는 저해상도(픽셀 하나 = zc/(f·s), 먼 바닥은 수 cm) → 픽셀 2개 크기만큼 여유
        # (고정 허용오차만 쓰면 비스듬히 보이는 먼 바닥이 '표면 뒤'로 세어져 사진 바닥 스플랫이 통째로 지워졌다)
        mg = tau + 2 * zc[idx] / (K[0, 0] * s)
        seen[idx[has]] += 1
        behind[idx[has & (zc[idx] > zm + mg)]] += 1
        front[idx[has & (zc[idx] < zm - 2 * mg)]] += 1
    lo, hi = mesh.bounds[0] - tau, mesh.bounds[1] + tau
    in_box = ((xyz >= lo) & (xyz <= hi)).all(1)  # 메쉬 범위 밖은 앞뒤를 판단할 근거가 약하다
    hidden = (seen >= 2) & (behind >= 0.8 * seen) & (d_mesh > tau) & (d_dense > tau) & in_box
    # 표면 앞 허공: 색(SH 0차)으로 밝기 → 어둡고 반투명한 잡티는 조금 덜 엄격하게
    lum = np.full(len(xyz), 0.5)
    if all(f"f_dc_{i}" in col for i in range(3)):
        rgb = 0.5 + 0.28209479 * a[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]].astype(np.float64)
        lum = rgb @ np.array([0.299, 0.587, 0.114])
    free = (seen >= 3) & (front >= 0.8 * seen) & in_box
    floating = free & (d_mesh > 3 * tau) & (d_dense > 3 * tau)
    floating |= in_box & (seen >= 3) & (front >= 0.6 * seen) & (lum < 0.25) & (op < 0.6) & (d_mesh > tau) & (d_dense > tau)
    floating &= ~hidden
    keep = ~(hidden | floating)
    write_splat(out, a[keep], props, comment="photo3d: fused (mesh-checked) splat, Z-up")
    w = op[keep]
    dk = d_mesh[keep]
    order = np.argsort(dk)
    cw = np.cumsum(w[order]) / max(w.sum(), 1e-12)

    def wq(q):
        return float(dk[order][min(len(order) - 1, np.searchsorted(cw, q))]) if len(order) else 0.0
    info = {"before": int(len(a)), "hidden_removed": int(hidden.sum()), "floating_removed": int(floating.sum()),
            "after": int(keep.sum()),
            "tau_m": round(tau, 4), "dist_p50_m": round(wq(0.5), 4), "dist_p90_m": round(wq(0.9), 4),
            "within_tau": round(float(w[dk <= tau].sum() / max(w.sum(), 1e-12)), 3)}
    log(f"  통합 정리: 측정 표면 뒤에 숨은 스플랫 {info['hidden_removed']:,}개 · 앞 허공에 뜬 것 "
        f"{info['floating_removed']:,}개 제거 → {info['after']:,}개 · "
        f"메쉬와 거리 중앙 {info['dist_p50_m'] * 1000:.0f}mm · 90% {info['dist_p90_m'] * 1000:.0f}mm · "
        f"허용({tau * 1000:.0f}mm) 안 {info['within_tau']:.0%}")
    return info


def pick_mesh(mvs_dir: Path) -> Path | None:
    """통합에 쓸 측정 메쉬: 평면 정리본 > 엄격 정리본 > 원본"""
    for n in ("scene_mesh_planar.ply", "scene_mesh_pruned.ply", "scene_mesh_poisson.ply", "scene_mesh.ply"):
        if (mvs_dir / n).exists():
            return mvs_dir / n
    return None


def _quat_from_z(n: np.ndarray) -> np.ndarray:
    """국소 z축을 법선 n 으로 돌리는 사원수(w,x,y,z) — 납작한 원판 스플랫용"""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(np.broadcast_to(z, n.shape), n)
    w = 1.0 + n[:, 2]
    q = np.concatenate([w[:, None], v], 1)
    bad = w < 1e-6  # n = -z
    q[bad] = [0.0, 1.0, 0.0, 0.0]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def make_best(out: Path, work: Path, max_backing: int = 1_500_000, log=print) -> dict | None:
    """'최적 합성' 모델(site_best.ply, 가우시안 스플랫 형식):
    스플랫의 장점(실사 같은 형태·얇은 물체·자연스러운 색) + 포토그래메트리의 장점(속이 꽉 찬 면).
    1) 측정 표면 뒤에 숨은 스플랫·표면에서 먼 잡티 제거(mesh_prune_and_agree)
    2) 표면 위인데 반투명한 스플랫은 불투명도를 올림(지붕·벽이 비쳐 보이는 것 줄이기)
    3) 표면 바로 뒤(허용오차 절반)에 메쉬 텍스처 색의 납작한 불투명 스플랫 층을 깐다 → 스플랫이 비는 곳은
       포토그래메트리 면이 보이고, 스플랫이 있는 곳은 스플랫이 앞에 보인다(겹쳐 보기를 파일 하나로)"""
    import trimesh
    from scipy.spatial import cKDTree
    from .splatclean import read_splat, write_splat
    from .survey import _tex_colors
    sp = out / "site_splat.ply"
    mesh_p = out / "site_model_zup.obj"
    if not sp.exists() or not mesh_p.exists():
        return None
    meshp = pick_mesh(work / "mvs")
    dense = work / "dense" / "fused.ply" if (work / "dense" / "fused.ply").exists() else work / "mvs" / "scene_dense.ply"
    tmp = out / "_best_tmp.ply"
    info = mesh_prune_and_agree(sp, meshp or mesh_p, work / "dense" / "sparse", dense if dense.exists() else None,
                                tmp, log=lambda *_: None)
    a, props, _ = read_splat(tmp)
    tmp.unlink(missing_ok=True)
    col = {q: i for i, q in enumerate(props)}
    tau = info["tau_m"]
    m = trimesh.load(mesh_p, process=False, force="mesh")
    surf, fi = trimesh.sample.sample_surface(m, 1_000_000, seed=1)
    d = cKDTree(surf).query(a[:, [col["x"], col["y"], col["z"]]].astype(np.float64), workers=-1)[0]
    # 2) 표면 위 반투명 → 불투명하게(로짓 공간: 0.35 미만을 0.35~0.75 로 끌어올림)
    op = 1 / (1 + np.exp(-a[:, col["opacity"]].astype(np.float64)))
    near = d < 1.5 * tau
    boost = near & (op > 0.05) & (op < 0.75)
    op2 = np.where(boost, np.clip(0.35 + 0.6 * op, 0, 0.9), op)
    a[:, col["opacity"]] = np.log(op2 / (1 - op2)).astype(np.float32)
    # 3) 메쉬 뒤 받침 층
    area = float(m.area)
    step = max(2 * tau / 3, np.sqrt(area / max_backing))
    n = int(min(max_backing, area / step ** 2))
    P, fidx = trimesh.sample.sample_surface_even(m, n, radius=step * 0.7, seed=2) if n < 400_000 else \
        trimesh.sample.sample_surface(m, n, seed=2)
    fidx = np.asarray(fidx)
    T = m.vertices[m.faces[fidx]]
    # 무게중심 좌표로 텍스처 색
    v0, v1, v2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0], P - T[:, 0]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
    bv, bw = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
    bary = np.clip(np.stack([1 - bv - bw, bv, bw], 1), 0, 1)
    rgb = _tex_colors(m, fidx, bary).astype(np.float32) / 255.0
    nrm = m.face_normals[fidx]
    # 법선을 사진 쪽으로 돌린 뒤 반대쪽(표면 뒤)으로 tau/2 밀어 둔다
    try:
        import pycolmap
        rec = pycolmap.Reconstruction(work / "dense" / "sparse")
        C = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])
        near_c = C[cKDTree(C).query(P, workers=-1)[1]]
        flip = ((near_c - P) * nrm).sum(1) < 0
        nrm[flip] *= -1
    except Exception:
        pass
    P = P - nrm * (0.5 * tau)
    b = np.zeros((len(P), len(props)), np.float32)
    b[:, col["x"]], b[:, col["y"]], b[:, col["z"]] = P.T
    for k in range(3):
        b[:, col[f"f_dc_{k}"]] = (rgb[:, k] - 0.5) / 0.28209479177
    b[:, col["opacity"]] = np.log(0.97 / 0.03)
    b[:, col["scale_0"]] = b[:, col["scale_1"]] = np.log(step * 0.75)
    b[:, col["scale_2"]] = np.log(max(step * 0.05, 1e-5))
    q = _quat_from_z(nrm)
    for k in range(4):
        b[:, col[f"rot_{k}"]] = q[:, k]
    for k in ("nx", "ny", "nz"):
        if k in col:
            b[:, col[k]] = 0
    dst = out / "site_best.ply"
    write_splat(dst, np.vstack([a, b]), props, comment="photo3d: best-of-both (splat + photogrammetry backing), Z-up")
    res = {"file": dst.name, "splats": int(len(a)), "backing": int(len(b)), "boosted": int(boost.sum()),
           "removed_hidden": info["hidden_removed"], "backing_step_m": round(float(step), 4), "tau_m": tau}
    log(f"  최적 합성: 스플랫 {len(a):,}개(숨은 것 {info['hidden_removed']:,}개 제거·반투명 {int(boost.sum()):,}개 불투명하게) + "
        f"포토그래메트리 면 받침 {len(b):,}개({step * 100:.1f}cm 간격) → {dst.name}")
    try:
        from .splatexport import write_spz
        write_spz(dst, out / "site_best.spz")
    except Exception:
        pass
    return res


def merge_meshes(pg_path: Path, gs_path: Path, dense_ply: Path | None, dst: Path, log=print) -> dict | None:
    """최적 합성 메쉬: 포토그래메트리 메쉬(측정점에 붙은 면)는 그대로 두고, 그것이 없는 곳(구멍·얇은 물체·형태가
    애매해 측정점이 빠진 곳)만 스플랫 메쉬로 채운다. 같은 곳에 둘 다 있으면 측정한 포토그래메트리가 우선.
    그 뒤 건물 정형화(평면·모서리)를 합친 메쉬에 다시 적용."""
    import trimesh
    from scipy.spatial import cKDTree
    pg = trimesh.load(pg_path, process=False, force="mesh")
    gs = trimesh.load(gs_path, process=False, force="mesh")
    pg.merge_vertices()
    gs.merge_vertices()
    if not len(pg.faces) or not len(gs.faces):
        return None
    surf = np.asarray(trimesh.sample.sample_surface(pg, 2_000_000, seed=0)[0])
    tree = cKDTree(surf)
    if dense_ply is not None and Path(dense_ply).exists():
        from .fill import read_points
        dp = np.asarray(read_points(dense_ply)[0], float)
        sub = dp[np.random.default_rng(0).choice(len(dp), min(len(dp), 20000), replace=False)]
        sp = float(np.median(cKDTree(dp).query(sub, k=2)[0][:, 1]))
    else:
        sp = float(np.median(tree.query(surf[:20000], k=2)[0][:, 1]))
    tau = 3 * sp
    d = tree.query(gs.triangles_center, workers=-1)[0]
    keep = d > 2 * tau  # 포토그래메트리 면이 없는 곳만
    add = gs.copy()
    add.update_faces(keep)
    add.remove_unreferenced_vertices()
    if len(add.faces):  # 몇 면짜리 부스러기는 버린다
        cc = trimesh.graph.connected_components(add.face_adjacency, nodes=np.arange(len(add.faces)))
        m = np.zeros(len(add.faces), bool)
        for c in cc:
            if len(c) >= 50:
                m[c] = True
        add.update_faces(m)
        add.remove_unreferenced_vertices()
    for g in (pg, add):  # 색: 정점 색 그대로(텍스처는 사진에서 다시 굽는다)
        if g.visual.kind != "vertex":
            g.visual = g.visual.to_color() if g.visual.kind else trimesh.visual.ColorVisuals(g)
    merged = trimesh.util.concatenate([pg, add]) if len(add.faces) else pg
    merged.export(dst)
    from .regularize import regularize_mesh
    out = regularize_mesh(dst, dense_ply, dst, log=log) if dense_ply is not None and Path(dense_ply).exists() else dst
    info = {"pg_faces": int(len(pg.faces)), "added_faces": int(len(add.faces)),
            "added_area_ratio": round(float(add.area / max(pg.area + add.area, 1e-12)), 3), "tau_m": round(tau, 4),
            "file": Path(out).name}
    log(f"  최적 합성 메쉬: 포토그래메트리 {len(pg.faces):,}면 + 스플랫 메쉬로 채운 곳 {len(add.faces):,}면"
        f"(표면의 {info['added_area_ratio']:.0%})")
    return info
