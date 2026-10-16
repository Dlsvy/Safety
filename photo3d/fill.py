"""빈 곳 메우기 — Poisson 재구성으로 닫힌 충돌 메쉬를 만들고, 측정되지 않은 '추정 영역'을 따로 표시한다.

사진측량은 사진에 찍힌 면만 만든다(지붕·윗면·가려진 곳은 빈다). 크레인 안전 판정에는 빈 곳이 없어야 하므로
  1) 조밀 점군(법선 포함) → Poisson 으로 닫힌 면 생성
  2) 촬영 범위 밖·지면 아래로 부푼 부분 제거
  3) 실제 측정점에서 est_dist 이상 떨어진 면 = 추정 영역 → 유니티에서 다른 색, 안전 판정 시 경고
추정 영역은 '측정값이 아니다'. 지붕처럼 중요한 곳은 높은 위치/드론 사진 추가 또는 도면 3D 와 결합으로 채워야 한다.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pycolmap
import trimesh
from scipy.spatial import cKDTree

_PLY_TYPES = {"float32": "f", "float": "f", "float64": "d", "double": "d", "uint8": "B", "uchar": "B",
              "int32": "i", "int": "i", "uint32": "I", "uint": "I", "int16": "h", "uint16": "H"}


def read_openmvs_ply(path: Path):
    """OpenMVS 조밀 점군(binary, view 목록 가변길이 포함)에서 xyz·rgb·법선만 읽는다."""
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property list"):
                _, _, ct, it, name = line.split()
                props.append(("list", name, _PLY_TYPES[ct], _PLY_TYPES[it]))
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append(("scalar", name, _PLY_TYPES[t]))
            elif line == "end_header":
                break
        data = f.read()
    names = [p[1] for p in props if p[0] == "scalar"]
    # 앞쪽 고정 길이 스칼라만 한 번에 읽고, 가변 리스트는 건너뛴다
    fixed = "<" + "".join(p[2] for p in props if p[0] == "scalar")
    fsize = struct.calcsize(fixed)
    lists = [p for p in props if p[0] == "list"]
    out = np.empty((n, len(names)), np.float64)
    off = 0
    unpack = struct.Struct(fixed).unpack_from
    for i in range(n):
        out[i] = unpack(data, off)
        off += fsize
        for _, _, ct, it in lists:
            cnt = struct.unpack_from("<" + ct, data, off)[0]
            off += struct.calcsize(ct) + cnt * struct.calcsize(it)
    col = {k: j for j, k in enumerate(names)}
    xyz = out[:, [col["x"], col["y"], col["z"]]]
    nrm = out[:, [col["nx"], col["ny"], col["nz"]]] if "nx" in col else None
    rgb = out[:, [col["red"], col["green"], col["blue"]]].astype(np.uint8) if "red" in col else None
    return xyz, nrm, rgb


def _write_ply(path: Path, xyz, nrm, rgb):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property float nx\nproperty float ny\nproperty float nz\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.zeros(len(xyz), dtype=[("p", "<f4", 3), ("n", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["n"], rec["c"] = xyz, nrm, rgb if rgb is not None else 160
        f.write(rec.tobytes())


def read_points(path: Path):
    """조밀 점군(xyz·법선·색). COLMAP fused.ply 는 trimesh 로, OpenMVS(가변 리스트 포함)는 직접 읽는다."""
    with open(path, "rb") as f:
        head = f.read(2048)
    if b"property list" not in head.split(b"end_header")[0]:
        pc = trimesh.load(path, process=False)
        xyz = np.asarray(pc.vertices, float)
        el = pc.metadata.get("_ply_raw", {}).get("vertex", {}).get("data")
        nrm = np.stack([el[k] for k in ("nx", "ny", "nz")], 1).astype(float) \
            if el is not None and "nx" in el.dtype.names else None
        cols = getattr(pc, "colors", None)
        rgb = np.asarray(cols)[:, :3].astype(np.uint8) if cols is not None and len(cols) else None
        return xyz, nrm, rgb
    return read_openmvs_ply(path)


def _paint_unmeasured(m: trimesh.Trimesh, c: np.ndarray, measured: np.ndarray, pool: int = 48,
                      seed: int = 0) -> np.ndarray:
    """메운 곳을 주변의 '실제 색'으로 칠한다(평균·혼합·늘어짐 없음).

    표면을 따라(측지 거리) 가장 가까운 측정 정점을 찾고, 그 주변 측정 정점 pool 개 중 하나의 색을 그대로 복사한다.
    → 색은 섞이지 않고(실제 관측색만 사용), 경계색 하나가 길게 늘어나지도 않으며, 벽은 벽색·바닥은 바닥색을 따른다.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    est = ~measured
    if not est.any() or not measured.any():
        return c
    e = m.edges_unique
    L = m.edges_unique_length
    n = len(m.vertices)
    g = coo_matrix((np.r_[L, L], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    src_ids = np.where(measured)[0]
    # 경계 근처 측정 정점만 출발점으로(전부 넣으면 느림) — 미측정 정점과 한 변으로 이어진 측정 정점
    nb = est[e[:, 0]] != est[e[:, 1]]
    border = np.unique(np.where(measured[e[nb, 0]], e[nb, 0], e[nb, 1]))
    if len(border) == 0:
        border = src_ids
    _, _, src = dijkstra(g, indices=border, min_only=True, return_predecessors=True)
    out = c.copy()
    ev = np.where(est & (src >= 0))[0]
    if len(ev) == 0:
        return out
    mt = cKDTree(m.vertices[src_ids])
    _, cand = mt.query(m.vertices[src[ev]], k=min(pool, len(src_ids)), workers=-1)
    pick = np.random.default_rng(seed).integers(0, cand.shape[1], len(ev))
    out[ev] = c[src_ids[cand[np.arange(len(ev)), pick]]]
    return out


def fill_holes(dense_ply: Path, out: Path, cam_centers: np.ndarray | None, depth: int = 10,
               est_dist: float | None = None, max_gap: float | None = None, smooth: int = 8,
               max_faces: int = 1_200_000, log=print) -> dict:
    """주변 형상을 따라 구멍을 메운 '연속 메쉬'를 만든다.

    - Poisson(trim 없음)은 점군 전체를 매끈한 하나의 면으로 이어 붙이므로 구멍이 주변 곡면을 따라 닫힌다.
    - 측정점에서 max_gap 보다 먼 면(데이터가 전혀 없는 허공의 부풀림)은 지운다 → 근거 있는 구멍만 메움.
    - 메운 면에도 가까운 측정점들의 색을 거리 가중으로 입혀 사진처럼 보이게 한다.
    - 측정점에서 est_dist 보다 먼 면 = '추정 영역'(측정 아님) → 따로 표시·내보내기.
    """
    xyz, nrm, rgb = read_points(dense_ply)
    if nrm is None:
        raise RuntimeError("점군에 법선이 없어 메우기를 할 수 없습니다")
    diag = float(np.linalg.norm(np.percentile(xyz, 98, 0) - np.percentile(xyz, 2, 0)))
    tree = cKDTree(xyz)
    spacing = float(np.median(tree.query(xyz[:: max(1, len(xyz) // 20000)], k=2)[0][:, 1]))
    est_dist = est_dist or max(8 * spacing, 0.005 * diag)
    max_gap = max_gap or 0.05 * diag
    log(f"  조밀 점군 {len(xyz):,}점 → Poisson(depth={depth}) · 메울 최대 간격 {max_gap:.2f} · 추정 판정 {est_dist:.3f}")
    clean = out / "_poisson_in.ply"
    _write_ply(clean, xyz, nrm, rgb)
    raw = out / "_poisson_raw.ply"
    opt = pycolmap.PoissonMeshingOptions()
    opt.depth = depth
    opt.trim = 0  # 잘라내지 않음 = 닫힌 면(구멍 메움)
    pycolmap.poisson_meshing(clean, raw, opt)
    m = trimesh.load(raw, process=False, force="mesh")
    clean.unlink(missing_ok=True)
    raw.unlink(missing_ok=True)
    d, _ = tree.query(m.triangles_center, workers=-1)
    keep = d <= max_gap
    if cam_centers is not None and len(cam_centers) >= 3:  # 축척 확정 시: 지면 아래 부풀림 제거
        keep &= m.triangles_center[:, 2] >= np.percentile(xyz[:, 2], 2) - 0.3
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    # 측정 영역과 이어지지 않은 떠다니는 조각 제거
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        big = max(p.area for p in parts)
        m = trimesh.util.concatenate([p for p in parts if p.area >= 0.02 * big])
    if smooth:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=smooth)
    if len(m.faces) > max_faces:
        from .export import decimate
        m = decimate(m, max_faces)
        # 단순화가 가끔 만드는 '바늘'(길고 아주 가는 삼각형)만 제거 — 평평한 곳의 큰 삼각형은 둔다
        el = np.linalg.norm(m.vertices[m.faces] - m.vertices[np.roll(m.faces, 1, 1)], axis=2).max(1)
        thin = el ** 2 / np.maximum(2 * m.area_faces, 1e-12)  # 최장변 / 그 변에 대한 높이
        m.update_faces(~((thin > 25) & (el > 4 * np.median(el))))
        m.remove_unreferenced_vertices()
    dv, idx = tree.query(m.vertices, k=4, workers=-1)
    measured = dv[:, 0] <= est_dist
    if rgb is not None:
        # 측정된 곳: 가까운 4점만 좁게 섞어 선명하게
        w = 1.0 / np.maximum(dv, 1e-9)
        w[dv > 2 * dv[:, :1] + 1e-9] = 0
        c = (rgb[idx].astype(np.float32) * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        c = _paint_unmeasured(m, c, measured)
    else:
        c = np.full((len(m.vertices), 3), 170.0)
    m.visual = trimesh.visual.ColorVisuals(
        m, vertex_colors=np.concatenate([np.clip(c, 0, 255), np.full((len(c), 1), 255)], 1).astype(np.uint8))
    est = dv[:, 0][m.faces].min(1) > est_dist  # 세 꼭짓점 모두 측정점에서 먼 면
    area = m.area_faces
    frac = float(area[est].sum() / max(area.sum(), 1e-9))
    log(f"  메운 메쉬 {len(m.faces):,}면, 추정 영역 {frac:.1%}")
    return {"mesh": m, "estimated_mask": est, "estimated_area_ratio": round(frac, 4)}


def run_stage(dense_ply: Path, out: Path, work: Path, cams_npy: Path | None, bake_tex: bool, log=print) -> dict:
    """구멍 메운 모델 단계 전체(메우기 → 사진 텍스처 → GLB). 메인 메쉬와 동시에 별도 프로세스로 돈다."""
    import json
    from . import export
    cams = np.load(cams_npy) if cams_npy and Path(cams_npy).exists() else None
    fr = fill_holes(dense_ply, out, cams, log=log)
    m, est = fr["mesh"], fr["estimated_mask"]
    est_m = m.submesh([np.where(est)[0]], append=True) if est.any() else None
    if bake_tex and (work / "dense" / "sparse").exists() and (work / "dense" / "images").exists():
        try:  # 측정된 면은 사진 텍스처, 메운 면은 주변 실제 색 유지
            from .texture import bake
            m, _ = bake(m, work / "dense" / "sparse", work / "dense" / "images", atlas=8192,
                        max_faces=len(m.faces) + 1, vertex_only=est.copy(), log=log)
        except Exception as e:
            log(f"  메운 모델 사진 텍스처 생략: {e}")
    m.apply_transform(export.ZUP_TO_YUP)
    m.export(out / "site_filled.glb")  # 보기용: 구멍 없이 이어진 전체
    if est_m is not None:
        est_m.apply_transform(export.ZUP_TO_YUP)
        est_m.export(out / "site_estimated.glb")
    res = {"estimated_area_ratio": fr["estimated_area_ratio"], "filled": "site_filled.glb",
           "file": "site_estimated.glb" if est_m is not None else None,
           "note": "추정 영역은 측정값이 아님 — 높은 위치/드론 사진 추가 또는 도면 3D 결합으로 대체 권장"}
    (out / "fill_result.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res
