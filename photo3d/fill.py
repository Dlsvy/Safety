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


def fill_holes(dense_ply: Path, out: Path, cam_centers: np.ndarray | None, depth: int = 11,
               est_dist: float = 0.10, crop_margin: float = 1.5, log=print) -> dict:
    xyz, nrm, rgb = read_openmvs_ply(dense_ply)
    log(f"  조밀 점군 {len(xyz):,}점 → Poisson(depth={depth})")
    clean = out / "_poisson_in.ply"
    _write_ply(clean, xyz, nrm, rgb)
    raw = out / "_poisson_raw.ply"
    opt = pycolmap.PoissonMeshingOptions()
    opt.depth = depth
    opt.trim = 0  # 잘라내지 않음 = 닫힌 면(구멍 메움)
    pycolmap.poisson_meshing(clean, raw, opt)
    m = trimesh.load(raw, process=False)
    # 촬영 범위 밖 / 지면 아래 부풀림 제거
    keep = np.ones(len(m.faces), bool)
    fc = m.triangles_center
    if cam_centers is not None and len(cam_centers) >= 3:
        c = cam_centers.mean(0)
        r = np.linalg.norm((cam_centers - c)[:, :2], axis=1).max() * crop_margin + 2.0
        keep &= np.linalg.norm((fc - c)[:, :2], axis=1) <= r
    ground = np.percentile(xyz[:, 2], 2)
    keep &= fc[:, 2] >= ground - 0.3
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    # 측정점과의 거리로 추정 영역 판정
    d, _ = cKDTree(xyz).query(m.triangles_center, workers=-1)
    est = d > est_dist
    area = m.area_faces
    frac = float(area[est].sum() / max(area.sum(), 1e-9))
    colors = np.tile(np.array([170, 170, 170, 255], np.uint8), (len(m.faces), 1))
    colors[est] = [255, 120, 0, 255]  # 주황 = 추정(측정 아님)
    m.visual = trimesh.visual.ColorVisuals(m, face_colors=colors)
    clean.unlink(missing_ok=True)
    raw.unlink(missing_ok=True)
    log(f"  닫힌 메쉬 {len(m.faces):,}면, 추정 영역 {frac:.1%} (측정점에서 {est_dist*100:.0f}cm 초과)")
    return {"mesh": m, "estimated_mask": est, "estimated_area_ratio": round(frac, 4)}
