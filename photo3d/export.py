"""정리·내보내기: 텍스처 메쉬 → 유니티용 GLB(보기용) + 충돌용 경량 메쉬.

- 작은 떠다니는 조각(노이즈) 제거, 촬영 범위 밖 먼 배경 잘라내기
- 좌표: 내부는 Z-up(m). glTF 규격은 Y-up 이므로 GLB 는 Y-up 으로 회전해서 저장
  (유니티 glTFast 가 오른손→왼손 변환을 처리 → 유니티에서 1 unit = 1 m, 위쪽 = +Y)
- OBJ(Z-up)는 라이노/캐드 확인용으로 함께 남긴다
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

ZUP_TO_YUP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)


def _load(obj: Path) -> trimesh.Trimesh:
    m = trimesh.load(obj, process=False, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


def clean_mesh(mesh: trimesh.Trimesh, cam_centers: np.ndarray | None,
               crop_margin: float = 1.5, min_component_ratio: float = 0.01, log=print):
    n0 = len(mesh.faces)
    keep = np.ones(n0, bool)
    if cam_centers is not None and len(cam_centers) >= 3:
        # 촬영 궤적의 수평 범위 × margin 바깥(하늘·먼 배경 노이즈)을 잘라낸다
        c = cam_centers.mean(0)
        r = np.linalg.norm((cam_centers - c)[:, :2], axis=1).max() * crop_margin + 2.0
        fc = mesh.triangles_center
        keep &= np.linalg.norm((fc - c)[:, :2], axis=1) <= r
        zmin = cam_centers[:, 2].min() - 50
        keep &= fc[:, 2] >= zmin
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    # 텍스처 메쉬는 UV 경계에서 정점이 복제돼 있어, 그대로 연결성을 보면 텍스처 조각마다 분리된다.
    # 같은 위치 정점을 합친 형상으로 연결 덩어리를 판정한다(면 순서 유지).
    _, inv = np.unique(np.round(mesh.vertices, 5), axis=0, return_inverse=True)
    geo = trimesh.Trimesh(np.zeros((inv.max() + 1, 3)), inv.reshape(-1)[mesh.faces], process=False)
    comps = trimesh.graph.connected_components(geo.face_adjacency, nodes=np.arange(len(mesh.faces)))
    sizes = np.array([len(cc) for cc in comps])
    if len(comps) > 1:
        thr = max(50, min_component_ratio * sizes.max())
        mask = np.zeros(len(mesh.faces), bool)
        for cc in comps:
            if len(cc) >= thr:
                mask[cc] = True
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()
    log(f"  메쉬 정리: {n0:,} → {len(mesh.faces):,} 면")
    return mesh


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=True)
    import fast_simplification
    ratio = 1 - target_faces / len(mesh.faces)
    v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32),
                                        mesh.faces.astype(np.int32), target_reduction=ratio)
    return trimesh.Trimesh(v, f, process=True)


def export_all(textured_obj: Path, out: Path, cam_centers: np.ndarray | None,
               visual_faces: int = 1_500_000, collider_faces: int = 150_000,
               meta: dict | None = None, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mesh = _load(textured_obj)
    mesh = clean_mesh(mesh, cam_centers, log=log)
    files = {}
    # 라이노/CAD 확인용 (Z-up, m)
    p = out / "site_model_zup.obj"
    mesh.export(p)
    files["obj_zup"] = p.name
    # 유니티 보기용 GLB (Y-up)
    vis = mesh
    if len(mesh.faces) > visual_faces:
        log(f"  주의: 면 {len(mesh.faces):,}개 — 유니티 보기용으로 무거울 수 있음")
    g = vis.copy()
    g.apply_transform(ZUP_TO_YUP)
    p = out / "site_model.glb"
    g.export(p)
    files["glb"] = p.name
    # 충돌 판정용 (텍스처 없음, Y-up)
    col = decimate(mesh, collider_faces)
    col.apply_transform(ZUP_TO_YUP)
    p = out / "site_collider.glb"
    col.export(p)
    files["collider_glb"] = p.name
    p = out / "site_collider.obj"
    col.export(p)
    files["collider_obj"] = p.name
    b = mesh.bounds
    info = {"faces": int(len(mesh.faces)), "collider_faces": int(len(col.faces)),
            "bounds_zup_m": b.round(3).tolist(), "size_m": (b[1] - b[0]).round(3).tolist(),
            "files": files}
    with open(out / "site_meta.json", "w", encoding="utf-8") as f:
        json.dump({**(meta or {}), "model": info}, f, ensure_ascii=False, indent=2, default=str)
    log(f"  내보내기: {out} (크기 {info['size_m']} m)")
    return info
