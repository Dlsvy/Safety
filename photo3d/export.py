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


_PT = {"float32": "<f4", "float": "<f4", "float64": "<f8", "double": "<f8", "uint8": "u1", "uchar": "u1",
       "int8": "i1", "char": "i1", "int16": "<i2", "short": "<i2", "uint16": "<u2", "ushort": "<u2",
       "int32": "<i4", "int": "<i4", "uint32": "<u4", "uint": "<u4"}


def load_openmvs_textured_ply(path: Path) -> trimesh.Trimesh | None:
    """OpenMVS TextureMesh 의 PLY(면마다 texcoord 리스트, v2.4 는 texnumber 추가)를 직접 읽는다.
    trimesh 기본 PLY 로더가 v2.4 형식의 UV 를 잘못 읽어 텍스처가 검게 깨지는 문제(PC 확인) 대응.
    해당 형식이 아니면 None."""
    from PIL import Image
    with open(path, "rb") as f:
        elems, cur, tex = [], None, []
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("format") and "binary_little_endian" not in line:
                return None
            if line.startswith("comment TextureFile"):
                tex.append(line.split(None, 2)[2])
            elif line.startswith("element"):
                _, name, n = line.split()
                cur = {"name": name, "n": int(n), "props": []}
                elems.append(cur)
            elif line.startswith("property list"):
                _, _, ct, it, name = line.split()
                cur["props"].append((name, "list", _PT[ct], _PT[it]))
            elif line.startswith("property"):
                _, t, name = line.split()
                cur["props"].append((name, "scalar", _PT[t]))
            elif line == "end_header":
                break
        data = f.read()
    if not tex:
        return None
    off, out = 0, {}
    for e in elems:
        if e["name"] == "vertex":
            dt = np.dtype([(p[0], p[2]) for p in e["props"]])
            arr = np.frombuffer(data, dt, e["n"], off)
            off += dt.itemsize * e["n"]
            out["v"] = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
        elif e["name"] == "face":
            # 리스트 길이가 일정하다고 보고(삼각형 3, texcoord 6) 고정 레코드로 읽은 뒤 확인
            fields = []
            for p in e["props"]:
                if p[1] == "list":
                    k = 3 if p[0] == "vertex_indices" else 6
                    fields += [(p[0] + "_n", p[2]), (p[0], p[3], (k,))]
                else:
                    fields.append((p[0], p[2]))
            dt = np.dtype(fields)
            arr = np.frombuffer(data, dt, e["n"], off)
            off += dt.itemsize * e["n"]
            if not (arr["vertex_indices_n"] == 3).all() or not (arr["texcoord_n"] == 6).all():
                return None
            out["f"] = arr["vertex_indices"].astype(np.int64)
            out["uv"] = arr["texcoord"].reshape(-1, 3, 2).astype(np.float64)
            out["tn"] = arr["texnumber"].astype(np.int64) if "texnumber" in arr.dtype.names else None
    imgs = [Image.open(path.parent / t).convert("RGB") for t in tex]
    uv = out["uv"]
    if len(imgs) > 1:  # 여러 장이면 가로로 이어 붙이고 u 를 다시 계산
        h = max(i.height for i in imgs)
        W = sum(i.width for i in imgs)
        atlas = Image.new("RGB", (W, h))
        x0s = np.cumsum([0] + [i.width for i in imgs[:-1]])
        for i, x0 in zip(imgs, x0s):
            atlas.paste(i, (int(x0), h - i.height))
        tn = out["tn"] if out["tn"] is not None else np.zeros(len(uv), np.int64)
        ws = np.array([i.width for i in imgs])[tn]
        hs = np.array([i.height for i in imgs])[tn]
        uv = uv.copy()
        uv[:, :, 0] = (x0s[tn][:, None] + uv[:, :, 0] * ws[:, None]) / W
        uv[:, :, 1] = uv[:, :, 1] * hs[:, None] / h
        img = atlas
    else:
        img = imgs[0]
    V = out["v"][out["f"]].reshape(-1, 3)  # 면 모서리마다 정점을 따로(UV 경계 보존)
    F = np.arange(len(V)).reshape(-1, 3)
    vis = trimesh.visual.TextureVisuals(uv=uv.reshape(-1, 2), image=img)
    return trimesh.Trimesh(V, F, visual=vis, process=False)


def _load(obj: Path) -> trimesh.Trimesh:
    if obj.suffix.lower() == ".ply":
        try:
            m = load_openmvs_textured_ply(obj)
            if m is not None:
                return m
        except Exception as e:  # 형식이 다르면 일반 로더로
            print(f"  (텍스처 PLY 직접 읽기 실패 → 일반 로더: {e})")
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
               meta: dict | None = None, extra: list | None = None, extra_fn=None, log=print) -> dict:
    """extra: 함께 내보낼 추가 메쉬(Z-up, 예: 사진 바닥 조각) — GLB·OBJ 에 별도 물체로, 충돌 메쉬에는 합쳐서."""
    out.mkdir(parents=True, exist_ok=True)
    mesh = _load(textured_obj)
    mesh = clean_mesh(mesh, cam_centers, log=log)
    files = {}
    # 라이노/CAD 확인용 (Z-up, m)
    if extra_fn is not None:  # 정리된 메쉬를 보고 추가 메쉬를 만든다(예: 측정이 빈 바닥만 사진 바닥으로)
        try:
            extra = list(extra or []) + list(extra_fn(mesh) or [])
        except Exception as e:
            log(f"  추가 메쉬 생략: {e}")
    extra = [e for e in (extra or []) if e is not None and len(e.faces)]
    p = out / "site_model_zup.obj"
    if extra:
        trimesh.Scene([mesh] + extra).export(p)
    else:
        mesh.export(p)
    files["obj_zup"] = p.name
    # 유니티 보기용 GLB (Y-up)
    vis = mesh
    if len(mesh.faces) > visual_faces:
        log(f"  주의: 면 {len(mesh.faces):,}개 — 유니티 보기용으로 무거울 수 있음")
    g = vis.copy()
    g.apply_transform(ZUP_TO_YUP)
    p = out / "site_model.glb"
    if extra:
        sc = trimesh.Scene([g])
        for i, e in enumerate(extra):
            e2 = e.copy()
            e2.apply_transform(ZUP_TO_YUP)
            sc.add_geometry(e2, geom_name=f"photo_floor_{i}")
        sc.export(p)
    else:
        g.export(p)
    files["glb"] = p.name
    # 충돌 판정용 (텍스처 없음, Y-up)
    geo = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    if extra:
        geo = trimesh.util.concatenate([geo] + [trimesh.Trimesh(e.vertices, e.faces, process=False) for e in extra])
    col = decimate(geo, collider_faces)
    col.apply_transform(ZUP_TO_YUP)
    p = out / "site_collider.glb"
    col.export(p)
    files["collider_glb"] = p.name
    p = out / "site_collider.obj"
    col.export(p)
    files["collider_obj"] = p.name
    b = mesh.bounds if not extra else np.array([np.minimum(mesh.bounds[0], np.min([e.bounds[0] for e in extra], 0)),
                                                 np.maximum(mesh.bounds[1], np.max([e.bounds[1] for e in extra], 0))])
    info = {"faces": int(len(mesh.faces)), "collider_faces": int(len(col.faces)),
            "bounds_zup_m": b.round(3).tolist(), "size_m": (b[1] - b[0]).round(3).tolist(),
            "files": files, "extra_parts": len(extra)}
    info["_extra"] = extra
    with open(out / "site_meta.json", "w", encoding="utf-8") as f:
        json.dump({**(meta or {}), "model": info}, f, ensure_ascii=False, indent=2, default=str)
    log(f"  내보내기: {out} (크기 {info['size_m']} m)")
    return info
