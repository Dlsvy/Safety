"""조밀복원: 정합된 SfM → OpenMVS (조밀 점군 → 메쉬 → 정제 → 텍스처).

OpenMVS 는 CPU 로도 돌고, CUDA 빌드가 있으면 DensifyPointCloud 가 GPU 를 쓴다.
OPENMVS_BIN 환경변수로 바이너리 폴더를 지정할 수 있다(윈도우 릴리스 zip 포함).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pycolmap

QUALITY = {
    # resolution-level: 0=원본, 1=1/2, 2=1/4 해상도에서 깊이맵 계산
    "draft":  dict(res=2, views=5, refine=False, decimate=0.5),
    "medium": dict(res=1, views=8, refine=False, decimate=1.0),
    "high":   dict(res=1, views=12, refine=True, decimate=1.0),
    "ultra":  dict(res=0, views=16, refine=True, decimate=1.0),
}


def _bin(name: str) -> str:
    cands = []
    if os.environ.get("OPENMVS_BIN"):
        cands.append(Path(os.environ["OPENMVS_BIN"]))
    cands += [Path("/opt/openmvs/bin/OpenMVS"), Path("/usr/local/bin/OpenMVS"),
              Path("C:/OpenMVS"), Path(sys.prefix) / "OpenMVS"]
    for d in cands:
        for ext in ("", ".exe"):
            p = d / f"{name}{ext}"
            if p.exists():
                return str(p)
    p = shutil.which(name)
    if p:
        return p
    raise FileNotFoundError(f"OpenMVS '{name}' 를 찾을 수 없습니다. OPENMVS_BIN 을 설정하세요.")


def _run(args: list[str], cwd: Path, log, logfile: Path):
    log("  $ " + " ".join(Path(args[0]).name if i == 0 else a for i, a in enumerate(args)))
    with open(logfile, "a") as f:
        f.write("\n$ " + " ".join(args) + "\n")
        f.flush()
        r = subprocess.run(args, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(args[0]).name} 실패 (로그: {logfile})")


def _color_from_points(mesh_path: Path, points_path: Path, mvs: Path, log, smooth: int = 8) -> Path:
    """메쉬 정점마다 주변 조밀 점들의 색을 입힌다(텍스처 실패 시 대체).

    정점 색은 정점 간격보다 세밀할 수 없어 각져 보인다 → ① Taubin 스무딩(부피·치수 보존형)으로
    Poisson 표면의 울퉁불퉁함을 줄이고 ② 한 번 세분해 색 해상도를 4배로 ③ 가장 가까운 1점 대신
    가까운 4점만 좁게 거리 가중해 얼룩은 줄이되 선명함은 유지한다.
    """
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    try:
        m = trimesh.load(mesh_path, process=False, force="mesh")
        pc = trimesh.load(points_path, process=False)
        cols = getattr(pc, "colors", None)
        if cols is None or len(cols) == 0:
            cols = pc.visual.vertex_colors
        cols = np.asarray(cols)[:, :3].astype(np.float32)
        if smooth:
            m.merge_vertices()
            trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=smooth)
        if len(m.faces) < 3_000_000:  # 메모리 한도 내에서 색 해상도 ×4
            v, f = trimesh.remesh.subdivide(m.vertices, m.faces)
            m = trimesh.Trimesh(v, f, process=False)
        d, idx = cKDTree(np.asarray(pc.vertices)).query(m.vertices, k=4, workers=-1)
        w = 1.0 / np.maximum(d, 1e-9)
        w[d > 2 * d[:, :1] + 1e-9] = 0  # 너무 먼 점(다른 면)은 제외
        c = (cols[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        rgba = np.concatenate([np.clip(c, 0, 255), np.full((len(c), 1), 255)], 1).astype(np.uint8)
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=rgba)
        out = mvs / "scene_mesh_colored.ply"
        m.export(out)
        log(f"  점군 색 입힘(스무딩 {smooth}회·세분·4점): {len(m.vertices):,}정점")
        return out
    except Exception as e:
        log(f"  점군 색 입히기 실패({e}) → 색 없는 메쉬 사용")
        return mesh_path


def prune_mesh(mesh_path: Path, points_path: Path, dst: Path, log=print) -> Path:
    """Poisson 메쉬에서 측정점과 동떨어진 면과 바늘처럼 긴 삼각형(가시)을 지운다."""
    import trimesh
    from scipy.spatial import cKDTree
    try:
        m = trimesh.load(mesh_path, process=False, force="mesh")
        pts = np.asarray(trimesh.load(points_path, process=False).vertices, float)
        tree = cKDTree(pts)
        sp = float(np.median(tree.query(pts[:: max(1, len(pts) // 20000)], k=2)[0][:, 1]))
        d, _ = tree.query(m.triangles_center, workers=-1)
        el = np.linalg.norm(m.vertices[m.faces] - m.vertices[np.roll(m.faces, 1, 1)], axis=2).max(1)
        thin = el ** 2 / np.maximum(2 * m.area_faces, 1e-12)
        far = d > 20 * sp  # 가시는 수십~수백 배 떨어져 있다(6배는 점이 성긴 실제 표면까지 지움)
        needle = (thin > 12) & (el > 8 * np.median(el))
        n0 = len(m.faces)
        m.update_faces(~(far | needle))
        m.remove_unreferenced_vertices()
        m.export(dst)
        log(f"  메쉬 가시 정리: {n0:,} → {len(m.faces):,}면 (측정점과 먼 면 {int(far.sum()):,}, 바늘 {int(needle.sum()):,})")
        return dst
    except Exception as e:
        log(f"  메쉬 가시 정리 생략({e})")
        return mesh_path


def add_prior_points(fused: Path, dense: Path, log=print) -> int:
    """조밀 점군(fused.ply)에 깊이 prior 로 채운 점을 더해 같은 파일로 저장. 실패해도 원래 점군 유지."""
    try:
        from . import priorfill
        from .fill import _write_ply, read_points
        if not priorfill.ensure_installed(log):
            log("  깊이 prior 설치 실패 → 건너뜀")
            return 0
        xyz, nrm, rgb = read_points(fused)
        pr = priorfill.fill(xyz, rgb, dense / "sparse", dense / "images", log=log)
        n = len(pr["xyz"])
        if n:
            if nrm is None:
                nrm = np.zeros_like(xyz)
            _write_ply(fused, np.vstack([xyz, pr["xyz"]]), np.vstack([nrm, pr["normal"]]),
                       np.vstack([rgb if rgb is not None else np.full((len(xyz), 3), 160, np.uint8), pr["rgb"]]))
            from .pointfilter import LAST
            LAST["prior_added"] = int(n)
        return n
    except Exception as e:
        log(f"  깊이 prior 생략: {e}")
        return 0


def densify_and_mesh(rec: pycolmap.Reconstruction, image_dir: Path, work: Path,
                     quality: str = "medium", threads: int = 0, engine: str = "openmvs",
                     colmap_bin: str | None = None, on_dense_ready=None, prior: bool = True,
                     log=print) -> dict:
    q = QUALITY[quality]
    sparse = work / "sparse_georef"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    rec.write_text(sparse)  # OpenMVS InterfaceCOLMAP 은 텍스트 모델을 확실히 읽는다

    dense = work / "dense"
    if dense.exists():
        shutil.rmtree(dense)
    log("  왜곡보정 이미지 생성")
    pycolmap.undistort_images(dense, sparse, image_dir)
    # undistort 결과 sparse 를 텍스트로도 기록
    und = pycolmap.Reconstruction(dense / "sparse")
    und.write_text(dense / "sparse")

    mvs = work / "mvs"
    mvs.mkdir(exist_ok=True)
    logf = work / "openmvs.log"
    common = ["-w", str(mvs), "-v", "2"] + (["--max-threads", str(threads)] if threads else [])
    if engine != "colmap":
        _run([_bin("InterfaceCOLMAP"), "-i", str(dense), "-o", str(mvs / "scene.mvs"),
              "--image-folder", str(dense / "images")] + common, mvs, log, logf)
    if engine == "colmap":
        # GPU 조밀복원: COLMAP PatchMatchStereo(CUDA) → StereoFusion → OpenMVS 로 가져와 메쉬·텍스처
        if not colmap_bin:
            raise RuntimeError("--dense colmap 은 CUDA 지원 COLMAP 실행파일(--colmap-bin)이 필요합니다")
        # VRAM 8GB 기준: 원본(-1)은 12MP 이상에서 메모리 부족 위험 → ultra 도 3200px
        size = {"draft": 1600, "medium": 2400, "high": 3200, "ultra": 3200}[quality]
        fast = quality == "draft"
        # 형상 일관성(geom_consistency)은 빠른 모드에서도 켠다: 끄면 하늘·그늘·비스듬한 지면의 틀린 깊이가
        # 걸러지지 않아 카메라 쪽으로 뻗는 '가시'가 생긴다(PC castle-P19). 빠른 모드는 반복만 줄인다.
        _run([colmap_bin, "patch_match_stereo", "--workspace_path", str(dense),
              "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "1",
              "--PatchMatchStereo.max_image_size", str(size), "--PatchMatchStereo.gpu_index", "0",
              "--PatchMatchStereo.filter_min_ncc", "0.15"]
             + (["--PatchMatchStereo.num_iterations", "4", "--PatchMatchStereo.window_step", "2"] if fast else []),
             mvs, log, logf)
        fused = dense / "fused_raw.ply"
        _run([colmap_bin, "stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
              "--input_type", "geometric", "--StereoFusion.max_image_size", str(size),
              "--StereoFusion.min_num_pixels", "4", "--StereoFusion.max_reproj_error", "1.5",
              "--StereoFusion.max_depth_error", "0.008", "--StereoFusion.max_normal_error", "8",
              "--output_path", str(fused)], mvs, log, logf)
        from .pointfilter import clean_ply
        cams = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])
        fused = clean_ply(fused, dense / "fused.ply", cams, log=log)
        if fused.name != "fused.ply":
            shutil.copy2(fused, dense / "fused.ply")
            fused = dense / "fused.ply"
        if prior:  # 무늬 없는 벽·바닥 채우기(MoGe-2 깊이 prior, 측정과 맞춘 뒤 다시점 일치만)
            add_prior_points(fused, dense, log)
        # (OpenMVS 장면 변환은 이 경로에서 쓰지 않음 — 시간 절약)
        shutil.copy2(fused, mvs / "scene_dense.ply")
        if on_dense_ready:  # 점군 완성 → 메운 모델(CPU 별도 프로세스)·스플랫(GPU)을 지금 동시에 시작
            on_dense_ready(dense, mvs / "scene_dense.ply")
    else:
        _run([_bin("DensifyPointCloud"), str(mvs / "scene.mvs"),
              "--resolution-level", str(q["res"]), "--number-views", str(q["views"]),
              "--number-views-fuse", "3", "-o", str(mvs / "scene_dense.mvs")] + common, mvs, log, logf)
    # v2.3: ReconstructMesh/RefineMesh 는 메쉬(.ply)만 쓴다 → 다음 단계에 --mesh-file 로 넘긴다
    dense_scene = mvs / "scene_dense.mvs"
    if engine == "colmap":
        # Windows OpenMVS v2.4.0 ReconstructMesh 가 이 점군에서 GPU판·CPU판 모두 종료(PC 3회)
        # → 메쉬는 COLMAP Poisson(정점 색 포함)으로 만든다
        mesh = mvs / "scene_mesh_poisson.ply"
        _run([colmap_bin, "poisson_mesher", "--input_path", str(dense / "fused.ply"),
              "--output_path", str(mesh), "--PoissonMeshing.depth", "10" if quality == "draft" else "11",
              "--PoissonMeshing.trim", "7"], mvs, log, logf)  # color 는 3.9=숫자, 4.x=on/off 라 지정 안 함
        mesh = prune_mesh(mesh, dense / "fused.ply", mvs / "scene_mesh_pruned.ply", log=log)
    else:
        _run([_bin("ReconstructMesh"), str(dense_scene),
              "-o", str(mvs / "scene_mesh.mvs")] + common, mvs, log, logf)
        mesh = mvs / "scene_mesh.ply"
    if q["refine"] and engine != "colmap":
        try:
            _run([_bin("RefineMesh"), str(dense_scene), "--mesh-file", str(mesh),
                  "--resolution-level", str(q["res"] + 1), "--max-face-area", "16",
                  "-o", str(mvs / "scene_refined.mvs")] + common, mvs, log, logf)
            if (mvs / "scene_refined.ply").exists():
                mesh = mvs / "scene_refined.ply"
        except RuntimeError as e:
            log(f"  메쉬 정제 건너뜀: {e}")
    if engine == "colmap":
        # Windows OpenMVS TextureMesh 는 검은 텍스처(PC 3회) → 직접 구현한 사진 텍스처 굽기 사용
        colored = _color_from_points(mesh, dense / "fused.ply", mvs, log)
        tex_info = None
        try:
            import trimesh
            from .texture import bake
            atlas = {"draft": 8192, "medium": 8192, "high": 8192, "ultra": 8192}[quality]
            maxf = {"draft": 600_000, "medium": 900_000, "high": 1_200_000, "ultra": 1_500_000}[quality]
            tm, _ = bake(trimesh.load(colored, process=False), dense / "sparse", dense / "images",
                         atlas=atlas, max_faces=maxf, log=log)
            obj = mvs / "scene_baked.glb"
            tex_info = tm.metadata.get("bake")
            tm.export(obj)
        except Exception as e:
            log(f"  사진 텍스처 굽기 실패 → 점군 색 메쉬 사용: {e}")
            obj = colored
        ply = mvs / "scene_dense.ply"
        from .pointfilter import LAST
        return {"textured_obj": obj, "dense_ply": ply if ply.exists() else None, "log": logf,
                "stats": {**LAST, "texture": tex_info}}
    try:
        _run([_bin("TextureMesh"), str(dense_scene), "--mesh-file", str(mesh), "--export-type", "ply",  # v2.3 SaveOBJ 세그폴트 → PLY(+png)
              "--empty-color", "8421504",  # 어느 사진에도 안 보인 면은 회색(기본 주황)
              "--resolution-level", str(max(0, q["res"] - 1)),
              "-o", str(mvs / "scene_textured.mvs")] + common, mvs, log, logf)
    except RuntimeError as e:
        log(f"  텍스처 실패 → 정점 색 메쉬로 대체: {e}")
    obj = mvs / "scene_textured.ply"
    if obj.exists():
        # Windows OpenMVS v2.4 는 사진을 못 읽어 검은 텍스처를 만들었다(PC 3회). 아틀라스 평균은 여백·경계색 때문에
        # 판정이 안 되므로, 면에 실제로 입혀진 색의 밝기를 원본 점군 색과 비교한다.
        try:
            import numpy as np
            import trimesh
            from .export import load_openmvs_textured_ply
            tm = load_openmvs_textured_ply(obj)
            pc = trimesh.load(dense / "fused.ply", process=False) if (dense / "fused.ply").exists() else None
            if tm is not None and pc is not None:
                img = np.asarray(tm.visual.material.image)
                uv = tm.visual.uv.reshape(-1, 3, 2).mean(1)
                h, w = img.shape[:2]
                sel = np.random.default_rng(0).choice(len(uv), min(200000, len(uv)), replace=False)
                px = img[np.clip(((1 - uv[sel, 1]) * h).astype(int), 0, h - 1),
                         np.clip((uv[sel, 0] * w).astype(int), 0, w - 1)][:, :3].astype(float)
                pcol = np.asarray(getattr(pc, "colors", pc.visual.vertex_colors))[:, :3].astype(float)
                tex_l, pt_l = float(px.mean()), float(pcol.mean())
                log(f"  텍스처 밝기 {tex_l:.0f} / 점군 밝기 {pt_l:.0f}")
                if tex_l < 0.5 * pt_l:
                    log("  경고: 텍스처가 사진보다 훨씬 어두움(사진 읽기 실패 의심) → 점군 색 메쉬로 대체")
                    obj = _color_from_points(mesh, dense / "fused.ply", mvs, log)
        except Exception as e:
            log(f"  텍스처 점검 생략: {e}")
    if not obj.exists():
        if engine == "colmap" and mesh.exists():
            obj = _color_from_points(mesh, dense / "fused.ply", mvs, log)  # 사진 텍스처 대신 점군 색
        else:
            raise RuntimeError("텍스처 메쉬가 생성되지 않았습니다")
    ply = mvs / "scene_dense.ply"
    return {"textured_obj": obj, "dense_ply": ply if ply.exists() else None, "log": logf}
