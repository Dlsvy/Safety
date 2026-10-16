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


def _color_from_points(mesh_path: Path, points_path: Path, mvs: Path, log) -> Path:
    """메쉬 정점마다 가장 가까운 조밀 점의 색을 입힌다(텍스처 실패 시 최후 대체)."""
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    try:
        m = trimesh.load(mesh_path, process=False, force="mesh")
        pc = trimesh.load(points_path, process=False)
        cols = getattr(pc, "colors", None)
        if cols is None or len(cols) == 0:
            cols = pc.visual.vertex_colors
        _, idx = cKDTree(np.asarray(pc.vertices)).query(m.vertices, workers=-1)
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.asarray(cols)[idx])
        out = mvs / "scene_mesh_colored.ply"
        m.export(out)
        log(f"  점군 색 입힘: {len(m.vertices):,}정점")
        return out
    except Exception as e:
        log(f"  점군 색 입히기 실패({e}) → 색 없는 메쉬 사용")
        return mesh_path


def densify_and_mesh(rec: pycolmap.Reconstruction, image_dir: Path, work: Path,
                     quality: str = "medium", threads: int = 0, engine: str = "openmvs",
                     colmap_bin: str | None = None, log=print) -> dict:
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
    _run([_bin("InterfaceCOLMAP"), "-i", str(dense), "-o", str(mvs / "scene.mvs"),
          "--image-folder", str(dense / "images")] + common, mvs, log, logf)
    if engine == "colmap":
        # GPU 조밀복원: COLMAP PatchMatchStereo(CUDA) → StereoFusion → OpenMVS 로 가져와 메쉬·텍스처
        if not colmap_bin:
            raise RuntimeError("--dense colmap 은 CUDA 지원 COLMAP 실행파일(--colmap-bin)이 필요합니다")
        # VRAM 8GB 기준: 원본(-1)은 12MP 이상에서 메모리 부족 위험 → ultra 도 3200px
        size = {"draft": 1600, "medium": 2400, "high": 3200, "ultra": 3200}[quality]
        fast = quality == "draft"  # 빠른 모드: 형상 일관성 단계 생략·반복 축소 → 조밀복원 시간 약 1/3
        _run([colmap_bin, "patch_match_stereo", "--workspace_path", str(dense),
              "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "0" if fast else "1",
              "--PatchMatchStereo.max_image_size", str(size), "--PatchMatchStereo.gpu_index", "0"]
             + (["--PatchMatchStereo.num_iterations", "3", "--PatchMatchStereo.window_step", "2"] if fast else []),
             mvs, log, logf)
        fused = dense / "fused.ply"
        _run([colmap_bin, "stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP",
              "--input_type", "photometric" if fast else "geometric", "--StereoFusion.max_image_size", str(size),
              "--output_path", str(fused)], mvs, log, logf)
        _run([_bin("InterfaceCOLMAP"), "-i", str(dense), "-p", str(fused),
              "-o", str(mvs / "scene_dense.mvs"), "--image-folder", str(dense / "images")] + common,
             mvs, log, logf)
        shutil.copy2(fused, mvs / "scene_dense.ply")
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
    try:
        _run([_bin("TextureMesh"), str(dense_scene), "--mesh-file", str(mesh), "--export-type", "ply",  # v2.3 SaveOBJ 세그폴트 → PLY(+png)
              "--empty-color", "8421504",  # 어느 사진에도 안 보인 면은 회색(기본 주황)
              "--resolution-level", str(max(0, q["res"] - 1)),
              "-o", str(mvs / "scene_textured.mvs")] + common, mvs, log, logf)
    except RuntimeError as e:
        log(f"  텍스처 실패 → 정점 색 메쉬로 대체: {e}")
    obj = mvs / "scene_textured.ply"
    if obj.exists():
        # 사진을 못 읽으면 TextureMesh 가 검은 아틀라스를 만든다(PC v2.4 에서 확인) → 밝기로 판정해 대체
        try:
            from PIL import Image
            import numpy as np
            pngs = sorted(mvs.glob("scene_textured*.png"))
            if pngs:
                im = np.asarray(Image.open(pngs[0]).convert("L").resize((256, 256)))
                bright = float(im.mean())
                log(f"  텍스처 평균 밝기 {bright:.0f}/255")
                if bright < 20 and mesh.exists():
                    log("  경고: 텍스처가 거의 검음(사진 로딩 실패 의심) → 점군 색 메쉬로 대체")
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
