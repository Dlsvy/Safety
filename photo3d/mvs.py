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


def densify_and_mesh(rec: pycolmap.Reconstruction, image_dir: Path, work: Path,
                     quality: str = "medium", threads: int = 0, log=print) -> dict:
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
    _run([_bin("DensifyPointCloud"), str(mvs / "scene.mvs"),
          "--resolution-level", str(q["res"]), "--number-views", str(q["views"]),
          "--number-views-fuse", "3", "-o", str(mvs / "scene_dense.mvs")] + common, mvs, log, logf)
    # v2.3: ReconstructMesh/RefineMesh 는 메쉬(.ply)만 쓴다 → 다음 단계에 --mesh-file 로 넘긴다
    dense_scene = mvs / "scene_dense.mvs"
    _run([_bin("ReconstructMesh"), str(dense_scene),
          "-o", str(mvs / "scene_mesh.mvs")] + common, mvs, log, logf)
    mesh = mvs / "scene_mesh.ply"
    if q["refine"]:
        try:
            _run([_bin("RefineMesh"), str(dense_scene), "--mesh-file", str(mesh),
                  "--resolution-level", str(q["res"] + 1), "--max-face-area", "16",
                  "-o", str(mvs / "scene_refined.mvs")] + common, mvs, log, logf)
            if (mvs / "scene_refined.ply").exists():
                mesh = mvs / "scene_refined.ply"
        except RuntimeError as e:
            log(f"  메쉬 정제 건너뜀: {e}")
    _run([_bin("TextureMesh"), str(dense_scene), "--mesh-file", str(mesh), "--export-type", "ply",  # v2.3 SaveOBJ 세그폴트 → PLY(+png)
          "--empty-color", "8421504",  # 어느 사진에도 안 보인 면은 회색(기본 주황)
          "--resolution-level", str(max(0, q["res"] - 1)),
          "-o", str(mvs / "scene_textured.mvs")] + common, mvs, log, logf)
    obj = mvs / "scene_textured.ply"
    ply = mvs / "scene_dense.ply"
    if not obj.exists():
        raise RuntimeError("텍스처 메쉬가 생성되지 않았습니다")
    return {"textured_obj": obj, "dense_ply": ply if ply.exists() else None, "log": logf}
