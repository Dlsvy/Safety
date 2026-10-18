"""MeshSplatting(CVPR 2026) — 가우시안 대신 '불투명 삼각형'을 직접 학습해 이어진 메쉬를 만든다.
저장소 meshsplatting/mesh-splatting(자체 코드 Apache-2.0, 바탕 3DGS 코드는 Inria 비상업 — 사용자: 비상업 용도).

역할 분담(통합 탭): 형태 정확도 = PGSR(+포토그래메트리) → site_best_mesh.glb(측정·충돌)
                   보이는 모델 = MeshSplatting → site_opaque_mesh.glb(투명 구간 없음, 유니티 보기 모델)
제한된 델로네 삼각측량으로 면을 이어 붙여 구멍·반투명 경계가 원리적으로 생기지 않는다(논문: MILo 대비 메모리 1/2·2배 빠름).

설치(Windows, NVIDIA): scripts/gpu/setup_meshsplat_windows.ps1 → third_party/meshsplat/mesh-splatting + venv
실행(저장소 README 그대로): python train.py -s <데이터> -m <출력> → point_cloud/iteration_N/point_cloud_state_dict.pt
                         python create_ply.py <그 폴더> --out mesh.ply (정점 색 메쉬, 학습 좌표 = 측량 좌표 그대로)
학습 스케줄(불투명화 24k·업샘플 20k·sigma 30k)이 30,000 스텝 기준이라 스텝 수는 바꾸지 않는다.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ITERATIONS = 30_000
# 학습 일정(train.py OptimizationParams 기본값, 30,000스텝 기준). 스텝을 줄이면 같은 비율로 당긴다.
# 간격값(densification_interval·intervall_add_triangles 500)·densify_from_iter 는 그대로 둔다.
SCHEDULE = {"position_lr_max_steps": 30_000, "densify_until_iter": 10_000, "iteration_mesh": 5_000,
            "start_opacity_floor": 5_000, "start_pruning": 4_000, "sigma_until": 30_000,
            "final_opacity_iter": 24_000, "start_upsampling": 20_000, "start_vertex_opt": 12_000}


def find_meshsplat() -> tuple[str, Path] | None:
    """(파이썬 실행파일, 저장소 폴더). 환경변수 MESHSPLAT_DIR/MESHSPLAT_PYTHON 또는 third_party/meshsplat"""
    d = Path(os.environ.get("MESHSPLAT_DIR", ROOT / "third_party" / "meshsplat" / "mesh-splatting"))
    if not (d / "train.py").exists() or not (d / "create_ply.py").exists():
        return None
    py = os.environ.get("MESHSPLAT_PYTHON")
    if not py:
        for c in (d.parent / "venv" / "Scripts" / "python.exe", d.parent / "venv" / "bin" / "python"):
            if c.exists():
                py = str(c)
                break
    return (py or sys.executable), d


def _crop(m, sparse: Path, pad: float = 0.05) -> int:
    """측정 범위(희소점 0.5~99.5%, 5% 여유) 밖의 면 — 하늘·먼 배경 삼각형 — 을 지운다. 지운 면 수"""
    import pycolmap
    P = np.array([p.xyz for p in pycolmap.Reconstruction(sparse).points3D.values()])
    if len(P) < 100:
        return 0
    lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
    g = (hi - lo) * pad
    lo, hi = lo - g, hi + g
    V = np.asarray(m.vertices)
    inside = ((V >= lo) & (V <= hi)).all(1)
    keep = inside[m.faces].all(1)
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    return int((~keep).sum())


def run(dense: Path, work: Path, max_side: int = 1600, iterations: int = ITERATIONS, log=print,
        timeout: float = 5 * 3600) -> Path | None:
    """MeshSplatting 학습 → 불투명 메쉬(PLY, 정점 색, 조밀복원 좌표 Z-up). 실패하면 None"""
    f = find_meshsplat()
    if f is None:
        return None
    py, d = f
    from .pgsr import prep_data
    data, r = prep_data(dense, work, max_side)
    model = work / "model"
    logf = open(work / "meshsplat.log", "w", encoding="utf-8", errors="replace")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    train = [py, "train.py", "-s", str(data), "-m", str(model), "-r", str(r), "--data_device", "cpu",
             "--iterations", str(iterations), "--test_iterations", "-1", "--save_iterations", str(iterations), "--quiet"]
    if iterations != ITERATIONS:
        for k, v in SCHEDULE.items():
            train += [f"--{k}", str(max(1, round(v * iterations / ITERATIONS)))]
    log(f"  $ MeshSplatting 학습 {iterations}스텝(불투명 삼각형 + 델로네 연결, 해상도 1/{r})")
    out = work / "mesh.ply"
    try:
        rc = subprocess.run(train, cwd=d, stdout=logf, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
        if rc != 0:
            raise RuntimeError(f"학습 종료코드 {rc}")
        it = model / "point_cloud" / f"iteration_{iterations}"
        if not (it / "point_cloud_state_dict.pt").exists():
            raise RuntimeError("학습 결과(point_cloud_state_dict.pt) 없음")
        rc = subprocess.run([py, "create_ply.py", str(it), "--out", str(out)], cwd=d, stdout=logf,
                            stderr=subprocess.STDOUT, env=env, timeout=1800).returncode
        if rc != 0 or not out.exists():
            raise RuntimeError(f"메쉬 내보내기 종료코드 {rc}")
    except Exception as e:
        logf.close()
        tail = (work / "meshsplat.log").read_text(encoding="utf-8", errors="replace").splitlines()[-4:]
        log(f"  MeshSplatting 실패: {e} | " + " | ".join(re.sub(r"\s+", " ", t)[:160] for t in tail))
        return None
    logf.close()
    return out


def opaque_model(mesh_ply: Path, dense: Path, glb: Path, log=print) -> dict | None:
    """MeshSplatting 메쉬 → 범위 밖 정리·부스러기 제거 → 사진 텍스처(정점 색보다 선명) → Y-up GLB"""
    import trimesh
    from .export import ZUP_TO_YUP
    from .texture import bake
    m = trimesh.load(mesh_ply, process=False, force="mesh")
    n0 = len(m.faces)
    m = trimesh.Trimesh(m.vertices, m.faces, vertex_colors=m.visual.vertex_colors, process=True)  # 겹친 정점 합치기
    cut = _crop(m, dense / "sparse")
    # 부스러기(50면 미만 덩어리)만 지운다. mvs._fix_topology 의 '나비 정점' 깎기는 OpenMVS 텍스처용인데,
    # 구멍 가장자리가 많은 MeshSplatting 메쉬는 5번 깎는 동안 거의 다 사라졌다(1Family 839,195 → 2,668면).
    # 덩어리는 '정점'으로 이어진 것으로 센다 — 삼각형들이 모서리 대신 꼭짓점을 공유해 모서리 기준이면 30만 조각으로 보인다
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    F, nv = np.asarray(m.faces), len(m.vertices)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]]])
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)), directed=False)
    m.update_faces(np.bincount(lab[F[:, 0]])[lab[F[:, 0]]] >= 50)
    m.remove_unreferenced_vertices()
    if len(m.faces) < 1000:
        log(f"  불투명 모델 생략: 정리 후 면 {len(m.faces)}개")
        return None
    tm, _ = bake(m, dense / "sparse", dense / "images", atlas=8192, max_faces=1_200_000, log=log)
    tm.apply_transform(ZUP_TO_YUP)
    tm.export(glb)
    log(f"  불투명 모델(MeshSplatting): 면 {n0:,} → {len(m.faces):,}(범위 밖 {cut:,} 제거) → {glb.name}")
    return {"faces_raw": n0, "faces": int(len(m.faces)), "cropped": cut, "file": glb.name}
