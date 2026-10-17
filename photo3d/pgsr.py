"""PGSR(Planar-based Gaussian Splatting, TVCG 2024) — 형태 정확도가 가장 높은 가우시안 표면 복원(연구용 라이선스,
사용자: 비상업 용도라 사용 허락). DTU Chamfer 0.47mm(2DGS 0.80, 3DGS 1.96, COLMAP 1.36), TnT F1 0.51.
가우시안을 납작한 '평면 조각'으로 학습하고 여러 사진의 기하·명암 일치(NCC)를 함께 맞춰 벽·바닥이 평평하고
모서리가 살아 있다 → 건물에 잘 맞는다. 결과 메쉬(TSDF)를 포토그래메트리 메쉬와 합친다(fuse.merge_meshes).

설치(Windows, NVIDIA): scripts/gpu/setup_pgsr_windows.ps1 → third_party/pgsr/PGSR + venv
여기서는 실행만: 조밀복원 폴더(dense/images, dense/sparse = 왜곡보정 PINHOLE)를 그대로 학습 데이터로 쓴다.
좌표는 학습 전후 그대로(측량 좌표, m) → 메쉬가 포토그래메트리와 바로 겹친다.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def find_pgsr() -> tuple[str, Path] | None:
    """(파이썬 실행파일, PGSR 폴더). 환경변수 PGSR_DIR/PGSR_PYTHON 또는 third_party/pgsr"""
    d = Path(os.environ.get("PGSR_DIR", ROOT / "third_party" / "pgsr" / "PGSR"))
    if not (d / "train.py").exists():
        return None
    py = os.environ.get("PGSR_PYTHON")
    if not py:
        for c in (d.parent / "venv" / "Scripts" / "python.exe", d.parent / "venv" / "bin" / "python"):
            if c.exists():
                py = str(c)
                break
    return (py or sys.executable), d


def _patch(d: Path) -> None:
    """한 번만 고치는 것: ① pytorch3d(윈도우 설치가 어려움)는 회전 함수 하나만 쓰므로 직접 구현
    ② TSDF 깊이를 16bit mm 로 저장해 65m 넘으면 넘침(드론 고도) → 장면 크기에 맞춘 단위"""
    gm = d / "scene" / "gaussian_model.py"
    s = gm.read_text(encoding="utf-8")
    if "from pytorch3d.transforms import quaternion_to_matrix" in s:
        s = s.replace("from pytorch3d.transforms import quaternion_to_matrix", '''def quaternion_to_matrix(q):  # photo3d: pytorch3d 대신
    import torch
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                     two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                     two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return o.reshape(q.shape[:-1] + (3, 3))''')
        gm.write_text(s, encoding="utf-8")
    rp = d / "render.py"
    s = rp.read_text(encoding="utf-8")
    if "PHOTO3D_DS" not in s:
        s = s.replace("(ref_depth*1000).astype(np.uint16)",
                      "(ref_depth*PHOTO3D_DS).astype(np.uint16)")
        s = s.replace("depth_scale=1000.0", "depth_scale=PHOTO3D_DS")
        s = s.replace("import os\n", "import os\nPHOTO3D_DS = float(os.environ.get('PHOTO3D_DS', '1000'))\n", 1)
        rp.write_text(s, encoding="utf-8")


def _scene_params(sparse: Path) -> dict:
    """측량 좌표(m) 장면에 맞춘 값: 이웃 사진 거리(여러 사진 일치용), TSDF 최대 깊이·복셀 크기"""
    import pycolmap
    from scipy.spatial import cKDTree
    rec = pycolmap.Reconstruction(sparse)
    C = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])
    nn = cKDTree(C).query(C, k=min(4, len(C)))[0][:, 1:].mean(1) if len(C) > 3 else np.array([1.0])
    step = float(np.median(nn))
    depths, gsd = [], []
    pts = {pid: p.xyz for pid, p in rec.points3D.items()}
    for iid in list(rec.reg_image_ids())[:: max(1, len(C) // 40)]:
        im = rec.images[iid]
        T = np.asarray((im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix())
        P = np.array([pts[p.point3D_id] for p in im.points2D if p.has_point3D()][:2000])
        if len(P) < 20:
            continue
        z = P @ T[2, :3] + T[2, 3]
        z = z[z > 0]
        depths.append(np.percentile(z, 98))
        gsd.append(np.median(z) / rec.cameras[im.camera_id].calibration_matrix()[0, 0])
    max_depth = float(np.median(depths) * 1.3) if depths else 50.0
    g = float(np.median(gsd)) if gsd else 0.01
    ext = float(np.ptp(np.array(list(pts.values())), 0).max()) if pts else 50.0
    voxel = max(3 * g, ext / 3000)  # 큰 현장은 메모리 때문에 거칠게(3000칸)
    return {"mv_max": 3 * step, "mv_min": 0.05 * step, "max_depth": max_depth, "voxel": voxel,
            "ds": 1000.0 if max_depth < 60 else 60000.0 / max_depth}


def run(dense: Path, work: Path, iterations: int = 30000, max_side: int = 1600, log=print,
        timeout: float = 4 * 3600) -> Path | None:
    """PGSR 학습 → 메쉬(TSDF). 반환: 메쉬 경로(조밀복원 좌표, Z-up) 또는 None"""
    f = find_pgsr()
    if f is None:
        return None
    py, d = f
    _patch(d)
    data = work / "data"
    if (work).exists():
        shutil.rmtree(work)
    (data / "sparse" / "0").mkdir(parents=True)
    for b in (dense / "sparse").glob("*.bin"):
        shutil.copy2(b, data / "sparse" / "0" / b.name)
    try:
        os.symlink(dense / "images", data / "images", target_is_directory=True)
    except OSError:
        shutil.copytree(dense / "images", data / "images")
    prm = _scene_params(dense / "sparse")
    import pycolmap
    longest = max(max(c.width, c.height) for c in pycolmap.Reconstruction(dense / "sparse").cameras.values())
    r = max(1, int(np.ceil(longest / max_side)))
    model = work / "model"
    logf = open(work / "pgsr.log", "w", encoding="utf-8", errors="replace")
    env = {**os.environ, "PHOTO3D_DS": str(prm["ds"]), "PYTHONIOENCODING": "utf-8"}
    train = [py, "train.py", "-s", str(data), "-m", str(model), "-r", str(r), "--iterations", str(iterations),
             "--max_abs_split_points", "0", "--opacity_cull_threshold", "0.05", "--data_device", "cpu",
             "--exposure_compensation", "--multi_view_max_dis", f"{prm['mv_max']:.4f}",
             "--multi_view_min_dis", f"{prm['mv_min']:.4f}", "--test_iterations", "-1", "--save_iterations", str(iterations), "--quiet"]
    log(f"  $ PGSR 학습 {iterations}스텝(평면 가우시안 + 여러 사진 일치, 해상도 1/{r}, 이웃 사진 거리 {prm['mv_max']:.2f}m)")
    try:
        rc = subprocess.run(train, cwd=d, stdout=logf, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
        if rc != 0:
            raise RuntimeError(f"학습 종료코드 {rc}")
        log(f"  $ PGSR 메쉬(TSDF 복셀 {prm['voxel'] * 100:.1f}cm, 최대 깊이 {prm['max_depth']:.0f}m)")
        rnd = [py, "render.py", "-m", str(model), "--max_depth", f"{prm['max_depth']:.3f}",
               "--voxel_size", f"{prm['voxel']:.5f}", "--use_depth_filter", "--skip_test", "--quiet"]
        rc = subprocess.run(rnd, cwd=d, stdout=logf, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
        if rc != 0:
            raise RuntimeError(f"메쉬 종료코드 {rc}")
    except Exception as e:
        logf.close()
        tail = (work / "pgsr.log").read_text(encoding="utf-8", errors="replace").splitlines()[-4:]
        log(f"  PGSR 실패: {e} | " + " | ".join(re.sub(r"\s+", " ", t)[:160] for t in tail))
        return None
    logf.close()
    m = model / "mesh" / "tsdf_fusion.ply"  # 후처리본(가장 큰 덩어리 1개만)이 아닌 원본 — 정리는 우리가
    return m if m.exists() else None
