"""적은 장수(2~15장) 카메라 자세 — MapAnything(Apache-2.0) + COLMAP 다듬기.

COLMAP 증분 SfM 은 사진 사이 겹침·시차가 충분해야 시작된다(보통 15~20장↑). 사진이 10장 안팎이면
카메라를 못 잡거나 일부만 잡는다. MapAnything 은 사진 묶음을 한 번에 보고 카메라 위치·렌즈·깊이를 바로 추정한다(미터 단위).
여기서는 그 결과를 '시작값'으로만 쓰고, 원본 해상도 SIFT 매칭으로 3D 점을 삼각측량한 뒤 번들 조정으로 다듬는다
→ MapAnything 의 518px 해상도 한계 대신 원본 사진 정밀도, 그리고 대략의 실제 크기(마커가 없을 때).

설치: scripts/gpu/setup_mapanything_windows.ps1 (third_party/mapanything). 없으면 None 을 돌려준다.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGES = 40  # 이보다 많으면 COLMAP 이 더 정확하고, MapAnything 은 VRAM(8GB)이 모자랄 수 있다


def find() -> Path | None:
    py = ROOT / "third_party" / "mapanything" / "venv" / ("Scripts/python.exe" if (ROOT / "third_party" / "mapanything" / "venv" / "Scripts").exists() else "bin/python")
    return py if py.exists() else None


def _env() -> dict:
    """가중치를 이미 받았으면 인터넷 확인 없이(HF 허브 확인이 느리거나 막힌 곳에서 몇 분씩 걸렸다)"""
    import os
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if (hub / "models--facebook--map-anything-apache").exists():
        env["HF_HUB_OFFLINE"] = "1"
    return env


def _infer(img_dir: Path, work: Path, log) -> list[dict] | None:
    py = find()
    if py is None:
        log("  적은 장수 보조(MapAnything) 미설치 — scripts/gpu/setup_mapanything_windows.ps1")
        return None
    names = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(names) > MAX_IMAGES:
        names = names[:: int(np.ceil(len(names) / MAX_IMAGES))]
    work.mkdir(parents=True, exist_ok=True)
    lst = json.dumps([str(p) for p in names])
    if (work / "poses.json").exists() and (work / "poses_dense.npz").exists() and (work / "list.json").exists() \
            and (work / "list.json").read_text(encoding="utf-8") == lst:  # 같은 사진이면 다시 추정하지 않는다
        log("  MapAnything: 이전 추정 결과 재사용")
        return json.loads((work / "poses.json").read_text(encoding="utf-8"))
    (work / "poses.json").unlink(missing_ok=True)
    (work / "list.json").write_text(lst, encoding="utf-8")
    log(f"  $ MapAnything: 사진 {len(names)}장으로 카메라 위치·렌즈 추정(GPU, 첫 실행은 가중치 내려받기)")
    with open(work / "mapanything.log", "w", encoding="utf-8", errors="replace") as lf:
        rc = subprocess.run([str(py), "-I", str(Path(__file__).with_name("fewshot_runner.py")),
                             str(work / "list.json"), str(work / "poses.json")],
                            stdout=lf, stderr=subprocess.STDOUT, timeout=3600,
                            env=_env()).returncode
    if rc != 0 or not (work / "poses.json").exists():
        tail = (work / "mapanything.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        log("  MapAnything 실패: " + " | ".join(tail))
        return None
    return json.loads((work / "poses.json").read_text(encoding="utf-8"))


def run(img_dir: Path, work: Path, use_gpu: bool = True, log=print):
    """반환: (pycolmap.Reconstruction, stats) 또는 None"""
    import pycolmap
    from .sfm import _finish
    poses = _infer(img_dir, work, log)
    if not poses or len(poses) < 2:
        return None
    by = {p["name"]: p for p in poses}
    db_path = work / "database.db"
    db_path.unlink(missing_ok=True)
    dev = pycolmap.Device.cuda if use_gpu else pycolmap.Device.auto
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = 3200
    ext.sift.max_num_features = 16000
    try:
        pycolmap.extract_features(db_path, img_dir, image_names=sorted(by), camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=ext, device=dev)
    except Exception:
        pycolmap.extract_features(db_path, img_dir, image_names=sorted(by), camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=ext)
    tv = pycolmap.TwoViewGeometryOptions()
    tv.min_num_inliers = 12  # 장수가 적고 시점 차이가 커서 짝마다 맞는 점이 적다
    try:
        pycolmap.match_exhaustive(db_path, verification_options=tv, device=dev)
    except Exception:
        pycolmap.match_exhaustive(db_path, verification_options=tv)
    db = pycolmap.Database.open(db_path)
    rec = pycolmap.Reconstruction()
    cams = {c.camera_id: c for c in db.read_all_cameras()}
    imgs = [im for im in db.read_all_images() if im.name in by]
    for cid, c in cams.items():
        ks = np.array([by[im.name]["K"] for im in imgs if im.camera_id == cid])
        f = float(np.median((ks[:, 0, 0] + ks[:, 1, 1]) / 2))
        c.params = [f, c.width / 2, c.height / 2, 0.0]  # SIMPLE_RADIAL: f, cx, cy, k
        db.update_camera(c)
        rec.add_camera_with_trivial_rig(c)
    for im in imgs:
        w2c = np.linalg.inv(np.array(by[im.name]["c2w"]))
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(w2c[:3, :3]), w2c[:3, 3])
        rec.add_image_with_trivial_frame(pycolmap.Image(name=im.name, camera_id=im.camera_id, image_id=im.image_id), pose)
    db.close()
    out = work / "sparse"
    out.mkdir(exist_ok=True)
    # MapAnything 위치는 회전 1~3°·초점 10~20% 오차(1Family 8장: 정답 대비) — 화면에서 수십 px 라 COLMAP 기본 기준(4px)이면
    # 점이 거의 다 버려진다(8장 → 11점). 넉넉한 기준으로 점을 만들고 → 번들 조정(위치·초점) → 기준을 조여 다시, 3번.
    ba = pycolmap.BundleAdjustmentOptions()
    ba.refine_principal_point = False
    for ang, px in ((10.0, 80.0), (4.0, 16.0), (2.0, 4.0)):
        opts = pycolmap.IncrementalPipelineOptions()
        opts.triangulation.ignore_two_view_tracks = False  # 장수가 적으면 두 장에만 보이는 점이 대부분
        opts.triangulation.create_max_angle_error = ang
        opts.triangulation.continue_max_angle_error = ang
        opts.triangulation.complete_max_reproj_error = px
        opts.triangulation.merge_max_reproj_error = px
        opts.mapper.filter_max_reproj_error = px
        opts.mapper.filter_min_tri_angle = 1.0
        rec = pycolmap.triangulate_points(rec, db_path, img_dir, out, clear_points=True, options=opts)
        if rec.num_points3D() < 30:
            log(f"  MapAnything 결과로 맞는 점이 {rec.num_points3D()}개뿐(기준 {px:.0f}px) → 사용 안 함")
            return None
        pycolmap.bundle_adjustment(rec, ba)
        rec.write(out)
    rec, st = _finish({0: rec}, len(list(img_dir.iterdir())), log, robust=False)
    st["fewshot"] = {"engine": "MapAnything(apache)", "images": len(poses), "scale": "metric_estimate"}
    log(f"  적은 장수 복원: 사진 {st['images_registered']}장 · 3D점 {st['points3D']:,} · 재투영오차 "
        f"{st['mean_reprojection_error_px']:.2f}px (크기는 MapAnything 추정 미터 — 마커가 있으면 마커가 우선)")
    return rec, st


def _sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """dst ≈ s·R·src + t (Umeyama). 카메라 위치끼리 맞춰 MapAnything 좌표 → 최종(정합·수평) 좌표"""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max((a ** 2).sum() / len(src), 1e-12))
    return s, R, md - s * R @ ms


def dense_points(work: Path, rec, dst: Path, keep_conf: float = 0.3, max_points: int = 3_000_000,
                 log=print) -> Path | None:
    """MapAnything 이 사진마다 예측한 조밀 점(픽셀 단위 깊이)을 최종 좌표로 옮겨 COLMAP fused.ply 형식으로 쓴다.
    사진이 적고 시점 차이가 크면 COLMAP 조밀복원(여러 사진 일치)이 거의 비어 나온다(8장: 3,584점) → 대신 이것을 쓴다.
    신뢰도 하위 keep_conf 비율·모델 마스크 밖은 버리고, 법선은 사진의 점 격자에서 바로 구해 카메라 쪽으로 향하게 한다."""
    npz, pj = work / "poses_dense.npz", work / "poses.json"
    if not npz.exists() or not pj.exists():
        return None
    poses = json.loads(pj.read_text(encoding="utf-8"))
    names = {rec.image(i).name: rec.image(i).projection_center() for i in rec.reg_image_ids()}
    pairs = [(np.array(p["c2w"])[:3, 3], names[p["name"]]) for p in poses if p["name"] in names]
    if len(pairs) < 3:
        return None
    s, R, t = _sim3(np.array([a for a, _ in pairs]), np.array([b for _, b in pairs]))
    d = np.load(npz)
    xyz, nrm, col = [], [], []
    for k, p in enumerate(poses):
        if f"pts{k}" not in d:
            continue
        P = d[f"pts{k}"].astype(np.float64)
        ok = d[f"mask{k}"] & np.isfinite(P).all(-1)
        cf = d[f"conf{k}"]
        if ok.any():
            ok &= cf >= np.quantile(cf[ok], keep_conf)
        dx = np.zeros_like(P)  # 격자 이웃으로 법선(마지막 행·열은 버림)
        dy = np.zeros_like(P)
        dx[:, :-1] = P[:, 1:] - P[:, :-1]
        dy[:-1] = P[1:] - P[:-1]
        n = np.cross(dx, dy)
        ok[-1, :] = False
        ok[:, -1] = False
        nn = np.linalg.norm(n, axis=-1)
        ok &= nn > 0
        n = n / np.maximum(nn, 1e-12)[..., None]
        cam = np.array(p["c2w"])[:3, 3]
        n[((cam - P) * n).sum(-1) < 0] *= -1
        xyz.append(P[ok])
        nrm.append(n[ok])
        col.append((d[f"rgb{k}"] if f"rgb{k}" in d else np.full(P.shape, 128, np.uint8))[ok])
    if not xyz:
        return None
    xyz, nrm, col = np.concatenate(xyz), np.concatenate(nrm), np.concatenate(col)
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, nrm, col = xyz[sel], nrm[sel], col[sel]
    xyz = s * xyz @ R.T + t
    nrm = nrm @ R.T
    data = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"),
                                      ("nz", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, k in enumerate(("x", "y", "z")):
        data[k] = xyz[:, i]
        data["n" + k] = nrm[:, i]
    data["red"], data["green"], data["blue"] = col[:, 0], col[:, 1], col[:, 2]
    from .pointfilter import write_ply_raw
    write_ply_raw(data, dst)
    err = float(np.median([np.linalg.norm(s * R @ a + t - b) for a, b in pairs]))
    log(f"  조밀 점(AI 깊이): {len(xyz):,}점 · 카메라 맞춤 오차 중앙 {err * 1000:.0f}mm (사진이 적어 COLMAP 조밀복원 대신)")
    return dst
