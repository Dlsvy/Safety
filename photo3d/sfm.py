"""SfM: 이미지 → 카메라 자세 + 렌즈 보정 + 희소 점군 (COLMAP / pycolmap).

정확도 포인트
- 같은 폰으로 찍은 사진은 카메라 1대(single_camera)로 묶어 렌즈 왜곡을 공동 추정한다.
- 폰 렌즈는 왜곡이 있으므로 OPENCV 모델(k1,k2,p1,p2)을 쓴다.
- 등록률·재투영오차를 품질 지표로 남긴다(재투영오차 1px 이하가 목표).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import os

import numpy as np
import pycolmap


SIFT_PEAK = 0.002


def _incr_options(robust: bool = True) -> "pycolmap.IncrementalPipelineOptions":
    o = pycolmap.IncrementalPipelineOptions()
    o.ba_refine_principal_point = False
    if robust:
        o.min_num_matches = 15
        o.mapper.abs_pose_min_num_inliers = 20
        o.structure_less_registration_fallback = True
    return o


FEATURES = {
    "sift": (pycolmap.FeatureExtractorType.SIFT, pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE),
    "sift_hq": (pycolmap.FeatureExtractorType.SIFT, pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE),
    "sift_lg": (pycolmap.FeatureExtractorType.SIFT, pycolmap.FeatureMatcherType.SIFT_LIGHTGLUE),
    "aliked": (pycolmap.FeatureExtractorType.ALIKED_N16ROT, pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE),
}


def _quat_wxyz(R) -> list[float]:
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(np.asarray(R)).as_quat()
    return [float(w), float(x), float(y), float(z)]


def _rig_cam_json(r: dict, ref: bool) -> dict:
    c = {"image_prefix": r["prefix"]}
    if ref:
        c["ref_sensor"] = True
    else:
        c["cam_from_rig_rotation"] = _quat_wxyz(r["cam_from_rig"])
        c["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
    return c


def _apply_rig(db: Path, rig: list) -> None:
    cams = []
    for i, r in enumerate(rig):
        c = pycolmap.RigConfigCamera()
        c.image_prefix = r["prefix"]
        c.ref_sensor = i == 0
        if i:
            c.cam_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(np.asarray(r["cam_from_rig"])), np.zeros(3))
        cams.append(c)
    cfg = pycolmap.RigConfig()
    cfg.cameras = cams
    with pycolmap.Database.open(str(db)) as database:
        pycolmap.apply_rig_config([cfg], database)


def _run_cli(colmap: str, image_dir: Path, work: Path, sequential: bool, single_camera: bool,
             camera_model: str, mapper: str, use_gpu: bool, log, features: str = "sift",
             mask_dir: Path | None = None, max_side: int = 3200,
             max_features: int = 12000, camera_params: str | None = None,
             rig: list | None = None) -> list[pycolmap.Reconstruction]:
    """COLMAP 4.x 실행파일(윈도우 CUDA 릴리스 등)로 SfM. GPU 로 수 배 빠르고 ALIKED/LightGlue 사용 가능."""
    import subprocess
    db = work / "database.db"
    g = "1" if use_gpu else "0"
    logf = open(work / "colmap.log", "a")

    def run(*args):
        log("  $ colmap " + " ".join(args[:1]))
        r = subprocess.run([colmap, *args], stdout=logf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"colmap {args[0]} 실패 (로그: {work / 'colmap.log'})")

    ftype = {"sift": ("SIFT", "SIFT_BRUTEFORCE"), "sift_hq": ("SIFT", "SIFT_BRUTEFORCE"),
             "sift_lg": ("SIFT", "SIFT_LIGHTGLUE"), "aliked": ("ALIKED_N16ROT", "ALIKED_LIGHTGLUE")}[features]
    ext = ["--FeatureExtraction.use_gpu", g, "--FeatureExtraction.type", ftype[0],
           "--FeatureExtraction.max_image_size", str(max_side)]
    ext += (["--AlikedExtraction.max_num_features", "8000"] if features == "aliked"
            else ["--SiftExtraction.max_num_features", str(max_features),
                  "--SiftExtraction.peak_threshold", str(SIFT_PEAK)])
    if features == "sift_hq":
        ext += ["--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"]
    mat = ["--FeatureMatching.use_gpu", g, "--FeatureMatching.type", ftype[1]]
    reader = ["--ImageReader.camera_model", camera_model,
              "--ImageReader.single_camera", "1" if single_camera else "0"]
    if rig:
        reader += ["--ImageReader.single_camera_per_folder", "1"]
    if mask_dir:
        reader += ["--ImageReader.mask_path", str(mask_dir)]
    if camera_params:
        reader += ["--ImageReader.camera_params", camera_params]
    run("feature_extractor", "--database_path", str(db), "--image_path", str(image_dir), *reader, *ext)
    rig_args = []
    if rig:
        import json
        cfg = work / "rig_config.json"
        cfg.write_text(json.dumps([{"cameras": [_rig_cam_json(r, i == 0) for i, r in enumerate(rig)]}]))
        run("rig_configurator", "--database_path", str(db), "--rig_config_path", str(cfg))
        mat += ["--FeatureMatching.rig_verification", "1"]
        rig_args = ["--Mapper.ba_refine_sensor_from_rig", "0", "--Mapper.ba_refine_focal_length", "0",
                    "--Mapper.ba_refine_extra_params", "0", "--Mapper.min_num_matches", "10",
                    "--Mapper.abs_pose_min_num_inliers", "15", "--Mapper.structure_less_registration_fallback", "1"]
        mapper = "incremental"  # 리그는 증분 매퍼가 안정적
    if rig:
        run("sequential_matcher", "--database_path", str(db), *mat, "--SequentialMatching.overlap", "8",
            "--SequentialMatching.quadratic_overlap", "0", "--SequentialMatching.expand_rig_images", "1",
            "--TwoViewGeometry.min_num_inliers", "15")
    elif sequential:
        run("sequential_matcher", "--database_path", str(db), *mat,
            "--SequentialMatching.overlap", "15", "--SequentialMatching.quadratic_overlap", "1")
    else:
        run("exhaustive_matcher", "--database_path", str(db), *mat)
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    relaxed = ([] if rig else ["--Mapper.min_num_matches", "15", "--Mapper.abs_pose_min_num_inliers", "20",
                               "--Mapper.structure_less_registration_fallback", "1"])

    def load(d):
        return {i: pycolmap.Reconstruction(x) for i, x in enumerate(sorted(d.iterdir()))
                if (x / "images.bin").exists()}

    def incremental(out):
        run("mapper", "--database_path", str(db), "--image_path", str(image_dir),
            "--output_path", str(out), "--Mapper.ba_refine_principal_point", "0", *relaxed, *rig_args,
            *(["--Mapper.ba_use_gpu", "1"] if use_gpu else []))
        return load(out)

    if mapper == "global":
        sg, si = sparse / "_global", sparse / "_incr"
        sg.mkdir(); si.mkdir()
        run("global_mapper", "--database_path", str(db), "--image_path", str(image_dir),
            "--output_path", str(sg),
            *(["--GlobalMapper.ba_ceres_use_gpu", "1", "--GlobalMapper.gp_use_gpu", "1"] if use_gpu else []))
        cands = [("global", load(sg))]
        try:  # 증분(완화)도 돌려 비교 — 글로벌이 가끔 위치를 통째로 틀리게 잡는 문제 대비
            cands.append(("incremental", incremental(si)))
        except RuntimeError as e:
            log(f"  증분 매퍼 생략: {e}")
        recs = pick_best(cands, log)
    else:
        recs = incremental(sparse)
    logf.close()
    return list(recs.values())


def run_sfm(image_dir: Path, work: Path, sequential: bool, single_camera: bool = True,
            camera_model: str = "OPENCV", mapper: str = "incremental", use_gpu: bool = False,
            colmap_bin: str | None = None, features: str = "sift", mask_dir: Path | None = None,
            refine_pp: bool = False, robust: bool = True, max_side: int = 3200,
            max_features: int = 12000, camera_params: str | None = None, rig: list | None = None,
            log=print) -> tuple[pycolmap.Reconstruction, dict]:
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    n_imgs = len([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if colmap_bin:
        sparse = work / "sparse"
        if sparse.exists():
            shutil.rmtree(sparse)
        recs = _run_cli(colmap_bin, image_dir, work, sequential, single_camera, camera_model,
                        mapper, use_gpu, log, features=features, mask_dir=mask_dir,
                        max_side=max_side, max_features=max_features, camera_params=camera_params,
                        rig=rig)
        if not recs:
            raise RuntimeError("SfM 실패: 이미지 간 겹침이 부족합니다(사진 간 60~80% 겹치게 촬영)")
        return _finish(dict(enumerate(recs)), n_imgs, log, refine_pp, robust,
                       min_img_points=25 if rig else 60)
    device = pycolmap.Device.cuda if (use_gpu and pycolmap.has_cuda) else pycolmap.Device.cpu

    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = max_features
    ext.max_image_size = max_side
    # 대비 낮은 흰 벽·천장에서도 특징점(기본 0.0067 → 0.002). 실험: 사진당 연결점 +40%, 위치 오차 감소
    ext.sift.peak_threshold = SIFT_PEAK
    mopt = pycolmap.FeatureMatchingOptions()
    # features: sift(기본) | sift_lg(SIFT+LightGlue) | aliked(ALIKED+LightGlue, 반복무늬·저질감에 강함)
    ext_type, match_type = FEATURES[features]
    ext.type = ext_type
    mopt.type = match_type
    if features == "sift_hq":
        # 어파인 형상 추정 + DSP-SIFT: 비스듬한 벽면·시점 변화에 강하고 매칭 정확도↑ (CPU 시간 2~3배)
        ext.sift.estimate_affine_shape = True
        ext.sift.domain_size_pooling = True
        mopt.guided_matching = True
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    if camera_params:
        reader.camera_params = camera_params
    if mask_dir:
        # 움직이는 사람·차량·크레인 마스크. 파일명 = <이미지명>.png, 검정(0) 픽셀은 특징점에서 제외
        reader.mask_path = str(mask_dir)
    log(f"  특징점 추출 ({n_imgs}장, {device.name})")
    pycolmap.extract_features(db, image_dir, camera_mode=(
        pycolmap.CameraMode.PER_FOLDER if rig else
        pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO),
        reader_options=reader, extraction_options=ext, device=device)
    if rig:
        _apply_rig(db, rig)
        mopt.rig_verification = True
        mapper = "incremental"
        log(f"  360° 리그: {len(rig)}방향을 한 위치로 묶음")

    tv = pycolmap.TwoViewGeometryOptions()
    if robust:
        # 반복 패턴(조적·거푸집·비계)에서 생기는 잘못된 짝을 거른다(DEGENSAC).
        # 인라이어 기준은 15(기본). 40 으로 올리면 실내에서 사진 절반이 빠졌다(ETH3D office 54%, pipes 50%)
        tv.min_num_inliers = 15
        tv.use_degensac = True
    if rig:
        # 360 걷기 촬영: 가까운 위치끼리만 매칭(먼 위치끼리의 반복무늬 오매칭 방지 — 합성 시험에서 18 m 오등록 원인),
        # 리그 제약이 기하 검증을 겸하므로 쌍별 인라이어 기준은 낮춘다. 합성 360(30곳·45 m): RMSE 16 mm
        log("  매칭: 360 리그 순차(위치 순서 ±8, 리그 확장)")
        so = pycolmap.SequentialPairingOptions()
        so.overlap = 8
        so.quadratic_overlap = False
        so.expand_rig_images = True
        tv.min_num_inliers = 15
        tv.use_degensac = False
        pycolmap.match_sequential(db, matching_options=mopt, pairing_options=so,
                                  verification_options=tv, device=device)
    elif sequential:
        log("  매칭: 순차(동영상) + 루프 검출 없음, overlap=15")
        so = pycolmap.SequentialPairingOptions()
        so.overlap = 15
        so.quadratic_overlap = True
        pycolmap.match_sequential(db, matching_options=mopt, pairing_options=so,
                                  verification_options=tv, device=device)
    elif n_imgs <= 300:
        log("  매칭: 전수(exhaustive)")
        pycolmap.match_exhaustive(db, matching_options=mopt, verification_options=tv, device=device)
    else:
        log("  매칭: 공간/어휘트리 대신 순차+전수 혼합이 필요 — 순차 overlap=30 사용")
        so = pycolmap.SequentialPairingOptions()
        so.overlap = 30
        pycolmap.match_sequential(db, matching_options=mopt, pairing_options=so,
                                  verification_options=tv, device=device)

    sparse = work / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()
    log(f"  매핑: {mapper}")
    if mapper == "global" and not rig:
        # 글로벌 + 증분(완화) 둘 다 돌려 비교 — 매핑은 매칭보다 훨씬 빠르다
        sg, si = sparse / "_global", sparse / "_incr"
        sg.mkdir(); si.mkdir()
        cands = [("global", pycolmap.global_mapping(db, image_dir, sg)),
                 ("incremental", pycolmap.incremental_mapping(db, image_dir, si, options=_incr_options(robust)))]
        recs = pick_best(cands, log)
    elif mapper == "global":
        recs = pycolmap.global_mapping(db, image_dir, sparse)
    else:
        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_refine_principal_point = False  # 초기 매핑 중에는 고정(불안정 방지)
        if rig:  # 360 에서 자른 사진: 렌즈·방향 관계를 정확히 앎 → 고정
            opts.ba_refine_sensor_from_rig = False
            opts.ba_refine_focal_length = False
            opts.ba_refine_extra_params = False
            opts.structure_less_registration_fallback = True  # 약한 연결 구간에서 모델이 끊기지 않게
        if robust:
            # 완화(15/20 + 구조 없는 등록 보조): 엄격(30/60)은 실내 등록률 50~77%, 완화는 97~100% (오차 비슷)
            opts.min_num_matches = 10 if rig else 15
            opts.mapper.abs_pose_min_num_inliers = 15 if rig else 20
            opts.structure_less_registration_fallback = True
        recs = pycolmap.incremental_mapping(db, image_dir, sparse, options=opts)
    if not recs:
        raise RuntimeError("SfM 실패: 이미지 간 겹침이 부족합니다(사진 간 60~80% 겹치게 촬영)")
    return _finish(recs, n_imgs, log, refine_pp, robust, min_img_points=25 if rig else 60)


def reject_bad_images(rec: pycolmap.Reconstruction, min_points: int = 60, log=print) -> list[str]:
    """잘못 등록됐을 가능성이 큰 사진을 등록 취소한다.
    - 관측 3D 점이 min_points 미만 (반복무늬로 우연히 맞춘 경우가 대부분)
    - 카메라 위치가 다른 카메라들과 동떨어짐 (중앙값 거리의 20배 초과)"""
    ids = list(rec.reg_image_ids())
    C = np.array([rec.image(i).projection_center() for i in ids])
    med = np.median(C, 0)
    d = np.linalg.norm(C - med, axis=1)
    lim = 20 * max(np.median(d), 1e-9)
    # 프레임(360 리그면 한 위치의 10방향) 단위로 판정 — 하늘을 향한 한 방향의 점이 적어도 위치 전체는 정상
    frames: dict = {}
    for i, di in zip(ids, d):
        im = rec.image(i)
        f = frames.setdefault(im.frame_id, {"pts": 0, "far": False, "names": []})
        f["pts"] += im.num_points3D
        f["far"] |= di > lim
        f["names"].append(im.name)
    names = []
    for fid, f in frames.items():
        if f["pts"] < min_points or f["far"]:
            names += f["names"]
            rec.deregister_frame(fid)
    if names:
        log(f"  잘못 등록 의심 {len(names)}장 제외: {', '.join(names[:8])}{'…' if len(names) > 8 else ''}")
    return names


def gradual_selection(rec: pycolmap.Reconstruction, iters: int = 2, max_err: float = 2.0,
                      min_angle_deg: float = 1.5, log=print) -> dict:
    """Metashape 'Gradual Selection' 대응: 나쁜 연결점을 지우고 번들조정을 다시 한다.
    - 재투영오차 > max_err px
    - 삼각측량각(점을 본 카메라 시선 사이 최대각) < min_angle_deg  → 깊이가 불확실한 점
    - 2장에서만 보이고 오차 > 1px 인 점
    사진마다 최소 60점은 남긴다(등록이 끊기지 않게)."""
    removed = 0
    ba = pycolmap.BundleAdjustmentOptions()
    ba.refine_principal_point = False
    centers = {i: rec.image(i).projection_center() for i in rec.reg_image_ids()}
    for _ in range(iters):
        rec.update_point_3d_errors()
        bad = []
        for pid, p in rec.points3D.items():
            els = p.track.elements
            if p.error > max_err or (len(els) <= 2 and p.error > 1.0):
                bad.append(pid)
                continue
            C = np.array([centers[e.image_id] for e in els if e.image_id in centers])
            if len(C) >= 2:
                r = C - p.xyz
                r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
                cmin = float((r @ r.T).min())
                if np.degrees(np.arccos(np.clip(cmin, -1, 1))) < min_angle_deg:
                    bad.append(pid)
        # 사진별 남는 점 수 보호
        cnt = {i: rec.image(i).num_points3D for i in centers}
        dropped = 0
        for pid in bad:
            els = rec.point3D(pid).track.elements
            if all(cnt[e.image_id] > 60 for e in els if e.image_id in cnt):
                for e in els:
                    if e.image_id in cnt:
                        cnt[e.image_id] -= 1
                rec.delete_point3D(pid)
                dropped += 1
        removed += dropped
        if dropped == 0:
            break
        pycolmap.bundle_adjustment(rec, ba)
    err = rec.compute_mean_reprojection_error()
    log(f"  연결점 정리(Gradual selection): {removed:,}점 제거 → 재투영오차 {err:.3f}px")
    return {"removed": removed, "reproj_after": err}


def _all_names(recs: dict) -> list[str]:
    names = set()
    for r in recs.values():
        names |= {im.name for im in r.images.values()}
    return sorted(names)


def pick_best(cands: list[tuple[str, dict]], log=print) -> dict:
    """여러 매퍼 결과 중 고르기: 가장 큰 모델의 등록 사진 수가 많은 쪽, 비슷하면(1장 차이 이내) 증분 매퍼.
    실험(ETH3D 실내 5곳): 글로벌 매퍼는 등록률은 높지만 가끔 카메라 위치가 통째로 틀어진다(중앙 오차 0.2~0.3 m,
    등록 수로는 감지 불가). 둘이 비슷한 수를 등록하면 안정적인 증분 쪽을 쓴다."""
    def size(recs):
        return max((r.num_reg_images() for r in recs.values()), default=0)
    best = None
    for name, recs in cands:
        n = size(recs)
        log(f"  매퍼 {name}: 가장 큰 모델 {n}장, 모델 {len(recs)}개")
        if best is None or n > best[1] + 1 or (abs(n - best[1]) <= 1 and name.startswith("incr") and not best[0].startswith("incr")):
            best = (name, n, recs)
    if best:
        log(f"  → {best[0]} 결과 사용")
    return best[2] if best else {}


def _finish(recs: dict, n_imgs: int, log, refine_pp: bool = False,
            robust: bool = True, min_img_points: int = 60) -> tuple[pycolmap.Reconstruction, dict]:
    # 가장 큰 모델 선택
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    # 실내·무늬 적은 현장은 사진당 점 수 자체가 적다 → 기준을 전체 평균의 15% 로 완화(최대 min_img_points)
    if robust and rec.num_reg_images():
        avg = float(np.mean([rec.image(i).num_points3D for i in rec.reg_image_ids()]))
        min_img_points = int(min(min_img_points, max(15, 0.15 * avg)))
    rejected = reject_bad_images(rec, min_points=min_img_points, log=log) if robust else []
    # 최종 전체 BA (렌즈 파라미터 포함)
    ba = pycolmap.BundleAdjustmentOptions()
    ba.refine_principal_point = False
    pycolmap.bundle_adjustment(rec, ba)
    gsel = gradual_selection(rec, log=log) if robust and os.environ.get("PHOTO3D_NO_GSEL") != "1" else None
    if refine_pp:
        # 마지막에 광학중심(cx,cy)까지 풀어서 한 번 더 — 중심이 어긋난 렌즈에서 접선왜곡항이 대신 흡수하는 것을 막는다
        ba.refine_principal_point = True
        pycolmap.bundle_adjustment(rec, ba)
    stats = {
        "images_input": n_imgs,
        "images_registered": rec.num_reg_images(),
        "images_rejected": rejected,
        "registration_ratio": rec.num_reg_images() / max(n_imgs, 1),
        "num_models": len(recs),
        "model_sizes": sorted((r.num_reg_images() for r in recs.values()), reverse=True),
        "images_unregistered": sorted(set(_all_names(recs)) - {rec.image(i).name for i in rec.reg_image_ids()}),
        "points3D": rec.num_points3D(),
        "mean_reprojection_error_px": rec.compute_mean_reprojection_error(),
        "mean_track_length": rec.compute_mean_track_length(),
        "gradual_selection": gsel,
        "cameras": {cid: {"model": c.model_name, "params": [round(v, 6) for v in c.params]}
                    for cid, c in rec.cameras.items()},
    }
    log(f"  등록 {stats['images_registered']}/{n_imgs}장, 3D점 {stats['points3D']}, "
        f"재투영오차 {stats['mean_reprojection_error_px']:.3f}px")
    if len(recs) > 1:
        log(f"  경고: 모델이 {len(recs)}개로 쪼개졌습니다(크기 {stats['model_sizes']}) — 가장 큰 것만 사용. "
            "빠진 사진 주변을 더 촘촘히(한두 걸음마다) 재촬영하세요")
    if stats["images_unregistered"]:
        u = stats["images_unregistered"]
        log(f"  미등록 {len(u)}장: {', '.join(u[:10])}{'…' if len(u) > 10 else ''}")
    return rec, stats
