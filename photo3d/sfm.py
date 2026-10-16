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


FEATURES = {
    "sift": (pycolmap.FeatureExtractorType.SIFT, pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE),
    "sift_hq": (pycolmap.FeatureExtractorType.SIFT, pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE),
    "sift_lg": (pycolmap.FeatureExtractorType.SIFT, pycolmap.FeatureMatcherType.SIFT_LIGHTGLUE),
    "aliked": (pycolmap.FeatureExtractorType.ALIKED_N16ROT, pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE),
}


def _run_cli(colmap: str, image_dir: Path, work: Path, sequential: bool, single_camera: bool,
             camera_model: str, mapper: str, use_gpu: bool, log, features: str = "sift",
             mask_dir: Path | None = None, max_side: int = 3200,
             max_features: int = 12000, camera_params: str | None = None) -> list[pycolmap.Reconstruction]:
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
            else ["--SiftExtraction.max_num_features", str(max_features)])
    if features == "sift_hq":
        ext += ["--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"]
    mat = ["--FeatureMatching.use_gpu", g, "--FeatureMatching.type", ftype[1]]
    reader = ["--ImageReader.camera_model", camera_model,
              "--ImageReader.single_camera", "1" if single_camera else "0"]
    if mask_dir:
        reader += ["--ImageReader.mask_path", str(mask_dir)]
    if camera_params:
        reader += ["--ImageReader.camera_params", camera_params]
    run("feature_extractor", "--database_path", str(db), "--image_path", str(image_dir), *reader, *ext)
    if sequential:
        run("sequential_matcher", "--database_path", str(db), *mat,
            "--SequentialMatching.overlap", "15", "--SequentialMatching.quadratic_overlap", "1")
    else:
        run("exhaustive_matcher", "--database_path", str(db), *mat)
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    if mapper == "global":
        run("global_mapper", "--database_path", str(db), "--image_path", str(image_dir),
            "--output_path", str(sparse),
            *(["--GlobalMapper.ba_ceres_use_gpu", "1", "--GlobalMapper.gp_use_gpu", "1"] if use_gpu else []))
    else:
        run("mapper", "--database_path", str(db), "--image_path", str(image_dir),
            "--output_path", str(sparse), "--Mapper.ba_refine_principal_point", "0",
            *(["--Mapper.ba_use_gpu", "1"] if use_gpu else []))
    logf.close()
    return [pycolmap.Reconstruction(d) for d in sorted(sparse.iterdir()) if (d / "images.bin").exists()]


def run_sfm(image_dir: Path, work: Path, sequential: bool, single_camera: bool = True,
            camera_model: str = "OPENCV", mapper: str = "incremental", use_gpu: bool = False,
            colmap_bin: str | None = None, features: str = "sift", mask_dir: Path | None = None,
            refine_pp: bool = False, robust: bool = True, max_side: int = 3200,
            max_features: int = 12000, camera_params: str | None = None,
            log=print) -> tuple[pycolmap.Reconstruction, dict]:
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    n_imgs = len([p for p in image_dir.iterdir() if p.is_file()])
    if colmap_bin:
        sparse = work / "sparse"
        if sparse.exists():
            shutil.rmtree(sparse)
        recs = _run_cli(colmap_bin, image_dir, work, sequential, single_camera, camera_model,
                        mapper, use_gpu, log, features=features, mask_dir=mask_dir,
                        max_side=max_side, max_features=max_features, camera_params=camera_params)
        if not recs:
            raise RuntimeError("SfM 실패: 이미지 간 겹침이 부족합니다(사진 간 60~80% 겹치게 촬영)")
        return _finish(dict(enumerate(recs)), n_imgs, log, refine_pp, robust)
    device = pycolmap.Device.cuda if (use_gpu and pycolmap.has_cuda) else pycolmap.Device.cpu

    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = max_features
    ext.max_image_size = max_side
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
        pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO),
        reader_options=reader, extraction_options=ext, device=device)

    tv = pycolmap.TwoViewGeometryOptions()
    if robust:
        # 반복 패턴(조적·거푸집·비계)에서 생기는 잘못된 짝을 거른다
        tv.min_num_inliers = 40
        tv.use_degensac = True
    if sequential:
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
    if mapper == "global":
        recs = pycolmap.global_mapping(db, image_dir, sparse)
    else:
        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_refine_principal_point = False  # 초기 매핑 중에는 고정(불안정 방지)
        if robust:
            opts.min_num_matches = 30
            opts.mapper.abs_pose_min_num_inliers = 60
        recs = pycolmap.incremental_mapping(db, image_dir, sparse, options=opts)
    if not recs:
        raise RuntimeError("SfM 실패: 이미지 간 겹침이 부족합니다(사진 간 60~80% 겹치게 촬영)")
    return _finish(recs, n_imgs, log, refine_pp, robust)


def reject_bad_images(rec: pycolmap.Reconstruction, min_points: int = 60, log=print) -> list[str]:
    """잘못 등록됐을 가능성이 큰 사진을 등록 취소한다.
    - 관측 3D 점이 min_points 미만 (반복무늬로 우연히 맞춘 경우가 대부분)
    - 카메라 위치가 다른 카메라들과 동떨어짐 (중앙값 거리의 20배 초과)"""
    ids = list(rec.reg_image_ids())
    C = np.array([rec.image(i).projection_center() for i in ids])
    med = np.median(C, 0)
    d = np.linalg.norm(C - med, axis=1)
    lim = 20 * max(np.median(d), 1e-9)
    bad = [i for i, di in zip(ids, d) if rec.image(i).num_points3D < min_points or di > lim]
    names = []
    for i in bad:
        im = rec.image(i)
        names.append(im.name)
        rec.deregister_frame(im.frame_id)
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


def _finish(recs: dict, n_imgs: int, log, refine_pp: bool = False,
            robust: bool = True) -> tuple[pycolmap.Reconstruction, dict]:
    # 가장 큰 모델 선택
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    rejected = reject_bad_images(rec, log=log) if robust else []
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
