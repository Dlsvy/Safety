"""photo3d 명령행.

예)
  python -m photo3d run 현장영상.mp4 --marker-size 0.40 --out out/현장A
  python -m photo3d run photos/ --marker-size 0.40 --gcp gcp.csv --check check.csv --quality high --gpu
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def run(a) -> dict:
    from . import export, frames, mvs, scale, sfm

    out = Path(a.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    logp = out / "run.log"
    t0 = time.time()

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    rep: dict = {"version": "0.1.0", "args": vars(a)}
    if a.from_colmap:
        # 외부 모델(MapAnything·VGGT 등 적은 장수용)이 계산한 카메라를 이어받는다: <dir>/images, <dir>/sparse
        src = Path(a.from_colmap)
        log(f"1-2/5 외부 카메라 결과 사용: {src}")
        img_dir = src / "images"
        sp = src / "sparse"
        if (sp / "0").exists():
            sp = sp / "0"
        import pycolmap
        rec = pycolmap.Reconstruction(sp)
        n_in = len([q for q in img_dir.iterdir() if q.is_file()])
        rep["input"] = {"images": n_in, "from_colmap": str(src)}
        from .sfm import _finish
        rec, st = _finish({0: rec}, n_in, log, a.refine_pp, robust=False)  # 우리 BA 로 한 번 더 다듬기
    else:
        log("1/5 입력 정리")
        img_dir = work / "images"
        prep = frames.prepare_inputs([Path(p) for p in a.inputs], img_dir, max_side=a.max_side,
                                     video_frames=a.video_frames, log=log)
        rep["input"] = {"images": len(prep["images"]), "sequential": prep["sequential"],
                        "skipped_blurry": prep["skipped_blurry"]}

        log("2/5 SfM (카메라 자세·렌즈 보정)")
        rec, st = sfm.run_sfm(img_dir, work / "sfm", sequential=prep["sequential"],
                              mapper=a.mapper, use_gpu=a.gpu,
                              colmap_bin=a.colmap_bin, features=a.features,
                              mask_dir=Path(a.masks) if a.masks else None, refine_pp=a.refine_pp,
                              robust=not a.no_robust, max_side=a.max_side,
                              max_features=a.max_features, log=log)
    rep["sfm"] = st

    log("3/5 축척·좌표 정합")
    geo = scale.georeference(rec, img_dir, a.marker_size, a.marker_dict,
                             Path(a.gcp) if a.gcp else None,
                             Path(a.check) if a.check else None,
                             scale_csv=Path(a.scale_dist) if a.scale_dist else None,
                             gcp_ba=not a.no_gcp_ba, log=log)
    rep["georef"] = geo
    (work / "sfm" / "georef").mkdir(parents=True, exist_ok=True)
    rec.write(work / "sfm" / "georef")
    cams = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])

    log(f"4/5 조밀복원·메쉬·텍스처 (품질 {a.quality})")
    res = mvs.densify_and_mesh(rec, img_dir, work, quality=a.quality, engine=a.dense,
                               colmap_bin=a.colmap_bin, log=log)

    log("5/5 정리·유니티 내보내기")
    info = export.export_all(res["textured_obj"], out, cams if not geo["method"].startswith("none") else None,
                             collider_faces=a.collider_faces, meta={"georef": geo, "sfm": st}, log=log)
    rep["model"] = info
    if a.fill and res.get("dense_ply"):
        from . import fill
        log("  구멍 메우기(주변 형상 Poisson) — 메운 부분은 '추정 영역'으로 따로 표시")
        try:
            fr = fill.fill_holes(res["dense_ply"], out, cams if not geo["method"].startswith("none") else None,
                                 log=log)
            m, est = fr["mesh"], fr["estimated_mask"]
            m.apply_transform(export.ZUP_TO_YUP)
            m.export(out / "site_filled.glb")  # 보기용: 구멍 없이 이어진 전체(색 포함)
            est_mesh = m.submesh([np.where(est)[0]], append=True) if est.any() else None
            if est_mesh is not None:
                est_mesh.export(out / "site_estimated.glb")  # 유니티: SiteEstimated 레이어로 경고
            rep["fill"] = {"estimated_area_ratio": fr["estimated_area_ratio"], "filled": "site_filled.glb",
                           "file": "site_estimated.glb" if est_mesh is not None else None,
                           "note": "추정 영역은 측정값이 아님 — 높은 위치/드론 사진 추가 또는 도면 3D 결합으로 대체 권장"}
        except Exception as e:
            log(f"  구멍 메우기 실패: {e}")
    rep["elapsed_min"] = round((time.time() - t0) / 60, 1)
    rep["grade"] = grade(rep)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    log(f"완료 — 등급 {rep['grade']['grade']}: {rep['grade']['reason']}")
    return rep


def grade(rep: dict) -> dict:
    """현장 사용 가능 여부를 단순 규칙으로 판정한다."""
    s, g = rep["sfm"], rep["georef"]
    issues = []
    if s["registration_ratio"] < 0.9:
        issues.append(f"등록률 {s['registration_ratio']:.0%} (90% 미만)")
    if s.get("num_models", 1) > 1:
        issues.append(f"모델 {s['num_models']}개로 분리(크기 {s.get('model_sizes')}) — 촬영 간격이 넓은 구간 재촬영")
    rej = len(s.get("images_rejected", []))
    if rej > 0.1 * s["images_input"]:
        issues.append(f"잘못 등록 의심 {rej}장 제외(반복무늬 현장 — 검사점 확인 필수)")
    if s["mean_reprojection_error_px"] > 1.2:
        issues.append(f"재투영오차 {s['mean_reprojection_error_px']:.2f}px")
    if g["method"].startswith("none"):
        return {"grade": "D", "reason": "축척 미확정 — 시뮬레이션 사용 불가"}
    errs = [abs(r["error_mm"]) for r in g.get("check_distances", [])]
    errs += [v * 1000 for v in g.get("checkpoint_error_m", {}).values()]
    if not errs:
        errs += [v * 1000 for v in g.get("checkpoint_loo_error_m", {}).values()]
    if errs:
        e = max(errs)
        lvl = "A" if e <= 30 else "B" if e <= 100 else "C"
        reason = f"검증 최대오차 {e:.0f}mm"
    else:
        lvl, reason = "B", "독립 검증거리 없음 — check.csv 권장"
    if g.get("method") == "marker_scale" and g.get("scale_source") == "marker_edges":
        # 가상현장 검증: 마커 한 변만으로 잡은 축척은 1.8~2.2% 오차(10 m 에 18~22 cm)
        lvl = chr(max(ord(lvl), ord("C")))
        reason += " / 축척이 마커 크기에만 의존(오차 ~2%) — 긴 줄자 거리(--scale-dist) 또는 GCP 필요"
    if issues:
        lvl = chr(max(ord(lvl), ord("B")))
        reason += " / " + ", ".join(issues)
    return {"grade": lvl, "reason": reason}


def main(argv=None):
    # 윈도우 한글 콘솔(cp949)은 '—' 등 일부 문자를 못 찍어 UnicodeEncodeError 로 멈춘다 → 대체문자로 출력
    import sys
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="photo3d", description="폰 사진/동영상 → 실측 3D")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="전체 파이프라인 실행")
    r.add_argument("inputs", nargs="*", help="사진 파일/폴더/동영상")
    r.add_argument("--from-colmap", help="외부에서 계산한 COLMAP 결과 폴더(images/, sparse/) — 적은 장수용 MapAnything 등")
    r.add_argument("--out", required=True)
    r.add_argument("--marker-size", type=float, default=None, help="ArUco 마커 검은 테두리 한 변 길이(m)")
    r.add_argument("--marker-dict", default="4x4_50")
    r.add_argument("--gcp", help="gcp.csv: marker_id,x,y,z (현장좌표, m)")
    r.add_argument("--check", help="check.csv: id1,id2,distance_m (줄자 실측)")
    r.add_argument("--no-gcp-ba", action="store_true", help="GCP 를 번들조정에 넣지 않고 유사변환만")
    r.add_argument("--scale-dist", help="scale.csv: id1,id2,distance_m — 축척 기준으로 쓸 줄자 실측(길수록 정확)")
    r.add_argument("--quality", default="medium", choices=["draft", "medium", "high", "ultra"])
    r.add_argument("--mapper", default="incremental", choices=["incremental", "global"])
    r.add_argument("--max-side", type=int, default=3200, help="입력 이미지 긴 변 최대 픽셀")
    r.add_argument("--video-frames", type=int, default=150)
    r.add_argument("--collider-faces", type=int, default=150_000)
    r.add_argument("--features", default="sift", choices=["sift", "sift_hq", "sift_lg", "aliked"],
                   help="sift(기본, 실사 벤치마크에서 가장 균형) / sift_hq(DSP+어파인, 2~7배 느림, 장면따라 ±) / aliked(ALIKED+LightGlue, ONNX 지원 COLMAP 필요)")
    r.add_argument("--no-robust", action="store_true",
                   help="강건 모드 끔(기본 켬: 반복무늬 오등록 방지, 실사 정확도 손실 없음)")
    r.add_argument("--refine-pp", action="store_true",
                   help="광학중심까지 보정(실사 벤치마크: 장면 따라 -30%%~+8%% — 기본 끔)")
    r.add_argument("--profile", choices=["default", "gpu-max", "fast"], default="default",
                   help="fast: GPU 로 3~4분 미리보기(1600px 조밀복원·정제 생략) / gpu-max: 원본 해상도·GPU 특징점/번들조정/조밀복원·ultra 품질 (NVIDIA + CUDA COLMAP 필요)")
    r.add_argument("--dense", choices=["openmvs", "colmap"], default="openmvs",
                   help="조밀복원 엔진: openmvs(CPU 가능) / colmap(PatchMatch, CUDA GPU 필요)")
    r.add_argument("--max-features", type=int, default=12000)
    r.add_argument("--no-fill", dest="fill", action="store_false",
                   help="구멍 메우기 끄기(기본: 주변 형상으로 메운 site_filled.glb + 추정 영역 site_estimated.glb 생성)")
    r.add_argument("--fill", action="store_true", default=True, help=argparse.SUPPRESS)
    r.add_argument("--masks", help="마스크 폴더(<이미지명>.png, 검정=제외) — 움직이는 사람·장비 제거")
    r.add_argument("--gpu", action="store_true", help="CUDA 지원 COLMAP 일 때 GPU 사용")
    r.add_argument("--colmap-bin", default=os.environ.get("COLMAP_BIN"),
                   help="COLMAP 실행파일 경로(윈도우 CUDA 릴리스: .../COLMAP.bat). 없으면 pycolmap 사용")
    a = ap.parse_args(argv)
    if a.cmd == "run" and a.profile == "gpu-max":
        # GPU 를 최대한 쓰는 설정 — 명시적으로 준 옵션보다 우선하지 않도록 기본값일 때만 바꾼다
        a.gpu = True
        if a.max_side == 3200:
            a.max_side = 6000
        if a.max_features == 12000:
            a.max_features = 16000
        if a.quality == "medium":
            a.quality = "ultra"
        if a.mapper == "incremental":
            a.mapper = "global"
        if a.video_frames == 150:
            a.video_frames = 400
        if a.dense == "openmvs" and a.colmap_bin:
            a.dense = "colmap"
        if a.collider_faces == 150_000:
            a.collider_faces = 300_000
        if not a.colmap_bin:
            print("경고: --colmap-bin(CUDA COLMAP)이 없어 특징점·조밀복원은 CPU 로 돈다")
    if a.cmd == "run" and a.profile == "fast":
        a.gpu = True
        if a.max_side == 3200:
            a.max_side = 2400
        if a.max_features == 12000:
            a.max_features = 8000
        if a.quality == "medium":
            a.quality = "draft"
        if a.mapper == "incremental":
            a.mapper = "global"
        # 조밀복원은 COLMAP PatchMatch(GPU). OpenMVS v2.4.0 CUDA 빌드의 DensifyPointCloud 는
        # RTX 50(SM 12.0)에서 GPU 계산 시작 시 종료(PC 확인) → 메쉬·텍스처만 OpenMVS(CPU판)
        if a.dense == "openmvs" and a.colmap_bin:
            a.dense = "colmap"
    if a.cmd == "run" and not a.inputs and not a.from_colmap:
        ap.error("사진/동영상 입력 또는 --from-colmap 이 필요합니다")
    if a.cmd == "run":
        run(a)
