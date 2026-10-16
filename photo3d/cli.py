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

    stage_t = []

    def log(msg):
        if len(msg) > 3 and msg[0].isdigit() and msg[1:3] == "/5":
            stage_t.append((msg, time.time() - t0))
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    rep: dict = {"version": "0.1.0", "args": vars(a)}
    # GPU 상태를 처음에 기록하고, VRAM 크기에 맞춰 조밀복원·스플랫 설정을 정한다
    from . import gpu
    try:
        g = gpu.query()
    except Exception:
        g = None
    log(gpu.summary_line(g))
    plan = gpu.vram_plan(g.get("mem_free_mb"), g.get("mem_total_mb")) if g and g.get("mem_total_mb") else None
    if plan:
        os.environ["PHOTO3D_PM_MAX"] = str(plan["pm_max"])
        # Postshot 식 품질 단계: 빠르게 = VRAM 상한의 절반(미리보기), 최고 품질 = VRAM 상한까지
        cap = plan["splats"] if a.profile == "gpu-max" else max(500_000, plan["splats"] // 2)
        plan["splats"] = cap
        os.environ["PHOTO3D_MAX_SPLATS"] = str(cap)
        os.environ["PHOTO3D_MOGE"] = plan["moge"]
        os.environ["PHOTO3D_MOGE_SIDE"] = str(plan["moge_side"])
        if a.splat_res == 1920:
            a.splat_res = plan["splat_res"]
            if a.profile == "gpu-max" and plan["vram_gb"] >= 7:  # 최고 품질: 학습 해상도↑(선명도), 8GB 에서 2560px
                a.splat_res = max(a.splat_res, 2560)
        log(f"  VRAM {plan['vram_gb']}GB 맞춤: 조밀복원 최대 {plan['pm_max']}px · 스플랫 최대 {plan['splats']:,}개 · "
            f"학습 해상도 {a.splat_res}px · 깊이 모델 {plan['moge']}" + (" (다른 프로그램이 VRAM 사용 중 → 낮춤)" if plan.get("busy") else ""))
    rep["gpu"] = {"info": g, "plan": plan}
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
        if not a.gcp:  # 입력 폴더 안의 측량 기준점 파일 자동 사용(gcp_list.txt)
            for p_ in a.inputs:
                for cand in ([Path(p_)] if Path(p_).is_file() else list(Path(p_).rglob("gcp*.txt"))):
                    if cand.suffix == ".txt" and scale.is_gcp_list(cand):
                        a.gcp = str(cand)
                        log(f"  기준점 파일 자동 사용: {cand.name}")
                        break
        img_dir = work / "images"
        prep = frames.prepare_inputs([Path(p) for p in a.inputs], img_dir, max_side=a.max_side,
                                     video_frames=a.video_frames, log=log)
        rep["input"] = {"images": len(prep["images"]), "sequential": prep["sequential"],
                        "skipped_blurry": prep["skipped_blurry"]}

        log("2/5 SfM (카메라 자세·렌즈 보정)")
        if prep.get("pano_views"):
            log(f"  360° 사진/영상에서 원근 사진 {prep['pano_views']}장 생성")
        rec, st = sfm.run_sfm(img_dir, work / "sfm", sequential=prep["sequential"],
                              single_camera=prep.get("single_camera", True),
                              camera_model=prep.get("camera_model") or "OPENCV",
                              camera_params=prep.get("camera_params"), rig=prep.get("rig"),
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
    side = {}  # 동시에 도는 작업: 메운 모델(CPU 별도 프로세스), 스플랫(GPU)

    def on_dense_ready(dense_dir: Path, dense_ply: Path):
        import subprocess
        import sys
        if a.fill:
            cn = work / "cams.npy"
            np.save(cn, cams if not geo["method"].startswith("none") else np.zeros((0, 3)))
            fl = open(work / "fill.log", "w", encoding="utf-8")
            side["fill"] = (subprocess.Popen([sys.executable, "-m", "photo3d", "fill-stage", str(dense_ply), str(out),
                                              str(work), str(cn), "1" if a.dense == "colmap" else "0"],
                                             stdout=fl, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"}), fl)
            log("  ▶ 구멍 메운 모델: 별도 CPU 프로세스로 동시에 시작")
        if a.splat_steps > 0:
            try:
                from . import splat
                root = Path(__file__).resolve().parents[1]
                sp_exe = splat.find_spirula(root, log=log) if a.splat_engine in ("auto", "spirula") else None
                if sp_exe:
                    side["splat"] = splat.start_splat_spirula(dense_dir, sp_exe, steps=a.splat_steps,
                                                              max_res=a.splat_res, log=log)
                    eng = "Spirula"
                else:
                    brush = splat.find_brush(root, log=log)
                    side["splat"] = splat.start_splat(dense_dir, brush, steps=a.splat_steps, max_res=a.splat_res,
                                                      log=log) if brush else None
                    eng = "Brush"
                if side.get("splat"):
                    side["splat_t0"] = time.time()
                    log(f"  ▶ 가우시안 스플래팅: GPU 로 동시에 시작({eng}, {a.splat_steps}스텝)")
                else:
                    side.pop("splat", None)
            except Exception as e:
                log(f"  스플래팅 시작 실패(메쉬 결과는 정상): {e}")

    res = mvs.densify_and_mesh(rec, img_dir, work, quality=a.quality, engine=a.dense,
                               colmap_bin=a.colmap_bin, on_dense_ready=on_dense_ready, prior=a.prior, log=log)
    if a.fill and "fill" not in side and res.get("dense_ply"):  # CPU(OpenMVS) 경로: 콜백 없음 → 지금 시작
        on_dense_ready(work / "dense", res["dense_ply"])

    rep["dense"] = res.get("stats")
    log("5/5 정리·유니티 내보내기")
    info = export.export_all(res["textured_obj"], out, cams if not geo["method"].startswith("none") else None,
                             collider_faces=a.collider_faces, meta={"georef": geo, "sfm": st}, log=log)
    rep["model"] = info
    try:  # 촬영 위치(뷰어·유니티에서 '어디서 찍었나' 표시) — GLB 와 같은 Y-up 좌표
        camj = []
        for iid in rec.reg_image_ids():
            im = rec.image(iid)
            pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            Rm = pose.matrix()[:, :3]
            c = export.ZUP_TO_YUP[:3, :3] @ im.projection_center()
            fwd = export.ZUP_TO_YUP[:3, :3] @ (Rm.T @ np.array([0.0, 0.0, 1.0]))
            up = export.ZUP_TO_YUP[:3, :3] @ (Rm.T @ np.array([0.0, -1.0, 0.0]))
            camj.append({"name": im.name, "pos": [round(float(v), 4) for v in c],
                         "fwd": [round(float(v), 4) for v in fwd], "up": [round(float(v), 4) for v in up]})
        with open(out / "cameras.json", "w", encoding="utf-8") as f:
            json.dump({"frame": "gltf_yup_m", "cameras": camj}, f, ensure_ascii=False)
    except Exception as e:
        log(f"  촬영 위치 저장 생략: {e}")
    if "fill" in side:
        log("  구멍 메운 모델 완료 대기…")
        proc, fl = side["fill"]
        rc = proc.wait()
        fl.close()
        fr = out / "fill_result.json"
        if rc == 0 and fr.exists():
            rep["fill"] = json.loads(fr.read_text(encoding="utf-8"))
            log(f"  구멍 메운 모델 완료 (추정 영역 {rep['fill']['estimated_area_ratio']:.0%})")
        else:
            tail = (work / "fill.log").read_text(encoding="utf-8", errors="replace").splitlines()[-4:]
            log("  구멍 메우기 실패: " + " | ".join(tail))
    if "splat" in side:
        log("  가우시안 스플래팅 완료 대기…")
        try:
            from . import splat
            # 빠른 모드: 본 처리가 끝난 뒤 최대 5분만 더 기다리고 중간본 사용(최고 품질은 끝까지)
            rep["splat"] = splat.finish_splat(side["splat"], out, log=log,
                                              max_wait=300 if a.profile == "fast" else None)
            rep["splat"]["minutes"] = round((time.time() - side["splat_t0"]) / 60, 1)
        except Exception as e:
            log(f"  스플래팅 실패(메쉬 결과는 정상): {e}")
    rep["elapsed_min"] = round((time.time() - t0) / 60, 1)
    ends = [t for _, t in stage_t[1:]] + [time.time() - t0]
    rep["timing_s"] = {name: round(e - b, 1) for (name, b), e in zip(stage_t, ends)}
    rep["grade"] = grade(rep)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    try:
        from . import report
        report.write(out, rep, work, log=log)
    except Exception as e:
        log(f"  품질 보고서 생략: {e}")
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
    if g.get("method") == "gps_exif":
        u = g.get("scale_uncertainty_pct", 99)
        lvl = "B" if u < 0.3 else "C"
        reason = f"사진 GPS 로 축척(추정 축척 오차 {u}%) — 정밀 측정은 기준점·마커·줄자 거리 필요"
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
    fs = sub.add_parser("fill-stage", help=argparse.SUPPRESS)  # 내부용: 메운 모델을 별도 프로세스로
    fs.add_argument("dense_ply"); fs.add_argument("out"); fs.add_argument("work"); fs.add_argument("cams")
    fs.add_argument("bake")
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
    r.add_argument("--splat-steps", type=int, default=-1,
                   help="가우시안 스플래팅 학습 스텝(0=끔). 기본: fast 7000 / gpu-max 30000 / 그 외 끔")
    r.add_argument("--no-prior", dest="prior", action="store_false",
                   help="깊이 prior(MoGe-2)로 무늬 없는 벽·바닥 채우기 끄기 (기본: GPU 경로에서 켬)")
    r.add_argument("--splat-engine", choices=["auto", "spirula", "brush"],
                   default=os.environ.get("PHOTO3D_SPLAT_ENGINE", "auto"),
                   help="스플랫 학습 프로그램: auto(Spirula 가 있으면 Spirula, 없으면 Brush)")
    r.add_argument("--splat-res", type=int, default=1920, help="스플래팅 학습 이미지 최대 변 길이(px)")
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
    if a.cmd == "run" and a.splat_steps < 0:
        a.splat_steps = {"fast": 10000, "gpu-max": 30000}.get(a.profile, 0)
    if a.cmd == "run" and not a.inputs and not a.from_colmap:
        ap.error("사진/동영상 입력 또는 --from-colmap 이 필요합니다")
    if a.cmd == "fill-stage":
        from . import fill
        def flog(m):
            print(m, flush=True)
        fill.run_stage(Path(a.dense_ply), Path(a.out), Path(a.work), Path(a.cams), a.bake == "1", log=flog)
        return
    if a.cmd == "run":
        run(a)
