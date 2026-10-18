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


def _jsonable(o):
    """체크포인트 저장용: numpy 값을 JSON 으로"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def run(a) -> dict:
    from . import export, fewshot, frames, mvs, scale, sfm

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
        # Spirula 는 압축 학습이라 같은 VRAM 에 더 많이(README: 8GB 에 1,000만). 최고 품질은 Spirula 의
        # quality=high/ultra 기준(300만·5만 스텝 / 1,000만·8만 스텝)으로 더 잘게 쪼개 세밀하게
        # 8GB 는 Spirula 자체 quality=high(300만·5만 스텝)와 같게: 500만으로 올렸더니 오히려 화질이 떨어졌다는
        # 사용자 보고(Spirula: "개수만 늘리고 학습을 그만큼 안 늘리면 다듬어지지 않은 스플랫만 늘어난다")
        if a.profile == "gpu-max":
            sp_cap, sp_steps = (10_000_000, 80_000) if plan["vram_gb"] >= 11 else (3_000_000, 50_000)
            os.environ["PHOTO3D_SPIRULA_CAP"] = str(sp_cap)
            os.environ["PHOTO3D_SPIRULA_STEPS"] = str(sp_steps)
            os.environ["PHOTO3D_SPIRULA_RES"] = "3200"
        elif a.profile == "instant":  # 아주 빠르게: 형태 확인용 — 30만 개·3천 스텝·학습 1000px(몇 분 안)
            os.environ["PHOTO3D_SPIRULA_CAP"] = "300000"
            os.environ["PHOTO3D_SPIRULA_STEPS"] = "3000"
            os.environ["PHOTO3D_SPIRULA_RES"] = "1000"
            os.environ["PHOTO3D_SPIRULA_SAVE"] = "1000"
            os.environ["PHOTO3D_SPIRULA_GEO"] = "0"
        else:  # 빠르게: 더 많이 학습(1만5천 스텝)하되 본 처리 끝난 뒤 최대 5분만 기다리고 마지막 저장본 사용
            os.environ["PHOTO3D_SPIRULA_CAP"] = str(max(1_000_000, cap))
            os.environ["PHOTO3D_SPIRULA_STEPS"] = "7000"
            os.environ["PHOTO3D_SPIRULA_SAVE"] = "1000"
            os.environ["PHOTO3D_SPIRULA_GEO"] = "0"
        os.environ["PHOTO3D_MOGE"] = plan["moge"]
        os.environ["PHOTO3D_MOGE_SIDE"] = str(plan["moge_side"])
        if a.splat_res == 1920:
            a.splat_res = plan["splat_res"]
            if a.profile == "gpu-max" and plan["vram_gb"] >= 7:  # 최고 품질: 학습 해상도↑(선명도), 8GB 에서 2560px
                a.splat_res = max(a.splat_res, 2560)
        log(f"  VRAM {plan['vram_gb']}GB 맞춤: 조밀복원 최대 {plan['pm_max']}px · 스플랫 최대 {plan['splats']:,}개 · "
            f"학습 해상도 {a.splat_res}px · 깊이 모델 {plan['moge']}" + (" (다른 프로그램이 VRAM 사용 중 → 낮춤)" if plan.get("busy") else ""))
    rep["gpu"] = {"info": g, "plan": plan}
    ck = work / "ckpt.json"  # 이어서 처리: 3단계(축척·좌표)까지 끝났으면 그 결과부터 다시 시작
    resumed = False
    if getattr(a, "resume", False) and ck.exists() and (work / "sfm" / "georef" / "images.bin").exists():
        try:
            import pycolmap
            c = json.loads(ck.read_text(encoding="utf-8"))
            rec = pycolmap.Reconstruction(work / "sfm" / "georef")
            img_dir = Path(c["img_dir"])
            st, geo = c["sfm"], c["georef"]
            rep["input"] = c.get("input")
            resumed = True
            log(f"이어서 처리: 1~3단계 결과 재사용(사진 {rec.num_reg_images()}장 연결)")
        except Exception as e:
            log(f"이어서 처리 불가 → 처음부터: {e}")
    if resumed:
        pass
    elif a.from_colmap:
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
        rep["input"] = {"images": len(prep["images"]), "sequential": prep["sequential"], "camera_models": prep.get("models"),
                        "skipped_blurry": prep["skipped_blurry"]}

        log("2/5 SfM (카메라 자세·렌즈 보정)")
        if prep.get("pano_views"):
            log(f"  360° 사진/영상에서 원근 사진 {prep['pano_views']}장 생성")
        n_img = len(prep["images"])
        rec = st = None
        if a.profile == "instant" and n_img > INSTANT_PICK and not prep.get("rig"):
            n_img = _pick_for_instant(img_dir, work / "images_unused", log)
        if a.profile == "instant" and n_img <= 15 and not prep.get("rig") and fewshot.find() is not None:
            fs = fewshot.run(img_dir, work / "fewshot", use_gpu=a.gpu, log=log)  # COLMAP 시도 없이 바로(시간 절약)
            if fs is not None:
                rec, st = fs
        mask_dir = Path(a.masks) if a.masks else (Path(prep["mask_dir"]) if prep.get("mask_dir") else None)
        if rec is None and mask_dir is None and not a.subject and not prep.get("rig") and os.environ.get("PHOTO3D_NO_DYNMASK") != "1":
            from . import dynmask  # 작업자·차량 위 특징점은 카메라 계산에서 뺀다(사람 스캔 모드는 제외)
            mask_dir = dynmask.make(img_dir, work / "masks_dynamic", log=log)
        if rec is None:  # 아주 빠르게에서 이미 카메라를 잡았으면 COLMAP 은 건너뛴다
            try:
                rec, st = sfm.run_sfm(img_dir, work / "sfm", sequential=prep["sequential"],
                                      single_camera=prep.get("single_camera", True),
                                      camera_model=prep.get("camera_model") or "OPENCV",
                                      camera_params=prep.get("camera_params"), rig=prep.get("rig"),
                                      mapper=a.mapper, use_gpu=a.gpu,
                                      colmap_bin=a.colmap_bin, features=a.features,
                                      mask_dir=mask_dir,
                                      per_folder=bool(prep.get("per_folder")), refine_pp=a.refine_pp,
                                      robust=not a.no_robust, max_side=a.max_side,
                                      max_features=a.max_features, log=log)
            except Exception as e:
                if n_img > fewshot.MAX_IMAGES or fewshot.find() is None:
                    raise
                log(f"  COLMAP 카메라 계산 실패({str(e)[:120]}) → 적은 장수 방식으로")
        # 사진이 적어(≤40장) COLMAP 이 일부만 연결하면 MapAnything 으로 시작값을 잡아 다시(8장 시험: COLMAP 2/8장 → 8/8장)
        if n_img <= fewshot.MAX_IMAGES and (st is None or st["images_registered"] < 0.8 * n_img) and not prep.get("rig") \
                and not (st or {}).get("fewshot"):
            fs = fewshot.run(img_dir, work / "fewshot", use_gpu=a.gpu, log=log)
            if fs is not None and (st is None or fs[1]["images_registered"] > st["images_registered"]):
                rec, st = fs
        if rec is None:
            raise RuntimeError("카메라 위치를 계산하지 못했습니다 — 사진 사이 겹침을 늘리거나(한 장씩 옆으로 조금씩) "
                               "적은 장수 보조(scripts/gpu/setup_mapanything_windows.ps1)를 설치하세요")
    rep["sfm"] = st

    if not resumed:
        log("3/5 축척·좌표 정합")
        geo = scale.georeference(rec, img_dir, a.marker_size, a.marker_dict,
                                 Path(a.gcp) if a.gcp else None,
                                 Path(a.check) if a.check else None,
                                 scale_csv=Path(a.scale_dist) if a.scale_dist else None,
                                 gcp_ba=not a.no_gcp_ba, log=log)
        if geo.get("method") == "none_leveled" and st.get("fewshot"):
            # 마커가 없어도 적은 장수 경로(MapAnything)는 미터 단위를 추정한다 — 축척 1 로 그대로 두었으므로 그 값을 쓴다
            geo["method"] = "ai_metric_estimate"
            geo.setdefault("warnings", []).append("크기는 AI(MapAnything) 추정 미터 — 수~10% 오차 가능(검증 전). "
                                                  "치수가 중요하면 마커나 줄자 거리 하나를 넣으세요")
            log("  축척: 마커 없음 → AI 추정 미터 사용(개략 치수)")
        (work / "sfm" / "georef").mkdir(parents=True, exist_ok=True)
        rec.write(work / "sfm" / "georef")
        ck.write_text(json.dumps({"stage": 3, "img_dir": str(img_dir), "sfm": st, "georef": geo,
                                  "input": rep.get("input")}, ensure_ascii=False, default=_jsonable),
                      encoding="utf-8")
    rep["georef"] = geo
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
        if a.splat_steps > 0 and a.pipeline != "fuse":  # 통합은 측정 메쉬가 나온 뒤 그 지도를 넣어 시작
            start_splat(dense_dir)

    def start_splat(dense_dir: Path, prior_dir: Path | None = None):
        try:
            from . import splat
            root = Path(__file__).resolve().parents[1]
            sp_exe = splat.find_spirula(root, log=log) if a.splat_engine in ("auto", "spirula") else None
            if sp_exe:
                side["splat"] = splat.start_splat_spirula(dense_dir, sp_exe, steps=a.splat_steps,
                                                          max_res=a.splat_res, log=log, subject=a.subject,
                                                          prior_dir=prior_dir)
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

    ai_dense = None
    if (st or {}).get("fewshot") and rec.num_reg_images() <= 15 and a.dense == "colmap":
        # 사진이 아주 적으면 COLMAP 조밀복원(여러 사진 일치)이 거의 비어 나온다(8장 45° 간격: 3,584점)
        # → MapAnything 이 사진마다 예측한 깊이 점을 최종 좌표로 옮겨 대신 쓴다(카메라 위치끼리 Sim3 정합)
        (work / "dense").mkdir(parents=True, exist_ok=True)
        ai_dense = fewshot.dense_points(work / "fewshot", rec, work / "ai_dense.ply", log=log)
    res = mvs.densify_and_mesh(rec, img_dir, work, quality=a.quality, engine=a.dense,
                               colmap_bin=a.colmap_bin, on_dense_ready=on_dense_ready, prior=a.prior and ai_dense is None,
                               subject=a.subject, resume=resumed, strict=a.pipeline in ("pg", "fuse"),
                               dense_points=ai_dense, log=log)
    if a.fill and "fill" not in side and res.get("dense_ply"):  # CPU(OpenMVS) 경로: 콜백 없음 → 지금 시작
        on_dense_ready(work / "dense", res["dense_ply"])
    if a.pipeline == "fuse" and a.splat_steps > 0 and "splat" not in side:
        # 통합: 측정 메쉬를 사진마다 그려 표면 방향(·깊이) 지도로 스플랫 학습에 넣는다 → 메쉬 정리·내보내기와 동시에 GPU 학습
        from . import fuse
        fm = fuse.pick_mesh(work / "mvs")
        prior = None
        if fm is not None and a.subject is False:
            try:
                prior = work / "fuse_maps"
                rep["fuse"] = {"mesh": fm.name, "maps": fuse.write_prior_maps(
                    fm, work / "dense" / "sparse", prior, depth=os.environ.get("PHOTO3D_FUSE_DEPTH") == "1", log=log)}
            except Exception as e:
                log(f"  통합 지도 생략(스플랫은 그대로 학습): {e}")
                prior = None
        start_splat(work / "dense", prior)

    rep["dense"] = res.get("stats")
    log("5/5 정리·유니티 내보내기")
    dense_pts = work / "dense" / "fused.ply" if (work / "dense" / "fused.ply").exists() else res.get("dense_ply")
    floor = {}

    def add_floor(mesh):
        """포토그래메트리: 비스듬히 찍혀 측정이 빈 바닥을 사진 정사영상 바닥으로 메움"""
        if a.pipeline not in ("pg", "fuse") or a.subject or not dense_pts:
            return []
        from .planes import floor_patch
        fm, fi = floor_patch(Path(dense_pts), work / "dense" / "sparse", work / "dense" / "images", mesh,
                             out_dir=out, log=log)
        floor.update(fi)
        return [fm] if fm is not None else []
    info = export.export_all(res["textured_obj"], out, cams if not geo["method"].startswith("none") else None,
                             collider_faces=a.collider_faces, meta={"georef": geo, "sfm": st},
                             extra_fn=add_floor, log=log)
    extra = info.pop("_extra", [])
    rep["model"] = info
    if floor:
        rep["floor_patch"] = floor
    main_mesh = None

    def cleaned():
        """GLB 와 같은 정리본 메쉬(바닥 조각 제외, Z-up) — 한 번만 읽는다"""
        nonlocal main_mesh
        if main_mesh is None:
            from .export import _load
            main_mesh = _load(out / "site_model_zup.obj") if not extra else None
            if main_mesh is None:  # 바닥 조각이 붙은 OBJ 는 재료가 둘 → 본 메쉬는 GLB 와 같은 정리본을 다시 읽음
                main_mesh = export.clean_mesh(_load(Path(res["textured_obj"])),
                                              cams if not geo["method"].startswith("none") else None, log=lambda *_: None)
        return main_mesh
    if info["faces"] > a.light_over and (work / "dense" / "sparse").exists():
        try:  # 면이 너무 많으면 보기용 가벼운 모델을 따로(유니티·웹·VR)
            rep["model"]["light"] = export.light_model(cleaned(), work / "dense" / "sparse", work / "dense" / "images",
                                                       out, extra=extra, faces=a.light_faces, log=log)
        except Exception as e:
            log(f"  가벼운 모델 생략: {e}")
    if a.pipeline in ("pg", "fuse"):  # 측량형 결과물: 정사영상·DSM·LAS 점군
        try:
            from . import survey
            m = cleaned()
            zc = survey.plan_cut_height([m], cams if len(cams) else None)
            rep["survey"] = {"ortho": survey.ortho_dsm([m] + extra, out, z_cut=zc, log=log,
                                                       photos=(work / "dense" / "sparse", work / "dense" / "images"))}
            if dense_pts:
                rep["survey"]["points"] = survey.export_points(Path(dense_pts), out, m.bounds, log=log)
        except Exception as e:
            log(f"  측량 결과물(정사영상·DSM·점군) 생략: {e}")
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
                                              max_wait=120 if a.profile in ("fast", "instant") and a.pipeline != "fuse" else None)
            rep["splat"]["minutes"] = round((time.time() - side["splat_t0"]) / 60, 1)
            if a.pipeline == "fuse":  # 측정 표면 뒤에 숨은 스플랫 제거 + 메쉬와 일치도
                from . import fuse
                fm = fuse.pick_mesh(work / "mvs")
                if fm is not None:
                    sp = out / "site_splat.ply"
                    fi = fuse.mesh_prune_and_agree(sp, fm, work / "dense" / "sparse",
                                                   Path(dense_pts) if dense_pts else None, sp, log=log)
                    rep.setdefault("fuse", {})["agree"] = fi
                    # 최적 합성 메쉬: 포토그래메트리 + (그것이 없는 곳만) 스플랫에서 뽑은 메쉬 → 정형화 → 사진 텍스처
                    try:
                        # 형태 메쉬: PGSR(평면 가우시안, 설치돼 있으면 — 가장 정확) → 없거나 실패하면 Spirula 메쉬
                        from . import pgsr
                        gm = None
                        if pgsr.find_pgsr() is not None and a.profile != "instant":
                            gm = pgsr.run(work / "dense", work / "pgsr",
                                          iterations=30000 if a.profile == "gpu-max" else 7000, log=log)
                            if gm is not None:
                                rep["fuse"]["shape_mesh"] = "pgsr"
                        if gm is None:
                            gm = splat.mesh_from_splat(side["splat"], work / "splat" / "gs_mesh", log=log)
                            if gm is not None:
                                rep["fuse"]["shape_mesh"] = "spirula"
                        if gm is not None:
                            dpl = Path(dense_pts) if dense_pts else None
                            mi = fuse.merge_meshes(fm, gm, dpl, work / "mvs" / "scene_mesh_best.ply", log=log)
                            if mi:
                                import trimesh
                                from .texture import bake
                                from .meshsplat import _crop
                                bm = trimesh.load(work / "mvs" / mi["file"], process=False, force="mesh")
                                cut = _crop(bm, work / "dense" / "sparse")  # 하늘·먼 배경 조각
                                if cut:
                                    log(f"  최적 합성 메쉬: 측정 범위 밖 면 {cut:,}개 제거")
                                tm, _ = bake(bm, work / "dense" / "sparse", work / "dense" / "images",
                                             atlas=8192, max_faces=1_200_000, log=log)
                                tm.apply_transform(export.ZUP_TO_YUP)
                                tm.export(out / "site_best_mesh.glb")
                                rep["fuse"]["best_mesh"] = mi
                    except Exception as e:
                        log(f"  최적 합성 메쉬 생략: {e}")
                    # 보이는 모델: MeshSplatting 불투명 메쉬(설치돼 있을 때 — 빠른 모드는 일정을 1/3 로 당겨 10,000스텝)
                    try:
                        from . import meshsplat
                        if meshsplat.find_meshsplat() is not None and a.profile != "instant":
                            mp = meshsplat.run(work / "dense", work / "meshsplat", log=log,
                                               iterations=10_000 if a.profile == "fast" else meshsplat.ITERATIONS)
                            oi = mp and meshsplat.opaque_model(mp, work / "dense", out / "site_opaque_mesh.glb", log=log)
                            if oi:
                                rep["fuse"]["opaque_mesh"] = oi
                                # 빠른 모드(10,000스텝) 결과는 삼각형이 흩어져 아직 거칠다(1Family 확인) → 화면 '불투명 모델'
                                # 로만 보고, 유니티 패키지 보기 모델로는 최적 합성 메쉬를 쓴다
                                if a.profile == "fast":
                                    (out / "site_opaque_mesh.preview").write_text("fast", encoding="utf-8")
                                else:
                                    (out / "site_opaque_mesh.preview").unlink(missing_ok=True)
                    except Exception as e:
                        log(f"  불투명 모델 생략: {e}")
                    rep["splat"]["splats"] = fi["after"]
                    try:
                        from .splatexport import write_spz, write_splat
                        write_spz(sp, out / "site_splat.spz")
                        write_splat(sp, out / "site_splat.splat")
                    except Exception as e:
                        log(f"  가벼운 스플랫 다시 만들기 생략: {e}")
        except Exception as e:
            log(f"  스플래팅 실패(메쉬 결과는 정상): {e}")
        if (out / "site_splat.ply").exists():
            try:  # 최적 합성: 스플랫(형태·실사) + 포토그래메트리 면(속이 꽉 찬 표면) — 뷰어 '최적 합성'
                from . import fuse
                rep["best"] = fuse.make_best(out, work, log=log)
            except Exception as e:
                log(f"  최적 합성 생략: {e}")
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
    try:  # 유니티에서 더블클릭으로 바로 가져가는 패키지(OBJ + 스크립트 + 자동 프리팹)
        from .unitypkg import write_unity_package
        write_unity_package(out, a.name or Path(a.out).resolve().parent.name, log=log)
    except Exception as e:
        log(f"  유니티 패키지 생략: {e}")
    log(f"완료 — 등급 {rep['grade']['grade']}: {rep['grade']['reason']}")
    return rep


INSTANT_PICK = 12  # 아주 빠르게: 사진이 많아도 이만큼만 골라 쓴다(MapAnything 한 번에, 8GB VRAM 에서 1~3분)


def _pick_for_instant(img_dir: Path, spare: Path, log=print) -> int:
    """아주 빠르게: 찍은 순서(이름 순)로 INSTANT_PICK 구간을 나누고 구간마다 가장 선명한 사진 한 장을 고른다
    → 흔들린 사진은 빠지고 현장을 고르게 덮는다. 나머지는 spare 로 옮겨 둔다(지우지 않음). 남은 장수를 돌려준다."""
    import shutil
    from . import frames
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(imgs) <= INSTANT_PICK:
        return len(imgs)
    sc = np.array([frames.sharpness(p) for p in imgs])
    edges = np.linspace(0, len(imgs), INSTANT_PICK + 1).round().astype(int)
    keep = {int(lo + np.argmax(sc[lo:hi])) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo}
    spare.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(imgs):
        if i not in keep:
            shutil.move(str(p), spare / p.name)
    log(f"  아주 빠르게: 사진 {len(imgs)}장 중 고르게 퍼진 선명한 {len(keep)}장만 사용(나머지는 images_unused)")
    return len(keep)


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
    if g["method"] == "ai_metric_estimate":
        return {"grade": "C", "reason": "AI 추정 축척(마커 없음) — 개략 배치·위험성 검토용, 정밀 치수는 마커·줄자 필요"}
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
    r.add_argument("--name", default=None, help="현장 이름(유니티 패키지 폴더·프리팹 이름)")
    r.add_argument("--light-over", type=int, default=1_500_000, help="면이 이보다 많으면 보기용 가벼운 모델(site_model_light.glb)도 만듦")
    r.add_argument("--light-faces", type=int, default=600_000, help="가벼운 모델의 면 수")
    r.add_argument("--features", default="sift", choices=["sift", "sift_hq", "sift_lg", "aliked"],
                   help="sift(기본, 실사 벤치마크에서 가장 균형) / sift_hq(DSP+어파인, 2~7배 느림, 장면따라 ±) / aliked(ALIKED+LightGlue, ONNX 지원 COLMAP 필요)")
    r.add_argument("--no-robust", action="store_true",
                   help="강건 모드 끔(기본 켬: 반복무늬 오등록 방지, 실사 정확도 손실 없음)")
    r.add_argument("--refine-pp", action="store_true",
                   help="광학중심까지 보정(실사 벤치마크: 장면 따라 -30%%~+8%% — 기본 끔)")
    r.add_argument("--profile", choices=["default", "gpu-max", "fast", "instant"], default="default",
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
    r.add_argument("--resume", action="store_true",
                   help="멈춘 작업 이어서: 1~3단계 결과·이미 계산한 깊이 지도·조밀 점군을 재사용")
    r.add_argument("--subject", action="store_true",
                   help="사람·물체 모드: 사진마다 배경을 자동으로 빼고(BiRefNet) 대상만 스플랫 학습·자르기")
    r.add_argument("--splat-engine", choices=["auto", "spirula", "brush"],
                   default=os.environ.get("PHOTO3D_SPLAT_ENGINE", "auto"),
                   help="스플랫 학습 프로그램: auto(Spirula 가 있으면 Spirula, 없으면 Brush)")
    r.add_argument("--splat-res", type=int, default=1920, help="스플래팅 학습 이미지 최대 변 길이(px)")
    r.add_argument("--splat-aa", action="store_true",
                   help="Mip-Splatting 안티앨리어싱으로 학습(확대·축소 때 반짝임·계단 줄임, Spirula). 웹 뷰어도 같은 방식으로 그림")
    r.add_argument("--pipeline", choices=["splat", "pg", "fuse"], default="splat",
                   help="splat: 메쉬 + 가우시안 스플래팅(실사 보기) / pg: 포토그래메트리 전용(측정용 메쉬·정사영상·DSM·점군, 스플랫 없음) / fuse: 통합 — 포토그래메트리 + 그 메쉬로 맞춘 스플랫")
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
    if a.cmd == "run" and a.profile == "instant":
        # 아주 빠르게 = 사진 10장 안팎으로 몇 분 안에 형태 확인: 15장 이하면 COLMAP 대신 MapAnything 으로 카메라,
        # 1600px 조밀복원, 깊이 보조·구멍 메우기·PGSR·MeshSplatting 생략, 스플랫 3천 스텝
        a.gpu = True
        if a.max_side == 3200:
            a.max_side = 1600
        a.fill = False
        a.prior = False
        if a.max_features == 12000:
            a.max_features = 8000
        if a.quality == "medium":
            a.quality = "draft"
        if a.dense == "openmvs" and a.colmap_bin:
            a.dense = "colmap"
    if a.cmd == "run" and a.profile == "fast":
        # 빠르게 = 프로토타입(목표 5~10분): 해상도 1600, 구멍 메우기(CPU 별도 처리) 생략, 텍스처 4096,
        # 스플랫 7천 스텝·표면 방향 지도 생략·본 처리 후 최대 2분 대기
        a.gpu = True
        if a.max_side == 3200:
            a.max_side = 1600
        a.fill = False
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
    if a.cmd == "run" and a.pipeline == "pg":
        # 포토그래메트리: GPU 는 스플랫 대신 조밀복원에 모두 쓴다 → 스플랫 끔, 구멍 메운 모델·평면 정리·측량 결과물 켬
        a.splat_steps = 0
        if a.profile == "fast":
            a.fill = True
            if a.max_side == 1600:
                a.max_side = 2400  # 측정용은 빠르게도 해상도를 덜 낮춘다(조밀점·모서리 정확도)
    if a.cmd == "run" and a.pipeline == "fuse":
        # 통합: 포토그래메트리(측정용 엄격 메쉬·측량 결과물) + 그 메쉬로 맞춘 스플랫(보기)
        if a.profile == "fast":
            a.fill = True
            if a.max_side == 1600:
                a.max_side = 2400
    if a.cmd == "run" and a.splat_aa:
        os.environ["PHOTO3D_SPLAT_MIP"] = "1"
    if a.cmd == "run" and a.splat_steps < 0:
        a.splat_steps = {"instant": 3000, "fast": 7000, "gpu-max": 30000}.get(a.profile, 0)
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
