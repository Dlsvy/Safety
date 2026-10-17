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

import numpy as np
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
        # 외부 프로그램 로그의 마지막 줄들을 오류에 붙여 화면에서 바로 원인을 보게 한다
        try:
            txt = Path(logfile).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
            tail = [ln.strip() for ln in txt.splitlines() if ln.strip() and "Estimated depth-maps" not in ln][-6:]
        except OSError:
            tail = []
        raise RuntimeError(f"{Path(args[0]).name} 실패 (종료코드 {r.returncode}, 로그: {logfile})\n"
                           + "\n".join("    " + t[:300] for t in tail))


def _color_from_points(mesh_path: Path, points_path: Path, mvs: Path, log, smooth: int = 8) -> Path:
    """메쉬 정점마다 주변 조밀 점들의 색을 입힌다(텍스처 실패 시 대체).

    정점 색은 정점 간격보다 세밀할 수 없어 각져 보인다 → ① Taubin 스무딩(부피·치수 보존형)으로
    Poisson 표면의 울퉁불퉁함을 줄이고 ② 한 번 세분해 색 해상도를 4배로 ③ 가장 가까운 1점 대신
    가까운 4점만 좁게 거리 가중해 얼룩은 줄이되 선명함은 유지한다.
    """
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    try:
        m = trimesh.load(mesh_path, process=False, force="mesh")
        pc = trimesh.load(points_path, process=False)
        cols = getattr(pc, "colors", None)
        if cols is None or len(cols) == 0:
            cols = pc.visual.vertex_colors
        cols = np.asarray(cols)[:, :3].astype(np.float32)
        if smooth:
            m.merge_vertices()
            trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=smooth)
        if len(m.faces) < 3_000_000:  # 메모리 한도 내에서 색 해상도 ×4
            v, f = trimesh.remesh.subdivide(m.vertices, m.faces)
            m = trimesh.Trimesh(v, f, process=False)
        d, idx = cKDTree(np.asarray(pc.vertices)).query(m.vertices, k=4, workers=-1)
        w = 1.0 / np.maximum(d, 1e-9)
        w[d > 2 * d[:, :1] + 1e-9] = 0  # 너무 먼 점(다른 면)은 제외
        c = (cols[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        rgba = np.concatenate([np.clip(c, 0, 255), np.full((len(c), 1), 255)], 1).astype(np.uint8)
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=rgba)
        out = mvs / "scene_mesh_colored.ply"
        m.export(out)
        log(f"  점군 색 입힘(스무딩 {smooth}회·세분·4점): {len(m.vertices):,}정점")
        return out
    except Exception as e:
        log(f"  점군 색 입히기 실패({e}) → 색 없는 메쉬 사용")
        return mesh_path


def _fix_topology(m, min_faces: int = 50, rounds: int = 5) -> None:
    """면을 지우고 나면 두 부채꼴이 꼭짓점 하나로만 이어진 '나비 정점'과 몇 면짜리 부스러기가 남는다.
    OpenMVS v2.3 TextureMesh 가 여기서 unordered_map::at 으로 죽는다(Pix4D building: 나비 정점 11,598개)."""
    import trimesh
    for _ in range(rounds):
        e = np.sort(m.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), 1)
        u, c = np.unique(e, axis=0, return_counts=True)
        bc = np.bincount(u[c == 1].ravel(), minlength=len(m.vertices))
        bad = bc > 2  # 경계 모서리가 3개 이상 닿은 정점 = 나비
        if not bad.any():
            break
        m.update_faces(~bad[m.faces].any(1))
    cc = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)))
    keep = np.zeros(len(m.faces), bool)
    for x in cc:
        if len(x) >= min_faces:
            keep[x] = True
    m.update_faces(keep)
    m.remove_unreferenced_vertices()


def prune_mesh(mesh_path: Path, points_path: Path, dst: Path, strict: float | None = None, log=print) -> Path:
    """Poisson 메쉬에서 측정점과 동떨어진 면과 바늘처럼 긴 삼각형(가시)을 지운다.
    strict(점 간격의 배수): 측정용(포토그래메트리). 면의 세 꼭짓점·중심이 모두 측정점에서 이 거리 안인 면만 남긴다
    → 측정점 사이를 건너 이은 추정 면 제거. ETH3D delivery_area: 5cm F1 58→71%, 중앙 오차 62→26mm(완전도 78→66%)."""
    import trimesh
    from scipy.spatial import cKDTree
    try:
        m = trimesh.load(mesh_path, process=False, force="mesh")
        from .fill import read_points  # COLMAP fused.ply·OpenMVS 점군 모두
        pts = np.asarray(read_points(points_path)[0], float)
        tree = cKDTree(pts)
        sp = float(np.median(tree.query(pts[:: max(1, len(pts) // 20000)], k=2)[0][:, 1]))
        d, _ = tree.query(m.triangles_center, workers=-1)
        el = np.linalg.norm(m.vertices[m.faces] - m.vertices[np.roll(m.faces, 1, 1)], axis=2).max(1)
        thin = el ** 2 / np.maximum(2 * m.area_faces, 1e-12)
        far = d > 20 * sp  # 가시는 수십~수백 배 떨어져 있다(6배는 점이 성긴 실제 표면까지 지움)
        if strict:
            # (점 밀도에 맞춰 성긴 곳은 4배까지 허용해 봤으나 ETH3D delivery_area 5cm F1 71.5→61.0 으로 나빠져 되돌림:
            #  성긴 곳을 건너 이은 면은 부정확하다 → 구멍은 그대로 두고 '구멍 메운 모델'(추정 표시)로 보여준다)
            dv, _ = tree.query(m.vertices, workers=-1)
            far |= np.maximum(dv[m.faces].max(1), d) > strict * sp
        needle = (thin > 12) & (el > 8 * np.median(el))
        n0 = len(m.faces)
        m.update_faces(~(far | needle))
        m.remove_unreferenced_vertices()
        if strict:
            _fix_topology(m)
        m.export(dst)
        log(f"  메쉬 가시 정리{'(측정용 엄격)' if strict else ''}: {n0:,} → {len(m.faces):,}면 "
            f"(측정점과 먼 면 {int(far.sum()):,}, 바늘 {int(needle.sum()):,})")
        return dst
    except Exception as e:
        log(f"  메쉬 가시 정리 생략({e})")
        return mesh_path


def subject_masks(dense: Path, log=print) -> Path | None:
    """dense/images 의 사진마다 주인공 마스크(BiRefNet). COLMAP 이름 규칙(<사진 이름>.png)과
    Spirula·Brush 규칙(<확장자 뺀 이름>.png)을 둘 다 쓴다. 실패하면 None(마스크 없이 진행)."""
    try:
        from . import splat
        exe = splat.find_spirula(Path(__file__).resolve().parents[1], log=log)
        if not exe:
            log("  배경 분리 생략: Spirula 없음")
            return None
        log("  사람·물체 모드: 사진마다 배경 자동 분리(BiRefNet)")
        tmpd = dense / "_subject"
        shutil.rmtree(tmpd, ignore_errors=True)
        (tmpd / "images").mkdir(parents=True)
        for q in (dense / "images").iterdir():  # make_subject_masks 는 <폴더>/images 를 읽는다
            try:
                os.link(q, tmpd / "images" / q.name)
            except OSError:
                shutil.copy2(q, tmpd / "images" / q.name)

        class _L:
            def write(self, m):
                if "완료" in m:
                    log("  " + m.strip())

            def flush(self):
                pass
        n = splat.make_subject_masks(exe, tmpd, _L())
        if not n:
            return None
        mdir = dense / "masks"
        shutil.rmtree(mdir, ignore_errors=True)
        mdir.mkdir()
        names = {q.stem: q.name for q in (dense / "images").iterdir()}
        for m in (tmpd / "masks").iterdir():
            shutil.copy2(m, mdir / f"{names.get(m.stem, m.stem)}.png")
        shutil.copytree(tmpd / "masks", dense / "masks_stem", dirs_exist_ok=True)
        shutil.rmtree(tmpd, ignore_errors=True)
        return mdir
    except Exception as e:
        log(f"  배경 분리 실패(마스크 없이 진행): {e}")
        return None


def add_prior_points(fused: Path, dense: Path, log=print) -> int:
    """조밀 점군(fused.ply)에 깊이 prior 로 채운 점을 더해 같은 파일로 저장. 실패해도 원래 점군 유지."""
    try:
        from . import priorfill
        from .fill import _write_ply, read_points
        from .pointfilter import LAST
        if not priorfill.ensure_installed(log):
            log("  깊이 prior 설치 실패 → 건너뜀")
            LAST["prior_status"] = "설치 실패(PyTorch/MoGe) — 처리 기록 확인"
            return 0
        xyz, nrm, rgb = read_points(fused)
        pr = priorfill.fill(xyz, rgb, dense / "sparse", dense / "images", log=log)
        n = len(pr["xyz"])
        if n:
            if nrm is None:
                nrm = np.zeros_like(xyz)
            _write_ply(fused, np.vstack([xyz, pr["xyz"]]), np.vstack([nrm, pr["normal"]]),
                       np.vstack([rgb if rgb is not None else np.full((len(xyz), 3), 160, np.uint8), pr["rgb"]]))
            LAST["prior_added"] = int(n)
        LAST["prior_status"] = f"사진 {pr.get('views_used', 0)}장 사용, {n:,}점 추가" if n else \
            f"추가된 점 없음(측정과 맞는 사진 {pr.get('views_used', 0)}장)"
        return n
    except Exception as e:
        log(f"  깊이 prior 생략: {e}")
        from .pointfilter import LAST
        LAST["prior_status"] = f"오류: {str(e)[:80]}"
        return 0


def densify_and_mesh(rec: pycolmap.Reconstruction, image_dir: Path, work: Path,
                     quality: str = "medium", threads: int = 0, engine: str = "openmvs",
                     colmap_bin: str | None = None, on_dense_ready=None, prior: bool = True,
                     subject: bool = False, resume: bool = False, strict: bool = False, log=print) -> dict:
    q = QUALITY[quality]
    sparse = work / "sparse_georef"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    rec.write_text(sparse)  # OpenMVS InterfaceCOLMAP 은 텍스트 모델을 확실히 읽는다

    dense = work / "dense"
    if resume and (dense / "undistort.done").exists():
        log("  이어서 처리: 왜곡보정 이미지 재사용")
    else:
        if dense.exists():
            shutil.rmtree(dense)
        log("  왜곡보정 이미지 생성")
        pycolmap.undistort_images(dense, sparse, image_dir)
        # undistort 결과 sparse 를 텍스트로도 기록
        und = pycolmap.Reconstruction(dense / "sparse")
        und.write_text(dense / "sparse")
        (dense / "undistort.done").write_text("ok")

    mvs = work / "mvs"
    mvs.mkdir(exist_ok=True)
    logf = work / "openmvs.log"
    common = ["-w", str(mvs), "-v", "2"] + (["--max-threads", str(threads)] if threads else [])
    if engine != "colmap":
        _run([_bin("InterfaceCOLMAP"), "-i", str(dense), "-o", str(mvs / "scene.mvs"),
              "--image-folder", str(dense / "images")] + common, mvs, log, logf)
    if engine == "colmap":
        # GPU 조밀복원: COLMAP PatchMatchStereo(CUDA) → StereoFusion → OpenMVS 로 가져와 메쉬·텍스처
        if not colmap_bin:
            raise RuntimeError("--dense colmap 은 CUDA 지원 COLMAP 실행파일(--colmap-bin)이 필요합니다")
        # 해상도: VRAM 맞춤 상한(gpu.vram_plan, 8GB → 4000px) 안에서 품질별로. 메모리 부족으로 실패하면
        # 한 단계씩 낮춰 다시(남는 VRAM 을 최대한 쓰되 처리는 실패하지 않게)
        cap = int(os.environ.get("PHOTO3D_PM_MAX", 3200))
        size = {"draft": 2000 if cap >= 3200 else 1600, "medium": 2400, "high": 3200, "ultra": cap}[quality]
        size = min(size, cap)
        fast = quality == "draft"
        from .gpu import ram_gb
        cache = str(max(4, int(ram_gb() * 0.5)))  # 깊이 지도 캐시는 PC 메모리(RAM)의 절반까지
        pm_size = dense / "pm_size.txt"
        if resume and pm_size.exists():  # 이어서: 끊기기 전과 같은 해상도로(이미 계산한 깊이 지도는 COLMAP 이 건너뜀)
            size = int(pm_size.read_text())
        ladder = [size] + [s_ for s_ in (int(size * 0.8), int(size * 0.6)) if s_ >= 1200]
        if resume and (dense / "pm.done").exists():
            ladder = []
            log(f"  이어서 처리: 깊이 계산 완료분 재사용({size}px)")
        for k, size in enumerate(ladder):
            keep = resume and k == 0 and pm_size.exists()
            if keep:
                n_done = len(list((dense / "stereo" / "depth_maps").glob("*.geometric.bin")))
                log(f"  이어서 처리: 깊이 지도 {n_done}장은 건너뛰고 나머지 계산")
            for sub in ("depth_maps", "normal_maps", "consistency_graphs"):
                if not keep:
                    shutil.rmtree(dense / "stereo" / sub, ignore_errors=True)
                (dense / "stereo" / sub).mkdir(parents=True, exist_ok=True)
            pm_size.write_text(str(size))
            try:
                # 형상 일관성(geom_consistency)은 빠른 모드에서도 켠다: 끄면 하늘·그늘·비스듬한 지면의 틀린 깊이가
                # 걸러지지 않아 카메라 쪽으로 뻗는 '가시'가 생긴다(PC castle-P19). 빠른 모드는 반복만 줄인다.
                _run([colmap_bin, "patch_match_stereo", "--workspace_path", str(dense),
                      "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "1",
                      "--PatchMatchStereo.max_image_size", str(size), "--PatchMatchStereo.gpu_index", "0",
                      "--PatchMatchStereo.filter_min_ncc", "0.15", "--PatchMatchStereo.cache_size", cache]
                     + (["--PatchMatchStereo.num_iterations", "4", "--PatchMatchStereo.window_step", "2"] if fast else []),
                     mvs, log, logf)
                (dense / "pm.done").write_text(str(size))
                break
            except RuntimeError as e:
                if k == len(ladder) - 1:
                    raise
                log(f"  조밀복원 {size}px 실패 → {ladder[k + 1]}px 로 다시 ({str(e).splitlines()[-1].strip()[:120]})")
        log(f"  조밀복원 해상도 {size}px")
        fused = dense / "fused_raw.ply"
        fused_done = resume and (dense / "fused.done").exists() and (dense / "fused.ply").exists()
        if fused_done:
            fused = dense / "fused.ply"
            log("  이어서 처리: 조밀 점군 재사용")
        else:
            mask_args = []
            if subject:  # 사람·물체 모드: 배경 분리 마스크 안쪽만 합쳐 대상만 메쉬로(배경 덩어리·번짐 방지)
                mdir = subject_masks(dense, log)
                if mdir:
                    mask_args = ["--StereoFusion.mask_path", str(mdir)]
                    prior = False  # 흰 벽 채우기는 배경용이라 끔
            _run([colmap_bin, "stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP", *mask_args,
                  "--input_type", "geometric", "--StereoFusion.max_image_size", str(size),
                  "--StereoFusion.cache_size", cache,
                  "--StereoFusion.min_num_pixels", "4", "--StereoFusion.max_reproj_error", "1.5",
                  "--StereoFusion.max_depth_error", "0.008", "--StereoFusion.max_normal_error", "8",
                  "--output_path", str(fused)], mvs, log, logf)
            from .pointfilter import clean_ply
            cams = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])
            fused = clean_ply(fused, dense / "fused.ply", cams, log=log, work=dense.parent)
            if fused.name != "fused.ply":
                shutil.copy2(fused, dense / "fused.ply")
                fused = dense / "fused.ply"
            if prior:  # 무늬 없는 벽·바닥 채우기(MoGe-2 깊이 prior, 측정과 맞춘 뒤 다시점 일치만)
                add_prior_points(fused, dense, log)
            (dense / "fused.done").write_text("ok")
        # (OpenMVS 장면 변환은 이 경로에서 쓰지 않음 — 시간 절약)
        shutil.copy2(fused, mvs / "scene_dense.ply")
        if on_dense_ready:  # 점군 완성 → 메운 모델(CPU 별도 프로세스)·스플랫(GPU)을 지금 동시에 시작
            on_dense_ready(dense, mvs / "scene_dense.ply")
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
        mesh = prune_mesh(mesh, dense / "fused.ply", mvs / "scene_mesh_pruned.ply", strict=3.0 if strict else None,
                          log=log)
    else:
        _run([_bin("ReconstructMesh"), str(dense_scene),
              "-o", str(mvs / "scene_mesh.mvs")] + common, mvs, log, logf)
        mesh = mvs / "scene_mesh.ply"
        if strict:  # 측정용: 측정점 사이를 건너 이은 면 제거(텍스처 전에)
            mesh = prune_mesh(mesh, mvs / "scene_dense.ply", mvs / "scene_mesh_pruned.ply", strict=3.0, log=log)
    if q["refine"] and engine != "colmap":
        try:
            _run([_bin("RefineMesh"), str(dense_scene), "--mesh-file", str(mesh),
                  "--resolution-level", str(q["res"] + 1), "--max-face-area", "16",
                  "-o", str(mvs / "scene_refined.mvs")] + common, mvs, log, logf)
            if (mvs / "scene_refined.ply").exists():
                mesh = mvs / "scene_refined.ply"
        except RuntimeError as e:
            log(f"  메쉬 정제 건너뜀: {e}")
    if not subject:
        # 건물 정형화: 평평한 벽·바닥은 평평하게(수직·수평·주축), 꺾이는 곳은 날카롭게, 창대·돌출부 깊이는 유지
        # (예전 planarize_mesh: 평면 10개·1.5 점간격 안만 → 대부분 안 맞춰졌고 OpenMVS 경로는 안 돌았음)
        from .regularize import regularize_mesh
        pts = dense / "fused.ply" if engine == "colmap" else mvs / "scene_dense.ply"
        mesh = regularize_mesh(mesh, pts, mvs / "scene_mesh_planar.ply", log=log)
    if engine == "colmap":
        # Windows OpenMVS TextureMesh 는 검은 텍스처(PC 3회) → 직접 구현한 사진 텍스처 굽기 사용
        colored = _color_from_points(mesh, dense / "fused.ply", mvs, log)
        tex_info = None
        try:
            import trimesh
            from .texture import bake
            atlas = {"draft": 4096, "medium": 8192, "high": 8192, "ultra": 8192}[quality]  # 빠르게: 굽기 시간 ↓
            maxf = {"draft": 250_000, "medium": 900_000, "high": 1_200_000, "ultra": 1_500_000}[quality]
            tm, _ = bake(trimesh.load(colored, process=False), dense / "sparse", dense / "images",
                         atlas=atlas, max_faces=maxf, log=log)
            obj = mvs / "scene_baked.glb"
            tex_info = tm.metadata.get("bake")
            tm.export(obj)
        except Exception as e:
            log(f"  사진 텍스처 굽기 실패 → 점군 색 메쉬 사용: {e}")
            obj = colored
        ply = mvs / "scene_dense.ply"
        from .pointfilter import LAST
        return {"textured_obj": obj, "dense_ply": ply if ply.exists() else None, "log": logf,
                "stats": {**LAST, "texture": tex_info}}
    try:
        _run([_bin("TextureMesh"), str(dense_scene), "--mesh-file", str(mesh), "--export-type", "ply",  # v2.3 SaveOBJ 세그폴트 → PLY(+png)
              "--empty-color", "8421504",  # 어느 사진에도 안 보인 면은 회색(기본 주황)
              "--resolution-level", str(max(0, q["res"] - 1)),
              "-o", str(mvs / "scene_textured.mvs")] + common, mvs, log, logf)
    except RuntimeError as e:
        log(f"  텍스처 실패 → 정점 색 메쉬로 대체: {e}")
    obj = mvs / "scene_textured.ply"
    if obj.exists():
        # Windows OpenMVS v2.4 는 사진을 못 읽어 검은 텍스처를 만들었다(PC 3회). 아틀라스 평균은 여백·경계색 때문에
        # 판정이 안 되므로, 면에 실제로 입혀진 색의 밝기를 원본 점군 색과 비교한다.
        try:
            import trimesh
            from .export import load_openmvs_textured_ply
            tm = load_openmvs_textured_ply(obj)
            pc = trimesh.load(dense / "fused.ply", process=False) if (dense / "fused.ply").exists() else None
            if tm is not None and pc is not None:
                img = np.asarray(tm.visual.material.image)
                uv = tm.visual.uv.reshape(-1, 3, 2).mean(1)
                h, w = img.shape[:2]
                sel = np.random.default_rng(0).choice(len(uv), min(200000, len(uv)), replace=False)
                px = img[np.clip(((1 - uv[sel, 1]) * h).astype(int), 0, h - 1),
                         np.clip((uv[sel, 0] * w).astype(int), 0, w - 1)][:, :3].astype(float)
                pcol = np.asarray(getattr(pc, "colors", pc.visual.vertex_colors))[:, :3].astype(float)
                tex_l, pt_l = float(px.mean()), float(pcol.mean())
                log(f"  텍스처 밝기 {tex_l:.0f} / 점군 밝기 {pt_l:.0f}")
                if tex_l < 0.5 * pt_l:
                    log("  경고: 텍스처가 사진보다 훨씬 어두움(사진 읽기 실패 의심) → 점군 색 메쉬로 대체")
                    obj = _color_from_points(mesh, dense / "fused.ply", mvs, log)
        except Exception as e:
            log(f"  텍스처 점검 생략: {e}")
    if not obj.exists() and mesh.exists():
        # TextureMesh 가 죽으면(v2.3 unordered_map::at — Pix4D building 36장에서 재현) 파이썬 사진 텍스처 굽기로
        pts = dense / "fused.ply" if (dense / "fused.ply").exists() else mvs / "scene_dense.ply"
        colored = _color_from_points(mesh, pts, mvs, log, smooth=0)  # 형상은 그대로(측정용) — 색은 안 보인 면 대체용
        obj = colored
        if (dense / "sparse").exists() and (dense / "images").exists():
            try:
                import trimesh
                from .texture import bake
                maxf = {"draft": 600_000, "medium": 900_000, "high": 1_200_000, "ultra": 1_500_000}[quality]
                tm, _ = bake(trimesh.load(colored, process=False), dense / "sparse", dense / "images",
                             atlas=8192, max_faces=maxf, log=log)
                obj = mvs / "scene_baked.glb"
                tm.export(obj)
            except Exception as e:
                log(f"  사진 텍스처 굽기 실패 → 점군 색 메쉬 사용: {e}")
                obj = colored
    if not obj.exists():
        raise RuntimeError("텍스처 메쉬가 생성되지 않았습니다")
    ply = mvs / "scene_dense.ply"
    return {"textured_obj": obj, "dense_ply": ply if ply.exists() else None, "log": logf}
