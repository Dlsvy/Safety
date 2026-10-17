"""photo3d 앱 서버 — 사진/영상 올리기 → 처리 대기열 → 3D 보기.

실행: python -m photo3d.app  (PC 에서는 APP.bat 더블클릭)
- 브라우저: http://localhost:8765  (같은 와이파이의 폰: http://<PC IP>:8765)
- 표준 라이브러리만 사용(추가 설치 없음). 처리는 한 번에 하나씩(GPU 하나).
- 업로드는 파일마다 원본 바이트를 PUT 으로 받아 디스크에 바로 쓴다(큰 동영상도 메모리에 올리지 않음).
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
DATA = Path(os.environ.get("PHOTO3D_DATA", ROOT / "app_data"))
JOBS = DATA / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)

STAGES = [("1/5", "입력 정리"), ("2/5", "카메라 위치 계산"), ("3/5", "축척·좌표"),
          ("4/5", "3D 생성"), ("5/5", "유니티 파일")]
MEDIA = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".webp",
         ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".insv", ".insp"}  # .insv/.insp: Insta360 원본

_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()


# ── 도구 위치 ─────────────────────────────────────────
def find_tools() -> dict:
    """PC: third_party 의 COLMAP(CUDA)·OpenMVS(CPU판). 없으면 None(→ pycolmap CPU 경로)."""
    tp = ROOT / "third_party"
    colmap = next(iter(sorted(tp.glob("colmap/**/COLMAP.bat"))), None)
    mvs = next(iter(sorted(tp.glob("openmvs_cpu/**/ReconstructMesh.exe"))), None)
    return {"colmap": str(colmap) if colmap else None, "openmvs": str(mvs.parent) if mvs else None}


# ── 예시 사진 ─────────────────────────────────────────
SAMPLE_REPOS = [  # 공개 시험 사진(연구·시험용 배포) — 없으면 '예시 받기'로 git clone
    ("strecha", "https://github.com/openMVG/SfM_quality_evaluation.git"),
    ("sceaux", "https://github.com/openMVG/ImageDataset_SceauxCastle.git"),
]
SAMPLE_INFO = {"fountain-P11": "분수대 벽면 11장", "Herz-Jesus-P8": "성당 정면 8장", "Herz-Jesus-P25": "성당 25장",
               "entry-P10": "건물 입구 10장", "castle-P19": "성 안뜰 19장", "castle-P30": "성 안뜰 한 바퀴 30장",
               "ImageDataset_SceauxCastle": "성 외벽 11장", "sceaux": "성 외벽 11장"}


# 현장 예시(건설현장·공장·대형 건물): 크기가 커서 하나씩 '받기'. 라이선스를 지켜 원 배포처에서 직접 받는다(재배포 안 함)
SITE_SAMPLES = [
    {"key": "caliterra", "title": "건설현장(택지 조성) 드론 77장", "kind": "git", "mb": 272,
     "url": "https://github.com/OpenDroneMap/odm_data_caliterra.git",
     "license": "CC0 — OpenDroneMap. 땅고르기·도로·콘크리트관 적치, 수직 촬영(GPS)"},
    {"key": "pix4d_construction", "title": "건설현장 드론 197장(13ha)", "kind": "zip", "mb": 1700,
     "url": "https://data.pix4d.com/misc/example_datasets/example_soda_bim.zip",
     "license": "Pix4D 예시 데이터 — 교육·업무 훈련용, 상업·홍보에 쓰면 'Courtesy of Pix4D' 표기"},
    {"key": "pix4d_industrial", "title": "산업 건물(공장) 드론 138장", "kind": "zip", "mb": 302,
     "url": "https://data.pix4d.com/misc/example_datasets/example_bebop2_industrial_building.zip",
     "license": "Pix4D 예시 데이터 — 교육·업무 훈련용, 상업·홍보에 쓰면 'Courtesy of Pix4D' 표기. 격자+선회(경사)"},
    {"key": "pix4d_office", "title": "사무 건물 드론 선회 85장", "kind": "zip", "mb": 416,
     "url": "https://data.pix4d.com/misc/example_datasets/dji_demo.zip",
     "license": "Pix4D 예시 데이터 — 교육·업무 훈련용, 상업·홍보에 쓰면 'Courtesy of Pix4D' 표기. 스플래팅에 적합"},
    {"key": "pix4d_building", "title": "건물 드론 선회 36장(빠른 시험)", "kind": "zip", "mb": 245,
     "url": "https://data.pix4d.com/misc/example_datasets/example_building.zip",
     "license": "Pix4D 예시 데이터 — 교육·업무 훈련용, 상업·홍보에 쓰면 'Courtesy of Pix4D' 표기"},
    {"key": "eth3d_pipes", "title": "플랜트 배관(실내) 14장", "kind": "7z", "mb": 110,
     "url": "https://www.eth3d.net/data/pipes_dslr_jpg.7z",
     "license": "ETH3D CC BY-NC-SA 4.0 — 비상업 시험용만. 레이저 정답 있음(tools/eval_eth3d.py)"},
    {"key": "eth3d_electro", "title": "전기 설비 현장 45장", "kind": "7z", "mb": 423,
     "url": "https://www.eth3d.net/data/electro_dslr_jpg.7z",
     "license": "ETH3D CC BY-NC-SA 4.0 — 비상업 시험용만. 레이저 정답 있음"},
]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def _image_dir(root: Path) -> tuple[Path | None, int]:
    """압축을 푼 폴더 안에서 사진이 가장 많은 폴더"""
    best, n = None, 0
    for d in [root] + [q for q in root.rglob("*") if q.is_dir()]:
        if ".git" in d.parts:
            continue
        c = sum(1 for q in d.iterdir() if q.suffix.lower() in IMG_EXT)
        if c > n:
            best, n = d, c
    return best, n


def list_samples() -> list[dict]:
    out = []
    data = ROOT / "data"
    site_keys = {x["key"] for x in SITE_SAMPLES}
    for x in SITE_SAMPLES:
        d = data / x["key"]
        if d.exists() and (d / ".done").exists():
            img, n = _image_dir(d)
            if img is not None and n >= 5:
                out.append({"name": x["key"], "path": str(img), "count": n, "desc": x["title"], "license": x["license"], "site": True})
                continue
        out.append({"name": x["key"], "remote": True, "mb": x["mb"], "desc": x["title"], "license": x["license"], "site": True})
    if not data.exists():
        return out
    for d in sorted(data.glob("**/images")):
        if any(k in d.parts for k in site_keys):
            continue
        n = len([q for q in d.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if n >= 5:
            name = d.parent.name
            out.append({"name": name, "path": str(d), "count": n, "desc": SAMPLE_INFO.get(name, f"{n}장")})
    return out


_sample_dl = {"state": "idle", "msg": ""}
_site_dl: dict = {}  # key → {"state", "msg"}


def download_site_sample(key: str) -> None:
    x = next(s for s in SITE_SAMPLES if s["key"] == key)
    st = _site_dl[key] = {"state": "running", "msg": "받는 중…"}
    dst = ROOT / "data" / key
    try:
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        if x["kind"] == "git":
            r = subprocess.run(["git", "clone", "--depth", "1", x["url"], str(dst)], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip()[-200:])
        else:
            import urllib.request
            arc = ROOT / "data" / f"_{key}.{x['kind']}"
            arc.unlink(missing_ok=True)
            total = 0
            for attempt in range(8):  # 연결이 중간에 끊기면 받은 데부터 이어 받는다
                got = arc.stat().st_size if arc.exists() else 0
                if total and got >= total:
                    break
                try:
                    req = urllib.request.Request(x["url"], headers={"Range": f"bytes={got}-"} if got else {})
                    with urllib.request.urlopen(req, timeout=60) as resp, open(arc, "ab" if got else "wb") as f:
                        if got and resp.status != 206:  # 이어받기 안 되는 서버 → 처음부터
                            f.seek(0)
                            f.truncate()
                            got = 0
                        total = total or got + int(resp.headers.get("Content-Length") or 0)
                        while True:
                            b = resp.read(1 << 20)
                            if not b:
                                break
                            f.write(b)
                            got += len(b)
                            st["msg"] = f"받는 중 {got >> 20}/{total >> 20 or x['mb']} MB"
                    if not total or got >= total:
                        break
                except Exception:
                    if attempt == 7:
                        raise
                    st["msg"] = f"연결 끊김 → 이어 받기({attempt + 1})"
                    time.sleep(2 * (attempt + 1))
            st["msg"] = "압축 푸는 중…"
            if x["kind"] == "zip":
                import zipfile
                with zipfile.ZipFile(arc) as z:
                    z.extractall(dst)
            else:
                try:
                    import py7zr
                except ImportError:
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py7zr"], check=True)
                    import py7zr
                with py7zr.SevenZipFile(arc) as z:
                    z.extractall(dst)
            arc.unlink(missing_ok=True)
        if _image_dir(dst)[1] < 5:
            raise RuntimeError("사진을 찾지 못했습니다")
        (dst / ".done").write_text("ok")
        st.update(state="done", msg="준비됨")
    except Exception as e:
        st.update(state="failed", msg=f"받기 실패: {e}")


def download_samples() -> None:
    _sample_dl.update(state="running", msg="예시 사진 받는 중…")
    try:
        for name, url in SAMPLE_REPOS:
            dst = ROOT / "data" / name
            if dst.exists() and any(dst.iterdir()):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["git", "clone", "--depth", "1", url, str(dst)], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip()[-200:])
        _sample_dl.update(state="done", msg=f"예시 {len(list_samples())}개 준비됨")
    except Exception as e:
        _sample_dl.update(state="failed", msg=f"받기 실패: {e}")


# ── 작업 저장 ─────────────────────────────────────────
def job_dir(jid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", jid):
        raise KeyError(jid)
    return JOBS / jid


def load_job(jid: str) -> dict:
    return json.loads((job_dir(jid) / "job.json").read_text(encoding="utf-8"))


def save_job(job: dict) -> None:
    p = job_dir(job["id"]) / "job.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def _depth_dir(d: Path) -> Path | None:
    p = d / "out" / "work" / "dense" / "stereo" / "depth_maps"
    return p if p.exists() else None


def _preview_source(d: Path) -> Path | None:
    """처리 중 미리보기용 점: 조밀 점군 > 계산 중인 깊이 지도(GPU) > 정합된 희소 점 > 첫 희소 모델"""
    w = d / "out" / "work"
    if (w / "dense" / "fused.ply").exists():
        return w / "dense" / "fused.ply"
    dd = _depth_dir(d)
    if dd and any(dd.glob("*.bin")):
        return dd
    if (w / "sfm" / "georef" / "points3D.bin").exists():
        return w / "sfm" / "georef" / "points3D.bin"
    hits = sorted((w / "sfm" / "sparse").glob("*/points3D.bin")) if (w / "sfm" / "sparse").exists() else []
    return hits[0] if hits else None


def _preview_rev(src: Path) -> str:
    """미리보기가 바뀌었는지 판단하는 값 — 깊이 지도는 한 장 늘 때마다 바뀐다"""
    if src.is_dir():
        fs = list(src.glob("*.bin"))
        return f"depth:{len(fs)}:{int(max(f.stat().st_mtime for f in fs))}" if fs else "depth:0"
    return f"{src.name}:{int(src.stat().st_mtime)}"


_DEPTH_CACHE: dict = {}  # 깊이 지도 파일 → (수정시각, 점, 색) — 매번 다시 읽지 않게


def _read_colmap_array(p: Path):
    import numpy as np
    with open(p, "rb") as f:
        head = b""
        while head.count(b"&") < 3:
            head += f.read(1)
        w, h, c = map(int, head.decode().split("&")[:3])
        a = np.fromfile(f, np.float32)
    return a.reshape((w, h, c), order="F").transpose(1, 0, 2).squeeze()


def _depth_points(dd: Path, per_map: int = 4000):
    """GPU 가 깊이 지도를 한 장씩 끝낼 때마다 그 장의 점을 3D 로 펼쳐 미리보기에 더한다"""
    import numpy as np
    import pycolmap
    from PIL import Image
    dense = dd.parent.parent
    key = ("rec", str(dense))
    if key not in _DEPTH_CACHE:
        rec = pycolmap.Reconstruction(dense / "sparse")
        _DEPTH_CACHE[key] = {rec.image(i).name: rec.image(i) for i in rec.reg_image_ids()}, rec
    by_name, rec = _DEPTH_CACHE[key]
    files = {}
    for f in dd.glob("*.bin"):  # 같은 사진은 geometric(최종) 을 photometric(1차) 보다 우선
        name, kind = f.name.rsplit(".", 2)[0], f.name.rsplit(".", 2)[1]
        if kind == "geometric" or name not in files:
            files[name] = f
    P, C, cams = [], [], []
    for name, f in files.items():
        im = by_name.get(name)
        if im is None:
            continue
        mt = f.stat().st_mtime
        hit = _DEPTH_CACHE.get(f)
        if not hit or hit[0] != mt:
            try:
                D = _read_colmap_array(f)
            except Exception:  # 아직 쓰는 중인 파일
                continue
            cam = rec.camera(im.camera_id)
            h, w = D.shape
            sx, sy = w / cam.width, h / cam.height
            fx, fy, cx, cy = cam.focal_length_x * sx, cam.focal_length_y * sy, cam.principal_point_x * sx, cam.principal_point_y * sy
            vs, us = np.nonzero(D > 0)
            if len(us) > per_map:
                k = np.random.default_rng(0).choice(len(us), per_map, replace=False)
                vs, us = vs[k], us[k]
            z = D[vs, us]
            Xc = np.stack([(us + 0.5 - cx) / fx * z, (vs + 0.5 - cy) / fy * z, z], 1)
            T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            R, t = T.rotation.matrix(), T.translation
            Xw = (Xc - t) @ R
            try:
                img = np.asarray(Image.open(dense / "images" / name).convert("RGB").resize((w, h)))
                col = img[vs, us]
            except Exception:
                col = np.full((len(Xw), 3), 200)
            hit = (mt, Xw, col)
            _DEPTH_CACHE[f] = hit
        P.append(hit[1]); C.append(hit[2]); cams.append(im.projection_center())
    n_geo = sum(1 for f in files.values() if ".geometric." in f.name)
    _DEPTH_CACHE["geo"] = n_geo
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3)), [], 0, 0
    return np.concatenate(P), np.concatenate(C), np.array(cams).round(3).tolist(), len(P), len(by_name)


def preview_points(d: Path, limit: int = 30000) -> dict:
    # 처리 중에는 파일이 막 쓰이거나 바뀌는 중일 수 있다 → 실패하면 잠시 뒤 다시(화면은 이전 그림 유지)
    try:
        return _preview_points(d, limit)
    except Exception:
        return {"n": 0}


def _preview_points(d: Path, limit: int = 30000) -> dict:
    import numpy as np
    src = _preview_source(d)
    if src is None:
        return {"n": 0}
    cams, kind, extra = [], "sparse", {}
    if src.is_dir():
        xyz, rgb, cams, done, total = _depth_points(src)
        kind, extra = "depth", {"done": done, "total": total, "geo": _DEPTH_CACHE.get("geo", 0)}
    elif src.suffix == ".ply":
        import trimesh
        pc = trimesh.load(src, process=False)
        xyz = np.asarray(pc.vertices, float)
        rgb = np.asarray(pc.colors)[:, :3] if getattr(pc, "colors", None) is not None and len(pc.colors) else np.full((len(xyz), 3), 200)
        kind = "dense"
    else:
        import pycolmap
        rec = pycolmap.Reconstruction(src.parent)
        pts = list(rec.points3D.values())
        xyz = np.array([p.xyz for p in pts], float).reshape(-1, 3)
        rgb = np.array([p.color for p in pts], float).reshape(-1, 3)
        cams = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()]).round(3).tolist()
    if len(xyz) > limit:
        sel = np.random.default_rng(0).choice(len(xyz), limit, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    if len(xyz) > 50:  # 튀는 점 제거(미리보기 범위 안정화)
        lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
        ok = ((xyz >= lo) & (xyz <= hi)).all(1)
        xyz, rgb = xyz[ok], rgb[ok]
    return {"n": len(xyz), "kind": kind, **extra,
            "xyz": xyz.round(3).ravel().tolist(), "rgb": rgb.astype(int).ravel().tolist(), "cams": cams}


CMD_LABELS = [
    ("patch_match_stereo", "GPU 깊이 계산 중 — 사진마다 1차(밝기)·2차(사진 간 일치) 두 번 계산"),
    ("stereo_fusion", "깊이 지도 합치기(조밀 점군 만들기)"),
    ("feature_extractor", "GPU 특징점 찾기"), ("exhaustive_matcher", "GPU 사진끼리 짝 맞추기"),
    ("sequential_matcher", "GPU 사진끼리 짝 맞추기"), ("global_mapper", "카메라 위치 계산(글로벌)"),
    ("mapper", "카메라 위치 계산(증분) — 글로벌 결과와 비교"), ("poisson_mesher", "메쉬 만들기"),
    ("ReconstructMesh", "메쉬 만들기"), ("TextureMesh", "사진 색 입히기"), ("DensifyPointCloud", "CPU 깊이 계산"),
]


def splat_progress(d: Path) -> tuple[int, int] | None:
    """Brush 진행 스텝: brush.log 에서 '현재/전체' 숫자를 찾는다(진행 막대 출력)"""
    bl = d / "out" / "work" / "splat" / "brush.log"
    run = d / "out" / "run.log"
    if not bl.exists():
        return None
    m = re.search(r"--total-steps (\d+)", run.read_text(encoding="utf-8", errors="replace")) if run.exists() else None
    total = int(m.group(1)) if m else None
    try:
        with open(bl, "rb") as f:
            f.seek(max(0, bl.stat().st_size - 20000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    hits = re.findall(r"(\d+)\s*/\s*(\d+)", tail)
    if total:
        hits = [h for h in hits if int(h[1]) == total]
    if not hits:
        return (0, total) if total else None
    cur, tot = map(int, hits[-1])
    return cur, tot


def side_tasks(d: Path, lines: list[str]) -> list[str]:
    """본 작업과 동시에 도는 작업(구멍 메우기: CPU, 스플랫: GPU)의 상태 한 줄씩"""
    text = "\n".join(lines)
    out = []
    if "구멍 메운 모델: 별도 CPU" in text:
        out.append("구멍 메우기(CPU): " + ("완료" if "구멍 메운 모델 완료 (" in text or "구멍 메우기 실패" in text else "진행 중"))
    if "가우시안 스플래팅: GPU" in text:
        if "스플랫 " in text and "site_splat" in text or "스플래팅 실패" in text:
            out.append("실사 스플랫(GPU): 완료")
        else:
            pr = splat_progress(d)
            t0 = next((float(m.group(1)) for m in (re.match(r"\[\s*([\d.]+)s\].*가우시안 스플래팅: GPU", ln) for ln in lines) if m), None)
            el = time.time() - (load_job(d.name).get("started") or time.time()) - (t0 or 0)
            if pr and pr[0] > 50:
                eta = el / pr[0] * (pr[1] - pr[0])
                out.append(f"실사 스플랫(GPU): {pr[0]:,}/{pr[1]:,}스텝 · 남은 시간 약 {max(1, round(eta / 60))}분")
            else:
                out.append(f"실사 스플랫(GPU): 학습 중 {round(el / 60)}분째")
    return out


HIST = DATA / "timing_history.json"
# 기록이 없을 때 쓰는 단계별 '사진 1장당 초'(RTX 5060 Ti 8GB 기준 어림값). 처리가 끝날 때마다 이 PC 의 실제 기록으로 바뀐다
DEFAULT_SPI = {"fast": {"1": 0.5, "2": 1.5, "3": 0.2, "4": 10.0, "5": 3.0},
               "best": {"1": 0.8, "2": 3.0, "3": 0.3, "4": 35.0, "5": 25.0},
               # 포토그래메트리: 스플랫 대기 없음, 대신 해상도↑·구멍 메우기·정사영상/DSM/점군
               "pg-fast": {"1": 0.6, "2": 2.0, "3": 0.2, "4": 14.0, "5": 4.0},
               "pg-best": {"1": 0.8, "2": 3.0, "3": 0.3, "4": 35.0, "5": 6.0},
               # 통합: 포토그래메트리 다음 그 메쉬로 맞춘 스플랫 학습을 끝까지 기다림
               "fuse-fast": {"1": 0.6, "2": 2.0, "3": 0.2, "4": 14.0, "5": 12.0},
               "fuse-best": {"1": 0.8, "2": 3.0, "3": 0.3, "4": 35.0, "5": 30.0}}


def _mkey(job: dict) -> str:
    """시간 기록·예측용 모드 이름(포토그래메트리 탭은 따로 기록)"""
    m = job.get("mode", "fast")
    return f"{job['kind']}-{m}" if job.get("kind") in ("pg", "fuse") else m


def _hist() -> list:
    try:
        return json.loads(HIST.read_text(encoding="utf-8"))
    except Exception:
        return []


def record_timing(d: Path, job: dict) -> None:
    """끝난 작업의 단계별 시간을 기록 — 다음 작업의 시간 예측에 쓴다(최근 40건)."""
    try:
        r = json.loads((d / "out" / "report.json").read_text(encoding="utf-8"))
        n = int((r.get("input") or {}).get("images") or r["sfm"]["images_input"])
        st = {}
        for name, sec in (r.get("timing_s") or {}).items():
            m = re.match(r"(\d)/5", name)
            if m:
                st[m.group(1)] = float(sec)
        if n and st:
            h = _hist()[-39:] + [{"mode": _mkey(job), "n": n, "s": st, "t": time.time()}]
            HIST.write_text(json.dumps(h, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _spi(mode: str) -> dict:
    """단계별 '사진 1장당 초': 같은 모드의 최근 기록 중앙값(없으면 기본값)"""
    base = dict(DEFAULT_SPI.get(mode, DEFAULT_SPI["fast"]))
    rows = [h for h in _hist() if h.get("mode") == mode][-10:]
    for k in base:
        vals = [h["s"][k] / h["n"] for h in rows if k in h.get("s", {}) and h.get("n")]
        if vals:
            base[k] = sorted(vals)[len(vals) // 2]
    return base


def _n_images(d: Path, lines: list[str], job: dict) -> int:
    for ln in reversed(lines):
        m = re.search(r"특징점 추출 \((\d+)장", ln) or re.search(r"사진 \d+장 중 (\d+)장 사용", ln) or \
            re.search(r"등록 \d+/(\d+)장", ln)
        if m:
            return int(m.group(1))
    src = Path(job["local_path"]) if job.get("local_path") else d / "input"
    try:
        n = 0
        for q in src.iterdir():
            ext = q.suffix.lower()
            n += 1 if ext in {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".webp", ".insp"} else \
                (60 if ext in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".insv"} else 0)
        return max(n, 1)
    except Exception:
        return 30


def overall_progress(d: Path, lines: list[str], step: int, job: dict) -> tuple[int, int | None]:
    """전체 진행률(%)과 남은 시간(분).
    예전: (지금까지 걸린 시간 ÷ 진행률) — 단계마다 걸리는 시간이 크게 달라(3D 생성이 대부분) 앞쪽에서 너무 짧게 잡혔다.
    지금: 단계별 예상 시간 = 사진 수 × 이 PC 의 과거 기록(사진 1장당 초, 같은 모드 중앙값) × 이번 작업의 속도 보정.
    진행 중인 단계는 실제 진행(깊이 지도 장수 등)이 있으면 그 속도로, 없으면 예상 시간에서 뺀다."""
    mode = _mkey(job)
    n = _n_images(d, lines, job)
    spi = _spi(mode)
    exp = {k: spi[k] * n for k in spi}
    starts = {}  # 단계 시작 시각(처리 시작 후 초)
    for ln in lines:
        m = re.match(r"\[\s*([\d.]+)s\]\s+(\d)/5\s", ln)
        if m:
            starts[m.group(2)] = float(m.group(1))
    el = time.time() - (job.get("started") or time.time())
    cur = str(min(max(step, 1), 5))
    # 이번 작업이 기록보다 빠른지 느린지(끝난 단계의 실제/예상 비율) → 남은 단계에 반영
    ks = sorted(starts)
    ratios = []
    for a, b in zip(ks, ks[1:]):
        if exp.get(a, 0) > 1:
            ratios.append((starts[b] - starts[a]) / exp[a])
    speed = min(3.0, max(0.33, sorted(ratios)[len(ratios) // 2])) if ratios else 1.0
    e_cur = el - starts.get(cur, el)
    frac = 0.0
    if cur == "4":
        dd = d / "out" / "work" / "dense" / "stereo" / "depth_maps"
        try:
            total = len(list((d / "out" / "work" / "dense" / "images").rglob("*.jpg")))
            if total and dd.exists():
                frac = 0.5 * (len(list(dd.glob("*.photometric.bin"))) + len(list(dd.glob("*.geometric.bin")))) / (2 * total)
        except Exception:
            pass
    elif cur == "5" and step == 7:
        pr = splat_progress(d)
        if pr and pr[1]:
            frac = pr[0] / pr[1]
    e_stage = exp.get(cur, 1) * speed
    if frac > 0.05:
        rem_cur = max(e_cur * (1 - frac) / frac, 0.0)
        rem_cur = 0.5 * rem_cur + 0.5 * max(e_stage - e_cur, 0.1 * e_stage)  # 실측 속도와 기록을 반반
    else:
        rem_cur = max(e_stage - e_cur, 0.15 * e_stage)  # 기록보다 오래 걸리면 '조금 남음'으로 버팀
    rem = rem_cur + sum(exp[k] * speed for k in exp if k > cur)
    pct = int(min(99, 100 * el / max(el + rem, 1)))
    return pct, max(1, int(round(rem / 60)))


def job_view(job: dict) -> dict:
    d = job_dir(job["id"])
    v = dict(job)
    log = d / "out" / "run.log"
    if job["status"] == "running" and log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        stage, step = None, 0
        for ln in lines:
            mm = re.search(r"\]\s+(\d)/5\s", ln)
            if mm:
                step = int(mm.group(1))
                stage = f"{step}/5 {dict(STAGES).get(f'{step}/5')}"
            elif "구멍 메운 모델 완료 대기" in ln:  # 본 작업(5/5)이 끝나고 동시 작업을 기다리는 중
                step, stage = 6, "구멍 메우기"
            elif "스플래팅 완료 대기" in ln:
                step, stage = 7, "실사(스플래팅) 학습"
        if stage:
            v["stage"] = stage
        v["step"] = step
        v["side"] = side_tasks(d, lines)
        cam = next((ln.split("촬영 기종:", 1)[1].strip() for ln in lines if "촬영 기종:" in ln), None)
        if cam:
            v["camera"] = cam
        v["pct"], v["eta_min"] = overall_progress(d, lines, step, job)
        human = [ln for ln in lines if "$ " not in ln and ln.strip()]  # 실행 명령(경로 덩어리)은 숨김
        v["last_log"] = re.sub(r"^\[\s*[\d.]+s\]\s*", "", human[-1]).strip()[-160:] if human else ""
        last = lines[-1] if lines else ""
        for key, label in CMD_LABELS:  # 마지막 줄이 실행 중인 외부 프로그램이면 무엇을 하는지 표시
            if "$ " in last and key in last:
                v["last_log"] = label
                break
        src = _preview_source(d)
        v["preview_rev"] = _preview_rev(src) if src else None
    if job["status"] == "queued":  # 시작 전 예상 총 시간(이 PC 의 과거 기록 기준)
        try:
            n = _n_images(d, [], job)
            v["eta_min"] = max(1, int(round(sum(_spi(_mkey(job)).values()) * n / 60)))
        except Exception:
            pass
    rep = d / "out" / "report.json"
    if rep.exists():
        try:
            r = json.loads(rep.read_text(encoding="utf-8"))
            v["grade"] = r.get("grade")
            v["size_m"] = r.get("model", {}).get("size_m")
            v["registered"] = f"{r['sfm']['images_registered']}/{r['sfm']['images_input']}"
            cm = (r.get("input") or {}).get("camera_models")
            if cm:
                v["camera"] = ", ".join(f"{k}({n})" for k, n in sorted(cm.items(), key=lambda x: -x[1]))
        except Exception:
            pass
    v["has_model"] = (d / "out" / "site_model.glb").exists()
    v["has_light"] = (d / "out" / "site_model_light.glb").exists()
    v["has_filled"] = (d / "out" / "site_filled.glb").exists()
    v["has_est"] = (d / "out" / "site_estimated.glb").exists()
    v["has_splat"] = (d / "out" / "site_splat.ply").exists()
    v["has_best"] = (d / "out" / "site_best.ply").exists()
    v["has_best_mesh"] = (d / "out" / "site_best_mesh.glb").exists()
    v["has_ortho"] = (d / "out" / "site_ground_ortho.jpg").exists()
    v["has_survey"] = (d / "out" / "site_ortho.jpg").exists()
    v["has_las"] = (d / "out" / "site_points.las").exists()
    v["has_report"] = (d / "out" / "quality_report.html").exists()
    v["has_cams"] = (d / "out" / "cameras.json").exists()
    v["files"] = len([p for p in (d / "input").iterdir()]) if (d / "input").exists() else 0
    return v


# ── 처리 작업자 ───────────────────────────────────────
def worker() -> None:
    while True:
        jid = _queue.get()
        _CURRENT.update(id=jid, t=time.time())
        try:
            run_job(jid)
        except Exception as e:  # 작업자가 죽지 않게
            try:
                job = load_job(jid)
                job.update(status="failed", error=str(e), finished=time.time())
                save_job(job)
            except Exception:
                pass
        finally:
            _CURRENT.update(id=None, t=0)


def run_job(jid: str) -> None:
    d = job_dir(jid)
    job = load_job(jid)
    if job.get("status") != "queued":  # 대기 중에 멈춤(취소)된 작업
        return
    resume = bool(job.pop("resume", False))
    if not resume:
        shutil.rmtree(d / "out", ignore_errors=True)  # 다시 처리: 이전 결과·미리보기를 지우고 새로
    job.update(status="running", started=time.time())
    save_job(job)
    tools = find_tools()
    # 흰 벽 채우기 준비(PyTorch·모델 설치)가 아직이면 기다리지 않고 그 기능만 빼고 바로 처리(예전: 최대 30분 대기 → '대기'로 멈춘 듯 보임)
    no_prior = _PRIOR["state"] != "준비됨"
    if no_prior and _PRIOR["state"] in ("확인 전", "준비 중"):
        (d / "out").mkdir(parents=True, exist_ok=True)
        with open(d / "out" / "run.log", "a", encoding="utf-8") as f:
            f.write("[    0.0s] 흰 벽 채우기 준비가 아직 끝나지 않아 이번 처리에서는 빼고 진행합니다\n")
    env = dict(os.environ)
    if tools["openmvs"]:
        env["OPENMVS_BIN"] = tools["openmvs"]
    env["PYTHONIOENCODING"] = "utf-8"
    src = job.get("local_path") or str(d / "input")
    profile = "gpu-max" if job.get("mode") == "best" else "fast"
    cmd = [sys.executable, "-m", "photo3d", "run", src, "--profile", profile, "--out", str(d / "out"),
           "--name", job.get("name") or job["id"]]
    if job.get("kind") in ("pg", "fuse"):
        cmd += ["--pipeline", job["kind"]]
    if tools["colmap"]:
        cmd += ["--colmap-bin", tools["colmap"]]
    if job.get("scale"):
        sc = d / "scale.csv"
        sc.write_text(job["scale"], encoding="utf-8")
        cmd += ["--scale-dist", str(sc)]
    if job.get("subject"):
        cmd += ["--subject"]
    if resume:  # 이어서 처리: 끝난 단계·계산한 깊이 지도·점군 재사용
        cmd += ["--resume"]
    if no_prior:
        cmd += ["--no-prior"]
    if job.get("marker_size"):
        cmd += ["--marker-size", str(job["marker_size"])]
    with open(d / "console.log", "a" if resume else "w", encoding="utf-8") as f:
        # 별도 프로세스 그룹으로 실행 → '멈춤' 때 COLMAP·Spirula 등 하위 프로그램까지 한 번에 끝낼 수 있게
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, **kw)
        _RUNNING[jid] = proc
        rc = proc.wait()
        _RUNNING.pop(jid, None)
    job = load_job(jid)
    ok = rc == 0 and (d / "out" / "site_model.glb").exists()
    job.update(status="done" if ok else "failed", finished=time.time())
    if ok and not resume:  # 이어서 처리한 작업은 단계 시간이 섞여 기록하지 않음
        record_timing(d, job)
    if jid in _STOPPED:
        _STOPPED.discard(jid)
        job.update(status="failed", error="멈춤 — '이어서 처리'를 누르면 끝난 단계는 건너뛰고 이어서 합니다")
    elif not ok:
        lines = (d / "console.log").read_text(encoding="utf-8", errors="replace").splitlines()
        job["error"] = explain_error(lines)
    save_job(job)


def app_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip() or "?"
    except Exception:
        return "?"


VERSION = app_version()
_RUNNING: dict = {}   # 작업 id → 실행 중인 처리 프로세스
_CURRENT: dict = {"id": None, "t": 0}  # 작업자가 지금 맡은 작업
_STOPPED: set = set()


def stop_job(jid: str) -> bool:
    """처리 멈추기: 처리 프로세스와 그 하위 프로그램(COLMAP·Spirula·Brush·구멍 메우기)을 모두 끝낸다."""
    proc = _RUNNING.get(jid)
    if not proc or proc.poll() is not None:
        return False
    _STOPPED.add(jid)
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.kill()
    return True


_HINTS = [
    ("out of memory", "GPU 메모리 부족 — 품질을 한 단계 낮추거나 사진 수를 줄여 다시 처리하세요"),
    ("cudaerror", "GPU(CUDA) 오류 — 그래픽 드라이버를 최신으로 업데이트한 뒤 PC 재시작"),
    ("no cuda", "CUDA 를 쓸 수 있는 GPU 를 찾지 못함 — 그래픽 드라이버 확인"),
    ("등록", "사진끼리 연결이 부족 — 겹치게(70~80%) 더 찍거나 무늬 있는 곳을 함께 찍으세요"),
    ("no space", "디스크 공간 부족 — C 드라이브 여유 공간을 20GB 이상 확보하세요"),
    ("memoryerror", "PC 메모리 부족 — 다른 프로그램을 닫고 다시 처리하세요"),
    ("permission", "파일 접근 권한 오류 — 결과 폴더가 다른 프로그램에서 열려 있지 않은지 확인"),
]


def explain_error(lines: list[str]) -> str:
    """실패 원인을 맨 위에 — 파이썬 호출 경로(Traceback)는 뒤로 보낸다"""
    tb = next((i for i, ln in enumerate(lines) if ln.startswith("Traceback")), None)
    if tb is None:
        return "\n".join(lines[-12:])
    body = lines[tb + 1:]
    k = next((i for i, ln in enumerate(body) if not ln.startswith(" ")), len(body))
    cause = body[k:] or body[-1:]
    progress = [ln for ln in lines[:tb] if ln.startswith("[")][-3:]
    low = " ".join(cause + lines[-40:]).lower()
    hint = next((h for key, h in _HINTS if key in low), "")
    out = ["원인: " + (cause[0] if cause else "알 수 없음")] + cause[1:]
    if hint:
        out += ["조치: " + hint]
    out += ["", "마지막 진행 단계:"] + progress + ["", "(자세한 기록: 결과 폴더의 console.log)"]
    return "\n".join(out)


# ── CPU·GPU 사용량 ─────────────────────────────────────
_SYS = {"cpu_prev": None, "gpu": None, "gpu_t": 0.0, "nvsmi": None}


def _cpu_percent() -> float | None:
    try:
        import psutil
        return float(psutil.cpu_percent(interval=None))
    except ImportError:
        pass
    try:
        if os.name == "nt":  # 추가 설치 없이 윈도우 API 로
            import ctypes
            from ctypes import wintypes
            idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
            v = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime  # noqa: E731
            cur = (v(idle), v(kern) + v(user))  # 커널 시간에 유휴 포함
        else:
            a = list(map(int, open("/proc/stat").readline().split()[1:]))
            cur = (a[3] + a[4], sum(a))
        prev, _SYS["cpu_prev"] = _SYS["cpu_prev"], cur
        if not prev or cur[1] == prev[1]:
            return None
        return round(100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1])), 1)
    except Exception:
        return None


def _ram() -> tuple[float, float] | None:
    try:
        import psutil
        m = psutil.virtual_memory()
        return m.used / 2**30, m.total / 2**30
    except ImportError:
        pass
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ax", ctypes.c_ulonglong)]
            m = MS(); m.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return (m.total - m.avail) / 2**30, m.total / 2**30
        d = {ln.split(":")[0]: int(ln.split()[1]) for ln in open("/proc/meminfo")}
        return (d["MemTotal"] - d["MemAvailable"]) / 2**20, d["MemTotal"] / 2**20
    except Exception:
        return None


def _gpu() -> dict | None:
    """nvidia-smi 로 GPU 사용률·메모리·온도 (1.5초 캐시)"""
    now = time.time()
    if now - _SYS["gpu_t"] < 0.9:  # 화면이 1초마다 그린다
        return _SYS["gpu"]
    _SYS["gpu_t"] = now
    if _SYS["nvsmi"] is None:
        cand = [shutil.which("nvidia-smi"), r"C:\Windows\System32\nvidia-smi.exe",
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]
        _SYS["nvsmi"] = next((c for c in cand if c and Path(c).exists()), "")
    if not _SYS["nvsmi"]:
        _SYS["gpu"] = None
        return None
    try:
        r = subprocess.run([_SYS["nvsmi"], "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        n, u, mu, mt, t = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        _SYS["gpu"] = {"name": n, "util": float(u), "mem_used": float(mu) / 1024, "mem_total": float(mt) / 1024,
                       "temp": float(t)}
    except Exception:
        _SYS["gpu"] = None
    return _SYS["gpu"]


_USAGE = {"t": 0.0, "v": None}


def system_usage() -> dict:
    # 화면이 1초마다 묻는다(폰·PC 여러 개면 더 자주) → 0.8초 안에는 같은 값(nvidia-smi 를 매번 띄우지 않게)
    if _USAGE["v"] is not None and time.time() - _USAGE["t"] < 0.8:
        return _USAGE["v"]
    ram = _ram()
    v = {"cpu": _cpu_percent(), "cores": os.cpu_count(),
         "ram_used": round(ram[0], 1) if ram else None, "ram_total": round(ram[1], 1) if ram else None,
         "gpu": _gpu()}
    _USAGE.update(t=time.time(), v=v)
    return v


def photo_dir(q: dict) -> Path | None:
    """사진 보기 창에 보여줄 폴더: 예시(sample=이름) 또는 작업(job=id)의 사진. 그 밖의 경로는 열지 않는다(같은 와이파이에서 접속 가능)"""
    if q.get("sample"):
        for x in list_samples():
            if x["name"] == q["sample"][0] and x.get("path"):
                return Path(x["path"])
        return None
    if q.get("job") and re.fullmatch(r"[0-9a-f]{12}", q["job"][0]):
        d = job_dir(q["job"][0])
        for c in (d / "out" / "work" / "images", d / "input"):
            if c.is_dir() and any(f.suffix.lower() in IMG_EXT for f in c.iterdir()):
                return c
        try:
            lp = json.loads((d / "job.json").read_text(encoding="utf-8")).get("local_path")
            if lp and Path(lp).is_dir():
                return Path(lp)
        except Exception:
            pass
    return None


def photo_thumb(src: Path, w: int) -> Path:
    """빠른 미리보기 그림(긴 변 w px) — JPEG 는 축소 해독(draft)이라 큰 사진도 빠르다. 한 번 만들면 저장해 둔다"""
    import hashlib
    from PIL import Image, ImageOps
    st = src.stat()
    key = hashlib.sha1(f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}|{w}".encode()).hexdigest()
    dst = DATA / "thumbs" / f"{key}.jpg"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(src)
        im.draft("RGB", (w, w))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((w, w))
        tmp = dst.with_suffix(".tmp")
        im.save(tmp, "JPEG", quality=82)
        os.replace(tmp, dst)
    return dst


# ── HTTP ──────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "photo3d"

    def log_message(self, fmt, *args):  # 조용히
        pass

    def handle_one_request(self):  # 요청 하나가 실패해도 앱은 계속 — 원인은 app.log 에 남긴다
        try:
            super().handle_one_request()
        except (ConnectionError, TimeoutError):
            pass
        except Exception:
            import traceback
            with open(JOBS.parent / "app.log", "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + self.requestline + "\n" + traceback.format_exc())

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _file(self, path: Path, ctype: str | None = None, download: str | None = None, cache: int = 0):
        if not path.is_file():
            return self._json({"error": "없음"}, 404)
        types = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css",
                 ".glb": "model/gltf-binary", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".json": "application/json", ".obj": "text/plain",
                 ".log": "text/plain; charset=utf-8", ".wasm": "application/wasm", ".svg": "image/svg+xml",
                 ".ply": "application/octet-stream"}
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "") or "")
        if rng and size:  # 이어받기(브라우저 '다시 시작')
            if rng.group(1):
                start = int(rng.group(1))
                end = int(rng.group(2)) if rng.group(2) else size - 1
            elif rng.group(2):
                start = max(0, size - int(rng.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype or types.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if cache:  # 사진 미리보기처럼 바뀌지 않는 것 — 휠로 넘길 때 다시 받지 않게
            self.send_header("Cache-Control", f"max-age={cache}")
        if download:  # 한글 파일 이름(RFC 5987)
            from urllib.parse import quote
            self.send_header("Content-Disposition", f"attachment; filename=\"download{path.suffix}\"; "
                                                    f"filename*=UTF-8''{quote(download)}")
        self.end_headers()
        left = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            while left > 0:
                b = f.read(min(1 << 20, left))
                if not b:
                    break
                self.wfile.write(b)
                left -= len(b)

    def _body_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        try:
            if p in ("/", "/index.html"):
                return self._file(WEB / "index.html")
            if p == "/api/gpu":
                from . import gpu
                g = gpu.query()
                exes = [str(Path(sys.executable).resolve())]
                cb = find_tools()["colmap"]
                if cb:  # COLMAP.bat 이 아니라 실제로 GPU 를 쓰는 bin\colmap.exe 를 본다
                    exe = Path(cb).parent / "bin" / "colmap.exe"
                    exes.append(str(exe if exe.exists() else Path(cb)))
                from .splat import find_brush
                b = find_brush(ROOT, download=False)
                if b:
                    exes.append(str(Path(b).resolve()))
                w = gpu.windows_settings(exes)
                ck = gpu.checks(g, w)
                lv = {"준비됨": "ok", "준비 중": "info", "실패": "bad"}.get(_PRIOR["state"], "info")
                ck.append({"level": lv, "item": "흰 벽 채우기(깊이 모델)", "value": _PRIOR["state"], "fix": _PRIOR["msg"]})
                return self._json({"gpu": g, "windows": w, "checks": ck,
                                   "plan": gpu.vram_plan(g.get("mem_free_mb"), g.get("mem_total_mb")) if g and g.get("mem_total_mb") else None})
            if p == "/api/sys":
                return self._json(system_usage())
            if p == "/api/photos":  # 사진 보기 창: 사진 목록
                d = photo_dir(parse_qs(u.query))
                if d is None:
                    return self._json({"items": []})
                names = sorted(f.name for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
                return self._json({"items": names, "dir": d.name})
            if p == "/api/photo_thumb":
                q = parse_qs(u.query)
                d = photo_dir(q)
                name = q.get("f", [""])[0]
                if d is None or "/" in name or "\\" in name or not (d / name).is_file():
                    return self._json({"error": "없음"}, 404)
                w = min(2048, max(64, int(q.get("w", ["320"])[0])))
                return self._file(photo_thumb(d / name, w), cache=3600)
            if p == "/api/samples":
                return self._json({"samples": list_samples(), "download": _sample_dl, "site": _site_dl})
            if p == "/api/jobs":
                jobs = []
                for jd in sorted(JOBS.iterdir(), reverse=True):
                    if (jd / "job.json").exists():
                        jobs.append(job_view(json.loads((jd / "job.json").read_text(encoding="utf-8"))))
                jobs.sort(key=lambda j: j.get("created", 0), reverse=True)
                # 대기 이유: 한 번에 하나씩(GPU 하나) — 앞에서 처리 중인 작업과 순서를 알려준다
                run = [j for j in jobs if j["status"] == "running"]
                waiting = sorted((j for j in jobs if j["status"] == "queued"), key=lambda j: j.get("created", 0))
                for k, j in enumerate(waiting):
                    if run:
                        j["wait_info"] = f"'{run[0]['name']}' 처리가 끝나면 시작합니다" + (f" (대기 {k + 1}번째)" if k else "")
                    elif _CURRENT["id"] and _CURRENT["id"] not in {x["id"] for x in jobs}:
                        j["wait_info"] = "지워진 이전 작업을 정리하는 중 — 잠시 후 시작합니다"
                        stop_job(_CURRENT["id"])
                    else:
                        j["wait_info"] = "곧 시작합니다"
                return self._json({"jobs": jobs, "tools": find_tools(), "lan": lan_urls(), "version": VERSION,
                                   "worker": {"busy": _CURRENT["id"],
                                              "for_s": int(time.time() - _CURRENT["t"]) if _CURRENT["id"] else 0}})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})", p)
            if m:
                return self._json(job_view(load_job(m.group(1))))
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/annotations", p)
            if m:
                f = job_dir(m.group(1)) / "out" / "annotations.json"
                return self._file(f) if f.exists() else self._json({"version": 1, "frame": "gltf_yup_m", "items": []})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/photo", p)
            if m:  # 원본(작업용) 사진 — 촬영 위치를 누르면 보여줌
                base = (job_dir(m.group(1)) / "out" / "work" / "images").resolve()
                q = (base / parse_qs(u.query).get("name", [""])[0]).resolve()
                if base not in q.parents or not q.is_file():
                    return self._json({"error": "없음"}, 404)
                return self._file(q)
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/unitypackage", p)
            if m:  # 지금 표시(annotations.json)까지 넣어 새로 만들어 내려준다
                job = load_job(m.group(1))
                from .unitypkg import write_unity_package
                f = write_unity_package(job_dir(m.group(1)) / "out", job.get("name") or job["id"], log=lambda *_: None)
                if f is None:
                    return self._json({"error": "모델이 없어 만들 수 없습니다"}, 404)
                return self._file(f, download=f"{job.get('name') or job['id']}.unitypackage")
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/rev", p)
            if m:  # 미리보기가 바뀌었는지만 가볍게(파일 목록·시각) — 화면이 1초마다 묻는다
                src = _preview_source(job_dir(m.group(1)))
                try:
                    return self._json({"rev": _preview_rev(src) if src else None})
                except Exception:
                    return self._json({"rev": None})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/preview", p)
            if m:
                return self._json(preview_points(job_dir(m.group(1))))
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/log", p)
            if m:
                return self._file(job_dir(m.group(1)) / "console.log", "text/plain; charset=utf-8")
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/thumb", p)
            if m:
                d = job_dir(m.group(1))
                lp = load_job(m.group(1)).get("local_path")
                src = Path(lp) if lp else d / "input"
                imgs = sorted((d / "out" / "work" / "images").glob("*.jpg")) or \
                    sorted(q for q in src.glob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
                return self._file(imgs[0]) if imgs else self._json({"error": "없음"}, 404)
            if p in ("/supersplat", "/supersplat/"):  # 스플랫 편집기(SuperSplat, MIT) — ?load=/files/<id>/site_splat.ply
                return self._file(WEB / "vendor" / "supersplat" / "index.html")
            m = re.fullmatch(r"/supersplat/([\w./\-]+)", p)
            if m and ".." not in m.group(1):
                return self._file(WEB / "vendor" / "supersplat" / m.group(1))
            m = re.fullmatch(r"/vendor/([\w./\-]+\.js)", p)
            if m and ".." not in m.group(1):  # 내장 three.js (현장 오프라인에서도 뷰어 동작)
                return self._file(WEB / "vendor" / m.group(1), "text/javascript; charset=utf-8")
            m = re.fullmatch(r"/files/([0-9a-f]{12})/([\w.\-]+)", p)
            if m:  # 결과 파일(site_model.glb, site_collider.glb, report.json ...) — ?dl=이름 이면 내려받기
                q = parse_qs(u.query).get("dl", [None])[0]
                return self._file(job_dir(m.group(1)) / "out" / m.group(2), download=q)
            return self._json({"error": "없는 경로"}, 404)
        except KeyError:
            return self._json({"error": "작업 없음"}, 404)

    def do_POST(self):
        p = unquote(urlparse(self.path).path)
        try:
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/open", p)
            if m:  # 이 PC 에서만: 결과 폴더를 탐색기로 열기(내려받기가 안 될 때 바로 복사)
                if self.client_address[0] not in ("127.0.0.1", "::1"):
                    return self._json({"error": "PC 에서만 쓸 수 있습니다"}, 403)
                d = job_dir(m.group(1)) / "out"
                if os.name == "nt":
                    os.startfile(str(d))  # noqa: S606
                return self._json({"path": str(d)})
            if p == "/api/gpu/tune":  # 이 PC 에서만: 관리자 확인 창을 띄워 GPU 최적 설정(scripts/gpu_tune.ps1)
                if self.client_address[0] not in ("127.0.0.1", "::1"):
                    return self._json({"error": "PC 에서만 쓸 수 있습니다"}, 403)
                if os.name != "nt":
                    return self._json({"error": "윈도우 PC 에서만 필요한 설정입니다"}, 400)
                ps1 = ROOT / "scripts" / "gpu_tune.ps1"
                subprocess.Popen(["powershell", "-NoProfile", "-Command",
                                  f"Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"{ps1}\"'"],
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                return self._json({"ok": True})
            if p == "/api/browse":  # 이 PC 에서만: 윈도우 폴더 선택 창
                if self.client_address[0] not in ("127.0.0.1", "::1"):
                    return self._json({"error": "PC 에서만 쓸 수 있습니다"}, 403)
                code = ("import tkinter as t, tkinter.filedialog as f; r=t.Tk(); r.withdraw(); "
                        "r.attributes('-topmost', True); print(f.askdirectory(title='사진 폴더 고르기') or '')")
                r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                   encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                path = (r.stdout or "").strip()
                if not path:
                    return self._json({"path": None})
                pp = Path(path)
                direct = lambda d: [q for q in d.iterdir() if q.is_file() and q.suffix.lower() in MEDIA]
                if not direct(pp):  # 상위 폴더를 고른 경우: 사진이 든 하위 폴더(예: images)를 쓴다
                    subs = sorted((d for d in pp.rglob("*") if d.is_dir() and direct(d)),
                                  key=lambda d: (d.name.lower() != "images", -len(direct(d))))
                    if subs:
                        pp = subs[0]
                cnt = len(direct(pp))
                return self._json({"path": str(pp), "name": pp.name if pp.name.lower() not in ("images", "image", "photos") else pp.parent.name, "count": cnt})
            if p == "/api/samples/download":
                key = self._body_json().get("key")
                if key:  # 현장 예시 하나
                    if not any(x["key"] == key for x in SITE_SAMPLES):
                        return self._json({"error": "모르는 예시"}, 400)
                    if _site_dl.get(key, {}).get("state") != "running":
                        _site_dl[key] = {"state": "running", "msg": "받는 중…"}
                        threading.Thread(target=download_site_sample, args=(key,), daemon=True).start()
                    return self._json({"site": _site_dl})
                if _sample_dl["state"] != "running":
                    threading.Thread(target=download_samples, daemon=True).start()
                    _sample_dl.update(state="running", msg="예시 사진 받는 중…")
                return self._json({"download": _sample_dl})
            if p == "/api/jobs":
                b = self._body_json()
                jid = uuid.uuid4().hex[:12]
                d = job_dir(jid)
                (d / "input").mkdir(parents=True)
                lp = (b.get("local_path") or "").strip().strip('"')
                if lp and not Path(lp).exists():
                    shutil.rmtree(d)
                    return self._json({"error": f"PC 에 없는 경로: {lp}"}, 400)
                job = {"id": jid, "name": (b.get("name") or "현장").strip()[:60], "status": "uploading",
                       "created": time.time(), "local_path": lp or None,
                       "scale": b.get("scale") or None, "marker_size": b.get("marker_size") or None,
                       "mode": "best" if b.get("mode") == "best" else "fast", "subject": bool(b.get("subject")),
                       "kind": b.get("kind") if b.get("kind") in ("pg", "fuse") else "splat"}
                save_job(job)
                return self._json(job, 201)
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/stop", p)
            if m:  # 처리 멈추기(대기 중이면 대기열에서 빼기)
                job = load_job(m.group(1))
                if job["status"] == "queued":
                    job.update(status="failed", error="멈춤(대기 중 취소) — '이어서 처리'로 다시 시작")
                    save_job(job)
                    return self._json(job)
                if not stop_job(job["id"]):
                    return self._json({"error": "처리 중인 작업이 아닙니다"}, 409)
                return self._json({"ok": True})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/resume", p)
            if m:  # 멈춘 곳부터 이어서
                job = load_job(m.group(1))
                if job["status"] != "failed":
                    return self._json({"error": "멈춘 작업만 이어서 처리할 수 있습니다"}, 409)
                job.update(status="queued", error=None, resume=True)
                save_job(job)
                _queue.put(job["id"])
                return self._json(job)
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/start", p)
            if m:
                job = load_job(m.group(1))
                if job["status"] not in ("uploading", "failed", "done"):
                    return self._json({"error": "이미 처리 중이거나 완료"}, 409)
                if not job.get("local_path") and not any((job_dir(job["id"]) / "input").iterdir()):
                    return self._json({"error": "올린 파일이 없습니다"}, 400)
                job.update(status="queued", error=None)
                save_job(job)
                _queue.put(job["id"])
                return self._json(job)
            return self._json({"error": "없는 경로"}, 404)
        except KeyError:
            return self._json({"error": "작업 없음"}, 404)

    def do_PUT(self):
        u = urlparse(self.path)
        ma = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/annotations", unquote(u.path))
        if ma:  # 현장 표시(거리·반경·구역·위험) 저장 → 유니티가 같은 좌표로 읽음
            try:
                b = self._body_json()
                if not isinstance(b.get("items"), list) or len(json.dumps(b)) > 2_000_000:
                    return self._json({"error": "형식 오류"}, 400)
                f = job_dir(ma.group(1)) / "out" / "annotations.json"
                f.write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
                return self._json({"ok": True, "n": len(b["items"])})
            except KeyError:
                return self._json({"error": "작업 없음"}, 404)
        m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/files", unquote(u.path))
        if not m:
            return self._json({"error": "없는 경로"}, 404)
        name = Path(parse_qs(u.query).get("name", ["file"])[0]).name
        ext = Path(name).suffix.lower()
        if ext not in MEDIA:
            from .cam360 import UNSUPPORTED_EXT
            hint = UNSUPPORTED_EXT.get(ext)
            return self._json({"error": f"지원하지 않는 형식: {ext}" + (f" — {hint}" if hint else "")}, 400)
        try:
            d = job_dir(m.group(1)) / "input"
        except KeyError:
            return self._json({"error": "작업 없음"}, 404)
        safe = re.sub(r"[^\w.\-]", "_", name)
        n = int(self.headers.get("Content-Length", 0) or 0)
        with open(d / safe, "wb") as f:
            left = n
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        return self._json({"ok": True, "name": safe, "bytes": n})

    def do_DELETE(self):
        m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})", unquote(urlparse(self.path).path))
        if not m:
            return self._json({"error": "없는 경로"}, 404)
        try:
            job = load_job(m.group(1))
        except KeyError:
            return self._json({"error": "작업 없음"}, 404)
        if job["status"] in ("running",):
            return self._json({"error": "처리 중에는 지울 수 없습니다"}, 409)
        shutil.rmtree(job_dir(job["id"]), ignore_errors=True)
        return self._json({"ok": True})


def lan_urls(port: int = 8765) -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # 실제로 보내지 않음, 바깥으로 나가는 인터페이스 IP 확인용
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return [f"http://{ip}:{port}" for ip in sorted(ips) if not ip.startswith("127.")]


def _no_quickedit() -> None:
    """윈도우 콘솔의 '빠른 편집' 끄기 — 검은 창을 클릭하면 프로그램이 멈춰 앱 연결이 끊긴다."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            k.SetConsoleMode(h, (mode.value & ~0x0040) | 0x0080)  # QUICK_EDIT 끔, EXTENDED_FLAGS 켬
    except Exception:
        pass


_PRIOR = {"state": "확인 전", "msg": ""}


def prepare_prior() -> None:
    """앱 시작 시 흰 벽 채우기(MoGe-2)에 필요한 PyTorch(GPU)·MoGe·모델 파일을 미리 준비 — 첫 처리 때
    설치하느라 실패·건너뛰는 일이 없게(PC '집': 흰 벽 채우기가 돌지 않아 벽이 비었음)"""
    from . import gpu
    if not gpu.nvidia_smi():
        _PRIOR.update(state="해당 없음", msg="NVIDIA GPU 없음 → CPU 처리에서는 사용 안 함")
        return
    _PRIOR.update(state="준비 중", msg="PyTorch(GPU)·MoGe-2 확인/설치 중 — 처음 한 번 수 분")
    msgs = []
    try:
        from . import priorfill
        if not priorfill.ensure_installed(msgs.append):
            _PRIOR.update(state="실패", msg="설치 실패: " + " / ".join(msgs[-2:]))
            return
        import torch
        if not torch.cuda.is_available():
            _PRIOR.update(state="실패", msg=f"PyTorch 가 GPU 를 못 찾음(torch {torch.__version__}) — CUDA 판 재설치 필요")
            return
        _PRIOR.update(msg="깊이 모델 파일 내려받는 중(약 1.3GB, 처음 한 번)")
        from huggingface_hub import snapshot_download
        snapshot_download("Ruicheng/moge-2-vitl-normal")
        _PRIOR.update(state="준비됨", msg=f"PyTorch {torch.__version__} · {torch.cuda.get_device_name(0)}")
    except Exception as e:
        _PRIOR.update(state="실패", msg=str(e)[:200])


def main():
    _no_quickedit()
    port = int(os.environ.get("PHOTO3D_PORT", "8765"))
    # 재시작 시 '처리 중'으로 남은 작업은 실패로 표시(대기열은 메모리라 사라짐)
    for jd in JOBS.iterdir():
        try:
            job = json.loads((jd / "job.json").read_text(encoding="utf-8"))
            if job["status"] in ("running", "queued"):
                job.update(status="failed", error="앱이 다시 시작되어 중단됨 — '다시 처리'를 누르세요")
                save_job(job)
        except Exception:
            pass
    if not HIST.exists():  # 처음: 이미 끝난 작업들의 단계 시간으로 예측 기록을 채운다
        for jd in sorted(JOBS.iterdir()):
            try:
                jb = json.loads((jd / "job.json").read_text(encoding="utf-8"))
                if jb.get("status") == "done":
                    record_timing(jd, jb)
            except Exception:
                pass
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=prepare_prior, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"photo3d 앱: http://localhost:{port}")
    for u in lan_urls(port):
        print(f"  같은 와이파이의 폰에서: {u}")
    print("  (끄려면 이 창을 닫으세요)")
    if os.environ.get("PHOTO3D_NO_BROWSER") != "1":
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
