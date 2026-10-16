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
         ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

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


def list_samples() -> list[dict]:
    out = []
    data = ROOT / "data"
    if not data.exists():
        return out
    for d in sorted(data.glob("**/images")):
        n = len([q for q in d.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if n >= 5:
            name = d.parent.name
            out.append({"name": name, "path": str(d), "count": n, "desc": SAMPLE_INFO.get(name, f"{n}장")})
    return out


_sample_dl = {"state": "idle", "msg": ""}


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
        kind, extra = "depth", {"done": done, "total": total}
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
        human = [ln for ln in lines if "$ " not in ln and ln.strip()]  # 실행 명령(경로 덩어리)은 숨김
        v["last_log"] = re.sub(r"^\[\s*[\d.]+s\]\s*", "", human[-1]).strip()[-160:] if human else ""
        src = _preview_source(d)
        v["preview_rev"] = _preview_rev(src) if src else None
    rep = d / "out" / "report.json"
    if rep.exists():
        try:
            r = json.loads(rep.read_text(encoding="utf-8"))
            v["grade"] = r.get("grade")
            v["size_m"] = r.get("model", {}).get("size_m")
            v["registered"] = f"{r['sfm']['images_registered']}/{r['sfm']['images_input']}"
        except Exception:
            pass
    v["has_model"] = (d / "out" / "site_model.glb").exists()
    v["has_filled"] = (d / "out" / "site_filled.glb").exists()
    v["has_est"] = (d / "out" / "site_estimated.glb").exists()
    v["has_splat"] = (d / "out" / "site_splat.ply").exists()
    v["has_report"] = (d / "out" / "quality_report.html").exists()
    v["has_cams"] = (d / "out" / "cameras.json").exists()
    v["files"] = len([p for p in (d / "input").iterdir()]) if (d / "input").exists() else 0
    return v


# ── 처리 작업자 ───────────────────────────────────────
def worker() -> None:
    while True:
        jid = _queue.get()
        try:
            run_job(jid)
        except Exception as e:  # 작업자가 죽지 않게
            try:
                job = load_job(jid)
                job.update(status="failed", error=str(e), finished=time.time())
                save_job(job)
            except Exception:
                pass


def run_job(jid: str) -> None:
    d = job_dir(jid)
    job = load_job(jid)
    shutil.rmtree(d / "out", ignore_errors=True)  # 다시 처리: 이전 결과·미리보기를 지우고 새로
    job.update(status="running", started=time.time())
    save_job(job)
    tools = find_tools()
    env = dict(os.environ)
    if tools["openmvs"]:
        env["OPENMVS_BIN"] = tools["openmvs"]
    env["PYTHONIOENCODING"] = "utf-8"
    src = job.get("local_path") or str(d / "input")
    profile = "gpu-max" if job.get("mode") == "best" else "fast"
    cmd = [sys.executable, "-m", "photo3d", "run", src, "--profile", profile, "--out", str(d / "out")]
    if tools["colmap"]:
        cmd += ["--colmap-bin", tools["colmap"]]
    if job.get("scale"):
        sc = d / "scale.csv"
        sc.write_text(job["scale"], encoding="utf-8")
        cmd += ["--scale-dist", str(sc)]
    if job.get("marker_size"):
        cmd += ["--marker-size", str(job["marker_size"])]
    with open(d / "console.log", "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    job = load_job(jid)
    ok = r.returncode == 0 and (d / "out" / "site_model.glb").exists()
    job.update(status="done" if ok else "failed", finished=time.time())
    if not ok:
        lines = (d / "console.log").read_text(encoding="utf-8", errors="replace").splitlines()
        job["error"] = explain_error(lines)
    save_job(job)


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
    if now - _SYS["gpu_t"] < 1.5:
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


def system_usage() -> dict:
    ram = _ram()
    return {"cpu": _cpu_percent(), "cores": os.cpu_count(),
            "ram_used": round(ram[0], 1) if ram else None, "ram_total": round(ram[1], 1) if ram else None,
            "gpu": _gpu()}


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

    def _file(self, path: Path, ctype: str | None = None):
        if not path.is_file():
            return self._json({"error": "없음"}, 404)
        types = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css",
                 ".glb": "model/gltf-binary", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".json": "application/json", ".obj": "text/plain",
                 ".log": "text/plain; charset=utf-8"}
        self.send_response(200)
        self.send_header("Content-Type", ctype or types.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 1 << 20)

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
                exes = [str(Path(sys.executable).resolve())] + [x for x in (find_tools()["colmap"],) if x]
                from .splat import find_brush
                b = find_brush(ROOT, download=False)
                if b:
                    exes.append(str(Path(b).resolve()))
                w = gpu.windows_settings(exes)
                return self._json({"gpu": g, "windows": w, "checks": gpu.checks(g, w),
                                   "plan": gpu.vram_plan(g.get("mem_free_mb"), g.get("mem_total_mb")) if g and g.get("mem_total_mb") else None})
            if p == "/api/sys":
                return self._json(system_usage())
            if p == "/api/samples":
                return self._json({"samples": list_samples(), "download": _sample_dl})
            if p == "/api/jobs":
                jobs = []
                for jd in sorted(JOBS.iterdir(), reverse=True):
                    if (jd / "job.json").exists():
                        jobs.append(job_view(json.loads((jd / "job.json").read_text(encoding="utf-8"))))
                jobs.sort(key=lambda j: j.get("created", 0), reverse=True)
                return self._json({"jobs": jobs, "tools": find_tools(), "lan": lan_urls()})
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
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/preview", p)
            if m:
                return self._json(preview_points(job_dir(m.group(1))))
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/log", p)
            if m:
                return self._file(job_dir(m.group(1)) / "console.log", "text/plain; charset=utf-8")
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/thumb", p)
            if m:
                d = job_dir(m.group(1))
                imgs = sorted((d / "out" / "work" / "images").glob("*.jpg")) or \
                    sorted(q for q in (d / "input").glob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
                return self._file(imgs[0]) if imgs else self._json({"error": "없음"}, 404)
            m = re.fullmatch(r"/vendor/([\w./\-]+\.js)", p)
            if m and ".." not in m.group(1):  # 내장 three.js (현장 오프라인에서도 뷰어 동작)
                return self._file(WEB / "vendor" / m.group(1), "text/javascript; charset=utf-8")
            m = re.fullmatch(r"/files/([0-9a-f]{12})/([\w.\-]+)", p)
            if m:  # 결과 파일(site_model.glb, site_collider.glb, report.json ...)
                return self._file(job_dir(m.group(1)) / "out" / m.group(2))
            return self._json({"error": "없는 경로"}, 404)
        except KeyError:
            return self._json({"error": "작업 없음"}, 404)

    def do_POST(self):
        p = unquote(urlparse(self.path).path)
        try:
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
                       "mode": "best" if b.get("mode") == "best" else "fast"}
                save_job(job)
                return self._json(job, 201)
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/start", p)
            if m:
                job = load_job(m.group(1))
                if job["status"] not in ("uploading", "failed"):
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
            return self._json({"error": f"지원하지 않는 형식: {ext}"}, 400)
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
    threading.Thread(target=worker, daemon=True).start()
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
