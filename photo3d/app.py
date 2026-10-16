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
         ".mp4", ".mov", ".m4v", ".avi", ".mkv"}

_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()


# ── 도구 위치 ─────────────────────────────────────────
def find_tools() -> dict:
    """PC: third_party 의 COLMAP(CUDA)·OpenMVS(CPU판). 없으면 None(→ pycolmap CPU 경로)."""
    tp = ROOT / "third_party"
    colmap = next(iter(sorted(tp.glob("colmap/**/COLMAP.bat"))), None)
    mvs = next(iter(sorted(tp.glob("openmvs_cpu/**/ReconstructMesh.exe"))), None)
    return {"colmap": str(colmap) if colmap else None, "openmvs": str(mvs.parent) if mvs else None}


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


def job_view(job: dict) -> dict:
    d = job_dir(job["id"])
    v = dict(job)
    log = d / "out" / "run.log"
    if job["status"] == "running" and log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        stage = None
        for ln in lines:
            mm = re.search(r"\]\s+(\d/5)\s", ln)
            if mm:
                stage = dict(STAGES).get(mm.group(1))
                stage = f"{mm.group(1)} {stage}"
        if stage:
            v["stage"] = stage
        v["last_log"] = lines[-1][-160:] if lines else ""
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
        tail = (d / "console.log").read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        job["error"] = "\n".join(tail)
    save_job(job)


# ── HTTP ──────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "photo3d"

    def log_message(self, fmt, *args):  # 조용히
        pass

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


def main():
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
    srv.serve_forever()


if __name__ == "__main__":
    main()
