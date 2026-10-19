"""폰 앱(현장 촬영 → PC 로 자동 업로드 → 3D 생성) 연결.

- 짝맞춤 토큰: PC 화면의 QR 로 폰에 전달. PC 밖에서 오는 요청은 토큰(쿠키 p3d)이 있어야 받는다.
- 폰 카메라(자동 촬영·흔들림 감지)는 보안 연결(HTTPS)에서만 열린다 →
  ① 같은 와이파이: 자체 서명 인증서로 https://<PC IP>:8766 (처음 한 번 '계속' 눌러 허용)
  ② 현장(LTE 등 어디서나): Cloudflare 임시 터널(cloudflared) → https://xxxx.trycloudflare.com (인증서 경고 없음)
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import os
import re
import secrets
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HTTPS_PORT = int(os.environ.get("PHOTO3D_HTTPS_PORT", "8766"))


def token(data: Path) -> str:
    f = data / "mobile_token.txt"
    if not f.exists():
        f.write_text(secrets.token_urlsafe(18), encoding="ascii")
    return f.read_text(encoding="ascii").strip()


def _ensure_crypto():
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cryptography"], capture_output=True)
        return r.returncode == 0


def cert(data: Path, ips: list[str]) -> tuple[Path, Path] | None:
    """LAN HTTPS 용 자체 서명 인증서(PC IP 가 바뀌면 다시 만든다)"""
    crt, key = data / "mobile_cert.pem", data / "mobile_key.pem"
    tag = data / "mobile_cert_ips.txt"
    want = ",".join(sorted(ips))
    if crt.exists() and key.exists() and tag.exists() and tag.read_text() == want:
        return crt, key
    if not _ensure_crypto():
        return None
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    k = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "photo3d")])
    san = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    now = dt.datetime.now(dt.timezone.utc)
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
         .not_valid_after(now + dt.timedelta(days=3650))
         .add_extension(x509.SubjectAlternativeName(san), critical=False).sign(k, hashes.SHA256()))
    crt.write_bytes(c.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
    tag.write_text(want)
    return crt, key


def serve_https(handler_cls, data: Path, ips: list[str], log=print) -> int | None:
    """같은 와이파이용 HTTPS 서버를 별도 스레드로. 성공하면 포트"""
    from http.server import ThreadingHTTPServer
    try:
        ck = cert(data, ips)
        if ck is None:
            log("  폰 HTTPS 생략: 인증서 도구(cryptography) 설치 실패")
            return None
        srv = ThreadingHTTPServer(("0.0.0.0", HTTPS_PORT), handler_cls)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(ck[0]), str(ck[1]))
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return HTTPS_PORT
    except Exception as e:
        log(f"  폰 HTTPS 생략: {e}")
        return None


# ---- 현장(인터넷)에서 접속: Cloudflare 임시 터널 ----
TUNNEL = {"state": "off", "url": None, "msg": ""}
_proc: subprocess.Popen | None = None
CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"


def _cloudflared(root: Path) -> Path | None:
    exe = root / "third_party" / ("cloudflared.exe" if os.name == "nt" else "cloudflared")
    if exe.exists():
        return exe
    if os.name != "nt":
        from shutil import which
        w = which("cloudflared")
        return Path(w) if w else None
    exe.parent.mkdir(parents=True, exist_ok=True)
    TUNNEL.update(msg="연결 프로그램(cloudflared, 약 60MB) 받는 중 — 처음 한 번")
    tmp = exe.with_suffix(".part")
    urllib.request.urlretrieve(CF_URL, tmp)  # noqa: S310
    tmp.replace(exe)
    return exe


def _reachable(url: str, timeout: float = 8) -> bool:
    """터널 주소가 바깥에서 실제로 열리는지(/m 이 200). 이 PC 의 DNS 는 '없는 주소'를 몇 분 기억하므로
    Cloudflare 공개 DNS(DoH)로 IP 를 찾아 그 IP 로 직접 접속해 본다."""
    import json
    import socket
    host = url.split("//", 1)[1].rstrip("/")
    try:
        req = urllib.request.Request(f"https://cloudflare-dns.com/dns-query?name={host}&type=A",
                                     headers={"accept": "application/dns-json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            ips = [a["data"] for a in json.load(r).get("Answer", []) if a.get("type") == 1]
        if not ips:
            return False
        with socket.create_connection((ips[0], 443), timeout=timeout) as sock:
            with ssl.create_default_context().wrap_socket(sock, server_hostname=host) as ss:
                ss.sendall(f"GET /m HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode())
                return b" 200 " in ss.recv(64).split(b"\r\n")[0] + b" "
    except Exception:
        return False


def tunnel_start(root: Path, port: int) -> None:
    """Cloudflare 임시 터널. 주소를 받으면 바깥에서 실제로 열리는지 확인한 뒤에만 'on'(QR 표시).
    임시 터널은 가끔 주소만 주고 등록에 실패한다(2026-10-19: DNS 에 없는 주소로 QR 이 떠서 폰에서 안 열림)
    → 안 열리면 새 터널로 다시(최대 4번), 켜져 있는 동안 1분마다 점검해 끊기면 자동으로 다시 연결."""
    if TUNNEL["state"] in ("starting", "on"):
        return
    gen = TUNNEL.get("gen", 0) + 1
    TUNNEL.update(state="starting", url=None, msg="현장 접속 주소 만드는 중…", gen=gen)

    def alive():
        return TUNNEL.get("gen") == gen and TUNNEL["state"] != "off"

    def run():
        global _proc
        exe = None
        try:
            exe = _cloudflared(root)
        except Exception as e:
            TUNNEL.update(state="error", url=None, msg=f"터널 실패: {e}")
            return
        if exe is None:
            TUNNEL.update(state="error", msg="cloudflared 를 찾지 못했습니다")
            return
        fails = 0
        while alive() and fails < 4:
            TUNNEL.update(state="starting", url=None,
                          msg="현장 접속 주소 만드는 중…" + (f" (다시 시도 {fails}/3)" if fails else ""))
            proc = subprocess.Popen([str(exe), "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace",
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _proc = proc
            found = {"url": None}

            def drain():  # 출력을 계속 읽어 줘야 cloudflared 가 멈추지 않는다
                for line in proc.stdout:
                    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                    if m and not found["url"]:
                        found["url"] = m.group(0)
            threading.Thread(target=drain, daemon=True).start()
            t0 = time.time()
            while alive() and not found["url"] and proc.poll() is None and time.time() - t0 < 60:
                time.sleep(0.5)
            ok = False
            if found["url"]:
                # 임시 터널 주소가 Cloudflare DNS 에 올라가기까지 요즘 약 1분(2026-10-19 실측 60초) → 3분까지 기다린다
                TUNNEL.update(msg="주소 등록 대기 중(보통 1~2분) — 열리는 것을 확인한 뒤 QR 이 나옵니다")
                t1 = time.time()
                while alive() and proc.poll() is None and time.time() - t1 < 180:
                    if _reachable(found["url"]):
                        ok = True
                        break
                    time.sleep(4)
            if ok and alive():
                fails = 0
                TUNNEL.update(state="on", url=found["url"], msg="현장(LTE 등)에서도 접속됩니다 — PC 와 이 앱은 켜 두세요")
                bad = 0
                while alive() and proc.poll() is None and bad < 2:  # 켜져 있는 동안 1분마다 점검
                    time.sleep(60)
                    bad = 0 if _reachable(found["url"]) else bad + 1
            else:
                fails += 1
            if proc.poll() is None:
                proc.terminate()
        if alive():
            TUNNEL.update(state="error", url=None, msg="현장 접속 주소를 만들지 못했습니다 — 같은 와이파이 QR 을 쓰거나 잠시 뒤 다시 켜 주세요")

    threading.Thread(target=run, daemon=True).start()


def tunnel_stop() -> None:
    global _proc
    TUNNEL.update(state="off", url=None, msg="")
    if _proc and _proc.poll() is None:
        _proc.terminate()
    _proc = None


def qr_svg(text: str) -> bytes | None:
    """QR 코드 SVG(segno — 순수 파이썬, 처음 한 번 설치)"""
    try:
        import segno
    except ImportError:
        if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segno"], capture_output=True).returncode:
            return None
        import segno
    import io
    b = io.BytesIO()
    segno.make(text, error="m").save(b, kind="svg", scale=6, border=2, dark="#000", light="#fff")
    return b.getvalue()
