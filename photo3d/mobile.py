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


def tunnel_start(root: Path, port: int) -> None:
    if TUNNEL["state"] in ("starting", "on"):
        return
    TUNNEL.update(state="starting", url=None, msg="현장 접속 주소 만드는 중…")

    def run():
        global _proc
        try:
            exe = _cloudflared(root)
            if exe is None:
                TUNNEL.update(state="error", msg="cloudflared 를 찾지 못했습니다")
                return
            _proc = subprocess.Popen([str(exe), "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace",
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            t0 = time.time()
            for line in _proc.stdout:
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if m and not TUNNEL["url"]:
                    TUNNEL.update(state="on", url=m.group(0), msg="현장(LTE 등)에서도 접속됩니다 — PC 와 이 앱은 켜 두세요")
                if TUNNEL["state"] == "starting" and time.time() - t0 > 60:
                    break
            if TUNNEL["state"] != "off":
                TUNNEL.update(state="error", url=None, msg="터널이 끊겼습니다 — 다시 켜 주세요")
        except Exception as e:
            TUNNEL.update(state="error", url=None, msg=f"터널 실패: {e}")

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
