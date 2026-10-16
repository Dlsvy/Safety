"""360 카메라 파일 처리 — 촬영 기종 알아내기, 원본(어안) 파일 나누기.

지원
- 이미 펼친 360 사진·영상(2:1 등장방형): 기존처럼 원근 사진으로 자름(frames.equirect_views)
- 앞뒤 어안 2개가 한 화면에 나란히(2:1, 네 귀퉁이 검정): Insta360 .insp/.insv, 리코 세타 원본 영상 등
  → 렌즈별로 잘라 lens0/·lens1/ 폴더(렌즈마다 카메라 하나, OPENCV_FISHEYE)로. 붙이기(스티칭) 없이 원본 그대로 써서
    이음매 왜곡이 없고, 렌즈 왜곡은 SfM 이 스스로 맞춘다(r/GaussianSplatting 권장: 펼친 영상보다 원본 어안)
- 렌즈 하나가 파일 하나(정사각형 어안): Insta360 X3·X4 5.7K 영상의 _00_/_10_ 파일 → 같은 방식
- GoPro MAX(.360)·DJI(.osv) 원본은 전용 투영(EAC 등)이라 → 제조사 프로그램에서 '360 MP4(등장방형)'로 내보내 올리라고 안내
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

RAW_IMG_EXT = {".insp"}
RAW_VID_EXT = {".insv"}
UNSUPPORTED_EXT = {".360": "GoPro MAX — GoPro Player 에서 '360 동영상(MP4)'으로 내보내 올려주세요",
                   ".osv": "DJI Osmo 360 — DJI Studio 에서 '360 동영상(MP4, 등장방형)'으로 내보내 올려주세요",
                   ".lrv": "저해상도 미리보기 파일입니다 — 같은 이름의 원본(.insv/.mp4)을 올려주세요"}

_MODEL_RE = re.compile(rb"(Insta360[ _][A-Za-z0-9 ]{1,20}|ONE ?X2|ONE ?RS|ONE ?R|GoPro ?MAX|GoPro ?Max|"
                       rb"RICOH THETA ?[A-Za-z0-9]{0,6}|QooCam[ A-Za-z0-9]{0,10}|Osmo ?360|Insta360 ?Ace[ A-Za-z0-9]{0,6})")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\x00", " ")).strip()


def detect_model(path: Path) -> str | None:
    """촬영 기종: 사진 EXIF(제조사·모델) → 영상 메타데이터(ffprobe) → 파일 앞뒤에서 알려진 360 카메라 이름 찾기."""
    path = Path(path)
    try:  # 사진(.jpg/.insp 등 JPEG 내용)
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            make, model = ex.get(271), ex.get(272)
            if model:
                make = _clean(str(make or ""))
                model = _clean(str(model))
                return model if make and make.lower().split()[0] in model.lower() else f"{make} {model}".strip()
    except Exception:
        pass
    try:  # 영상 메타데이터
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "default=nw=1",
                            str(path)], capture_output=True, text=True, timeout=20).stdout
        tags = dict(ln.split("=", 1) for ln in o.splitlines() if "=" in ln)
        for k in ("com.apple.quicktime.model", "model", "TAG:model"):
            if tags.get(k):
                mk = tags.get("com.apple.quicktime.make") or tags.get("make") or ""
                return _clean(f"{mk} {tags[k]}")
        hn = " ".join(v for k, v in tags.items() if "handler" in k.lower() or "encoder" in k.lower())
        if "GoPro" in hn:
            return "GoPro (" + _clean(hn)[:40] + ")"
    except Exception:
        pass
    try:  # 360 카메라는 메타데이터가 파일 끝(Insta360 트레일러)이나 앞에 문자열로 들어 있는 경우가 많다
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(4 << 20)
            f.seek(max(0, size - (4 << 20)))
            tail = f.read()
        m = _MODEL_RE.search(tail) or _MODEL_RE.search(head)
        if m:
            return _clean(m.group(1).decode("ascii", "ignore"))
    except Exception:
        pass
    return None


def classify(img: np.ndarray) -> str:
    """'equirect'(펼친 360) / 'dual_fisheye'(어안 2개 나란히) / 'fisheye'(어안 1개) / 'normal'"""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    c = max(4, min(h, w) // 40)

    def dark(y, x):
        return float(g[y:y + c, x:x + c].mean()) < 12
    corners = [dark(0, 0), dark(0, w - c), dark(h - c, 0), dark(h - c, w - c)]
    ar = w / max(h, 1)
    if 1.9 < ar < 2.1:
        mid = dark(0, w // 2 - c // 2) and dark(h - c, w // 2 - c // 2)  # 두 원 사이 위·아래도 검정
        if all(corners) and mid:
            return "dual_fisheye"
        return "equirect"
    if 0.95 < ar < 1.05 and all(corners):
        return "fisheye"
    return "normal"


def circle_mask(size: int, shrink: float = 0.97) -> np.ndarray:
    """어안 원 안쪽만 흰색(가장자리 3% 는 번짐이 커서 뺀다)"""
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), int(size / 2 * shrink), 255, -1)
    return m


def split_dual(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    s = min(h, w // 2)
    return [img[(h - s) // 2:(h - s) // 2 + s, 0:s], img[(h - s) // 2:(h - s) // 2 + s, w // 2:w // 2 + s]]


def lens_index_from_name(name: str) -> int | None:
    """Insta360 X3·X4 등: 렌즈별 파일 이름 VID_..._00_xxx.insv / _10_ → 0 / 1"""
    m = re.search(r"_(00|10)_\d+", name)
    return {"00": 0, "10": 1}[m.group(1)] if m else None


def write_lens_images(img: np.ndarray, kind: str, out_dir: Path, mask_dir: Path, stem: str, max_side: int,
                      lens_hint: int | None = None) -> list[Path]:
    """어안 사진(1개 또는 2개 나란히)을 렌즈별 폴더로 저장 + 원 바깥 마스크(COLMAP: <사진 경로>.png)"""
    parts = split_dual(img) if kind == "dual_fisheye" else [img]
    out = []
    for k, part in enumerate(parts):
        lens = lens_hint if (kind == "fisheye" and lens_hint is not None) else k
        s = part.shape[0]
        if s > max_side:
            part = cv2.resize(part, (max_side, max_side), interpolation=cv2.INTER_AREA)
            s = max_side
        rel = Path(f"lens{lens}") / f"{stem}.jpg"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / rel), part, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (mask_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_dir / f"{rel}.png"), circle_mask(s))
        out.append(out_dir / rel)
    return out


KIND_KO = {"equirect": "360 펼친 영상(2:1)", "dual_fisheye": "360 원본(앞뒤 어안 2개)",
           "fisheye": "360 원본(렌즈별 어안)", "normal": "일반"}
