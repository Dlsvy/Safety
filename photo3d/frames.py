"""입력 정리: 사진 폴더 또는 동영상 → 복원에 쓸 선명한 이미지 세트.

동영상은 ffmpeg 로 촘촘히 뽑은 뒤, 구간마다 가장 선명한(라플라시안 분산이 큰) 프레임
하나만 남긴다. 흔들린 프레임이 섞이면 특징점 매칭과 조밀복원 정확도가 함께 떨어진다.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".webp"}
VID_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def sharpness(path: Path, max_side: int = 1024) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    s = max_side / max(img.shape)
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


_DET = None


def _has_marker(path: Path) -> bool:
    global _DET
    if _DET is None:
        _DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    _, ids, _ = _DET.detectMarkers(img)
    return ids is not None


def _resize_copy(src: Path, dst: Path, max_side: int) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"이미지를 읽을 수 없음: {src}")
    s = max_side / max(img.shape[:2])
    if max_side and s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst.with_suffix(".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif src.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(src, dst.with_suffix(".jpg"))  # EXIF(초점거리) 보존
    else:
        cv2.imwrite(str(dst.with_suffix(".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _motion(a: Path, b: Path, side: int = 480) -> float:
    """두 프레임 사이 화면 이동량(이미지 폭 대비 비율) — 특징점 광류의 중앙값."""
    A = cv2.imread(str(a), cv2.IMREAD_GRAYSCALE)
    B = cv2.imread(str(b), cv2.IMREAD_GRAYSCALE)
    if A is None or B is None:
        return 0.0
    s = side / max(A.shape)
    A = cv2.resize(A, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    B = cv2.resize(B, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
    p0 = cv2.goodFeaturesToTrack(A, 400, 0.01, 8)
    if p0 is None or len(p0) < 20:
        return 0.05
    p1, st, _ = cv2.calcOpticalFlowPyrLK(A, B, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 20:
        return 0.2  # 추적 실패 = 크게 움직임
    return float(np.median(np.linalg.norm((p1 - p0)[ok].reshape(-1, 2), axis=1)) / A.shape[1])


def select_keyframes(cands: list[Path], scores: np.ndarray, target: int, log=print) -> list[int]:
    """시차(화면 이동) 기반 키프레임 선택 — 제자리에 서 있던 구간의 중복 프레임은 버리고,
    누적 이동이 기준(폭의 τ)을 넘을 때마다 그 근처에서 가장 선명한 프레임을 고른다.
    (PIX4Dcatch·Terra 영상 모델링 권장: 겹침 80~90% ≈ 프레임 간 이동 10~20%)"""
    n = len(cands)
    if n <= 2:
        return list(range(n))
    mv = np.array([_motion(cands[i], cands[i + 1]) for i in range(n - 1)])
    cum = np.r_[0, np.cumsum(mv)]
    blur_thr = 0.35 * np.median(scores)
    total = cum[-1]
    tau = max(0.06, min(0.2, total / max(target, 1)))  # 이동 6~20% 마다 한 장
    keep, last = [], -1e9
    i = 0
    while i < n:
        if cum[i] - last >= tau or not keep:
            # 다음 τ/2 이동 구간 안에서 가장 선명한 프레임
            j = i
            while j + 1 < n and cum[j + 1] - cum[i] < tau / 2:
                j += 1
            win = np.arange(i, j + 1)
            best = int(win[np.argmax(scores[win])])
            if scores[best] >= blur_thr:
                keep.append(best)
                last = cum[best]
            i = j + 1
        else:
            i += 1
    log(f"  누적 이동 {total:.1f}화면폭 · 기준 {tau:.0%} → 키프레임 {len(keep)}장 (정지 구간 중복·흔들림 제외)")
    return keep


def _video_size(video: Path) -> tuple[int, int] | None:
    try:
        o = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=width,height", "-of", "csv=p=0", str(video)]).decode().strip()
        w, h = (int(x) for x in o.split(",")[:2])
        return w, h
    except Exception:
        return None


def extract_video(video: Path, out_dir: Path, target_frames: int, max_side: int,
                  log=print) -> list[Path]:
    """동영상에서 target_frames 장을 고른다(구간별 최선명 프레임)."""
    tmp = out_dir.parent / "_raw_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(video)]).decode().strip() or 0)
    # 고를 프레임 1장당 후보 4장을 뽑는다
    fps = max(0.5, min(30.0, 4 * target_frames / max(dur, 1e-3)))
    scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
                    f"fps={fps:.3f},{scale}", "-qscale:v", "2",
                    str(tmp / "f_%05d.jpg")], check=True)
    cands = sorted(tmp.glob("f_*.jpg"))
    log(f"  동영상 {dur:.1f}s → 후보 {len(cands)}장 (fps={fps:.2f})")
    if not cands:
        raise RuntimeError("동영상에서 프레임을 뽑지 못했습니다")
    scores = np.array([sharpness(p) for p in cands])
    keep = select_keyframes(cands, scores, target_frames, log=log)
    out = []
    for n, i in enumerate(keep):
        dst = out_dir / f"v_{n:05d}.jpg"
        shutil.move(str(cands[i]), dst)
        out.append(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"  키프레임 {len(out)}장 사용")
    return out


def is_equirect(w: int, h: int) -> bool:
    return w >= 2000 and abs(w / h - 2.0) < 0.03


# 360° 한 장 → 원근 사진 여러 장 (FOV 90°). 수평 6방향 + 아래로 35° 기운 4방향(바닥·슬래브용).
# 바로 아래(촬영자·셀카봉)와 하늘 쪽은 쓰지 않는다.
PANO_SIZES: set = set()
PANO_VIEWS = [(y, 0) for y in range(0, 360, 60)] + [(y, -35) for y in (45, 135, 225, 315)]


def pano_view_size(W: int) -> int:
    """원근 사진 한 변: 90°가 파노라마 폭의 1/4 이므로 W/4 (최대 1600, 짝수)"""
    return int(min(1600, W // 4)) // 2 * 2


def equirect_views(src: Path, dst_prefix: Path, size: int | None = None) -> list[Path]:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return []
    H, W = img.shape[:2]
    size = size or pano_view_size(W)
    PANO_SIZES.add(size)
    f = size / 2.0  # FOV 90°
    u, v = np.meshgrid(np.arange(size) + 0.5 - size / 2, np.arange(size) + 0.5 - size / 2)
    d = np.stack([u / f, v / f, np.ones_like(u)], -1)  # 카메라 좌표(x 오른쪽, y 아래, z 앞)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    out = []
    for yaw, pitch in PANO_VIEWS:
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
        Rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])     # 아래(-pitch)로 기울이기
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        w = d @ (Ry @ Rx).T
        lon = np.arctan2(w[..., 0], w[..., 2])
        lat = np.arcsin(np.clip(w[..., 1], -1, 1))
        mx = ((lon / (2 * np.pi) + 0.5) * W).astype(np.float32)
        my = ((lat / np.pi + 0.5) * H).astype(np.float32)
        view = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        p = dst_prefix.with_name(f"{dst_prefix.name}_y{yaw:03d}p{abs(pitch):02d}.jpg")
        cv2.imwrite(str(p), view, [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append(p)
    return out


def prepare_inputs(inputs: list[Path], out_dir: Path, max_side: int = 3200,
                   video_frames: int = 150, log=print) -> dict:
    """사진·폴더·동영상을 섞어 받아 out_dir 에 복원용 이미지를 모은다.

    반환: {"images": [...], "sequential": bool} — 동영상만 들어오면 순차 매칭을 쓴다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    PANO_SIZES.clear()
    photos, videos = [], []
    for p in inputs:
        p = Path(p)
        if p.is_dir():
            for q in sorted(p.rglob("*")):
                if q.suffix.lower() in IMG_EXT:
                    photos.append(q)
                elif q.suffix.lower() in VID_EXT:
                    videos.append(q)
        elif p.suffix.lower() in IMG_EXT:
            photos.append(p)
        elif p.suffix.lower() in VID_EXT:
            videos.append(p)
        else:
            raise ValueError(f"지원하지 않는 입력: {p}")
    images: list[Path] = []
    blurry = []
    pano_n = 0
    if photos:
        scores = {p: sharpness(p) for p in photos}
        med = float(np.median(list(scores.values())))
        for k, p in enumerate(photos):
            # 바닥만 찍힌 마커 근접 사진은 질감이 적어 '흐림'으로 오판되기 쉽다 → 마커가 보이면 항상 유지
            if scores[p] < 0.2 * med and not _has_marker(p):
                blurry.append(p.name)
                continue
            dst = out_dir / f"p_{k:05d}_{p.stem}"
            hw = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
            if hw is not None and is_equirect(hw.shape[1] * 8, hw.shape[0] * 8):
                views = equirect_views(p, dst)
                images += views
                pano_n += len(views)
                continue
            _resize_copy(p, dst, max_side)
            images.append(dst.with_suffix(".jpg"))
        log(f"  사진 {len(photos)}장 중 {len(images)}장 사용"
            + (f" (흔들림 제외: {', '.join(blurry[:5])}{'…' if len(blurry) > 5 else ''})" if blurry else ""))
    for v in videos:
        sub = out_dir
        per = max(30, video_frames // max(1, len(videos)))
        w_h = _video_size(v)
        if w_h and is_equirect(*w_h):
            # 360° 영상: 위치 수를 줄이고(한 위치당 10방향) 각 프레임을 원근 사진으로 자른다
            raw = extract_video(v, sub, max(12, per // 4), 99999, log=log)
            frames = []
            for f in raw:
                frames += equirect_views(f, f.with_suffix(""))
                f.unlink()
            pano_n += len(frames)
            log(f"  360° 영상 → 위치 {len(raw)}곳 × {len(PANO_VIEWS)}방향 = {len(frames)}장")
        else:
            frames = extract_video(v, sub, per, max_side, log=log)
        # 동영상이 여러 개면 이름 충돌을 막는다
        if len(videos) > 1:
            renamed = []
            for f in frames:
                g = f.with_name(f"{v.stem}_{f.name}")
                f.rename(g)
                renamed.append(g)
            frames = renamed
        images += frames
    if len(images) < 3:
        raise RuntimeError(f"사용할 수 있는 이미지가 {len(images)}장뿐입니다(최소 3장, 권장 50장 이상)")
    only_pano = pano_n == len(images) and pano_n > 0
    return {"images": images, "sequential": bool(videos) and not photos and not pano_n,
            "skipped_blurry": blurry, "pano_views": pano_n,
            # 360 에서 자른 사진만 있으면 렌즈를 정확히 안다(FOV 90°, 왜곡 없음)
            "camera_model": "PINHOLE" if only_pano else None,
            "camera_params": (lambda z: f"{z / 2},{z / 2},{z / 2},{z / 2}")(next(iter(PANO_SIZES)))
            if only_pano and len(PANO_SIZES) == 1 else None,
            "single_camera": (only_pano and len(PANO_SIZES) == 1) or pano_n == 0}
