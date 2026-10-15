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
    bins = np.array_split(np.arange(len(cands)), min(target_frames, len(cands)))
    keep = []
    for b in bins:
        i = b[np.argmax(scores[b])]
        keep.append(i)
    # 전체 중앙값의 35% 미만인 프레임은 흔들림으로 보고 버린다
    thr = 0.35 * np.median(scores)
    keep = [i for i in keep if scores[i] >= thr]
    out = []
    for n, i in enumerate(keep):
        dst = out_dir / f"v_{n:05d}.jpg"
        shutil.move(str(cands[i]), dst)
        out.append(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"  선명도 선별 → {len(out)}장 사용 (블러 제외 {len(bins) - len(out)}장)")
    return out


def prepare_inputs(inputs: list[Path], out_dir: Path, max_side: int = 3200,
                   video_frames: int = 150, log=print) -> dict:
    """사진·폴더·동영상을 섞어 받아 out_dir 에 복원용 이미지를 모은다.

    반환: {"images": [...], "sequential": bool} — 동영상만 들어오면 순차 매칭을 쓴다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
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
    if photos:
        scores = {p: sharpness(p) for p in photos}
        med = float(np.median(list(scores.values())))
        for k, p in enumerate(photos):
            # 바닥만 찍힌 마커 근접 사진은 질감이 적어 '흐림'으로 오판되기 쉽다 → 마커가 보이면 항상 유지
            if scores[p] < 0.2 * med and not _has_marker(p):
                blurry.append(p.name)
                continue
            dst = out_dir / f"p_{k:05d}_{p.stem}"
            _resize_copy(p, dst, max_side)
            images.append(dst.with_suffix(".jpg"))
        log(f"  사진 {len(photos)}장 중 {len(images)}장 사용"
            + (f" (흔들림 제외: {', '.join(blurry[:5])}{'…' if len(blurry) > 5 else ''})" if blurry else ""))
    for v in videos:
        sub = out_dir
        per = max(30, video_frames // max(1, len(videos)))
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
    return {"images": images, "sequential": bool(videos) and not photos,
            "skipped_blurry": blurry}
