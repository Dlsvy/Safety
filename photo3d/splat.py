"""가우시안 스플래팅(GPU) — Brush 로 사진처럼 부드러운 '보기용' 3D 를 만든다.

- Brush(Apache-2.0, https://github.com/ArthurBrussee/brush): WebGPU 기반이라 CUDA 빌드 없이
  RTX 50 포함 NVIDIA/AMD GPU 에서 바로 돈다. COLMAP 결과를 그대로 읽는다.
- 입력: 왜곡보정된 COLMAP 작업폴더(dense/images + dense/sparse) — 메쉬와 같은 좌표(정합·축척 반영)
- 출력: site_splat.ply (표준 3DGS 속성 순서, Z-up 그대로. 메쉬 GLB 는 Y-up 이므로 뷰어/유니티에서 X축 -90° 회전)
스플랫은 '보기용'이고 충돌·거리 판정은 메쉬(site_collider.glb)로 한다.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

BRUSH_VERSION = "v0.3.0"
_ASSET = {"win32": "brush-app-x86_64-pc-windows-msvc.zip",
          "linux": "brush-app-x86_64-unknown-linux-gnu.tar.xz"}


def find_brush(root: Path, download: bool = True, log=print) -> str | None:
    """third_party/brush 에서 brush_app 실행파일을 찾고, 없으면 내려받는다."""
    if os.environ.get("BRUSH_BIN") and Path(os.environ["BRUSH_BIN"]).exists():
        return os.environ["BRUSH_BIN"]
    tp = root / "third_party" / "brush"
    for name in ("brush_app.exe", "brush.exe", "brush_app", "brush"):
        hit = next(iter(sorted(p for p in tp.glob(f"**/{name}") if p.is_file())), None)
        if hit:
            return str(hit)
    asset = _ASSET.get(sys.platform)
    if not download or not asset:
        return None
    import urllib.request
    tp.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/ArthurBrussee/brush/releases/download/{BRUSH_VERSION}/{asset}"
    log(f"  Brush 내려받는 중: {url}")
    arc = tp / asset
    urllib.request.urlretrieve(url, arc)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(arc) as z:
            z.extractall(tp)
    else:
        import tarfile
        with tarfile.open(arc) as t:
            t.extractall(tp)
    arc.unlink(missing_ok=True)
    for p in tp.glob("**/brush_app*"):
        if p.is_file() and p.suffix in ("", ".exe"):
            p.chmod(0o755)
            return str(p)
    return None


def _std_order(src: Path, dst: Path) -> int:
    """Brush 는 속성을 이름순(f_rest_10 이 f_rest_2 앞)으로 쓴다 → 모든 뷰어·유니티가 읽는 표준 순서로 다시 쓴다."""
    b = src.read_bytes()
    i = b.index(b"end_header\n")
    head = b[:i].decode("ascii", "ignore").splitlines()
    props = [ln.split()[2] for ln in head if ln.startswith("property")]
    n = int(next(ln for ln in head if ln.startswith("element vertex")).split()[2])
    a = np.frombuffer(b[i + 11:], np.float32, n * len(props)).reshape(n, len(props))
    rest = sorted((p for p in props if p.startswith("f_rest_")), key=lambda s: int(s[7:]))
    order = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + rest + \
        ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    cols = [a[:, props.index(p)] for p in order]
    cols[3:3] = [np.zeros(n, np.float32)] * 3  # nx ny nz (INRIA 형식 호환)
    names = order[:3] + ["nx", "ny", "nz"] + order[3:]
    out = np.stack(cols, 1).astype("<f4")
    ok = np.isfinite(out).all(1)
    out = out[ok]
    with open(dst, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\ncomment photo3d: Brush splat, Z-up\n"
                 f"element vertex {len(out)}\n" + "".join(f"property float {p}\n" for p in names) +
                 "end_header\n").encode())
        f.write(out.tobytes())
    return len(out)


def _growth_flags(brush: str, steps: int) -> list[str]:
    """가우시안이 더 많이·오래 자라게(r/GaussianSplatting: 1만 개대는 '쓸 수 없는' 수준, 수십만~수백만 권장).
    Brush 버전마다 옵션 이름이 달라 --help 에 있는 것만 넣는다."""
    try:
        h = subprocess.run([brush, "--help"], capture_output=True, text=True, timeout=30,
                           encoding="utf-8", errors="replace").stdout
    except Exception:
        return []
    # 주의: Brush v0.3.0 의 growth-grad-threshold 기본값은 0.00004(main 브랜치와 단위가 다름). 예전에 0.0015 를 넣어
    # 증식이 거의 막혔다(PC '집' 10만 개·바늘 모양) → 기본값을 쓰고 상한·선택 비율만 조정
    want = [("--max-splats", os.environ.get("PHOTO3D_MAX_SPLATS", "3000000")),  # VRAM 맞춤(8GB 300만)
            ("--growth-select-fraction", "0.2"),     # 기본 0.1 → 더 많이 증식
            ("--growth-stop-iter", str(int(steps * 0.8)))]
    out = []
    for k, v in want:
        if k in h:
            out += [k, v]
    return out


def _seed_from_dense(dense: Path, sp: Path, log, limit: int = 300_000) -> None:
    """스플랫 시작점을 희소 점(수천 개) 대신 조밀 점군(수십만)으로 — 점이 적으면 가우시안이 몇천 개에서
    자라지 못해 흐릿하고 방사형 줄무늬가 생긴다(PC '집': 5,875개)."""
    fused = dense / "fused.ply"
    if not fused.exists():
        return
    try:
        import numpy as np
        import pycolmap
        from .fill import read_points
        xyz, _, rgb = read_points(fused)
        if len(xyz) < 1000:
            return
        if len(xyz) > limit:
            k = np.random.default_rng(0).choice(len(xyz), limit, replace=False)
            xyz, rgb = xyz[k], (rgb[k] if rgb is not None else None)
        rec = pycolmap.Reconstruction(sp)
        for pid in list(rec.point3D_ids()):
            rec.delete_point3D(pid)
        col = rgb if rgb is not None else np.full((len(xyz), 3), 160, np.uint8)
        for x, c in zip(xyz, col.astype(np.uint8)):
            rec.add_point3D(x, pycolmap.Track(), c)
        rec.write(sp)
        from .fill import _write_ply
        _write_ply(sp.parent.parent / "init.ply", xyz, np.zeros_like(xyz), col.astype(np.uint8))
        log(f"  스플랫 시작점: 조밀 점 {len(xyz):,}개")
    except Exception as e:
        log(f"  스플랫 시작점 교체 생략: {e}")


def start_splat(dense: Path, brush: str, steps: int = 30000, max_res: int = 1920, log=print) -> dict:
    """Brush 학습을 백그라운드로 시작(바로 반환). 메쉬·텍스처(CPU)와 동시에 GPU 로 돈다."""
    work = dense.parent / "splat"
    if work.exists():
        shutil.rmtree(work)
    data = work / "data"
    (data / "sparse" / "0").mkdir(parents=True)
    for f in (dense / "sparse").glob("*.bin"):
        shutil.copy2(f, data / "sparse" / "0" / f.name)
    _seed_from_dense(dense, data / "sparse" / "0", log)
    try:
        os.symlink(dense / "images", data / "images", target_is_directory=True)
    except OSError:  # 윈도우 일반 계정은 심볼릭 링크 불가 → 복사
        shutil.copytree(dense / "images", data / "images")
    exp = work / "export"
    exp.mkdir()
    # 중간 결과를 1/3 마다 저장 → 빠른 모드에서 시간 상한에 걸리면 마지막 중간본을 쓴다
    cmd = [brush, str(data), "--total-steps", str(steps), "--export-every", str(max(1000, steps // 3)),
           "--export-path", str(exp), "--export-name", "splat_{iter}.ply",
           "--max-resolution", str(max_res), "--eval-every", str(steps + 1)]
    cmd += _growth_flags(brush, steps)
    log(f"  $ brush {data.name} --total-steps {steps} --max-resolution {max_res} (백그라운드 GPU)")
    logf = open(work / "brush.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, cwd=work, stdout=logf, stderr=subprocess.STDOUT)
    return {"proc": proc, "logf": logf, "work": work, "exp": exp, "steps": steps}


def finish_splat(h: dict, out: Path, log=print, max_wait: float | None = None) -> dict:
    """학습 끝을 기다린다. max_wait(초)를 넘기면 멈추고 마지막 중간 저장본을 쓴다(빠른 모드)."""
    stopped = False
    try:
        r = h["proc"].wait(timeout=max_wait)
    except subprocess.TimeoutExpired:
        h["proc"].terminate()
        try:
            h["proc"].wait(timeout=20)
        except subprocess.TimeoutExpired:
            h["proc"].kill()
        r, stopped = 0, True
    h["logf"].close()
    plys = sorted((p for p in h["exp"].glob("splat_*.ply") if p.stat().st_size > 0),
                  key=lambda p: int(p.stem.split("_")[-1]))
    if stopped:
        log(f"  스플랫 시간 상한 도달 → 중간 저장본 사용({plys[-1].stem.split('_')[-1] if plys else '없음'}스텝)")
    if r != 0 or not plys:
        tail = (h["work"] / "brush.log").read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
        raise RuntimeError("Brush 실패: " + " | ".join(tail))
    dst = out / "site_splat.ply"
    raw = h["work"] / "splat_raw.ply"
    n = _std_order(plys[-1], raw)
    clean = None
    dense_ply = h["work"].parent / "dense" / "fused.ply"
    try:  # 빛 번짐·떠다니는 잡티·바늘 제거(측정된 조밀 점 기준)
        from .splatclean import clean_file
        cams = None
        try:
            import pycolmap
            r = pycolmap.Reconstruction(h["work"].parent / "dense" / "sparse")
            cams = np.array([r.image(i).projection_center() for i in r.reg_image_ids()])
        except Exception:
            pass
        clean = clean_file(raw, dense_ply, dst, cams, log=log) if dense_ply.exists() else None
    except Exception as e:
        log(f"  스플랫 정리 생략: {e}")
    if clean is None:
        shutil.copy2(raw, dst)
    else:
        n = clean["after"]
    log(f"  스플랫 {n:,}개 → {dst.name} ({dst.stat().st_size / 1e6:.0f} MB)")
    return {"file": dst.name, "splats": n, "clean": clean, "steps": int(plys[-1].stem.split("_")[-1]), "up_axis": "Z",
            "note": "보기용. 메쉬 GLB(Y-up)와 맞추려면 X축 -90° 회전"}


def train_splat(dense: Path, out: Path, brush: str, steps: int = 30000, max_res: int = 1920,
                log=print) -> dict:
    return finish_splat(start_splat(dense, brush, steps, max_res, log), out, log)
