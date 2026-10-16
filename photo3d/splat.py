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


SPIRULA_VERSION = "2026.10.16"
_SP_ASSET = {"win32": f"spirula-{SPIRULA_VERSION}-windows-vulkan-x86_64.zip",
             "linux": f"spirula-{SPIRULA_VERSION}-ubuntu-vulkan-x86_64.zip"}


def find_spirula(root: Path, download: bool = True, log=print) -> str | None:
    """Spirula Studio(GPLv3, 별도 프로그램으로 실행만 함) — 사진별 노출·화이트밸런스 보정(PPISP·bilateral grid),
    표면 방향(normal) 지도 학습, 8GB 에서 수백만 개. third_party/spirula 에 없으면 공식 릴리스에서 내려받는다."""
    if os.environ.get("SPIRULA_BIN") and Path(os.environ["SPIRULA_BIN"]).exists():
        return os.environ["SPIRULA_BIN"]
    tp = root / "third_party" / "spirula"
    for name in ("spirula.exe", "spirula"):
        hit = next(iter(sorted(q for q in tp.glob(f"**/{name}") if q.is_file())), None)
        if hit:
            return str(hit)
    asset = _SP_ASSET.get(sys.platform)
    if not download or not asset:
        return None
    import urllib.request
    tp.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/harry7557558/spirula-studio/releases/download/v{SPIRULA_VERSION}/{asset}"
    log(f"  Spirula 내려받는 중: {url}")
    arc = tp / asset
    try:
        urllib.request.urlretrieve(url, arc)
        with zipfile.ZipFile(arc) as z:
            z.extractall(tp)
    except Exception as e:
        log(f"  Spirula 내려받기 실패: {e}")
        return None
    finally:
        arc.unlink(missing_ok=True)
    for q in tp.glob("**/spirula*"):
        if q.is_file() and q.suffix in ("", ".exe"):
            q.chmod(0o755)
            return str(q)
    return None


class _Chain:
    """여러 명령을 차례로 백그라운드 실행(Popen 처럼 wait/terminate/kill 지원)."""

    def __init__(self, cmds: list[list[str]], cwd: Path, logf):
        import threading
        self.cmds, self.cwd, self.logf = cmds, cwd, logf
        self.cur, self.rc, self.stop = None, None, False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        for c in self.cmds:
            if self.stop:
                break
            if callable(c):  # 파이썬 단계(예: 배경 분리 마스크 만들기)
                try:
                    c()
                except Exception as e:
                    self.logf.write(f"단계 실패: {e}\n")
                continue
            self.logf.write("$ " + " ".join(c) + "\n")
            self.logf.flush()
            self.cur = subprocess.Popen(c, cwd=self.cwd, stdout=self.logf, stderr=subprocess.STDOUT)
            r = self.cur.wait()
            if r != 0 and c is self.cmds[-1]:
                self.rc = r
                return
        self.rc = 0 if not self.stop else -1

    def wait(self, timeout=None):
        self.th.join(timeout)
        if self.th.is_alive():
            raise subprocess.TimeoutExpired("spirula", timeout)
        return self.rc

    def terminate(self):
        self.stop = True
        if self.cur and self.cur.poll() is None:
            self.cur.terminate()

    def kill(self):
        self.stop = True
        if self.cur and self.cur.poll() is None:
            self.cur.kill()


def make_subject_masks(exe: str, data: Path, logf, model: str = "birefnet") -> int:
    """사람·물체 모드: 사진마다 주인공(가장 눈에 띄는 대상)을 자동으로 찾아 마스크(흰색=대상)를 만든다.
    BiRefNet(MIT) — Spirula 'sam segment' 로 실행. 결과: data/masks/<사진이름(확장자 제외)>.png (Spirula·Brush 공용)."""
    masks = data / "masks"
    masks.mkdir(exist_ok=True)
    tmp = data.parent / "mask_tmp"
    imgs = sorted(q for q in (data / "images").iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    n = 0
    for k, im in enumerate(imgs):
        shutil.rmtree(tmp, ignore_errors=True)
        for mdl in (model, "birefnet-lite"):
            r = subprocess.run([exe, "sam", "segment", "--model", mdl, "--accept-license", "birefnet=yes",
                                "--image", str(im), "--out", str(tmp)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            if r.returncode == 0 and (tmp / "seg_mask.png").exists():
                shutil.move(str(tmp / "seg_mask.png"), str(masks / f"{im.stem}.png"))
                n += 1
                break
        logf.write(f"배경 분리 {k + 1}/{len(imgs)}\n")
        logf.flush()
    shutil.rmtree(tmp, ignore_errors=True)
    logf.write(f"배경 분리 완료: {n}/{len(imgs)}장\n")
    return n


def subject_points(dense: Path, masks: Path, min_ratio: float = 0.6) -> np.ndarray | None:
    """조밀 점 중 '보이는 사진의 60% 이상에서 마스크 안'인 점 = 대상 점(스플랫 자르기 기준)."""
    import cv2
    import pycolmap
    from .fill import read_points
    xyz = np.asarray(read_points(dense / "fused.ply")[0], float)
    r = pycolmap.Reconstruction(dense / "sparse")
    inside = np.zeros(len(xyz), np.int32)
    seen = np.zeros(len(xyz), np.int32)
    for i in r.reg_image_ids():
        im = r.image(i)
        mp = masks / f"{Path(im.name).stem}.png"
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        cam = r.camera(im.camera_id)
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        pc = xyz @ T.rotation.matrix().T + T.translation
        ok = pc[:, 2] > 0
        uv = np.zeros((len(xyz), 2))
        uv[ok] = np.asarray(cam.img_from_cam(pc[ok]))[:, :2]
        sx, sy = m.shape[1] / cam.width, m.shape[0] / cam.height
        u, v = (uv[:, 0] * sx).astype(int), (uv[:, 1] * sy).astype(int)
        ok &= (u >= 0) & (v >= 0) & (u < m.shape[1]) & (v < m.shape[0])
        seen[ok] += 1
        inside[np.where(ok)[0][m[v[ok], u[ok]] > 127]] += 1
    keep = (seen >= 2) & (inside >= min_ratio * np.maximum(seen, 1))
    return xyz[keep] if keep.sum() > 500 else None


def start_splat_spirula(dense: Path, exe: str, steps: int = 30000, max_res: int = 1920, log=print,
                        subject: bool = False, depth_sup: bool = False) -> dict:
    """Spirula 학습을 백그라운드로: ① 표면 방향 지도(MoGe-2, MIT) ② 학습(노출 보정·잡티 억제·조밀 점 시작)."""
    import math
    work = dense.parent / "splat"
    if work.exists():
        shutil.rmtree(work)
    data = work / "data"
    (data / "sparse" / "0").mkdir(parents=True)
    for f in (dense / "sparse").glob("*.bin"):
        shutil.copy2(f, data / "sparse" / "0" / f.name)
    _seed_from_dense(dense, data / "sparse" / "0", log)  # data/init.ply = 조밀 점(최대 30만)
    try:
        os.symlink(dense / "images", data / "images", target_is_directory=True)
    except OSError:
        shutil.copytree(dense / "images", data / "images")
    longest = 0
    try:
        import pycolmap
        r = pycolmap.Reconstruction(data / "sparse" / "0")
        longest = max(max(c.width, c.height) for c in r.cameras.values())
    except Exception:
        pass
    steps = int(os.environ.get("PHOTO3D_SPIRULA_STEPS", steps))
    max_res = int(os.environ.get("PHOTO3D_SPIRULA_RES", max_res))
    div = max(1, math.ceil(longest / max_res)) if longest else 1
    cap = os.environ.get("PHOTO3D_SPIRULA_CAP") or os.environ.get("PHOTO3D_MAX_SPLATS", "3000000")
    runs = work / "runs"
    geo = [exe, "geometry", str(data), "--max-size", "1064", "--normal-format", "jpg"] + (["--depth"] if depth_sup else [])
    train = [exe, "train", "centered-object" if subject else "3dgs", "--data", str(data), "--num-iterations", str(steps), "--disable-viewer", "1",
             "--scene-center", "none", "--seed-pointcloud", "init.ply", "--cap-max", cap,
             "--floater-suppression", "mild", "--train-resolution-divisor", str(div),
             "--output-dir-prefix", str(runs), "--output-dir-name", "run",
             "--steps-per-save", os.environ.get("PHOTO3D_SPIRULA_SAVE", "2000")] \
        + (["--depth-supervision-weight", "0.05"] if depth_sup else [])  # 최고 품질: 깊이도 지도(흰 벽 평평하게)
    log(f"  $ spirula train --total-steps {steps} (표면 방향 지도 → 학습, 해상도 1/{div}, 최대 {int(cap):,}개, 백그라운드 GPU)")
    logf = open(work / "brush.log", "w", encoding="utf-8", errors="replace")  # 진행 표시가 같은 파일을 읽음
    steps_ = [geo, train]
    if subject:  # 사람·물체 모드: 배경 분리 마스크 → 대상만 학습(centered-object), 배경은 빈 공간으로
        if (dense / "masks_stem").exists():  # 조밀복원 단계에서 이미 만든 마스크 재사용
            shutil.copytree(dense / "masks_stem", data / "masks")
            log("  사람·물체 모드: 배경 분리 마스크 재사용, 대상만 학습")
        else:
            log("  사람·물체 모드: 사진마다 배경 자동 분리(BiRefNet) 후 대상만 학습")
            steps_ = [lambda: make_subject_masks(exe, data, logf), geo, train]
    if os.environ.get("PHOTO3D_SPIRULA_GEO") == "0":  # 빠르게: 표면 방향 지도 생략
        steps_ = [c for c in steps_ if c is not geo]
    proc = _Chain(steps_, work, logf)
    return {"proc": proc, "logf": logf, "work": work, "exp": runs, "steps": steps, "engine": "spirula",
            "subject": subject, "data": data}


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


def _link_as_step(p: Path) -> Path:
    """Spirula 출력 이름(step-N.ckpt/splat.ply)을 Brush 와 같은 splat_N.ply 로 맞춘다(뒤 처리 공용)."""
    n = int(p.parent.name.split("-")[1].split(".")[0])
    dst = p.parent.parent / f"splat_{n}.ply"
    if not dst.exists():
        shutil.copy2(p, dst)
    return dst


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
    if h.get("engine") == "spirula":  # runs/run/step-000030000.ckpt/splat.ply
        plys = sorted((p for p in h["exp"].glob("**/step-*.ckpt/splat.ply") if p.stat().st_size > 0),
                      key=lambda p: int(p.parent.name.split("-")[1].split(".")[0]))
        plys = [_link_as_step(p) for p in plys]
    else:
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
        support = dense_ply
        if h.get("subject") and dense_ply.exists():
            pts = subject_points(dense_ply.parent, h["data"] / "masks")
            if pts is not None:
                from .fill import _write_ply
                support = h["work"] / "subject_points.ply"
                _write_ply(support, pts, np.zeros_like(pts), np.full((len(pts), 3), 200, np.uint8))
                log(f"  대상 점 {len(pts):,}개 기준으로 배경 잘라냄")
                cams = None  # 카메라 위치까지 범위에 넣지 않음(대상만)
        clean = clean_file(raw, support, dst, cams, log=log) if dense_ply.exists() else None
    except Exception as e:
        log(f"  스플랫 정리 생략: {e}")
    if clean is None:
        shutil.copy2(raw, dst)
    else:
        n = clean["after"]
    log(f"  스플랫 {n:,}개 → {dst.name} ({dst.stat().st_size / 1e6:.0f} MB)")
    light = {}
    try:  # 가벼운 형식: .spz(보관·공유, 약 1/10~1/16) · .splat(웹 뷰어)
        from .splatexport import write_spz, write_splat
        write_spz(dst, out / "site_splat.spz")
        write_splat(dst, out / "site_splat.splat")
        light = {f: round((out / f).stat().st_size / 1e6, 1) for f in ("site_splat.spz", "site_splat.splat")}
        log(f"  가벼운 스플랫: .spz {light['site_splat.spz']} MB · .splat {light['site_splat.splat']} MB")
    except Exception as e:
        log(f"  가벼운 스플랫 내보내기 생략: {e}")
    return {"file": dst.name, "splats": n, "clean": clean, "light_mb": light, "steps": int(plys[-1].stem.split("_")[-1]), "up_axis": "Z",
            "note": "보기용. 메쉬 GLB(Y-up)와 맞추려면 X축 -90° 회전"}


def train_splat(dense: Path, out: Path, brush: str, steps: int = 30000, max_res: int = 1920,
                log=print) -> dict:
    return finish_splat(start_splat(dense, brush, steps, max_res, log), out, log)
