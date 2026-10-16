"""깊이 prior 로 빈 곳 채우기 — 실내 흰 벽·천장·바닥처럼 무늬가 없어 MVS 가 비우는 곳을 메운다.

MoGe-2 (Microsoft, MIT, https://github.com/microsoft/MoGe): 사진 한 장 → 미터 깊이·법선.
처리(사진마다):
 1) MoGe-2 로 깊이 추정(언디스토트 사진, 화각 고정)
 2) 실제 측정된 조밀 점을 그 사진에 투영 → 측정 깊이와 비교해 prior 의 축척·이동을 강건하게 맞춤
    (맞춤 오차가 크면 그 사진의 prior 는 버림)
 3) 측정이 없는 픽셀만 후보로 삼아 3D 점 생성
 4) 다른 사진들의 (맞춘) prior 깊이와도 일치하고, 측정된 표면보다 앞(빈 공간)에 놓이지 않는 점만 채택
결과 점에는 '추정(prior)' 표시를 남겨 측정 점과 구분한다.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

_MODEL = None


def release() -> None:
    """깊이 모델을 GPU 메모리에서 내린다 — 바로 뒤에 스플랫 학습(Brush)이 같은 GPU 를 쓰므로"""
    global _MODEL
    if _MODEL is None:
        return
    _MODEL = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def ensure_installed(log=print) -> bool:
    """PC 첫 실행: PyTorch(CUDA, RTX 50 대응 cu128)·MoGe 자동 설치"""
    import subprocess
    import sys
    try:
        import torch  # noqa: F401
    except ImportError:
        log("  깊이 prior 준비: PyTorch(GPU) 설치 중 — 처음 한 번만, 수 분 걸립니다")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision",
                            "--index-url", "https://download.pytorch.org/whl/cu128"])
        if r.returncode != 0:
            return False
    try:
        import moge  # noqa: F401
    except ImportError:
        log("  깊이 prior 준비: MoGe-2(MIT) 설치 중")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/microsoft/MoGe.git"])
        if r.returncode != 0:
            return False
    return True


def _model(name: str | None, log=print):
    global _MODEL
    if _MODEL is None:
        import torch
        if name is None:  # GPU 면 큰 모델(정확), CPU 면 작은 모델(빠름) — VRAM 작으면 gpu.vram_plan 이 vitb 지정
            big = torch.cuda.is_available() and os.environ.get("PHOTO3D_MOGE", "vitl") == "vitl"
            name = "Ruicheng/moge-2-vitl-normal" if big else "Ruicheng/moge-2-vitb-normal"
        if torch.cuda.is_available():
            # Ampere 이후 GPU 의 텐서코어 사용(TF32) + 입력 크기가 같은 반복 추론에 맞는 cuDNN 알고리즘 자동 선택
            try:  # PyTorch 2.9+ 새 방식(옛 allow_tf32 와 섞어 쓰지 말 것 — PyTorch CUDA 문서)
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            except AttributeError:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # 사진 크기가 모두 같아 같은 알고리즘 재사용
        from moge.model.v2 import MoGeModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"  깊이 prior 모델 불러오기: {name} ({dev})")
        _MODEL = (MoGeModel.from_pretrained(name).to(dev).eval(), dev)
    return _MODEL


def _zbuffer(uv: np.ndarray, z: np.ndarray, w: int, h: int) -> np.ndarray:
    D = np.full(h * w, np.inf, np.float32)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < w) & (uv[:, 1] < h)
    idx = uv[ok, 1].astype(np.int64) * w + uv[ok, 0].astype(np.int64)
    np.minimum.at(D, idx, z[ok].astype(np.float32))
    D[~np.isfinite(D)] = 0
    return D.reshape(h, w)


def fill(dense_xyz: np.ndarray, dense_rgb: np.ndarray | None, sparse: Path, img_dir: Path,
         side: int | None = None, model: str | None = None, rel_tol: float = 0.03,
         stride: int = 2, neighbors: int = 8, log=print) -> dict:
    import cv2
    import torch
    from scipy.ndimage import binary_dilation
    from .texture import _cameras, _project
    net, dev = _model(model, log)
    if side is None:
        side = int(os.environ.get("PHOTO3D_MOGE_SIDE", "768" if dev == "cuda" else "640"))
    cams = _cameras(sparse, img_dir)
    views, fit_stats = [], []
    try:  # SfM 점(3장 이상에서 보인 것)도 축척 맞춤에 사용
        import pycolmap
        _r = pycolmap.Reconstruction(sparse)
        sparse_xyz = np.array([p.xyz for p in _r.points3D.values() if p.track.length() >= 3]).reshape(-1, 3)
    except Exception:
        sparse_xyz = None
    for ci, cam in enumerate(cams):
        im = cv2.imread(str(cam["path"]), cv2.IMREAD_COLOR)
        if im is None:
            continue
        s = side / max(im.shape[:2])
        w, h = int(round(cam["w"] * s)), int(round(cam["h"] * s))
        small = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
        K = cam["K"].copy()
        K[:2] *= s
        fov = math.degrees(2 * math.atan(w / (2 * K[0, 0])))
        with torch.no_grad():
            out = net.infer(torch.from_numpy(small).permute(2, 0, 1).float().div(255).to(dev), fov_x=fov,
                            use_fp16=dev == "cuda")
        zp = out["depth"].float().cpu().numpy()
        mk = out["mask"].cpu().numpy().astype(bool) & np.isfinite(zp) & (zp > 0)
        # 측정 깊이(조밀 점 투영) — 축척·이동 맞춤과 '빈 곳' 판정에 사용
        uv, z = _project({**cam, "K": K}, dense_xyz)
        Dm = _zbuffer(np.floor(uv), z, w, h)
        both = (Dm > 0) & mk
        a, b, ok, err = 1.0, 0.0, False, None
        # 측정 깊이: 조밀 점 + SfM 점. 흰 벽 위주 실내는 조밀 점이 사진당 수백 개뿐이라(PC '집' 2.8만 점/39장)
        # 예전 기준(200점 이상)으로는 대부분 사진이 빠졌다 → 30점부터, 적으면 축척만 맞춘다
        if sparse_xyz is not None and len(sparse_xyz):
            uv2, z2 = _project({**cam, "K": K}, sparse_xyz)
            ui, vi = np.floor(uv2[:, 0]).astype(int), np.floor(uv2[:, 1]).astype(int)
            okp = (z2 > 0) & (ui >= 0) & (vi >= 0) & (ui < w) & (vi < h)
            xs, ys = zp[vi[okp], ui[okp]], z2[okp]
            m2 = mk[vi[okp], ui[okp]]
            xs, ys = xs[m2], ys[m2]
        else:
            xs, ys = np.zeros(0), np.zeros(0)
        x = np.concatenate([zp[both], xs])
        y = np.concatenate([Dm[both], ys])
        if len(x) >= 30:
            a = float(np.median(y / x))
            inl = np.abs(a * x - y) / y < 0.1
            if len(x) >= 150:
                for _ in range(3):  # 강건 최소제곱(축척·이동)
                    if inl.sum() < 60:
                        break
                    A = np.stack([x[inl], np.ones(inl.sum())], 1)
                    a, b = np.linalg.lstsq(A, y[inl], rcond=None)[0]
                    inl = np.abs(a * x + b - y) / y < 0.05
            err = float(np.median(np.abs(a * x + b - y) / y))
            ok = a > 0 and err < 0.06 and inl.mean() > 0.4
        fit_stats.append((len(x), err, ok))
        Za = np.where(mk, a * zp + b, 0).astype(np.float32)
        # 법선: 맞춘 깊이 지도를 3D 로 펴서 이웃 차분으로 계산, 카메라를 향하게
        K_ = K
        uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        Xg = np.stack([(uu - K_[0, 2]) / K_[0, 0] * Za, (vv - K_[1, 2]) / K_[1, 1] * Za, Za], -1)
        du = np.gradient(Xg, axis=1)
        dv = np.gradient(Xg, axis=0)
        nrm = np.cross(du, dv)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
        flip = (nrm * Xg).sum(-1) > 0  # 카메라(원점) 쪽을 보게
        nrm[flip] *= -1
        views.append({"ci": ci, "cam": cam, "K": K, "w": w, "h": h, "Za": Za, "Dm": Dm, "ok": ok, "img": small,
                      "nrm": nrm, "cover": binary_dilation(Dm > 0, iterations=3)})
    good = [v for v in views if v["ok"]]
    errs = [e for _, e, _ in fit_stats if e is not None]
    log(f"  깊이 prior: 사진 {len(views)}장 중 측정과 잘 맞는 {len(good)}장 사용 "
        f"(사진당 측정점 중앙 {int(np.median([n for n, _, _ in fit_stats])) if fit_stats else 0}개, "
        f"맞춤 오차 중앙 {np.median(errs) * 100:.1f}%)" if errs else
        f"  깊이 prior: 사진 {len(views)}장 모두 측정점이 너무 적어 맞추지 못함")
    pts, cols, nrms, nsup = [], [], [], []
    Cg = np.array([v["cam"]["C"] for v in good]) if good else np.zeros((0, 3))
    Fg = np.array([v["cam"]["R"][2] for v in good]) if good else np.zeros((0, 3))  # 보는 방향
    for gi, v in enumerate(good):
        # 대조할 이웃: 보는 방향이 비슷한(70° 이내) 사진 중 가까운 순 N장 — 사진이 많아도 빠르게
        d = np.linalg.norm(Cg - Cg[gi], axis=1)
        d[(Fg @ Fg[gi]) < np.cos(np.radians(70))] = np.inf
        d[gi] = np.inf
        nb = [good[j] for j in np.argsort(d)[:neighbors] if np.isfinite(d[j])]
        cand = (v["Za"] > 0) & ~v["cover"]
        grid = np.zeros_like(cand)
        grid[::stride, ::stride] = True
        cand &= grid
        vs, us = np.nonzero(cand)
        if len(us) == 0:
            continue
        z = v["Za"][vs, us]
        K = v["K"]
        Xc = np.stack([(us + 0.5 - K[0, 2]) / K[0, 0] * z, (vs + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
        R, t = v["cam"]["R"], v["cam"]["t"]
        Xw = (Xc - t) @ R  # R^T (Xc - t)
        sup = np.zeros(len(Xw), np.int32)
        bad = np.zeros(len(Xw), bool)
        for o in nb:
            uv, zo = _project({**o["cam"], "K": o["K"]}, Xw)
            ui, vi = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
            inside = (zo > 0) & (ui >= 0) & (vi >= 0) & (ui < o["w"]) & (vi < o["h"])
            idx = np.where(inside)[0]
            za = o["Za"][vi[idx], ui[idx]]
            dm = o["Dm"][vi[idx], ui[idx]]
            agree = (za > 0) & (np.abs(za - zo[idx]) / zo[idx] < rel_tol)
            sup[idx[agree]] += 1
            # 측정된 표면보다 확실히 앞에 놓이면(그 자리는 빈 공간으로 관측됨) 버림
            bad[idx[(dm > 0) & (zo[idx] < dm * (1 - 2 * rel_tol))]] = True
        keep = (sup >= 1) & ~bad
        pts.append(Xw[keep])
        cols.append(v["img"][vs[keep], us[keep]])
        if v["nrm"] is not None:
            nrms.append(v["nrm"][vs[keep], us[keep]] @ R)
        nsup.append(sup[keep])
    if not pts:
        release()
        return {"xyz": np.zeros((0, 3)), "rgb": np.zeros((0, 3), np.uint8), "normal": None, "views_used": len(good)}
    P = np.concatenate(pts)
    C = np.concatenate(cols).astype(np.uint8)
    N = np.concatenate(nrms) if nrms else None
    log(f"  깊이 prior 로 채운 점 {len(P):,}개 (다른 사진과 일치·빈 공간 위반 없음)")
    release()
    return {"xyz": P, "rgb": C, "normal": N, "views_used": len(good)}
