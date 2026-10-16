"""스플랫 정리 — 학습이 끝난 가우시안에서 '빛나는 안개'·떠다니는 잡티·바늘을 지운다.

원인과 대응(PC castle-P30, '집' 결과):
- 하늘·먼 배경·사진이 적은 가장자리를 학습이 아주 큰 반투명 가우시안으로 메운다 → 모델 둘레가 하얗게 번져 빛나 보임
  → 측정된 장면 크기에 비해 너무 큰 가우시안 제거
- 측정된 표면(조밀 점)에서 멀리 떨어진 가우시안 = 떠다니는 잡티 → 표면 근처만 남김
- 한 방향으로만 긴 가우시안 = 바늘·줄무늬 → 비율 기준 제거
- 거의 투명한 가우시안 → 제거(보기에는 영향 없고 파일만 커짐)
splat-transform(PlayCanvas, MIT)의 -N(NaN 제거)·-V opacity·-F(floater 제거)·-B(박스 자르기)와 같은 목적을
Node.js 설치 없이 파이썬으로 한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_splat(p: Path) -> tuple[np.ndarray, list[str], str]:
    b = p.read_bytes()
    i = b.index(b"end_header\n")
    head = b[:i].decode("ascii", "ignore")
    props = [ln.split()[2] for ln in head.splitlines() if ln.startswith("property")]
    n = int(next(ln for ln in head.splitlines() if ln.startswith("element vertex")).split()[2])
    a = np.frombuffer(b[i + 11:], "<f4", n * len(props)).reshape(n, len(props)).copy()
    return a, props, head


def write_splat(p: Path, a: np.ndarray, props: list[str], comment: str = "photo3d: cleaned splat, Z-up") -> None:
    with open(p, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\ncomment {comment}\nelement vertex {len(a)}\n"
                 + "".join(f"property float {q}\n" for q in props) + "end_header\n").encode())
        f.write(a.astype("<f4").tobytes())


def clean_mask(a: np.ndarray, props: list[str], dense_xyz: np.ndarray, cams: np.ndarray | None = None,
               min_opacity: float = 0.03, support_k: float = 4.0) -> tuple[np.ndarray, dict]:
    from scipy.spatial import cKDTree
    col = {q: i for i, q in enumerate(props)}
    xyz = a[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    sc = np.exp(a[:, [col["scale_0"], col["scale_1"], col["scale_2"]]].astype(np.float64))
    op = 1 / (1 + np.exp(-a[:, col["opacity"]].astype(np.float64)))
    smax = sc.max(1)
    smid = np.median(sc, 1)

    # 장면 크기: 측정된 조밀 점의 1~99% 범위
    lo, hi = np.percentile(dense_xyz, 1, 0), np.percentile(dense_xyz, 99, 0)
    L = float(np.linalg.norm(hi - lo))
    tree = cKDTree(dense_xyz)
    # 조밀 점 간격(표면 지지 거리 기준)
    sample = dense_xyz[np.random.default_rng(0).choice(len(dense_xyz), min(20000, len(dense_xyz)), replace=False)]
    spacing = float(np.median(tree.query(sample, k=2)[0][:, 1]))

    ok = np.isfinite(a).all(1)
    why = {}

    def drop(name, bad):
        nonlocal ok
        bad = bad & ok
        why[name] = int(bad.sum())
        ok = ok & ~bad
    drop("거의 투명", op < min_opacity)
    # 너무 큼: 장면 대각선의 3% 이상 또는 표면 근처 가우시안 크기의 99% 값의 8배 이상
    d_surf, _ = tree.query(xyz, workers=-1)
    near = d_surf < max(support_k * spacing, 0.005 * L)
    ref = np.percentile(smax[near & ok], 99) if (near & ok).sum() > 100 else 0.01 * L
    drop("너무 큼(빛 번짐)", smax > min(0.03 * L, max(8 * ref, 0.01 * L)))
    # 바늘: 가장 긴 축이 중간 축의 25배 이상이면서 눈에 띄게 긴 것
    drop("바늘·줄무늬", (smax / np.maximum(smid, 1e-9) > 25) & (smax > max(3 * spacing, 0.002 * L)))
    # 떠다니는 잡티: 측정된 표면에서 멀리 떨어짐(가우시안 크기만큼은 허용)
    drop("표면에서 먼 잡티", d_surf > np.maximum(max(support_k * spacing, 0.01 * L), 3 * smax))
    # 측정 범위 밖(10% 여유, 카메라 위치 포함)
    box_pts = dense_xyz if cams is None or not len(cams) else np.vstack([dense_xyz, cams])
    blo, bhi = np.percentile(box_pts, 0.5, 0), np.percentile(box_pts, 99.5, 0)
    pad = 0.1 * (bhi - blo)
    drop("측정 범위 밖", ((xyz < blo - pad) | (xyz > bhi + pad)).any(1))
    info = {"before": int(len(a)), "after": int(ok.sum()), "removed": why,
            "scene_diag_m": round(L, 2), "dense_spacing_m": round(spacing, 4)}
    return ok, info


def clean_file(splat: Path, dense_ply: Path, out: Path, cams: np.ndarray | None = None, log=print) -> dict:
    from .fill import read_points
    a, props, _ = read_splat(splat)
    xyz, _, _ = read_points(dense_ply)
    ok, info = clean_mask(a, props, np.asarray(xyz, float), cams)
    write_splat(out, a[ok], props)
    rm = ", ".join(f"{k} {v:,}" for k, v in info["removed"].items() if v)
    log(f"  스플랫 정리: {info['before']:,} → {info['after']:,}개 ({rm or '제거 없음'})")
    return info
