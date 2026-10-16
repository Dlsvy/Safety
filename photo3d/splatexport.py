"""스플랫 가벼운 형식 내보내기 — 별도 설치 없이 파이썬으로.

- .spz (Niantic, MIT 형식, v2): 원본 PLY 의 약 1/10. SuperSplat·splat-transform·여러 뷰어가 읽음(보관·공유용)
- .splat (antimatter15 웹 뷰어 형식): 32바이트/개, 색은 기본색만(시점별 색 변화 없음) — 웹 페이지에 띄우기용
형식은 splat-transform(PlayCanvas, MIT)의 읽기 코드로 확인했다(read-spz.ts).
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

from .splatclean import read_splat

SH_C0 = 0.28209479177387814


def _cols(path: Path):
    a, props, _ = read_splat(path)
    col = {q: i for i, q in enumerate(props)}
    g = lambda k: a[:, col[k]].astype(np.float64)  # noqa: E731
    xyz = np.stack([g("x"), g("y"), g("z")], 1)
    dc = np.stack([g("f_dc_0"), g("f_dc_1"), g("f_dc_2")], 1)
    n_rest = len([q for q in props if q.startswith("f_rest_")])
    rest = np.stack([g(f"f_rest_{i}") for i in range(n_rest)], 1) if n_rest else np.zeros((len(a), 0))
    op = g("opacity")
    sc = np.stack([g("scale_0"), g("scale_1"), g("scale_2")], 1)
    q = np.stack([g("rot_0"), g("rot_1"), g("rot_2"), g("rot_3")], 1)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return xyz, dc, rest, op, sc, q


def _u8(v):
    return np.clip(np.round(v), 0, 255).astype(np.uint8)


def write_spz(src: Path, dst: Path, frac_bits: int = 12) -> int:
    xyz, dc, rest, op, sc, q = _cols(src)
    n = len(xyz)
    sh_deg = {0: 0, 9: 1, 24: 2, 45: 3}.get(rest.shape[1], 0)
    k = {0: 0, 1: 3, 2: 8, 3: 15}[sh_deg]
    # 위치: 24비트 고정소수점(1/2^frac_bits m 단위). 범위 ±(2^23/2^12)=±2048 m
    p = np.clip(np.round(xyz * (1 << frac_bits)), -(1 << 23), (1 << 23) - 1).astype(np.int32)
    pb = p.astype("<i4").view(np.uint8).reshape(n, 3, 4)[:, :, :3].reshape(n, 9)
    alpha = _u8(1 / (1 + np.exp(-op)) * 255)
    color = _u8((dc * 0.15 + 0.5) * 255)
    scale = _u8((sc + 10.0) * 16.0)
    qq = q.copy()
    qq[qq[:, 0] < 0] *= -1  # w ≥ 0 으로 맞추고 x,y,z 만 저장(v2)
    rot = _u8((qq[:, 1:4] + 1.0) * 127.5)
    if k:  # PLY: 채널별 [R0..Rk-1, G0.., B0..] → spz: 계수별 [R,G,B]
        r = rest[:, :3 * k].reshape(n, 3, k).transpose(0, 2, 1).reshape(n, 3 * k)
        shb = _u8(r * 128 + 128)
    else:
        shb = np.zeros((n, 0), np.uint8)
    head = struct.pack("<IIIBBBB", 0x5053474E, 2, n, sh_deg, frac_bits, 0, 0)
    body = b"".join(x.tobytes() for x in (pb, alpha, color, scale, rot, shb))
    with gzip.open(dst, "wb", compresslevel=6) as f:
        f.write(head + body)
    return n


def write_splat(src: Path, dst: Path) -> int:
    xyz, dc, _, op, sc, q = _cols(src)
    a = 1 / (1 + np.exp(-op))
    s = np.exp(sc)
    order = np.argsort(-(s.prod(1) * a))  # 크고 진한 것부터(뷰어가 먼저 그림)
    rec = np.zeros(len(xyz), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"] = xyz[order]
    rec["s"] = s[order]
    rgb = 0.5 + SH_C0 * dc[order]
    rec["c"] = np.concatenate([_u8(rgb * 255), _u8(a[order] * 255)[:, None]], 1)
    rec["r"] = _u8(q[order] * 128 + 128)
    dst.write_bytes(rec.tobytes())
    return len(xyz)
