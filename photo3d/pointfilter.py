"""조밀 점군 정리 — 메쉬를 만들기 전에 '시선 방향 가시'와 하늘·허공 점을 없앤다.

증상(PC castle-P19): 안뜰 가운데(촬영 위치)를 향해 벽·지면에서 뾰족한 가시가 방사형으로 뻗고,
벽 위쪽 하늘에 흰·하늘색 덩어리가 생김.
원인: 깊이 추정이 틀린 픽셀(하늘, 그늘, 무늬 없는 벽, 비스듬한 지면)은 점이 '카메라에서 그 픽셀로 가는
시선' 위의 엉뚱한 깊이에 놓인다. 여러 장에서 조금씩 다른 깊이로 놓인 점들이 시선을 따라 줄지어 늘어서고
(= 선 모양 점 무리), Poisson 이 그 줄을 따라 면을 늘려 가시가 된다.

정리 순서
 1) 지지 이미지 수: COLMAP fused.ply.vis 가 있으면 2장 이하에서만 보인 점 제거(우연 일치)
 2) 통계적 이상점(SOR): 이웃까지 평균 거리가 전체 분포에서 크게 벗어난 점 제거
 3) 선형성(가시) 필터: 이웃 점들의 분포를 주성분으로 보아, 면(2차원)이 아니라 선(1차원)처럼 늘어선 점 제거.
    실제 표면은 λ1≈λ2≫λ3, 시선 줄무늬는 λ1≫λ2.
 4) 카메라 시선 정렬 검사: 선형 무리의 방향이 가장 가까운 카메라를 향하면(시선 줄무늬) 더 엄격히 제거
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


LAST: dict = {}  # 마지막 정리 결과(품질 보고서용)


def _read_vis(path: Path, n: int) -> np.ndarray | None:
    """COLMAP fused.ply.vis → 점마다 보인 이미지 수"""
    try:
        b = path.read_bytes()
        cnt = struct.unpack_from("<Q", b, 0)[0]
        if cnt != n:
            return None
        out = np.empty(n, np.int32)
        off = 8
        for i in range(n):
            k = struct.unpack_from("<I", b, off)[0]
            out[i] = k
            off += 4 + 4 * k
        return out
    except Exception:
        return None


def clean_points(xyz: np.ndarray, cams: np.ndarray | None = None, vis: np.ndarray | None = None,
                 k: int = 16, sor_std: float = 6.0, lin_thr: float = 0.88, log=print) -> np.ndarray:
    """남길 점의 불리언 마스크를 돌려준다."""
    n = len(xyz)
    keep = np.ones(n, bool)
    if vis is not None:
        keep &= vis >= 3
    tree = cKDTree(xyz)
    d, idx = tree.query(xyz, k=k + 1, workers=-1)
    md = d[:, 1:].mean(1)
    mu, sd = np.median(md), 1.4826 * np.median(np.abs(md - np.median(md)))
    sor = md > mu + sor_std * max(sd, 1e-12)
    keep &= ~sor
    # 이웃 공분산의 고유값으로 선형성 계산 (배치로 처리해 메모리 제한)
    lin = np.zeros(n, np.float32)
    dirv = np.zeros((n, 3), np.float32)
    for s in range(0, n, 200000):
        nb = xyz[idx[s:s + 200000, 1:]]                     # (b,k,3)
        nb = nb - nb.mean(1, keepdims=True)
        cov = np.einsum("bki,bkj->bij", nb, nb) / k
        w, v = np.linalg.eigh(cov)                          # 오름차순
        l1, l2 = w[:, 2], w[:, 1]
        lin[s:s + 200000] = (l1 - l2) / np.maximum(l1, 1e-18)
        dirv[s:s + 200000] = v[:, :, 2]
    spike = lin > lin_thr
    if cams is not None and len(cams):
        # 선 방향이 가장 가까운 카메라를 향하면 시선 줄무늬 → 기준을 낮춰 더 잡는다
        _, ci = cKDTree(cams).query(xyz, workers=-1)
        ray = cams[ci] - xyz
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)
        along = np.abs((ray * dirv).sum(1)) > 0.9
        spike |= (lin > lin_thr - 0.2) & along
    keep &= ~spike
    LAST.update(raw=int(n), kept=int(keep.sum()))
    log(f"  점군 정리: {n:,} → {keep.sum():,}점 (적은 지지 {0 if vis is None else int((vis < 3).sum()):,}, "
        f"이상점 {int(sor.sum()):,}, 가시·시선 줄무늬 {int(spike.sum()):,})")
    return keep


def clean_ply(src: Path, dst: Path, cams: np.ndarray | None = None, log=print, work: Path | None = None) -> Path:
    """COLMAP fused.ply(법선·색 포함) → 정리된 PLY. 실패 시 원본 경로를 돌려준다."""
    import trimesh
    try:
        pc = trimesh.load(src, process=False)
        xyz = np.asarray(pc.vertices, float)
        raw = pc.metadata.get("_ply_raw", {}).get("vertex", {}).get("data")
        vis = _read_vis(Path(str(src) + ".vis"), len(xyz))
        keep = clean_points(xyz, cams, vis, log=log)
        if work is not None:  # 여러 사진 깊이에서 '허공'인 점(얇은 기둥이 시선 방향으로 늘어난 가시) 제거
            from .visfilter import free_space_mask
            bad = free_space_mask(xyz, work, log=log)
            if bad is not None:
                keep &= ~bad
        if raw is not None:
            data = raw[keep]
            names = data.dtype.names
            with open(dst, "wb") as f:
                hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(data)
                for nm in names:
                    t = data.dtype[nm]
                    hdr += f"property {'float' if t.kind == 'f' and t.itemsize == 4 else 'double' if t.kind == 'f' else 'uchar' if t.itemsize == 1 else 'int'} {nm}\n"
                f.write((hdr + "end_header\n").encode())
                f.write(data.astype(data.dtype.newbyteorder("<")).tobytes())
        else:
            trimesh.PointCloud(xyz[keep], colors=np.asarray(pc.colors)[keep]).export(dst)
        return dst
    except Exception as e:
        log(f"  점군 정리 생략({e})")
        return src
