"""건물 메쉬 정형화 — 평평한 벽·바닥은 평평하게, 꺾이는 곳(벽 모서리·지붕 마루·처마·바닥-벽·창대)은 날카롭게.

예전 planes.planarize_mesh 의 한계: 현장 전체에서 평면 10개(무한 평면)만 찾고 그 1.5 점간격 안의 정점만 옮겨,
실제 건물에서는 대부분의 벽이 안 맞춰지고 Poisson·Delaunay 가 둥글린 모서리는 그대로였다(OpenMVS 경로는 아예 안 돌았음).

방식(문헌의 표준 조합을 단순화):
 1) 면 법선을 '경계 보존 필터'로 다듬는다 — Zheng et al. 2011 Bilateral Normal Filtering(가까운 법선끼리만 평균,
    꺾인 곳 건너편은 섞지 않음) → 잡음은 줄고 모서리는 유지
 2) 법선이 비슷한 이웃 면끼리 묶어 '평면 조각'(region growing). 넓이가 작아도 창대·돌출부처럼 일정 넓이 이상이면
    따로 조각이 된다 → 벽과 그 위 돌출면(창대 앞면·윗면)이 서로 다른 평면으로 남아 깊이가 보존된다
    조각마다 평면을 맞추고 평평하지 않으면(곡면·비정형) 건드리지 않는다
 3) 건물 규칙(Manhattan/Atlanta world): 거의 수직(±4°)인 벽은 정확히 수직, 거의 수평인 바닥·지붕 평부는 정확히 수평,
    벽 방향이 건물 주축과 ±3° 안이면 주축에 맞춘다 — 맞춘 뒤 오차가 커지면 되돌린다
 4) 정점 옮기기: 한 조각 안 정점 → 그 평면 위로. 두 조각이 만나는 정점 → 두 평면의 교선 위로(곧은 모서리),
    세 조각 → 세 평면의 교점(모서리 꼭짓점). 둥글게 뭉개진 전이 부분(어느 조각에도 안 든 정점)은 가까운 두 조각의
    평면을 각자 '자기 쪽'으로 연장한 꺾인 면 위로 옮겨 날카로운 모서리를 다시 세운다
 5) 안전장치: 어떤 정점도 R(측정 오차 범위: 점 간격 4배·메쉬 변 3배 중 큰 값) 넘게 움직이지 않는다
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _face_graph(F: np.ndarray, nF: int):
    """면 인접(모서리 공유) 쌍 (a, b)"""
    import trimesh
    adj = trimesh.graph.face_adjacency(faces=F)
    return adj[:, 0], adj[:, 1]


def filter_normals(N: np.ndarray, A: np.ndarray, a: np.ndarray, b: np.ndarray, iters: int = 6,
                   sigma: float = 0.35) -> np.ndarray:
    """경계 보존 법선 필터: 이웃 법선을 넓이 × exp(-|ni-nj|²/2σ²) 로 평균(자기 자신 포함)."""
    import scipy.sparse as sp
    n = N.copy()
    nF = len(N)
    rows = np.r_[a, b]
    cols = np.r_[b, a]
    for _ in range(iters):
        d2 = ((n[rows] - n[cols]) ** 2).sum(1)
        w = A[cols] * np.exp(-d2 / (2 * sigma ** 2))
        W = sp.csr_matrix((w, (rows, cols)), shape=(nF, nF))
        m = W @ n + A[:, None] * n
        n = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return n


def segment(Nf: np.ndarray, a: np.ndarray, b: np.ndarray, nF: int, cos_t: float):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    keep = (Nf[a] * Nf[b]).sum(1) > cos_t
    G = sp.coo_matrix((np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(nF, nF))
    return connected_components(G, directed=False)[1]


def _fit(Pc: np.ndarray, w: np.ndarray):
    c = (Pc * w[:, None]).sum(0) / w.sum()
    X = (Pc - c) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[2]
    return n / np.linalg.norm(n), c


def _planar_parts(idx: np.ndarray, C: np.ndarray, A: np.ndarray, Nf: np.ndarray, min_area: float, tol: float,
                  max_parts: int = 12, seed: int = 0):
    """법선이 이어져 한 덩어리가 된 면 묶음(완만하게 꺾이는 땅·지붕+벽이 이어진 경우)을 평평한 부분들로 나눈다.
    전체가 이미 평평하면 그대로 하나. 순차 RANSAC(면 중심, 거리 2·tol, 법선 20° 이내)."""
    n, c = _fit(C[idx], A[idx])
    if np.sqrt(np.average(((C[idx] - c) @ n) ** 2, weights=A[idx])) <= tol:
        return [idx]
    rng = np.random.default_rng(seed)
    left = idx
    out = []
    for _ in range(max_parts):
        if A[left].sum() < min_area or len(left) < 20:
            break
        P, Nl = C[left], Nf[left]
        best, best_a = None, 0.0
        for _ in range(60):
            k = left[rng.integers(len(left))]
            nn, d0 = Nf[k], -(Nf[k] @ C[k])
            inl = (np.abs(P @ nn + d0) < 2 * tol) & ((Nl @ nn) > np.cos(np.radians(20)))
            ar = A[left[inl]].sum()
            if ar > best_a:
                best, best_a = inl, ar
        if best is None or best_a < min_area:
            break
        part = left[best]
        n, c = _fit(C[part], A[part])  # 다듬은 평면으로 다시 고른다
        inl = (np.abs((P - c) @ n) < 2 * tol) & (np.abs(Nl @ n) > np.cos(np.radians(20)))
        part = left[inl]
        if A[part].sum() < min_area:
            break
        out.append(part)
        left = left[~inl]
    return out


def _manhattan(planes: list[dict], log) -> int:
    """수직 벽은 정확히 수직, 수평면은 정확히 수평, 벽 방향은 건물 주축(±3°)에 맞춘다. 바뀐 평면 수."""
    up = np.array([0.0, 0.0, 1.0])
    walls = [p for p in planes if abs(p["n"] @ up) < np.sin(np.radians(4))]
    axis = None
    if walls:  # 넓이 가중 방위각(90° 주기) 히스토그램의 최빈값 = 건물 주축
        az = np.array([np.degrees(np.arctan2(p["n"][1], p["n"][0])) % 90 for p in walls])
        w = np.array([p["area"] for p in walls])
        h, e = np.histogram(az, bins=90, range=(0, 90), weights=w)
        h = h + np.roll(h, 1) + np.roll(h, -1)
        axis = float(e[int(np.argmax(h))] + 0.5)
    changed = 0
    for p in planes:
        n = p["n"].copy()
        if abs(n @ up) > np.cos(np.radians(4)):
            n = up * np.sign(n @ up)
        elif abs(n @ up) < np.sin(np.radians(4)):
            n[2] = 0
            n /= np.linalg.norm(n)
            if axis is not None:
                az = np.degrees(np.arctan2(n[1], n[0]))
                k = np.round((az - axis) / 90.0)
                tgt = axis + 90.0 * k
                if abs(az - tgt) < 3:
                    n = np.array([np.cos(np.radians(tgt)), np.sin(np.radians(tgt)), 0.0])
        else:
            continue
        # 맞춘 평면으로 오차가 눈에 띄게 커지면 되돌린다
        r_new = np.sqrt(np.average(((p["P"] - p["c"]) @ n) ** 2, weights=p["w"]))
        if r_new <= 1.5 * p["rms"] + 0.25 * p["tol"]:
            p["n"] = n
            p["rms"] = float(r_new)
            changed += 1
    return changed


def regularize(V: np.ndarray, F: np.ndarray, spacing: float, log=print, angle: float = 12.0,
               min_area: float | None = None) -> tuple[np.ndarray, dict]:
    """정점 좌표를 정형화한 새 배열과 통계. V(Z-up, 수평 맞춤된 좌표), F(삼각형), spacing(조밀 점 간격)."""
    from scipy.spatial import cKDTree
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    nF, nV = len(F), len(V)
    T = V[F]
    cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    A = np.linalg.norm(cr, axis=1) / 2
    N0 = cr / np.maximum(2 * A[:, None], 1e-18)
    C = T.mean(1)
    le = float(np.median(np.linalg.norm(T[:, 1] - T[:, 0], axis=1)))
    tol = max(spacing, 0.5 * le)  # 측정 잡음 수준
    R = max(4 * spacing, 3 * le)  # 정점이 움직일 수 있는 최대 거리(둥근 모서리 반경 정도)
    if min_area is None:
        min_area = (12 * max(spacing, le)) ** 2  # 창대 앞면(5cm×1m)도 점 간격 1cm면 조각이 된다
    a, b = _face_graph(F, nF)
    Nf = filter_normals(N0, A, a, b)
    lab = segment(Nf, a, b, nF, np.cos(np.radians(angle)))

    # 조각마다 평면 맞추기
    order = np.argsort(lab, kind="stable")
    ls = lab[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ls)) + 1]
    ends = np.r_[starts[1:], len(ls)]
    area_by = np.add.reduceat(A[order], starts)
    planes, pid_of_lab = [], {}
    for s, e, ar in zip(starts, ends, area_by):
        if ar < min_area or e - s < 20:
            continue
        for idx in _planar_parts(order[s:e], C, A, Nf, min_area, tol):
            n, c = _fit(C[idx], A[idx])
            # 조각 정점까지 거리로 평평한지 판단(곡면·비정형은 건드리지 않는다)
            vid = np.unique(F[idx])
            dv = (V[vid] - c) @ n
            rms = float(np.sqrt(np.mean(dv ** 2)))
            if rms > 1.5 * tol or np.percentile(np.abs(dv), 95) > 4 * tol:
                continue
            # 법선이 조각의 실제 면 법선과 같은 쪽을 보게
            if (N0[idx] * A[idx, None]).sum(0) @ n < 0:
                n = -n
            pid_of_lab[int(ls[s])] = len(planes)
            planes.append({"n": n, "c": c, "rms": rms, "area": float(A[idx].sum()), "P": C[idx], "w": A[idx], "tol": tol,
                           "faces": idx})
    if not planes:
        return V.copy(), {"planes": 0}
    mh = _manhattan(planes, log)
    PN = np.array([p["n"] for p in planes])
    PD = np.array([-(p["n"] @ p["c"]) for p in planes])
    PR = np.array([p["rms"] for p in planes])
    fp = np.full(nF, -1, np.int64)
    for i, p in enumerate(planes):
        fp[p["faces"]] = i

    # 정점별 이웃 조각(최대 3개, 넓이 큰 순)
    vf_v = F.ravel()
    vf_p = np.repeat(fp, 3)
    m = vf_p >= 0
    pairs = np.unique(np.stack([vf_v[m], vf_p[m]], 1), axis=0)
    cnt = np.bincount(pairs[:, 0], minlength=nV)
    has_unlab = np.zeros(nV, bool)
    has_unlab[vf_v[~m]] = True
    newV = V.copy()
    moved_kind = np.zeros(nV, np.int8)

    # (1) 한 조각에만 속한 정점 → 그 평면 위로
    one = (cnt == 1) & ~has_unlab
    v1 = pairs[np.isin(pairs[:, 0], np.flatnonzero(one))]
    d = (V[v1[:, 0]] * PN[v1[:, 1]]).sum(1) + PD[v1[:, 1]]
    ok = np.abs(d) < 3 * PR[v1[:, 1]] + tol
    newV[v1[ok, 0]] -= d[ok, None] * PN[v1[ok, 1]]
    moved_kind[v1[ok, 0]] = 1

    # 조각 쌍의 '자기 쪽' 부호: 조각 i 는 평면 j 의 어느 쪽에 있나
    def side(i, j):
        return np.sign(np.average((planes[i]["P"] @ PN[j]) + PD[j], weights=planes[i]["w"]))

    # (2) 정점 주변 조각들(직접 닿은 조각 + 둥근 전이부는 가까운 조각) → 교선·교점·꺾인 면
    multi = np.flatnonzero((cnt >= 2) | ((cnt == 1) & has_unlab) | ((cnt == 0) & np.isin(np.arange(nV), F[fp < 0].ravel())))
    lab_v = np.flatnonzero(moved_kind == 1)
    if len(lab_v) and len(multi):
        tree = cKDTree(V[lab_v])
        vlab = np.full(nV, -1, np.int64)
        vlab[v1[ok, 0]] = v1[ok, 1]
        dd, ii = tree.query(V[multi], k=24, distance_upper_bound=R, workers=-1)
        cand = np.where(np.isfinite(dd), vlab[lab_v[np.minimum(ii, len(lab_v) - 1)]], -1)
    else:
        cand = np.full((len(multi), 1), -1)
    # 직접 닿은 조각도 후보에 넣는다
    direct = {}
    for v, p in pairs[np.isin(pairs[:, 0], multi)]:
        direct.setdefault(int(v), []).append(int(p))
    side_cache = {}
    n_line = n_pt = n_env = 0
    for row, v in enumerate(multi):
        ids = list(direct.get(int(v), []))
        c = cand[row]
        c = c[c >= 0]
        if len(c):
            u, k = np.unique(c, return_counts=True)
            o = np.argsort(-k)
            ids += [int(x) for x, kk in zip(u[o], k[o]) if kk >= 3]  # 둘레에 3개 이상 보인 조각만
        seen, sel = set(), []
        for i in ids:
            if i in seen:
                continue
            seen.add(i)
            # 거의 평행한 조각(벽과 창대 앞면처럼 단차)은 따로 — 비평행만 모서리 후보
            if all(abs(PN[i] @ PN[j]) < 0.94 for j in sel):
                sel.append(i)
            if len(sel) == 3:
                break
        p = V[v]
        dist = PN[sel] @ p + PD[sel] if sel else np.zeros(0)
        sel = [s for s, dv in zip(sel, dist) if abs(dv) < R]
        if not sel:
            continue
        if len(sel) == 1:
            q = p - (PN[sel[0]] @ p + PD[sel[0]]) * PN[sel[0]]
        else:
            i, j = sel[0], sel[1]
            if (i, j) not in side_cache:
                side_cache[(i, j)] = (side(i, j), side(j, i))
            sij, sji = side_cache[(i, j)]
            qi = p - (PN[i] @ p + PD[i]) * PN[i]
            qj = p - (PN[j] @ p + PD[j]) * PN[j]
            vi = np.sign(PN[j] @ qi + PD[j]) == sij  # qi 가 조각 i 의 실제 쪽에 있나
            vj = np.sign(PN[i] @ qj + PD[i]) == sji
            if len(sel) == 3:
                M = PN[sel]
                if abs(np.linalg.det(M)) > 0.3:
                    q3 = np.linalg.solve(M, -PD[sel])
                    if np.linalg.norm(q3 - p) < R:
                        newV[v] = q3
                        moved_kind[v] = 3
                        n_pt += 1
                        continue
            if vi and vj:
                q = qi if np.linalg.norm(qi - p) <= np.linalg.norm(qj - p) else qj
                n_env += 1
            elif vi:
                q = qi
                n_env += 1
            elif vj:
                q = qj
                n_env += 1
            else:  # 두 평면의 교선 위 가장 가까운 점
                Am = np.stack([PN[i], PN[j]])
                r = Am @ p + PD[[i, j]]
                q = p - Am.T @ np.linalg.solve(Am @ Am.T, r)
                n_line += 1
        if np.linalg.norm(q - p) < R:
            newV[v] = q
            moved_kind[v] = 2
    mv = np.linalg.norm(newV - V, axis=1)
    planar_area = float(sum(p["area"] for p in planes) / max(A.sum(), 1e-12))
    st = {"planes": len(planes), "planar_area": round(planar_area, 3), "manhattan": mh,
          "on_plane": int((moved_kind == 1).sum()), "edge_or_env": int((moved_kind == 2).sum()),
          "corner_pts": int((moved_kind == 3).sum()), "edge_lines": n_line,
          "move_p95_m": round(float(np.percentile(mv[mv > 0], 95)) if (mv > 0).any() else 0.0, 4),
          "R_m": round(R, 4), "tol_m": round(tol, 4),
          "wall_rms_mm": round(1000 * float(np.average(PR, weights=[p["area"] for p in planes])), 2)}
    return newV, st


def regularize_mesh(mesh_path: Path, points_path: Path, dst: Path, log=print) -> Path:
    """mvs 에서 부르는 진입점: 메쉬 파일 → 정형화한 메쉬 파일(실패하면 원본 경로)."""
    import trimesh
    from scipy.spatial import cKDTree
    try:
        m = trimesh.load(mesh_path, process=False, force="mesh")
        m.merge_vertices()
        from .fill import read_points
        pts = np.asarray(read_points(points_path)[0], float)
        sub = pts[:: max(1, len(pts) // 20000)]
        sp = float(np.median(cKDTree(pts).query(sub, k=2)[0][:, 1]))
        V, st = regularize(np.asarray(m.vertices), np.asarray(m.faces), sp, log=log)
        if not st.get("planes"):
            log("  건물 정형화: 평평한 면을 찾지 못해 생략")
            return mesh_path
        m.vertices = V
        # 모서리로 모인 정점 때문에 넓이 0 이 된 면 제거
        ar = m.area_faces
        m.update_faces(ar > 1e-12)
        m.remove_unreferenced_vertices()
        from .mvs import _fix_topology  # 나비 정점이 남으면 OpenMVS TextureMesh 가 죽는다
        _fix_topology(m)
        m.export(dst)
        import json
        Path(dst).with_suffix(".json").write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        log(f"  건물 정형화: 평면 조각 {st['planes']}개(표면의 {st['planar_area']:.0%}) · 수직/수평/주축 맞춤 {st['manhattan']}개 · "
            f"평면 위 정점 {st['on_plane']:,} · 모서리·꺾임 {st['edge_or_env']:,} · 꼭짓점 {st['corner_pts']:,} · "
            f"평면 오차 {st['wall_rms_mm']}mm · 이동 95% {st['move_p95_m'] * 1000:.0f}mm(한도 {st['R_m'] * 1000:.0f}mm)")
        return dst
    except Exception as e:
        log(f"  건물 정형화 생략({e})")
        return mesh_path
