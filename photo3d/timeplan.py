"""시간 기준 처리 계획 — 모드마다 목표 시간 안에 끝나도록 쓸 사진 수·설정을 정하고, 단계별 예상 시간을 낸다.

목표: 아주 빠르게 10~15분 · 빠르게 25~30분 · 최고 품질 50~80분(사진이 많으면 고르게 퍼진 선명한 사진만 골라 쓴다).
계수는 이 PC(RTX 5060 Ti 8GB) 실제 기록에서 맞췄다(2026-10-18, 1Family 152장·2Francis 302장·8장 시험):
  짝 맞추기 192초/152장(사진 수 제곱), 증분 매퍼 190초/152장(1.5제곱), PatchMatch 1600px·이웃 20장 10.8초/장
  (2000px 13.3초/장, 이웃 7장 6.7초/장 → 장당 3.15 + 0.508×이웃 수, 해상도에 비례), MoGe 깊이 보조 0.8초/장,
  MapAnything 12장 97~123초(불러오기 포함, 사진 수 제곱), PGSR 7000스텝 1209초(152장), MeshSplatting 1만 스텝 432초.
처리가 끝날 때마다 실제/예상 비율을 기록해(app.record_timing) 다음 예상에 곱한다(speed).
"""
from __future__ import annotations

import math

TARGET_MIN = {"instant": (10, 15), "fast": (25, 30), "best": (50, 80)}
# 모드별 고정 설정(사진 수만 계획으로 정한다). 흰 벽 채우기(MoGe)는 모두 켠다 — 타설 후 콘크리트 내부·지하처럼
# 무늬 없는 벽이 주 대상이라 구멍을 줄이는 효과가 크다.
# 아주 빠르게(2026-10-19 사용자: AI 만으로는 품질이 떨어짐): COLMAP 카메라·조밀복원(1600px)로, 카메라를 못 잡으면 MapAnything.
SETTINGS = {
    "instant": {"sfm": "colmap", "dense": "colmap", "pm_size": 1600, "pm_src": 8, "prior": True, "splat_steps": 7000,
                "splat_cap": 1_000_000, "extras": False, "pgsr_steps": 0, "ms_steps": 0, "best_mesh": False},
    "fast": {"sfm": "colmap", "dense": "colmap", "pm_size": 2000, "pm_src": 12, "prior": True, "splat_steps": 10000,
             "splat_cap": 1_500_000, "extras": False, "pgsr_steps": 7000, "ms_steps": 10000, "best_mesh": True},
    "best": {"sfm": "colmap", "dense": "colmap", "pm_size": 2400, "pm_src": 16, "prior": True, "splat_steps": 30000,
             "splat_cap": 3_000_000, "extras": True, "pgsr_steps": 15000, "ms_steps": 20000, "best_mesh": True},
}
MIN_IMAGES = {"instant": 15, "fast": 40, "best": 60}  # 품질 하한 — 시간이 넘쳐도 이 아래로는 줄이지 않고 "목표보다 김"으로 알린다
MAX_IMAGES = {"instant": 40, "fast": 150, "best": 400}


def stages(n: int, mode: str, kind: str = "fuse", extras: bool | None = None, mp: float = 2.0) -> dict:
    """단계별 예상 초. n = 실제로 쓸 사진 수, mp = 사진 한 장 화소(백만), kind = splat/pg/fuse"""
    s = dict(SETTINGS[mode])
    if extras is not None:
        s["extras"] = extras
    n = max(1, n)
    st = {}
    st["입력 정리"] = 5 + 0.06 * n * max(1.0, mp / 2)
    if s["sfm"] == "ai":
        st["카메라 위치(AI)"] = 20 + 0.5 * n * n
    else:
        st["사람·차량 가림"] = 0.2 * n
        st["카메라 위치"] = 0.06 * n + 0.0083 * n * n + 0.1 * n ** 1.5 + 0.28 * n + (0.08 * n ** 1.5 if mode == "best" else 0)
    if s["dense"] == "ai":
        st["조밀 점(AI)"] = 5
    else:
        src = min(s["pm_src"], n - 1)
        st["조밀복원"] = n * (3.15 + 0.508 * src) * (s["pm_size"] / 2000) + 0.4 * n  # + 융합·정리
    if s["prior"]:
        st["흰 벽 채우기"] = 30 + 0.8 * n
    # 메쉬·텍스처: 1Family 74장 422초(Poisson 깊이 11 — 빠르게는 이제 10), 12장 약 80초
    st["메쉬·텍스처"] = (150 + 3.0 * n) if mode == "best" else (60 + 1.6 * n)
    # 끝부분(실측, 사진 수에 비례): 내보내기·바닥 정사영상 74장 190초·12장 60초, 스플랫 마무리(바닥 교체·통합 정리)
    # 74장 150초, 빈 곳 메운 최적 합성 메쉬 74장 233초·12장 50초, 최적 합성 74장 80초(정리 재사용 후)
    st["내보내기·유니티"] = 40 + 2.2 * n
    if kind in ("splat", "fuse"):
        train = s["splat_steps"] * 0.028 * (1.5 if s["splat_cap"] > 2_000_000 else 1.0)  # 7000스텝 약 196초(100만 개)
        # 통합(fuse)은 측정 메쉬로 만든 표면 방향 지도를 쓰므로 메쉬가 끝난 뒤 학습을 시작한다 → 내보내기와만 겹친다
        overlap = st["내보내기·유니티"] + (0 if kind == "fuse" else st["메쉬·텍스처"] + st.get("흰 벽 채우기", 0))
        st["실사(스플랫) 마무리"] = max(0.0, train - overlap) + ((80 + 0.9 * n) if kind == "fuse" else 40)
    if kind == "fuse" and s["extras"]:
        # PGSR: 1000스텝 54초(불러오기·TSDF 포함), 다른 프로그램이 GPU 를 쓰면 7000스텝 20분까지 늘었다 → 보통 값으로 두고 speed 로 보정
        st["PGSR 평면 메쉬"] = 30 + s["pgsr_steps"] * 0.06 * min(1.0, 0.4 + n / 250)
    if kind == "fuse" and (s["best_mesh"] or s["extras"]):
        st["최적 합성 메쉬"] = 40 + 2.7 * n
    if kind == "fuse" and s["extras"]:
        st["MeshSplatting"] = 30 + s["ms_steps"] * 0.043 * min(1.0, 0.5 + n / 300)
    st["마무리"] = ((20 + 0.8 * n) if kind == "fuse" else 10) + 15  # 최적 합성·보고서(유니티 패키지는 받을 때 만든다)
    return st


def plan(n_avail: int, mode: str, kind: str = "fuse", n_use: int | None = None, extras: bool | None = None,
         mp: float = 2.0, speed: float = 1.0) -> dict:
    """쓸 사진 수(n_use 를 주면 그대로, 아니면 목표 시간 상한 안에서 가장 많이)와 단계별 예상 시간(분)"""
    lo, hi = TARGET_MIN[mode]
    n_avail = max(1, n_avail)
    cap = min(n_avail, MAX_IMAGES[mode])

    def total(n, ex):
        return sum(stages(n, mode, kind, ex, mp).values()) * speed / 60

    ex = SETTINGS[mode]["extras"] if extras is None else extras
    if n_use is None and extras is None and ex and total(min(cap, 40), ex) > hi:
        ex = False  # 추가 기술만으로 목표를 넘으면 빼고 사진 수를 지킨다
    if n_use is None:
        n = cap
        # 목표 상한을 넘으면 사진 수를 줄인다(최고 품질은 먼저 추가 기술의 스텝을 지키고, 그래도 넘으면 사진 수)
        goal = lo + 0.7 * (hi - lo)  # 상한에 딱 맞추면 조금만 늦어도 넘는다 → 범위의 70% 지점을 겨냥
        while n > MIN_IMAGES[mode] and total(n, ex) > goal:
            n = max(MIN_IMAGES[mode], int(n * 0.92))
        auto = True
    else:
        n = max(1, min(n_use, n_avail))
        auto = False
    st = stages(n, mode, kind, ex, mp)
    mins = {k: round(v * speed / 60, 1) for k, v in st.items()}
    return {"mode": mode, "n_avail": n_avail, "n_use": n, "auto": auto, "extras": bool(ex), "target_min": [lo, hi],
            "settings": {**SETTINGS[mode], "extras": bool(ex)}, "stages_min": mins,
            "total_min": round(sum(mins.values()), 1), "speed": round(speed, 2)}


def curve(n_avail: int, mode: str, kind: str = "fuse", extras: bool | None = None, mp: float = 2.0,
          speed: float = 1.0, points: int = 24) -> list[tuple[int, float]]:
    """화면 그래프용: 쓸 사진 수 → 예상 분"""
    cap = max(1, min(n_avail, MAX_IMAGES[mode]))
    lo = min(MIN_IMAGES[mode], cap)
    ns = sorted({int(round(lo + (cap - lo) * i / max(1, points - 1))) for i in range(points)})
    return [(n, plan(n_avail, mode, kind, n, extras, mp, speed)["total_min"]) for n in ns]


def fmt(p: dict) -> str:
    top = sorted(p["stages_min"].items(), key=lambda kv: -kv[1])[:3]
    return (f"계획: 사진 {p['n_use']}/{p['n_avail']}장 · 예상 {p['total_min']:.0f}분(목표 {p['target_min'][0]}~{p['target_min'][1]}분)"
            f" · 큰 단계 " + ", ".join(f"{k} {v:.0f}분" for k, v in top) + (f" · 이 PC 보정 ×{p['speed']}" if p["speed"] != 1 else ""))


def _selftest():
    for mode, n in (("instant", 152), ("fast", 152), ("fast", 40), ("best", 152), ("best", 302)):
        p = plan(n, mode)
        print(mode, n, "→", p["n_use"], p["total_min"], p["stages_min"])


if __name__ == "__main__":
    _selftest()
    print(math.pi and "ok")
