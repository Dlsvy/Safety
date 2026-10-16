"""NVIDIA GPU 점검·자동 설정 — 드라이버·VRAM·전원·윈도우 설정을 읽어 처리 옵션을 GPU 에 맞춘다.

쓰는 곳
- probe(): 앱 'GPU 점검' 화면, 처리 시작 시 로그
- vram_plan(): VRAM 크기에 맞춘 조밀복원·스플랫·깊이 prior 설정 (8GB 기준으로 맞춰 둔 값을 늘리거나 줄임)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

FIELDS = ["name", "driver_version", "compute_cap", "memory.total", "memory.used", "memory.free",
          "utilization.gpu", "temperature.gpu", "power.draw", "power.limit", "power.max_limit", "pstate",
          "clocks.sm", "clocks.max.sm", "clocks.mem", "clocks.max.mem",
          "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max",
          "display_active"]


def nvidia_smi() -> str | None:
    for c in (shutil.which("nvidia-smi"), r"C:\Windows\System32\nvidia-smi.exe",
              r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if c and Path(c).exists():
            return c
    return None


def _run(args: list[str], timeout: float = 5) -> str:
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace",
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.stdout


def query() -> dict | None:
    """첫 번째 NVIDIA GPU 의 상태. 지원하지 않는 항목은 None."""
    smi = nvidia_smi()
    if not smi:
        return None
    out = {}
    # 항목마다 따로 물으면 느리므로 한 번에, 구버전 드라이버가 모르는 항목이 있으면 하나씩
    try:
        line = _run([smi, "--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits"]).strip().splitlines()[0]
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(FIELDS):
            raise ValueError
        out = dict(zip(FIELDS, vals))
    except Exception:
        for f in FIELDS:
            try:
                out[f] = _run([smi, f"--query-gpu={f}", "--format=csv,noheader,nounits"]).strip().splitlines()[0].strip()
            except Exception:
                out[f] = None
    try:  # CUDA 버전은 기본 출력 머리글에만 있다
        head = _run([smi])
        import re
        m = re.search(r"CUDA Version:\s*([\d.]+)", head)
        out["cuda_version"] = m.group(1) if m else None
    except Exception:
        out["cuda_version"] = None

    def num(k):
        try:
            return float(out.get(k))
        except (TypeError, ValueError):
            return None
    g = {"name": out.get("name"), "driver": out.get("driver_version"), "cuda": out.get("cuda_version"),
         "compute_cap": out.get("compute_cap"), "pstate": out.get("pstate"), "display": out.get("display_active")}
    for k, nk in [("memory.total", "mem_total_mb"), ("memory.used", "mem_used_mb"), ("memory.free", "mem_free_mb"),
                  ("utilization.gpu", "util"), ("temperature.gpu", "temp"), ("power.draw", "power_w"),
                  ("power.limit", "power_limit_w"), ("power.max_limit", "power_max_w"), ("clocks.sm", "sm_mhz"),
                  ("clocks.max.sm", "sm_max_mhz"), ("clocks.mem", "mem_mhz"), ("clocks.max.mem", "mem_max_mhz"),
                  ("pcie.link.gen.current", "pcie_gen"), ("pcie.link.gen.max", "pcie_gen_max"),
                  ("pcie.link.width.current", "pcie_width"), ("pcie.link.width.max", "pcie_width_max")]:
        g[nk] = num(k)
    return g


def _reg(path: str, name: str, hive: str = "HKLM"):
    if os.name != "nt":
        return None
    try:
        import winreg
        h = winreg.HKEY_LOCAL_MACHINE if hive == "HKLM" else winreg.HKEY_CURRENT_USER
        with winreg.OpenKey(h, path) as k:
            return winreg.QueryValueEx(k, name)[0]
    except OSError:
        return None


GFX = r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
PREF = r"Software\Microsoft\DirectX\UserGpuPreferences"


def windows_settings(exes: list[str] | None = None) -> dict:
    if os.name != "nt":
        return {}
    s = {"tdr_delay": _reg(GFX, "TdrDelay"), "tdr_ddi_delay": _reg(GFX, "TdrDdiDelay"),
         "tdr_level": _reg(GFX, "TdrLevel"), "hags": _reg(GFX, "HwSchMode")}
    try:
        plan = _run(["powercfg", "/getactivescheme"])
        s["power_plan"] = plan.strip().split("(")[-1].rstrip(")") if "(" in plan else plan.strip()
        s["power_plan_guid"] = plan.split(":")[1].split()[0] if ":" in plan else None
    except Exception:
        s["power_plan"] = None
    if exes:
        s["gpu_pref"] = {Path(e).name: _reg(PREF, e, "HKCU") for e in exes}
    return s


def checks(g: dict | None, w: dict) -> list[dict]:
    """점검 결과: ok/warn/bad + 조치. 근거는 NVIDIA·Microsoft·COLMAP 문서(앱 'GPU 점검' 화면에 표시)."""
    out = []

    def add(level, item, value, fix=""):
        out.append({"level": level, "item": item, "value": value, "fix": fix})
    if not g:
        add("bad", "NVIDIA GPU", "찾지 못함", "NVIDIA 드라이버 설치(nvidia-smi 가 있어야 함)")
        return out
    add("ok", "GPU", f"{g['name']} · VRAM {g['mem_total_mb'] / 1024:.0f}GB" if g.get("mem_total_mb") else g["name"])
    try:
        cc = float(g.get("compute_cap") or 0)
        drv = float((g.get("driver") or "0").split(".")[0])
    except ValueError:
        cc, drv = 0, 0
    if cc >= 12 and drv and drv < 570:
        add("bad", "드라이버", g["driver"], "RTX 50(Blackwell)은 드라이버 570 이상 필요 — NVIDIA 앱에서 최신 Studio 드라이버 설치")
    elif drv and drv < 580:
        add("warn", "드라이버", g["driver"], "CUDA 13 으로 빌드된 COLMAP 은 580 이상 필요(NVIDIA CUDA 릴리스 노트) — 최신 드라이버 권장")
    else:
        add("ok", "드라이버", f"{g.get('driver')} (CUDA {g.get('cuda')}, 연산능력 {g.get('compute_cap')})")
    if g.get("pcie_gen") and g.get("pcie_gen_max") and g.get("util", 0) > 30 and g["pcie_gen"] < g["pcie_gen_max"]:
        add("warn", "PCIe 연결", f"Gen{g['pcie_gen']:.0f} x{g.get('pcie_width') or 0:.0f} (최대 Gen{g['pcie_gen_max']:.0f})",
            "그래픽카드가 메인보드 첫 번째 x16 슬롯에 꽂혀 있는지 확인")
    elif g.get("pcie_gen"):
        add("ok", "PCIe 연결", f"Gen{g['pcie_gen']:.0f} x{g.get('pcie_width') or 0:.0f} (쉬는 중엔 낮게 표시되는 게 정상)")
    if g.get("power_limit_w") and g.get("power_max_w") and g["power_limit_w"] < 0.9 * g["power_max_w"]:
        add("warn", "전력 제한", f"{g['power_limit_w']:.0f}W / 최대 {g['power_max_w']:.0f}W",
            "제조사 튜닝 프로그램에서 전력 제한이 낮게 잡혀 있음 — 기본값으로")
    if g.get("mem_total_mb"):
        add("info", "공유 GPU 메모리", "PC 메모리를 GPU 가 빌려 쓰는 영역(PCIe 경유라 전용 VRAM 보다 수십 배 느림)",
            "앱은 전용 VRAM 을 최대로 쓰고, 부족하면 해상도를 자동으로 낮춰 다시 함. 권장: NVIDIA 제어판 → 3D 설정 관리 → "
            "'CUDA - 시스템 메모리 대체 정책' = '시스템 메모리 대체 사용 안 함'(느려지는 대신 즉시 재시도)")
    if g.get("temp") and g["temp"] >= 83:
        add("warn", "온도", f"{g['temp']:.0f}°C", "온도 때문에 속도가 떨어질 수 있음 — 케이스 통풍·먼지 확인")
    if os.name == "nt":
        td = w.get("tdr_delay")
        if (td or 2) < 30 and w.get("tdr_level") != 0:
            add("bad" if g.get("display") in ("Enabled", "1") else "warn", "GPU 시간 제한(TDR)",
                f"{td or 2}초 (윈도우 기본)",
                "화면도 같은 GPU 로 출력하면 오래 걸리는 GPU 계산(조밀복원)이 2초 넘으면 윈도우가 드라이버를 재시작해 "
                "처리가 실패함 — 'GPU 최적 설정' 실행(관리자, 재부팅 1회)")
        else:
            add("ok", "GPU 시간 제한(TDR)", f"{td}초")
        if w.get("hags") == 2:
            add("ok", "하드웨어 가속 GPU 예약", "켜짐")
        else:
            add("info", "하드웨어 가속 GPU 예약", "꺼짐", "계산 속도 영향에 대한 공식 근거 없음 — 그대로 두어도 됨")
        plan = (w.get("power_plan") or "")
        if any(k in plan for k in ("고성능", "High performance", "최고의 성능", "Ultimate")):
            add("ok", "윈도우 전원 계획", plan)
        else:
            add("info", "윈도우 전원 계획", plan or "알 수 없음", "'GPU 최적 설정'이 고성능으로 바꿈(CPU 단계인 메쉬·텍스처가 절전으로 느려지지 않게)")
        prefs = w.get("gpu_pref") or {}
        miss = [k for k, v in prefs.items() if not v or "GpuPreference=2" not in str(v)]
        if miss:
            add("info", "앱별 고성능 GPU 지정", f"미지정: {', '.join(miss)}", "내장 그래픽이 켜진 PC 에서만 의미(Brush 가 내장 그래픽을 고르는 것 방지)")
        elif prefs:
            add("ok", "앱별 고성능 GPU 지정", "지정됨")
    return out


def vram_plan(free_mb: float | None = None, total_mb: float | None = None) -> dict:
    """VRAM 에 맞춘 처리 설정. 8GB 에서 시험한 값을 기준으로 늘리거나 줄인다."""
    if total_mb is None:
        g = query()
        total_mb = (g or {}).get("mem_total_mb") or 0
        free_mb = (g or {}).get("mem_free_mb") or total_mb
    gb = (total_mb or 0) / 1024
    if gb >= 15:
        p = {"pm_max": 4000, "splats": 6_000_000, "splat_res": 2560, "moge": "vitl", "moge_side": 1024}
    elif gb >= 11:
        p = {"pm_max": 3200, "splats": 4_000_000, "splat_res": 2560, "moge": "vitl", "moge_side": 896}
    elif gb >= 7:  # 8GB: 폰 12MP 원본 크기(4000px)까지 시도, 부족하면 mvs 가 3200 → 2400 으로 자동 재시도
        p = {"pm_max": 4000, "splats": 3_000_000, "splat_res": 1920, "moge": "vitl", "moge_side": 768}
    elif gb >= 5:
        p = {"pm_max": 2000, "splats": 1_500_000, "splat_res": 1600, "moge": "vitb", "moge_side": 640}
    else:
        p = {"pm_max": 1600, "splats": 800_000, "splat_res": 1280, "moge": "vitb", "moge_side": 512}
    # 다른 프로그램(게임·브라우저 영상)이 VRAM 을 많이 쓰고 있으면 한 단계 낮춘다
    if free_mb and total_mb and free_mb < 0.5 * total_mb:
        p["pm_max"] = int(p["pm_max"] * 0.75)
        p["splats"] = int(p["splats"] * 0.6)
        p["busy"] = True
    p["vram_gb"] = round(gb, 1)
    return p


def ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except ImportError:
        pass
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("rest", ctypes.c_ulonglong * 6)]
            m = MS(); m.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total / 2**30
        return int(open("/proc/meminfo").readline().split()[1]) / 2**20
    except Exception:
        return 16.0


def summary_line(g: dict | None) -> str:
    if not g:
        return "GPU: NVIDIA 없음 → CPU 처리"
    return (f"GPU: {g['name']} · VRAM {(g.get('mem_free_mb') or 0) / 1024:.1f}/{(g.get('mem_total_mb') or 0) / 1024:.1f}GB 여유 · "
            f"드라이버 {g.get('driver')} · CUDA {g.get('cuda')} · {g.get('temp') or 0:.0f}°C")
