"""품질 보고서(HTML) — Pix4D Quality Report·DJI Terra 품질 보고서를 참고한 신호등식 판정.

report.json(처리 기록) + SfM 모델(카메라·연결점)로 만든다. 각 항목은 좋음/주의/나쁨 + 아이콘 + 글자로 표시
(색만으로 뜻을 전하지 않음). 겹침 지도: 위에서 본 연결점을 '몇 장의 사진에 찍혔는지'로 칠한다(Pix4D 오버랩 지도).
기준값
- 등록률: 95% 이상 좋음, 75% 이상 주의 (Pix4D)
- 재투영오차: 1.0px 이하 좋음, 1.5px 이하 주의
- 평균 트랙 길이(점 하나를 본 사진 수): 3.0 이상 좋음, 2.5 이상 주의
- 사진당 관측 수: 1000 이상 좋음, 300 이상 주의 (Pix4D '이미지당 매칭' 기준을 단순화)
- 축척 검증 오차: 30mm 이하 좋음, 100mm 이하 주의 (현장 판정용)
- 겹침: 5장 이상에서 보인 점 비율 50% 이상 좋음, 25% 이상 주의
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

GOOD, WARN, BAD, INFO = "good", "warn", "bad", "info"
ICON = {GOOD: "✓", WARN: "!", BAD: "✕", INFO: "i"}
WORD = {GOOD: "좋음", WARN: "주의", BAD: "나쁨", INFO: "참고"}


def _lvl(v, good, warn, higher_better=True):
    if v is None:
        return INFO
    if higher_better:
        return GOOD if v >= good else WARN if v >= warn else BAD
    return GOOD if v <= good else WARN if v <= warn else BAD


def _overlap_svg(model_dir: Path, w: int = 640, h: int = 420) -> tuple[str, dict]:
    import pycolmap
    rec = pycolmap.Reconstruction(model_dir)
    C = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])
    names = [rec.image(i).name for i in rec.reg_image_ids()]
    pts = list(rec.points3D.values())
    if not pts:
        return "", {}
    P = np.array([p.xyz for p in pts])
    T = np.array([p.track.length() for p in pts])
    sel = np.random.default_rng(0).choice(len(P), min(6000, len(P)), replace=False)
    P, T = P[sel], T[sel]
    lo = np.minimum(np.percentile(P[:, :2], 1, 0), C[:, :2].min(0))
    hi = np.maximum(np.percentile(P[:, :2], 99, 0), C[:, :2].max(0))
    span = np.maximum(hi - lo, 1e-9)
    s = min((w - 40) / span[0], (h - 40) / span[1])
    off = np.array([(w - span[0] * s) / 2, (h - span[1] * s) / 2])

    def xy(a):
        q = (a[:, :2] - lo) * s + off
        q[:, 1] = h - q[:, 1]
        return q

    q = xy(P)
    cls = np.where(T >= 5, "o-good", np.where(T >= 3, "o-warn", "o-bad"))
    dots = "".join(f'<circle class="{c}" cx="{x:.1f}" cy="{y:.1f}" r="1.6"/>' for (x, y), c in zip(q, cls))
    qc = xy(C)
    cams = "".join(f'<g class="cam"><circle cx="{x:.1f}" cy="{y:.1f}" r="5"/><title>{html.escape(n)}</title></g>'
                   for (x, y), n in zip(qc, names))
    ratio = {"ge5": float((T >= 5).mean()), "3to4": float(((T >= 3) & (T < 5)).mean()), "le2": float((T < 3).mean())}
    svg = (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="위에서 본 겹침 지도: 점 색은 그 점을 찍은 사진 수, '
           f'검은 원은 사진 위치">{dots}{cams}</svg>')
    return svg, ratio


def write(out: Path, rep: dict, work: Path, log=print) -> Path:
    s, g = rep.get("sfm", {}), rep.get("georef", {})
    rows = []
    svg, ratio = "", {}
    for cand in (work / "sfm" / "georef", work / "dense" / "sparse"):
        if (cand / "points3D.bin").exists():
            try:
                svg, ratio = _overlap_svg(cand)
            except Exception as e:
                log(f"  겹침 지도 생략: {e}")
            break

    def row(section, name, value, level, note=""):
        rows.append((section, name, value, level, note))

    rr = s.get("registration_ratio")
    row("사진 정렬", "등록된 사진", f"{s.get('images_registered')}/{s.get('images_input')}장 ({rr:.0%})" if rr is not None else "-",
        _lvl(rr, 0.95, 0.75), "95% 이상이면 좋음. 빠진 사진은 주변을 더 촘촘히 찍으세요")
    nm = s.get("num_models", 1)
    row("사진 정렬", "모델 분리", f"{nm}개", GOOD if nm == 1 else BAD, "2개 이상이면 촬영 간격이 넓은 구간이 있음")
    re_ = s.get("mean_reprojection_error_px")
    row("사진 정렬", "평균 재투영오차", f"{re_:.2f}px" if re_ is not None else "-", _lvl(re_, 1.0, 1.5, False),
        "사진과 3D 점이 얼마나 잘 맞는지(작을수록 좋음)")
    tl = s.get("mean_track_length")
    row("사진 정렬", "점 하나를 본 사진 수(평균)", f"{tl:.2f}장" if tl else "-", _lvl(tl, 3.0, 2.5),
        "3장 이상이면 깊이가 안정적")
    if s.get("points3D") and s.get("images_registered"):
        obs = s["points3D"] * (tl or 0) / max(s["images_registered"], 1)
        row("사진 정렬", "사진당 연결점", f"{obs:,.0f}개", _lvl(obs, 1000, 300), "무늬가 적은 면은 적게 나옴")
    if ratio:
        row("사진 정렬", "겹침(5장 이상에서 보인 점)", f"{ratio['ge5']:.0%}", _lvl(ratio["ge5"], 0.5, 0.25),
            "Terra 권장 겹침: 옆 60%·앞 65% 이상")

    gs = s.get("gradual_selection")
    if gs:
        row("사진 정렬", "나쁜 연결점 정리", f"{gs['removed']:,}점 제거", INFO, "오차 큰 점·각도 작은 점 제거 후 재조정")
    rej = s.get("images_rejected") or []
    if rej:
        row("사진 정렬", "잘못 등록 의심 제외", f"{len(rej)}장", WARN, ", ".join(rej[:6]))

    m = g.get("method", "")
    if m.startswith("none"):
        row("축척·정확도", "실제 크기", "미확정(임의 단위)", BAD, "마커·줄자 거리·GCP 중 하나가 필요합니다")
    else:
        row("축척·정확도", "축척 방법", m, INFO, "")
        errs = [abs(r["error_mm"]) for r in g.get("check_distances", [])]
        errs += [v * 1000 for v in g.get("checkpoint_error_m", {}).values()]
        if errs:
            e = max(errs)
            row("축척·정확도", "검증 최대오차", f"{e:.0f}mm", _lvl(e, 30, 100, False), "독립 검증거리·검사점 기준")
        else:
            row("축척·정확도", "독립 검증", "없음", WARN, "check.csv(검증 거리)를 넣으면 정확도를 증명할 수 있음")
    mk = g.get("markers_triangulated") or []
    if not m.startswith("none") or mk:
        n_mk = len(mk) if isinstance(mk, (list, dict)) else 0
        # 엔젤스윙·Terra: 기준점 최소 4개, 넓은 현장은 50m×50m 마다 1개(고르게)
        bb = (rep.get("model") or {}).get("size_m") or [0, 0, 0]
        need = max(4, int(np.ceil(float(bb[0]) * float(bb[1]) / 2500.0))) if len(bb) >= 2 else 4
        row("축척·정확도", "기준점(마커·GCP) 수", f"{n_mk}개 / 권장 {need}개 이상",
            GOOD if n_mk >= need else WARN if n_mk >= 2 else BAD,
            "최소 4개, 넓은 현장은 50m×50m 마다 1개를 고르게 (엔젤스윙·DJI Terra 기준)")
    if g.get("marker_corner_reproj_px") is not None:
        row("축척·정확도", "마커 모서리 재투영", f"{g['marker_corner_reproj_px']:.2f}px",
            _lvl(g["marker_corner_reproj_px"], 1.0, 2.0, False), "")

    d = rep.get("dense") or {}
    if d.get("raw"):
        k = d["kept"] / d["raw"]
        row("3D 생성", "조밀 점(정리 후)", f"{d['kept']:,}점 ({k:.0%} 유지)", INFO, f"가시·이상점 {d['raw'] - d['kept']:,}점 제거")
    if d.get("prior_status") and not d.get("prior_added"):
        row("3D 생성", "흰 벽 채우기(깊이 prior)", "안 됨", WARN, d["prior_status"])
    if d.get("prior_added"):
        row("3D 생성", "깊이 prior 로 채운 점", f"{d['prior_added']:,}점", INFO,
            "무늬 없는 벽·바닥을 MoGe-2 깊이로 채움(측정과 맞추고 다른 사진과 일치하는 점만)")
    tx = d.get("texture") or {}
    if tx:
        row("3D 생성", "사진 텍스처가 입혀진 면", f"{tx['seen']:.1%}", _lvl(tx["seen"], 0.95, 0.85),
            f"패치 {tx['patches']:,}개, 사진 1px → 텍스처 {tx['scale']:.2f}px")
    md = rep.get("model") or {}
    if md.get("faces"):
        row("3D 생성", "메쉬", f"{md['faces']:,}면", INFO, f"충돌용 {md.get('collider_faces', 0):,}면")
    fl = rep.get("fill") or {}
    if "estimated_area_ratio" in fl:
        er = fl["estimated_area_ratio"]
        row("3D 생성", "메운(추정) 면적", f"{er:.0%}", _lvl(er, 0.15, 0.35, False), "측정이 아닌 부분 — 유니티에서 경고 레이어")
    sp = rep.get("splat") or {}
    if sp.get("splats"):
        n = sp["splats"]
        row("3D 생성", "실사(가우시안 스플래팅)", f"{n:,}개 · {sp.get('minutes', '?')}분",
            WARN if n < 300_000 else INFO,
            f"{sp['steps']:,}스텝" + (" · 30만 개 미만이면 흐릿함 — 사진을 더 촘촘히" if n < 300_000 else ""))
        cl = sp.get("clean") or {}
        if cl.get("before"):
            rm = ", ".join(f"{k} {v:,}" for k, v in cl["removed"].items() if v)
            row("3D 생성", "스플랫 정리(빛 번짐·잡티)", f"{cl['before'] - cl['after']:,}개 제거", INFO, rm or "제거 없음")

    fp = rep.get("floor_patch") or {}
    if fp.get("area_m2"):
        row("측량 결과물", "사진 바닥 조각", f"{fp['area_m2']} m²", INFO,
            "비스듬히 찍혀 측정이 빈 바닥을 사진 정사영상으로 메움(높이는 바닥 평면, 색은 사진) — 바닥 위 물체 높이는 측정 아님")
    sv = rep.get("survey") or {}
    oi = sv.get("ortho") or {}
    if oi.get("m_per_px"):
        gsd = oi["m_per_px"] * 100
        row("측량 결과물", "정사영상 해상도", f"{gsd:.1f} cm/px", _lvl(gsd, 2.0, 5.0, False),
            f"{oi['size_px'][0]}×{oi['size_px'][1]}px · 덮인 비율 {oi['coverage']:.0%} · site_ortho.jpg(+.jgw)")
        row("측량 결과물", "DSM 높이 범위", f"{oi['z_range_m'][0]:.2f} ~ {oi['z_range_m'][1]:.2f} m", INFO,
            "site_dsm.tif(높이 m, 빈 곳 -9999) · 보기용 site_dsm_color.png")
    pt = sv.get("points") or {}
    if pt.get("points"):
        row("측량 결과물", "점군", f"{pt['points']:,}점", INFO, "site_points.las — CloudCompare·Civil3D·Revit·ReCap")

    gr = rep.get("grade", {})
    counts = {k: sum(1 for r in rows if r[3] == k) for k in (GOOD, WARN, BAD)}
    body = []
    sec = None
    for section, name, value, level, note in rows:
        if section != sec:
            if sec is not None:
                body.append("</tbody></table>")
            body.append(f"<h2>{html.escape(section)}</h2><table><tbody>")
            sec = section
        body.append(f'<tr><th scope="row">{html.escape(name)}</th><td class="v">{html.escape(str(value))}</td>'
                    f'<td><span class="st st-{level}"><b aria-hidden="true">{ICON[level]}</b>{WORD[level]}</span></td>'
                    f'<td class="n">{html.escape(note)}</td></tr>')
    body.append("</tbody></table>")
    legend = ('<div class="lg"><span><i class="sw o-good"></i>5장 이상</span><span><i class="sw o-warn"></i>3~4장</span>'
              '<span><i class="sw o-bad"></i>2장 이하</span><span><i class="sw camk"></i>사진 위치</span></div>')
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>품질 보고서</title>
<style>
:root{{--bg:#f6f5f2;--panel:#fff;--ink:#1d1f22;--muted:#5d636b;--line:#e3e1dc;--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;--info:#5d636b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151719;--panel:#1d2023;--ink:#eceae6;--muted:#a2a7ad;--line:#33373c}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,"Malgun Gothic",sans-serif}}
main{{max-width:980px;margin:0 auto;padding:20px 16px 40px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:15px;margin:22px 0 8px}}
.sum{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;display:flex;gap:18px;flex-wrap:wrap;align-items:center}}
.grade{{font-size:34px;font-weight:700;width:52px;text-align:center}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{padding:8px 10px;border-top:1px solid var(--line);text-align:left;vertical-align:top;font-size:14px}}
tr:first-child th,tr:first-child td{{border-top:0}}
th{{font-weight:500;width:28%}} .v{{font-variant-numeric:tabular-nums;font-weight:600;width:22%}} .n{{color:var(--muted);font-size:13px}}
.st{{display:inline-flex;gap:6px;align-items:center;font-size:13px;font-weight:600}}
.st b{{display:inline-grid;place-items:center;width:18px;height:18px;border-radius:50%;color:#fff;font-size:11px}}
.st-good b{{background:var(--good)}} .st-warn b{{background:var(--warn);color:#1d1f22}} .st-bad b{{background:var(--bad)}} .st-info b{{background:var(--info)}}
.map{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}}
.map svg{{width:100%;height:auto;display:block}}
.ortho{{display:grid;grid-template-columns:1fr 1fr;gap:8px}} .ortho img{{width:100%;height:auto;display:block;border-radius:6px}}
@media (max-width:620px){{.ortho{{grid-template-columns:1fr}}}}
.o-good{{fill:var(--good)}} .o-warn{{fill:var(--warn)}} .o-bad{{fill:var(--bad)}}
.cam circle{{fill:var(--ink);stroke:var(--panel);stroke-width:2}} .cam:hover circle{{r:7}}
.lg{{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin:6px 2px 0}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}}
.sw.o-good{{background:var(--good)}} .sw.o-warn{{background:var(--warn)}} .sw.o-bad{{background:var(--bad)}} .sw.camk{{background:var(--ink)}}
@media (max-width:620px){{th{{width:auto}} .n{{display:none}}}}
</style></head><body><main>
<h1>품질 보고서</h1>
<div class="sum"><div class="grade" aria-label="등급">{html.escape(str(gr.get('grade', '-')))}</div>
<div><div>{html.escape(gr.get('reason', ''))}</div>
<div class="n">좋음 {counts[GOOD]} · 주의 {counts[WARN]} · 나쁨 {counts[BAD]} · 처리 {rep.get('elapsed_min', '?')}분</div></div></div>
{''.join(body)}
{f'<h2>겹침 지도 (위에서 본 모습)</h2><div class="map">{svg}{legend}</div>' if svg else ''}
{'<h2>정사영상 · 높이 지도</h2><div class="map ortho"><img src="site_ortho.jpg" alt="정사영상"><img src="site_dsm_color.png" alt="DSM 높이 지도"></div>' if (out / "site_ortho.jpg").exists() else ''}
</main></body></html>"""
    p = out / "quality_report.html"
    p.write_text(page, encoding="utf-8")
    log(f"  품질 보고서: {p.name} (좋음 {counts[GOOD]}·주의 {counts[WARN]}·나쁨 {counts[BAD]})")
    return p
