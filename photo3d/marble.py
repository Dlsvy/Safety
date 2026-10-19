"""World Labs Marble API — 찍은 사진으로 'AI 세계'(가우시안 스플랫·충돌 메쉬·파노라마)를 클라우드에서 만든다.

World Labs 의 다음 세대 모델 ATLAS 는 아직 일부 파트너만 쓰는 얼리 액세스(API·가중치 없음, 2026-10). 지금 쓸 수 있는 것은
같은 회사의 Marble(ATLAS 가 앞으로 이 제품을 구동할 예정) API 다. 사진 최대 8장(복원 모드) + 방향 각도(azimuth)를 받아
약 5분 뒤 .spz 스플랫·충돌 메쉬(GLB)·파노라마·실측 축척값(metric_scale_factor)을 준다.
주의: 생성형이라 사진에 없는 곳을 '그럴듯하게 만든다' → 측정·치수용이 아니라 보기·설명·빈 곳 미리보기용(앱에 표시).
사진이 PC 밖(World Labs 서버)으로 나간다. 유료(크레딧). API 키: platform.worldlabs.ai → app_data/worldlabs_key.txt

python -m photo3d.marble <작업 out 폴더> [--images N] [--model marble-1.1]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

API = "https://api.worldlabs.ai/marble/v1"


def api_key(data: Path | None = None) -> str | None:
    k = os.environ.get("WORLDLABS_API_KEY")
    if k:
        return k.strip()
    for d in ([data] if data else []) + [Path(__file__).resolve().parents[1] / "app_data"]:
        f = Path(d) / "worldlabs_key.txt"
        if f.exists() and f.read_text(encoding="utf-8").strip():
            return f.read_text(encoding="utf-8").strip()
    return None


def _call(method: str, path: str, key: str, body: dict | None = None, timeout: float = 60) -> dict:
    req = urllib.request.Request(API + path if path.startswith("/") else path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "WLT-Api-Key": key})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return json.load(r)


def _upload(path: Path, key: str) -> str:
    ext = path.suffix.lower().lstrip(".") or "jpg"
    r = _call("POST", "/media-assets:prepare_upload", key, {"file_name": path.name, "kind": "image", "extension": ext})
    info = r["upload_info"]
    hdr = dict(info.get("required_headers") or {})
    hdr.setdefault("Content-Type", "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}")
    req = urllib.request.Request(info["upload_url"], data=path.read_bytes(), method=info.get("upload_method", "PUT"),
                                 headers=hdr)
    urllib.request.urlopen(req, timeout=300).close()  # noqa: S310
    return r["media_asset"]["id"]


def pick_views(out: Path, n: int = 8) -> list[tuple[Path, float]]:
    """작업 사진 중 방향이 고르게 퍼진 n 장과 방향 각도(첫 장 기준, 도). cameras.json(gltf Y-up)의 fwd 로 계산"""
    work_imgs = out / "work" / "images"
    cams = json.loads((out / "cameras.json").read_text(encoding="utf-8")).get("cameras", []) \
        if (out / "cameras.json").exists() else []
    rows = []
    for c in cams:
        f = c.get("fwd")
        p = work_imgs / c["name"]
        if f and p.exists():
            rows.append((p, math.degrees(math.atan2(f[0], -f[2]))))
    if not rows:  # 카메라가 없으면 이름 순으로 고르게, 방향은 모름
        imgs = sorted(q for q in work_imgs.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
        step = max(1, len(imgs) // n)
        return [(q, None) for q in imgs[::step][:n]]
    a0 = rows[0][1]
    rows = [(p, (a - a0) % 360) for p, a in rows]
    # 360° 를 n 칸으로 나눠 칸마다 한 장(칸 가운데에 가까운 것) — 빈 칸은 건너뜀
    out_rows = []
    for i in range(n):
        lo, hi = 360 * i / n, 360 * (i + 1) / n
        cand = [r for r in rows if lo <= r[1] < hi]
        if cand:
            mid = (lo + hi) / 2
            out_rows.append(min(cand, key=lambda r: abs(r[1] - mid)))
    if len(out_rows) < min(n, len(rows)):  # 한쪽만 찍은 경우: 남는 자리는 순서대로 채움
        rest = [r for r in rows if r not in out_rows]
        out_rows += rest[:: max(1, len(rest) // (n - len(out_rows)))][: n - len(out_rows)]
    return out_rows[:n]


def generate(out: Path, n_images: int = 8, model: str = "marble-1.1", name: str = "photo3d", log=print) -> dict:
    """사진으로 Marble 세계를 만들고 결과를 out/ai_world.* 로 받는다. 상태는 out/marble.json"""
    out = Path(out)
    key = api_key()
    if not key:
        raise RuntimeError("World Labs API 키가 없습니다 — PC 화면 'AI 세계' 설정에 넣으세요(platform.worldlabs.ai)")
    status = out / "marble.json"

    def save(**kw):
        cur = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
        cur.update(kw, t=time.time())
        status.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")

    views = pick_views(out, max(1, min(8, n_images)))
    save(state="uploading", msg=f"사진 {len(views)}장 올리는 중", images=[p.name for p, _ in views])
    log(f"  Marble: 사진 {len(views)}장 올리는 중")
    items = []
    for p, az in views:
        it = {"content": {"source": "media_asset", "media_asset_id": _upload(p, key)}}
        if az is not None:
            it["azimuth"] = round(az, 1)
        items.append(it)
    body = {"display_name": name[:64], "model": model,
            "world_prompt": {"type": "multi-image", "multi_image_prompt": items,
                             "reconstruct_images": len(items) > 4}}
    op = _call("POST", "/worlds:generate", key, body)
    op_id = op.get("operation_id") or op.get("id")
    save(state="generating", msg="World Labs 에서 세계를 만드는 중(약 5분)", operation_id=op_id)
    log(f"  Marble: 생성 중(operation {op_id})")
    t0 = time.time()
    while True:
        time.sleep(10)
        r = _call("GET", f"/operations/{op_id}", key)
        if r.get("error"):
            save(state="error", msg=f"생성 실패: {r['error']}")
            raise RuntimeError(f"Marble 생성 실패: {r['error']}")
        if r.get("done"):
            break
        wid = (r.get("metadata") or {}).get("world_id")
        save(state="generating", msg=f"World Labs 에서 세계를 만드는 중 — {int(time.time() - t0)}초", world_id=wid)
        if time.time() - t0 > 30 * 60:
            save(state="error", msg="30분 넘게 끝나지 않음")
            raise RuntimeError("Marble 생성 시간 초과")
    wid = (r.get("response") or {}).get("id") or (r.get("metadata") or {}).get("world_id")
    world = _call("GET", f"/worlds/{wid}", key)["world"]
    a = world.get("assets") or {}
    spz = (a.get("splats") or {}).get("spz_urls") or {}
    got = {}
    for k, url, dst in (("splat", spz.get("full_res") or spz.get("500k"), "ai_world.spz"),
                        ("splat_light", spz.get("500k") or spz.get("100k"), "ai_world_500k.spz"),
                        ("collider", (a.get("mesh") or {}).get("collider_mesh_url"), "ai_world_collider.glb"),
                        ("pano", (a.get("imagery") or {}).get("pano_url"), "ai_world_pano.jpg")):
        if url:
            urllib.request.urlretrieve(url, out / dst)  # noqa: S310
            got[k] = dst
    sem = (a.get("splats") or {}).get("semantics_metadata") or {}
    save(state="done", msg="AI 세계 완성(생성형 — 보기용, 측정은 측정 메쉬로)", world_id=wid, files=got,
         metric_scale_factor=sem.get("metric_scale_factor"), ground_plane_offset=sem.get("ground_plane_offset"),
         marble_url=world.get("world_marble_url"), minutes=round((time.time() - t0) / 60, 1))
    log(f"  Marble: 완료 — {', '.join(got.values())}")
    return got


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--model", default="marble-1.1")
    ap.add_argument("--name", default="photo3d")
    a = ap.parse_args()
    try:
        generate(Path(a.out), a.images, a.model, a.name)
    except Exception as e:
        print(f"Marble 실패: {e}", file=sys.stderr)
        sys.exit(1)
