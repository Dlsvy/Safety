"""같은 내용의 파일을 하드 링크로 한 벌만 남긴다(기준 venv 를 다른 venv 들이 가리킴). --apply 없으면 계산만."""
# 사용: python scripts/gpu/dedupe_venvs.py third_party/pgsr/venv third_party/meshsplat/venv,third_party/mapanything/venv --apply
# 도구를 다시 설치(pip)하면 바뀐 파일만 링크가 풀리고 나머지는 그대로라 안전. 설치 뒤 다시 돌리면 또 줄어든다.
import hashlib, os, sys
from pathlib import Path
base, others, apply = Path(sys.argv[1]), [Path(p) for p in sys.argv[2].split(",")], "--apply" in sys.argv

def h(p):
    d = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            d.update(b)
    return d.hexdigest()

base_idx = {}
for p in base.rglob("*"):
    if p.is_file() and not p.is_symlink():
        st = p.stat()
        if st.st_size >= 65536:
            base_idx[str(p.relative_to(base)).lower()] = (p, st.st_size)
saved, n, cache = 0, 0, {}
for o in others:
    for q in o.rglob("*"):
        if not q.is_file() or q.is_symlink():
            continue
        rel = str(q.relative_to(o)).lower()
        b = base_idx.get(rel)
        if not b or q.stat().st_size != b[1]:
            continue
        if os.path.samefile(q, b[0]):
            continue  # 이미 링크
        hb = cache.get(rel) or cache.setdefault(rel, h(b[0]))
        if h(q) != hb:
            continue
        saved += b[1]; n += 1
        if apply:
            tmp = q.with_name(q.name + ".lnk_tmp")
            os.link(b[0], tmp)
            os.replace(tmp, q)
print(f"{'링크로 바꿈' if apply else '바꿀 수 있음'}: 파일 {n}개 · {saved / 1e9:.1f} GB")
