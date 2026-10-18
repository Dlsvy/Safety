"""유니티 패키지(.unitypackage) — 더블클릭하면 유니티가 바로 가져가는 현장 파일.

GLB 는 유니티가 기본으로 못 읽어 glTFast 설치가 필요했다 → 유니티가 기본 지원하는 OBJ(+MTL·텍스처)로 넣고,
스크립트(Photo3D/Runtime·Editor)까지 함께 넣어 가져오기만 하면 자동으로 현장 프리팹(Site_<이름>.prefab:
보기 모델 + 숨긴 충돌 메쉬 + 추정 영역 + 현장 표시)이 만들어지게 한다(unity/Editor/Photo3DAutoSetup.cs).

좌표: OBJ 는 GLB 와 같은 Y-up(m). 유니티 OBJ 가져오기는 X 를 뒤집는데, annotations.json 을 읽는 스크립트도
(-x, y, z) 로 바꾸므로 모델·충돌·표시가 서로 맞는다.
형식: tar.gz 안에 <guid>/asset, <guid>/asset.meta, <guid>/pathname. guid 는 경로에서 만들어 같은 현장을 다시 넣으면 덮어쓴다.
"""
from __future__ import annotations

import hashlib
import io
import shutil
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _guid(path: str) -> str:
    return hashlib.md5(("photo3d:" + path).encode("utf-8")).hexdigest()


def _meta(guid: str, folder: bool = False) -> bytes:
    if folder:
        return (f"fileFormatVersion: 2\nguid: {guid}\nfolderAsset: yes\nDefaultImporter:\n  externalObjects: {{}}\n"
                "  userData: \n  assetBundleName: \n  assetBundleVariant: \n").encode()
    return f"fileFormatVersion: 2\nguid: {guid}\n".encode()  # 나머지 가져오기 설정은 유니티 기본값


def _glb_to_obj(glb: Path, dst_dir: Path, stem: str) -> list[Path]:
    """GLB(Y-up) → OBJ(+MTL·텍스처). 여러 물체(본 메쉬 + 사진 바닥 조각)는 한 OBJ 의 여러 그룹으로."""
    import trimesh
    sc = trimesh.load(glb, force="scene")
    for g in sc.geometry.values():  # 텍스처 없는 충돌 메쉬도 OBJ 로 잘 나가게
        if getattr(g.visual, "kind", None) is None:
            g.visual = trimesh.visual.ColorVisuals(g)
    files = trimesh.exchange.obj.export_obj(sc, include_texture=True, return_texture=True,
                                            mtl_name=f"{stem}.mtl", include_normals=True)
    text, res = files if isinstance(files, tuple) else (files, {})
    out = []
    p = dst_dir / f"{stem}.obj"
    p.write_text(text, encoding="utf-8")
    out.append(p)
    for name, data in (res or {}).items():
        q = dst_dir / name
        q.write_bytes(data)
        out.append(q)
    return out


def write_unity_package(out: Path, name: str, log=print) -> Path | None:
    """결과 폴더(out)의 모델로 <out>/site_unity.unitypackage 를 만든다. 실패하면 None."""
    out = Path(out)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60] or "site"
    tmp = out / "_unitypkg"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        assets: list[tuple[str, Path]] = []  # (유니티 경로, 파일)
        base = f"Assets/Photo3DSites/{safe}"
        # 보기 모델: MeshSplatting 불투명 메쉬 > 최적 합성 메쉬 > 가벼운/원본 모델. 충돌 메쉬는 항상 측정 메쉬(site_collider)
        # (빠른 모드 불투명 메쉬는 미리보기 표시 파일 site_opaque_mesh.preview 가 있으면 건너뛴다)
        order = ("site_best_mesh.glb", "site_model_light.glb", "site_model.glb")
        if not (out / "site_opaque_mesh.preview").exists():
            order = ("site_opaque_mesh.glb",) + order
        vis = next((out / f for f in order if (out / f).exists()), out / "site_model.glb")
        if not vis.exists() or not (out / "site_collider.glb").exists():
            return None
        for glb, stem in ((vis, "site_model"), (out / "site_collider.glb", "site_collider"),
                          (out / "site_estimated.glb", "site_estimated")):
            if glb.exists():
                for f in _glb_to_obj(glb, tmp, stem):
                    assets.append((f"{base}/{f.name}", f))
        for f in ("site_meta.json", "annotations.json", "cameras.json", "report.json", "quality_report.html"):
            if (out / f).exists():
                assets.append((f"{base}/{f}", out / f))
        for cs in sorted((ROOT / "unity").rglob("*.cs")):
            assets.append(("Assets/Photo3D/" + cs.relative_to(ROOT / "unity").as_posix(), cs))
        folders = sorted({str(Path(a).parent.as_posix()) for a, _ in assets} | {"Assets/Photo3DSites", "Assets/Photo3D"})
        dst = out / "site_unity.unitypackage"
        part = dst.with_suffix(".tmp")
        with tarfile.open(part, "w:gz") as tar:
            def add(name: str, data: bytes):
                ti = tarfile.TarInfo(name)
                ti.size, ti.mtime = len(data), int(time.time())
                tar.addfile(ti, io.BytesIO(data))

            for d in folders:
                if d == "Assets":
                    continue
                g = _guid(d)
                add(f"{g}/pathname", d.encode("utf-8"))
                add(f"{g}/asset.meta", _meta(g, folder=True))
            for rel, f in assets:
                g = _guid(rel)
                add(f"{g}/pathname", rel.encode("utf-8"))
                add(f"{g}/asset.meta", _meta(g))
                add(f"{g}/asset", f.read_bytes())
        part.replace(dst)
        log(f"  유니티 패키지: {dst.name} ({dst.stat().st_size / 1e6:.0f} MB) — 더블클릭하면 유니티가 가져가고 "
            f"Assets/Photo3DSites/{safe}/Site_{safe}.prefab 이 자동으로 만들어짐")
        return dst
    except Exception as e:
        log(f"  유니티 패키지 생략: {e}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
