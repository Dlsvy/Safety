"""메쉬 설정 비교(ETH3D 정답): 같은 조밀 점군에서 Poisson 깊이·trim·흰 벽 채우기 유무를 바꿔 정확도·완전도 비교.
python tools/mesh_sweep.py SCENE_DIR OUT_DIR [--out sweep.json]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pycolmap, trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from photo3d import fill, priorfill
from photo3d.mvs import prune_mesh
from eval_eth3d import evaluate

ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("out"); ap.add_argument("--json", default="sweep.json")
ap.add_argument("--depths", type=int, nargs="+", default=[10, 11]); ap.add_argument("--trims", type=float, nargs="+", default=[5, 7, 9])
a = ap.parse_args()
scene, out = Path(a.scene), Path(a.out); W = out / "work"
xyz, nrm, rgb = fill.read_points(W / "mvs" / "scene_dense.ply")
cache = W / "prior_cache.npz"
if cache.exists():
    z = np.load(cache); pr = {k: z[k] for k in z.files}
else:
    pr = priorfill.fill(xyz, rgb, W / "dense" / "sparse", W / "dense" / "images", model="Ruicheng/moge-2-vitb-normal")
    np.savez(cache, xyz=pr["xyz"], rgb=pr["rgb"], normal=pr["normal"])
sets = {"측정만": (xyz, nrm, rgb), "흰벽채움": (np.vstack([xyz, pr["xyz"]]), np.vstack([nrm, pr["normal"]]), np.vstack([rgb, pr["rgb"]]))}
res = json.loads(Path(a.json).read_text()) if Path(a.json).exists() else {}
tags = []
for sname, (P, N, C) in sets.items():
    ply = W / f"sw_{'m' if sname == '측정만' else 'p'}.ply"; fill._write_ply(ply, P, N, C)
    for d in a.depths:
        for t in a.trims:
            tag = f"sw_{'m' if sname == '측정만' else 'p'}_d{d}_t{t:g}"
            tags.append((f"{scene.name}|{sname}|d{d}|t{t:g}", tag))
            if (out / f"{tag}.obj").exists(): continue
            o = pycolmap.PoissonMeshingOptions(); o.depth = d; o.trim = t
            pycolmap.poisson_meshing(ply, W / f"{tag}.ply", o)
            pm = prune_mesh(W / f"{tag}.ply", ply, W / f"{tag}_p.ply", log=lambda *x: None)
            trimesh.load(pm, force="mesh", process=False).export(out / f"{tag}.obj")
ref = "sw_p_d10_t7.obj"  # 완전도 분모(정답 범위)를 모든 설정에 같게
for key, tag in tags:
    if key in res: continue
    r = evaluate(scene, out, f"{tag}.obj", near_ref=ref)
    res[key] = {k: r[k] for k in ("2cm", "5cm", "10cm")}
    res[key]["faces"] = len(trimesh.load(out / f"{tag}.obj", force="mesh", process=False).faces)
    print(key, {k: res[key][k]["F1"] for k in ("2cm", "5cm", "10cm")}, flush=True)
    Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1))
