"""같은 조밀 점군으로 '깊이 prior 채우기' 전/후 메쉬를 만들어 ETH3D 정답과 비교 (A/B)."""
import json, sys
from pathlib import Path
import numpy as np, pycolmap, trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from photo3d import fill, priorfill
from photo3d.mvs import prune_mesh
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_eth3d import evaluate

scene, out = Path(sys.argv[1]), Path(sys.argv[2])
model = sys.argv[3] if len(sys.argv) > 3 else "Ruicheng/moge-2-vitb-normal"
W = out / "work"
xyz, nrm, rgb = fill.read_points(W / "mvs" / "scene_dense.ply")
pr = priorfill.fill(xyz, rgb, W / "dense" / "sparse", W / "dense" / "images", model=model)
res = {}
for name, P, N, C in [("base", xyz, nrm, rgb),
                      ("prior", np.vstack([xyz, pr["xyz"]]), np.vstack([nrm, pr["normal"]]), np.vstack([rgb, pr["rgb"]]))]:
    ply = W / f"ab_{name}.ply"; fill._write_ply(ply, P, N, C)
    raw = W / f"ab_{name}_mesh.ply"
    o = pycolmap.PoissonMeshingOptions(); o.depth = 10; o.trim = 7
    pycolmap.poisson_meshing(ply, raw, o)
    pm = prune_mesh(raw, ply, W / f"ab_{name}_pruned.ply")
    m = trimesh.load(pm, force="mesh", process=False); m.export(out / f"ab_{name}_zup.obj")
    res[name] = evaluate(scene, out, f"ab_{name}_zup.obj")
    res[name]["points"] = int(len(P))
print(json.dumps(res, ensure_ascii=False, indent=1))
