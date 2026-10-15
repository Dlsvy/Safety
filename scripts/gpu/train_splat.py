"""photo3d 결과로 가우시안 스플래팅 학습 (GPU PC, gsplat).

python scripts/gpu/train_splat.py OUT_DIR [--steps 30000] [--mode 3dgs|2dgs]
  OUT_DIR: photo3d run --out 폴더. work/dense(왜곡보정 이미지 + 마커로 축척 맞춘 카메라)를 그대로 쓴다.
결과: OUT_DIR/splat/site_splat.ply  (좌표 = site_model_zup.obj 와 동일, Z-up, m)
유니티: aras-p/UnityGaussianSplatting 으로 PLY 를 에셋화 → 오브젝트를 X축 -90° 회전(Z-up→Y-up).
       충돌·이격거리는 site_collider 메쉬가 담당하고 스플랫은 화면 표시만 한다.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--mode", default="3dgs", choices=["3dgs", "2dgs"])
ap.add_argument("--gsplat-repo", default=str(Path(__file__).resolve().parents[2] / "third_party" / "gsplat"))
a = ap.parse_args()

out = Path(a.out)
dense = out / "work" / "dense"
if not (dense / "images").exists():
    sys.exit(f"{dense} 가 없습니다 — 먼저 photo3d run 을 끝까지 실행하세요")
# gsplat 예제는 <data>/images, <data>/sparse/0 구조를 기대한다
data = out / "splat" / "data"
(data / "sparse" / "0").mkdir(parents=True, exist_ok=True)
for f in (dense / "sparse").glob("*.bin"):
    shutil.copy2(f, data / "sparse" / "0" / f.name)
if not (data / "images").exists():
    shutil.copytree(dense / "images", data / "images")
res = out / "splat" / "result"
trainer = Path(a.gsplat_repo) / "examples" / ("simple_trainer.py" if a.mode == "3dgs" else "simple_trainer_2dgs.py")
cmd = [sys.executable, str(trainer)]
if a.mode == "3dgs":
    cmd += ["default"]
cmd += ["--data_dir", str(data), "--data_factor", "1", "--result_dir", str(res),
        "--max_steps", str(a.steps), "--disable_viewer"]
if a.mode == "3dgs":
    cmd += ["--save-ply"]
# 주의: gsplat 예제의 normalize_world_space 가 켜져 있으면 결과가 정규화 좌표다. 축척 유지를 위해 끈다.
cmd += ["--no-normalize-world-space"]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
plys = sorted(res.rglob("*.ply"))
if plys:
    shutil.copy2(plys[-1], out / "splat" / "site_splat.ply")
    print("저장:", out / "splat" / "site_splat.ply")
else:
    print("PLY 를 찾지 못했습니다 — gsplat 버전의 저장 옵션을 확인하세요:", res)
