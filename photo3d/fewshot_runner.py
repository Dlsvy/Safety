"""MapAnything 실행기 — third_party/mapanything/venv 의 파이썬으로 `python -I fewshot_runner.py <목록.json> <결과.json>` 실행.

입력 목록: 이미지 경로 리스트. 결과: 사진마다 원본 해상도 기준 내부 행렬 K, 카메라→세계 4x4 (미터 단위 추정).
MapAnything 은 사진을 518px 근처로 줄이고(덮도록 확대·축소) 가운데를 잘라 쓰므로 그 변환을 되돌려 원본 K 로 바꾼다.
"""
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image


def main(src: str, dst: str) -> None:
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    paths = json.load(open(src, encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained("facebook/map-anything-apache").to(dev).eval()  # Apache-2.0 가중치
    views = load_images(paths)
    with torch.no_grad():
        outs = model.infer(views, memory_efficient_inference=True, minibatch_size=1, use_amp=True,
                           amp_dtype="bf16", apply_mask=True, mask_edges=True)
    res = []
    for p, v, o in zip(paths, views, outs):
        th, tw = v["img"].shape[-2:]
        W, H = Image.open(p).size
        s = max(tw / W, th / H) + 1e-8
        left, top = (np.floor(W * s) - tw) * 0.5, (np.floor(H * s) - th) * 0.5
        K = o["intrinsics"][0].float().cpu().numpy().astype(float)
        Ko = np.array([[K[0, 0] / s, 0, (K[0, 2] + left) / s], [0, K[1, 1] / s, (K[1, 2] + top) / s], [0, 0, 1]])
        c2w = o["camera_poses"][0].float().cpu().numpy().astype(float)
        conf = o.get("conf")
        res.append({"path": p, "name": os.path.basename(p), "W": W, "H": H, "K": Ko.tolist(), "c2w": c2w.tolist(),
                    "conf": float(conf.float().mean()) if conf is not None else None})
    json.dump(res, open(dst, "w", encoding="utf-8"))
    print(f"MapAnything: {len(res)} views")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
