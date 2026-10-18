"""움직이는 것(사람·차량) 자동 가림 — 카메라 계산(SfM)에서 그 위의 특징점을 뺀다.

현장 사진에는 작업자·트럭·승용차가 찍히고, 사진마다 위치가 달라 잘못된 짝(오매칭)을 만든다.
torchvision Mask R-CNN(COCO)으로 사람·자전거·차·오토바이·버스·트럭을 찾아 COLMAP 마스크(<사진이름>.png,
검정=제외)를 쓴다. 가린 곳은 카메라 계산에서만 빠지고 사진 자체·조밀복원·텍스처는 그대로다.
(업그레이드 문서의 'SAM/LLaVA 동적 객체 마스킹' 요구 — 같은 목적을 설치 없이 되는 모델로)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

COCO_MOVING = {1: "사람", 2: "자전거", 3: "차", 4: "오토바이", 6: "버스", 8: "트럭"}


def make(img_dir: Path, out_dir: Path, score: float = 0.5, side: int = 800, log=print) -> Path | None:
    """가린 사진이 하나라도 있으면 마스크 폴더, 아니면 None. 실패해도 None(마스크 없이 진행)"""
    try:
        import cv2
        import torch
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    except Exception as e:
        log(f"  사람·차량 가림 생략(도구 없음: {e})")
        return None
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not imgs:
        return None
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(dev).eval()
        out_dir.mkdir(parents=True, exist_ok=True)
        counts, n_masked, frac, masks = {}, 0, [], []
        for p in imgs:
            im = cv2.imread(str(p))
            if im is None:
                continue
            H, W = im.shape[:2]
            s = min(1.0, side / max(H, W))
            small = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else im
            x = torch.from_numpy(small[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).to(dev)
            with torch.no_grad():
                r = model([x])[0]
            keep = (r["scores"] >= score) & torch.isin(r["labels"], torch.tensor(list(COCO_MOVING), device=r["labels"].device))
            m = np.zeros(small.shape[:2], bool)
            for lab, sc, mk in zip(r["labels"][keep].tolist(), r["scores"][keep].tolist(), r["masks"][keep]):
                one = mk[0].cpu().numpy() > 0.5
                if one.mean() > 0.2:  # 화면의 20% 넘게 차지 = 가까이 찍은 주인공(동상·사람 스캔)일 가능성 → 가리지 않음
                    continue
                if lab in (6, 8) and (sc < 0.8 or one.mean() > 0.03):
                    # 버스·트럭: 위에서 본 컨테이너 현장사무소·자재 더미를 버스/트럭으로 잘못 잡는다(pix4d 공사현장) → 엄격하게
                    continue
                m |= one
                counts[COCO_MOVING[lab]] = counts.get(COCO_MOVING[lab], 0) + 1
            if m.any():  # 경계 특징점(그림자·흐림)도 빼도록 화면 크기의 1.5% 만큼 넓힌다
                k = max(3, int(0.015 * max(m.shape)) | 1)
                m = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
                n_masked += 1
            frac.append(float(m.mean()))
            masks.append((p, m, W, H))
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        # 대부분의 사진에 같은 것이 크게 찍혀 있으면 지나가는 사람·차가 아니라 찍으려는 대상(동상·서 있는 장비)이다
        # (1Family 동상 8장: '사람' 58개·면적 14% → 가리면 주인공 특징점이 사라진다)
        if n_masked >= 0.6 * len(imgs) and float(np.mean(frac)) > 0.05:
            log(f"  사람·차량 가림 안 함: 사진 {n_masked}/{len(imgs)}장에 크게(평균 {100 * float(np.mean(frac)):.0f}%) 찍혀 "
                "찍으려는 대상(동상·고정 장비)으로 판단")
            return None
        for p, m, W, H in masks:
            full = cv2.resize((~m).astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out_dir / f"{p.name}.png"), full)  # COLMAP 규칙: <사진 이름>.png, 흰색=사용·검정=제외
        if not n_masked:
            log(f"  사람·차량 가림: 사진 {len(imgs)}장에서 찾지 못함(가림 없이 진행)")
            return None
        log(f"  사람·차량 가림: 사진 {n_masked}/{len(imgs)}장 · " + ", ".join(f"{k} {v}" for k, v in counts.items())
            + f" · 가린 면적 평균 {100 * float(np.mean(frac)):.1f}% (카메라 계산에서만 제외)")
        return out_dir
    except Exception as e:
        log(f"  사람·차량 가림 생략({e})")
        return None
