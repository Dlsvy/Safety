# GPU PC 확장 (NVIDIA, Windows) — 최신 오픈소스 연동

> 이 문서의 스크립트·명령은 GPU 가 없는 클라우드에서 만들었고 **아직 실기 검증 전**이다.
> 상업 이용 가능한 라이선스만 사용한다(2026-10 기준 각 저장소 LICENSE 확인).

## 원칙: 정확도 기준축은 바꾸지 않는다
2025~26 독립 벤치마크(arXiv 2603.17358, 2507.14798, 2607.26646)에서 COLMAP/GLOMAP 이 VGGT·π³·Depth Anything 3
보다 카메라 자세가 정확했다(VGGT 는 2~3 cm 오차 하한). 그래서
**COLMAP + 마커/GCP 번들조정 = 치수·충돌의 기준**, 최신 학습형 모델 = 보조(구멍 메우기, 실패 시 대체, 보기 품질).

## 1. 가우시안 스플래팅 보기 레이어 (P1) — gsplat (Apache-2.0)
```powershell
.\scripts\gpu\setup_splat_windows.ps1
.\.venv-splat\Scripts\python.exe scripts\gpu\train_splat.py D:\현장\out          # 3DGS
.\.venv-splat\Scripts\python.exe scripts\gpu\train_splat.py D:\현장\out --mode 2dgs  # 표면이 더 또렷
```
- 입력은 photo3d 의 `work/dense` (마커로 축척을 맞춘 카메라) → 스플랫이 메쉬와 **같은 좌표(m, Z-up)**.
- `--no-normalize-world-space` 로 실측 좌표를 유지한다(기본값은 정규화되어 축척이 깨짐).
- 대안: **Brush** (Apache-2.0, 아무 GPU, COLMAP 입력), **LichtFeld Studio** (GPLv3, CUDA).
- 클라우드 CPU 에서는 **OpenSplat**(AGPL-3.0)으로 시험 가능(느림): `scripts/splat_cpu.sh`.

**유니티 표시**: [aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) (MIT, 2023-12 이후 유지보수 중단, Unity 2022.3 기준)
→ Tools > Gaussian Splats > Create GaussianSplatAsset 으로 `site_splat.ply` 반입 → 오브젝트 X축 -90° 회전.
충돌·이격거리는 `site_collider` 메쉬가 담당하고, 스플랫은 렌더링만 한다.
Unity 6 에서 메쉬와 깊이 가림 처리가 필요하면 SplatBridge(유료) 검토.

## 2. 질감 없는 면 구멍 메우기 (P1) — Depth Anything 3 (Apache 모델만)
- 저장소: https://github.com/ByteDance-Seed/Depth-Anything-3 (코드 Apache-2.0)
- **상업 가능 가중치: DA3-Base, DA3-Small, DA3METRIC-Large, DA3MONO-Large** (Large/Giant/Nested 는 CC-BY-NC → 사용 금지)
- COLMAP 폴더를 입력으로 받아 **우리 카메라 자세·축척을 유지한 채** 조밀 깊이를 낸다(pose conditioning).
- 사용처: 흰 콘크리트·도장면처럼 OpenMVS 가 비우는 곳의 깊이 → Open3D TSDF 로 융합 → 메쉬 보강.
- 검증 방법: `tools/eval_strecha.py` 와 가상현장 `tools/eval_accuracy.py` 로 OpenMVS 단독 대비 개선을 숫자로 확인 후 채택.

## 3. SfM 실패 시 대체 경로 (P1) — MapAnything (Meta)
- https://github.com/facebookresearch/map-anything — 코드 Apache-2.0, **`map-anything-apache` 체크포인트만 상업 가능**
  (`map-anything` 은 CC-BY-NC).
- `scripts/demo_colmap.py` 로 COLMAP 형식 출력 → photo3d 의 축척·GCP 단계(`scale.georeference`)를 그대로 적용.
- 사진이 적거나 겹침이 부족해 COLMAP 이 모델을 못 이을 때만 쓴다. 결과는 반드시 GCP/검사점으로 확인.

## 4. 학습형 특징점 (P1) — ALIKED + LightGlue / LoMa (COLMAP 4.x 내장)
```powershell
.\scripts\run_windows.ps1 photos -MarkerSize 0.25 -Out out   # 기본 SIFT
python -m photo3d run photos --features aliked --gpu --colmap-bin $env:COLMAP_BIN ...
```
반복무늬·저질감 현장에서 등록률 개선 기대. SIFT 대비 효과는 현장 데이터로 비교 후 기본값 결정.

## 쓰지 않는 것 (비상업 라이선스)
MASt3R/MASt3R-SfM, π³ 가중치, CUT3R, StreamVGGT, Fast3R, MVSAnywhere, MILo, 2DGS(Inria), GOF, RaDe-GS, PGSR,
VGGT 기본 가중치(상업용은 신청제 VGGT-1B-Commercial 만).

## 클라우드에서 가중치 쓰기 (필요할 때)
클라우드는 Hugging Face·GitHub 릴리스·LFS 가 막혀 있다. 라이선스가 재배포를 허용하는(Apache/MIT) 가중치에 한해
PC 에서 받아 **개인 비공개 GitHub 저장소**에 일반 파일로 올리고(100 MB 초과 시 분할) 클라우드에서 `git clone`.
