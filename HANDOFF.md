# HANDOFF — 사용자 PC(Windows + NVIDIA GPU)에서 이어서 할 일

이 문서는 클라우드 세션(GPU 없음)에서 만든 photo3d 를 **사용자 PC 의 Claude Code 세션**이 이어받기 위한 것이다.
README.md 에 전체 설명·검증 결과가 있다. 먼저 README 를 읽어라.

## 현재 상태 (2026-10-15, 클라우드에서 검증된 것)
- 파이프라인: 사진/동영상 → COLMAP(SfM, 강건 모드) → ArUco/줄자/GCP 축척 → OpenMVS(조밀·메쉬·텍스처) → GLB/OBJ.
- 실사(Strecha) 카메라 오차 4–9 mm(15–32 m 장면), 한쪽 면 4–6장, 한 바퀴 15–20장이면 연결.
- 가상현장 전과정: 건물 18 m 길이 오차 −17~−21 mm, 표면 오차 중앙값 ~10 mm.
- OpenMVS v2.3.0 우회: (1) `libs/Common/Util.h` formatTime 의 `_sntprintf(buf+len, 128,` → `128-len` 패치(리눅스 빌드에서 필요),
  (2) TextureMesh 는 `--export-type ply` 로 저장(OBJ 저장 세그폴트), (3) RefineMesh/TextureMesh 에 `--mesh-file` 로 메쉬 전달.

## 검증 안 된 것 — PC 에서 할 일 (순서대로)
1. **설치**: `scripts/setup_windows.ps1` (COLMAP CUDA 릴리스 + OpenMVS 릴리스 + venv). 아직 실제 Windows 에서 돌려본 적 없음.
   - COLMAP 4.x CLI 옵션 이름은 `FeatureExtraction.use_gpu`, `FeatureExtraction.type` 등(소스에서 확인함).
   - OpenMVS 윈도우 릴리스에 위 (1)(2) 버그가 있는지 확인. 있으면 photo3d/mvs.py 가 이미 PLY 저장으로 우회함.
2. **GPU 실행 확인**: `scripts/run_windows.ps1 <사진폴더> -MarkerSize 0.25 -Out out` → report.json 등급·소요시간 확인.
   - 시험 데이터: README 의 Strecha 데이터는 `git clone https://github.com/openMVG/SfM_quality_evaluation`
     → `python tools/eval_strecha.py <scene> <out> --keep 6` 로 정확도 비교(클라우드 결과: fountain 6장 3.7 mm).
3. **가우시안 스플래팅(GPU)**: `scripts/gpu/setup_splat_windows.ps1` → `scripts/gpu/train_splat.py <out>`
   (gsplat, `--no-normalize-world-space` 로 실측 좌표 유지). 클라우드 CPU(OpenSplat) 결과: 1,420회에 미사용 시점 PSNR 27.8 dB.
4. **적은 장수(3장 이하) 시험**: MapAnything `--apache` 체크포인트(Apache-2.0)로 `scripts/demo_colmap.py` →
   `python -m photo3d run --from-colmap <출력> ...`. 클라우드에선 가중치 다운로드가 막혀 미검증.
5. **유니티**: `unity/` 를 Assets 로 복사, glTFast 설치, 레이어 `Site`·`SiteEstimated` 추가, 메뉴 Photo3D > Import Site Folder.
   스크립트는 컴파일 검증 전.

## GPU 최대 사용 (사용자 요청: "성능이 남아돈다, GPU 를 최대로")
- 사용자 PC: NVIDIA GeForce RTX(전용 VRAM 8 GB), RAM 32 GB, 설치 직후 GPU 사용률 8%.
- `python -m photo3d run <입력> --profile gpu-max --colmap-bin %COLMAP_BIN% ...` (run_windows.ps1 기본값)
  = 원본 해상도(최대 6000px)·특징점 16k·GPU 추출/매칭·GLOMAP+GPU 번들조정·COLMAP PatchMatch(GPU) 조밀복원·ultra.
- 실행 중 `nvidia-smi -l 2` 로 사용률·VRAM 확인. VRAM 부족이면 `--quality high` 또는 PatchMatch max_image_size 를 낮춘다.
- 스플래팅은 `scripts/gpu/train_splat.py` (8 GB 에서 30k 스텝 가능 예상, 부족하면 `--steps 15000`).
- 클라우드에선 GPU 가 없어 이 경로를 한 번도 실행하지 못했다 — 첫 실행에서 오류 나면 고쳐서 커밋할 것.

## 지켜야 할 것
- 상업 이용 가능 라이선스만(비상업: MASt3R, π³ 가중치, MILo, 2DGS, PGSR, VGGT 기본 가중치 등 사용 금지 — docs/GPU_PC.md).
- 새 기법은 실사 벤치마크(eval_strecha.py)로 숫자 확인 후 채택(클라우드에서 sift_hq·refine_pp 는 효과 없어 옵션으로만 둠).
- 무거운 작업(특징점·조밀복원·스플래팅)은 동시에 돌리지 말 것(클라우드에서 메모리 부족으로 두 번 중단됨).
