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

## 포토그래메트리 탭 (2026-10-16, `--pipeline pg`)
- 화면 맨 위 탭 '가우시안 스플래팅 | 포토그래메트리'. 포토그래메트리는 스플랫 없이 측정용 3D 전용.
- 처리: 강건 SfM(DEGENSAC) → 드론·100장↑+GPS 는 GPS 가까운 사진끼리만 매칭(`sfm.gps_pairs`) → 조밀복원 →
  메쉬 **엄격 정리**(측정점에서 점 간격 3배 넘는 면 제거) → 평면·모서리 정리 → 측정이 빈 평평한 바닥만 사진 정사영상 바닥 조각
  (항공·경사지는 생략) → 내보내기 + `survey.py`: `site_ortho.jpg/.png(+.jgw/.pgw)`, `site_dsm.tif(+.tfw)`, `site_dsm_color.png`,
  `site_points.las`(LAS 1.2), `site_ortho.json`. 지붕·천장 아래 현장은 카메라 높이 위를 잘라 평면도처럼.
- 검증(클라우드 CPU, OpenMVS draft — PC 의 GPU PatchMatch 보다 낮은 해상도):
  - ETH3D delivery_area(야외 산업현장, 레이저 정답): 엄격 정리로 5cm F1 58→71%, 중앙 오차 62→26mm. 등록 44/44, 0.38px.
  - ETH3D office: 5cm 안 비율 75.7→90.8%, 중앙 오차 19.9→9.4mm, F1 유지(64.9→66.1).
  - ODM Aukerman(드론 77장, GPS): 등록 77/77, 0.83px, GPS 축척 맞춤 오차 1.44m(일반 GPS), 정사영상 8cm/px.
- 2026-10-16 수정: 충돌 메쉬(site_collider)가 텍스처 이음새마다 찢어져 있던 문제(OpenMVS 경로) 수정 — 그 전 결과는 다시 돌릴 것.
  면 150만↑이면 `site_model_light.glb`(60만 면, 사진에서 다시 구운 텍스처)를 함께 만들고 웹·유니티 보기에 사용.
- PC 에서 확인할 것: COLMAP 4.2.1 `--TwoViewGeometry.use_degensac 1`·`matches_importer` 실제 동작, pg 모드 처리 시간.

## 2026-10-17 바뀐 것 (클라우드 검증 범위 표시)
- **통합 탭** (`--pipeline fuse`): 포토그래메트리(엄격 메쉬·측량 결과물) + 그 메쉬로 사진마다 그린 표면 방향 지도로
  스플랫 학습(`fuse.write_prior_maps`, 깊이 지도는 `PHOTO3D_FUSE_DEPTH=1` 때만) + 측정 표면 뒤 스플랫 제거·일치도 보고.
  가상 잡티 92.5% 제거·실제 표면 100% 유지(클라우드). **실제 GPU 학습 화질 비교는 PC 에서 필요.**
- 뷰어: 더블클릭=회전 중심, 클릭 후 F=확대, R=처음, 나란히 비교(같이 회전), 겹쳐 보기(메쉬를 허용오차만큼 뒤로).
  스플랫 SH 2차 표시(400만↑ 1차), `--splat-aa`(Mip 안티앨리어싱 학습+뷰어) 선택.
- 최고 품질 스플랫: MoGe 깊이 지도 학습 끔, 8GB 는 300만·5만 스텝(Spirula high). **PC 에서 화질 확인 필요.**
- **건물 정형화** `regularize.py`(기존 planarize 대체, OpenMVS 경로 포함): 경계 보존 법선 필터 → 평면 조각 → 수직·수평·주축
  → 교선·교점으로 모서리 날카롭게, 평행 단차(창대) 유지. 가상 건물 모서리 1.9→0.0mm, ETH3D delivery 정확도 유지.
- **GPS 높이 고장 대응**(Caliterra: GPS 고도 317→431m 로 모델 42° 기울어짐 = 사용자 결과 DSM -106~-1m 증상):
  GPS 맞춤이 사진 배치의 '위'와 5° 넘게 어긋나면 수평 GPS 만 사용. 축척 오차는 표준오차로 표시.
- **참 정사영상**: 위에서 찍은 사진이면 원본 사진에서 직접 색(사진 GSD). 비스듬한 사진만 있으면 메쉬 텍스처 방식.
- 마커 오검출 제외(재투영 4px·정사각형 검사), 충돌 메쉬 이음새 찢어짐 수정(**그 전 결과는 다시 처리**),
  OpenMVS TextureMesh 실패 시 파이썬 굽기로 대체, 가벼운 보기용 GLB(150만 면↑).
- 화면: 오른쪽 사진 보기 창(휠), 사용량 1초 그래프(작업 카드에도), 실시간 3D 1초 확인, 현장 예시 7개(하나씩 받기).
- 해 본 뒤 되돌린 것: 점 밀도 적응 가지치기(ETH3D 5cm F1 71.5→61.0).
- 해 봤지만 효과 없음(기본값 유지): OpenMVS ReconstructMesh --smooth 0·-d 1.5(office 2cm F1 56.1→53.0),
  Poisson point_weight 4(55.6→55.5).
- 최적 합성: site_best.ply(스플랫 + 메쉬 받침, 반투명 막음, 뷰어 기본), 통합 탭은 spirula mesh 로 스플랫 메쉬를 만들어
  포토그래메트리가 없는 곳만 채운 site_best_mesh.glb(→ 정형화·사진 텍스처). **spirula mesh 는 CUDA 전용 — PC 에서 확인.**
- 유니티: site_unity.unitypackage(OBJ+스크립트, 가져오면 Site_<이름>.prefab 자동) + 뷰어 '유니티 패키지'.
  **유니티 에디터에서 실제 가져오기는 아직 확인 못 함.**
- 가시(삐죽 솟은 선): COLMAP 경로 점군에 빈 공간 검사(visfilter.py) — CPU 에서는 재현 안 됨, PC 결과로 확인 필요.
- Caliterra(드론 77장) 재처리: 수평 맞음(DSM 높이폭 7.5m, 예전 105m), 정사영상 2.8cm/px, 축척 표준오차 4.09%.

## PGSR (2026-10-17, PC 에서 확인 필요)
- 설치: `scripts/gpu/setup_pgsr_windows.ps1` → third_party/pgsr. 통합 탭(--pipeline fuse)에서 스플랫 학습이 끝난 뒤 순서대로 실행
  (빠름 7000 / 최고 30000 스텝) → TSDF 메쉬를 포토그래메트리 메쉬와 합침(fuse.merge_meshes). report.json fuse.shape_mesh 로 어느 쪽이 쓰였는지 확인.
- 클라우드 확인: 패치(pytorch3d 제거·깊이 단위) 문법·회전 함수 수치, 장면 값(건물 이웃거리 2.5m/복셀 8mm, Caliterra 24.9m/8.5cm).
  실제 학습·메쉬는 GPU 가 없어 미검증 → 실패하면 work/pgsr/pgsr.log 확인.

## MeshSplatting (2026-10-17, PC 에서 확인 필요)
- 설치: `scripts/gpu/setup_meshsplat_windows.ps1` → third_party/meshsplat. 통합 탭에서 PGSR 다음에 실행 → 범위 밖(하늘) 면 제거 →
  사진 텍스처 → site_opaque_mesh.glb(뷰어 '불투명 모델' 버튼, 유니티 패키지 보기 모델 1순위; 충돌 메쉬는 측정 메쉬 그대로).
- 클라우드 확인: 가짜 학습기 + 저장소의 create_ply.py 로 전 과정(내보내기·자르기·텍스처·GLB). 실제 학습은 미검증 → work/meshsplat/meshsplat.log.
- 2026-10-17 사용자 PC(RTX 5060 Ti, CUDA 12.8, VS2022 17.14, torch 2.7.1) PGSR·MeshSplatting 둘 다 설치 성공(torch 2.11 은 C2872 로 실패 → 2.7.1 고정). 윈도우 빌드 우회 2가지:
  torch_win_fix.py(torch 헤더 std 모호성 C2872), meshsplat_win_fix.py(or·uint·sqrt(int) 리눅스 전용 문법).
- 저장소 compile.bat 은 옛 폴더명이라 안 됨 → 설치 스크립트가 세 확장을 직접 빌드. effrdel 은 src 경로 .pth 추가.

## 지켜야 할 것
- 사용자는 비상업 용도 → 연구용 라이선스(PGSR 등)도 사용 가능(docs/GPU_PC.md). 상업 용도로 바뀌면 빼야 함.
- 새 기법은 실사 벤치마크(eval_strecha.py)로 숫자 확인 후 채택(클라우드에서 sift_hq·refine_pp 는 효과 없어 옵션으로만 둠).
- 무거운 작업(특징점·조밀복원·스플래팅)은 동시에 돌리지 말 것(클라우드에서 메모리 부족으로 두 번 중단됨).
