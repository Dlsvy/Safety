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

## 2026-10-18 로컬 PC 세션에서 고친 것 (이 PC 에서 실제로 돌려 확인)
- **품질이 안 오른 진짜 원인(1Family 결과)**: 측정 메쉬 엄격 정리가 701,145면 → 724면(99.9% 삭제). 배경 점 때문에 점군 범위가
  386m 라 Poisson 칸이 38cm 로 거칠어졌고, 점 간격(6mm)×3 기준에 모두 걸렸다 → 최적 메쉬 99% 가 스플랫 메쉬였다.
  수정: mvs._poisson_input — 카메라 상자 ± max(카메라 폭, 카메라→점 거리 중앙값) ∩ 점군 0.5~99.5% 로 자르고 깊이를 점 간격에 맞춤
  (+1, 최대 12), prune_mesh(min_step=칸/2), 90% 넘게 지우려 하면 가시 정리만. 1Family: 724면 → 3,765,911면, Poisson 46초.
- **PGSR 은 매번 실패했었다**: 저장소는 data/sparse/ 에서 읽는데 sparse/0 에만 복사 → 둘 다 복사(pgsr.prep_data). 1000스텝 54초로 확인.
- **PGSR 빠른 모드(7000)는 평면·다시점 손실이 한 번도 안 켜졌다**(기본 7000스텝부터, 로그 Single=0 Geo=0 Pho=0) →
  pgsr.SCHEDULE 을 스텝 수에 비례해 당김. 사진 8장에서 TSDF 메쉬가 비어 실패한 것도 이 때문으로 보임(재시험 결과 아래).
- **PGSR 메쉬가 늘 비어 있던 진짜 원인: PGSR venv 의 open3d 0.20.0(윈도우) TSDF 가 고장**(평평한 벽 합성 시험도 삼각형 0개,
  0.19.0 은 27,170개). 1Family 1000스텝 메쉬 80면, 8장 56면이었다 → open3d==0.19.0 고정(setup_pgsr_windows.ps1, 이 PC 설치 완료).
  재추출: 8장 정점 354 → 1,346,204. pgsr.run 은 면 5,000 미만이면 실패로 처리(형태 메쉬로 안 씀).
- MeshSplatting 빠른 모드: 10,000스텝 + 학습 일정 1/3(meshsplat.SCHEDULE). 1Family 152장: 6분 50초, 839,195면(손실 0.108).
  불투명 모델 정리가 mvs._fix_topology(나비 정점 5회 깎기)로 2,668면만 남기던 것 → 정점 연결 기준 부스러기(50면 미만)만 제거 → 810,268면.
  단, 빠른 모드 결과는 삼각형이 흩어져 거칠다(눈으로 확인) → site_opaque_mesh.preview 표시를 남겨 유니티 패키지 보기 모델에서 제외
  (화면 '불투명 모델' 버튼으로만). 8장은 학습 자체가 안 됨(28k면).
- 드론+폰·사진+영상처럼 렌즈가 섞이면 single_camera=False(전에는 360 이 아니면 무조건 카메라 하나 → 공중·지상 통합 불가).
  기종 판별도 앞 40장 → 전체에서 고르게.
- 텍스처: 패치 여백 2→4px + 빈 텍셀 push-pull 채움(texture._fill_empty, 8192² 4.5초) → 축소 시 검은 띠 원인 제거.
- 스플랫 정리: 표면 '앞' 허공 스플랫 제거 추가 + 판정 여유에 픽셀 크기(zc/f·s) 반영 + 메쉬 범위 밖은 판정 안 함
  (고정 19mm 만 쓰면 비스듬한 먼 바닥이 '뒤'로 세어져 사진 바닥 스플랫 134만 개가 지워졌음). 최적 메쉬도 측정 범위 밖 면 자름.
- **폰 동영상이 전부 실패한 원인: 이 PC 에 ffmpeg/ffprobe 가 없었다** → frames.ffmpeg_exe(PATH 없으면 imageio-ffmpeg 자동 설치),
  ffprobe 안 씀(길이는 ffmpeg 출력, webm N/A 는 끝까지 읽어서). 안드로이드 webm·아이폰 mp4 둘 다 51장 추출, 폰 경로(터널·조각 업로드)로
  영상만 올려 51/51장 등록 확인.
- **QR**: 폰 앱이 연결 코드를 쿠키에만 두고 주소에서 지워 카톡·네이버 앱 브라우저·아이폰 홈 화면 앱에서 '연결 필요'로 막혔다 →
  localStorage + X-P3D-Token 헤더, 주소에 ?t= 유지, manifest start_url 에 코드(app.py), 앱 안 브라우저는 크롬/사파리로 넘김.
  PC '폰 연결' 창을 열면 Cloudflare 터널 자동 시작 → QR 하나(정식 https, 와이파이·LTE 모두). 켠 상태는 mobile_tunnel_on.txt 로 기억.
- **적은 장수(≤40장, COLMAP 등록 80% 미만이면 자동)**: fewshot.py — MapAnything(apache 가중치, third_party/mapanything,
  scripts/gpu/setup_mapanything_windows.ps1)으로 자세·렌즈 → 원본 해상도 SIFT 로 삼각측량(80→16→4px 단계)·BA.
  1Family 8장: COLMAP 만 2/8장 → 8/8장·2,528점·0.36px, 초점 1160(정답 1166). 전 과정 14분(fast·fuse). 크기는 AI 추정 미터 →
  georef.method=ai_metric_estimate, 등급 C. 첫 실행 때 가중치 내려받기(약 8분).
- 글로벌 매퍼 종료코드 1(사진 적을 때)이 작업 전체를 죽이던 것 → 증분 매퍼로 계속.
- 사람·차량 자동 가림(dynmask.py, torchvision Mask R-CNN): SfM 특징점에서만 제외. 사진 60%↑에 5%↑로 찍히면 대상(동상)으로 보고 안 가림,
  한 물체가 화면 20%↑면 제외, 버스·트럭은 점수 0.8↑·3%↓만(위에서 본 컨테이너 사무소를 버스로 오인). pix4d 공사현장 30장: 차 167·사람 47.
  끄기: PHOTO3D_NO_DYNMASK=1.
- 폰 앱 '공간 스캔'(메타포트 방식): 제자리 한 바퀴 30°마다 12장(나침반, 없으면 화면 이동 70%) → 1.5~2m 이동 → 반복. 파일명 p3d_sNN_….
  실제 폰에서는 아직 안 찍어 봄.
- 사용량 표시 확인: 메모리(psutil)·GPU 메모리(nvidia-smi)·GPU% 모두 같은 순간 실측과 일치(작업 중·쉴 때).
- **아주 빠르게(--profile instant, 앱 mode 'instant', PC·폰 화면)**: 사진이 12장 넘으면 찍은 순서 12구간마다 가장 선명한 1장
  (나머지는 work/images_unused) → 15장 이하는 COLMAP 없이 MapAnything → 그 사진별 깊이 점을 조밀 점군으로(COLMAP 조밀복원 생략)
  → Poisson·텍스처 → 스플랫 30만·3천 스텝·1000px, PGSR·MeshSplatting·깊이 보조 생략. 1Family 152장 6분(빠르게 69분), 2Francis 302장 7.7분.
  AI 깊이 점 정합: 사진마다 '그 사진의 다듬은 카메라' 기준 + 그 사진에 보이는 삼각측량 점으로 깊이 축척(0.5~2배 밖이면 그 사진 제외).
  삼각측량 점까지 거리 중앙: 2Francis 1,376 → 193mm, 1Family 617 → 205mm (전체 Sim3 하나만 쓸 때 대비).
  MapAnything 은 GPU 가 비어 있으면 8장 46초(불러오기 12 + 추론 28), Unity 에디터가 GPU 를 쓰면 3~8배 느려진다. 가중치를 받았으면 HF_HUB_OFFLINE=1.
- 같은 파일(바이트 동일) 사진은 입력에서 한 장만(1Family 폴더: 240장이 8가지 사진 복사본 → 글로벌 매퍼가 4시간 반 멈춤).
- single_camera 는 사진 크기가 모두 같을 때만(크기 다르면 COLMAP 이 CAMERA_SINGLE_DIM_ERROR 로 건너뜀 → 8/160장 등록).
- 주의: APP.bat 창이 두 개 뜨면 같은 app_data 를 같이 써서, 나중에 켜진 앱이 처리 중 작업을 '앱이 다시 시작되어 중단됨'으로 바꾼다.

## ▶ 지금 이어서 할 일 (2026-10-18 사용자 피드백 — 로컬 PC 세션에서 진행)
사용자 결과 캡처(사무 건물 드론, 왼쪽 실사/스플랫, 오른쪽 메쉬): "품질 향상이 크게 안 느껴짐" + "빠른 작업에서도 새 기술(PGSR·MeshSplatting) 돌게".

1) 메쉬의 검은 테두리(창·벽 경계마다 검은 띠) — 원인 거의 확실: 텍스처 아틀라스 여백
   - photo3d/texture.py 의 패치 여백 pad=2px(240행), 아틀라스 빈 곳은 검정 → 뷰어/유니티가 멀리서 볼 때(mipmap) 검정이 번짐.
   - 할 일: 굽기 끝에 빈 픽셀을 이웃 색으로 채우기(push-pull 또는 cv2.inpaint/반복 dilate, 패치 마스크 기준) + pad 4~8px.
   - 확인: site_best_mesh.glb / site_opaque_mesh.glb 를 축소해서 볼 때 검은 띠가 사라지는지.
2) 실사(스플랫) 벽의 검은 얼룩·하늘 조각·전봇대 늘어짐
   - 검은 얼룩: 어두운 반투명 스플랫이 벽 앞에 뜸 → fuse.mesh_prune_and_agree 의 "표면 앞" 허공 스플랫도 지우기(지금은 뒤만),
     또는 메쉬 표면 근처인데 색이 사진과 크게 다른(어두운) 스플랫의 불투명도 낮추기.
   - 먼 건물·하늘 조각: 측정 범위(희소점 백분위) 밖 스플랫/메쉬 자르기(meshsplat._crop 과 같은 방식을 스플랫·best_mesh 에도).
3) 빠른 모드에서도 새 기술
   - 지금: cli.py — PGSR 은 빠른 모드 7000스텝으로 돌지만 MeshSplatting 은 `a.profile == "fast"` 면 생략.
   - MeshSplatting 은 학습 일정이 30000 기준(train.py: sigma_until·final_opacity_iter·start_upsampling·start_vertex_opt·
     densify_until_iter·iteration_mesh·start_pruning·start_opacity_floor·position_lr_max_steps). 이 값들은 arguments 의
     OptimizationParams 라 그대로 `--이름 값` 으로 넘길 수 있음 → 빠른 모드는 iterations=10000 과 위 값들을 ×1/3 로 같이 넘기기
     (간격값 densification_interval·intervall_add_triangles 500 은 그대로). meshsplat.run 에 iterations 인자 추가.
   - 앱(app.py)은 mode 가 fast/best 뿐: best → profile gpu-max, 그 외 → fast. ("기본" 품질 없음)
4) 효과 확인 방법: 같은 사진으로 이전 결과와 나란히(뷰어 '나란히 비교') + report.json 의 fuse.shape_mesh(pgsr/spirula),
   fuse.opaque_mesh 가 있는지, work/pgsr/pgsr.log · work/meshsplat/meshsplat.log 로 실제로 돌았는지 먼저 확인.
   (사용자 캡처가 빠른 모드였다면 MeshSplatting 은 안 돈 상태.)

## 로컬 개발 메모
- 저장소 Dlsvy/Safety, 브랜치 photo3d. PC 경로 C:\photo3d (APP.bat 이 git pull 후 scripts/app_run.bat 실행).
- 외부 도구: third_party/pgsr(torch 2.7.1+cu128, PGSR venv), third_party/meshsplat(같은 torch), Spirula/OpenMVS/COLMAP 은 setup_windows.ps1.
- 윈도우 CUDA 확장 빌드 우회: scripts/gpu/torch_win_fix.py(torch 헤더 C2872), scripts/gpu/meshsplat_win_fix.py(리눅스 전용 문법).
  torch 2.11 은 헤더 패치로도 C2872 → 2.7.1 고정.
- 확인 도구: pyflakes, node --check(index.html/m.html 의 script 추출), tools/eval_eth3d.py(ETH3D 정확도), Playwright(헤드리스 화면).
- 폰 앱 시험: PHOTO3D_PORT/PHOTO3D_HTTPS_PORT/PHOTO3D_DATA 환경변수로 따로 띄워 시험 가능. 원격 요청 흉내는 헤더 `Cf-Ray: x`.

## 폰 앱 (2026-10-17, PC 에서 확인 필요)
- PC 앱 오른쪽 위 '📱 폰 연결' → QR. 폰 화면 /m (photo3d/web/m.html, 홈 화면 추가 가능한 웹앱). 서버: photo3d/mobile.py.
- 짝맞춤 토큰(app_data/mobile_token.txt): PC 밖 요청은 쿠키 p3d 필요. 터널은 localhost 로 들어오므로 Cf-Ray 등 헤더로 원격 판정(_is_local).
- 카메라는 https 에서만: 같은 와이파이 https://<IP>:8766(자체 서명, cryptography 자동 설치) / 현장은 cloudflared 임시 터널(자동 내려받기).
- 영상 + 자동 사진(기본 추천): 4K 화면을 녹화하면서 같은 화면에서 좋은 순간을 사진으로(녹화가 끊기지 않게 ImageCapture 안 씀). 영상은 p3d_fill_* 로 올라가 frames.py 가 사진 수의 1/3 이하만 뽑아 빈 곳을 메움. webm 길이 N/A 대응(_duration).
- 자동 촬영: 프레임 간 이동 누적으로 겹침 75% 에서 촬영, 자이로 25°/s 넘으면 대기, 라플라시안 선명도 중앙값 60% 미만 건너뜀,
  첫 장 뒤 색온도 고정, 줌 최소, 화면 꺼짐 방지, 방향별 한 바퀴 링, GPS → input/_gps.json(frames.exif_gps 가 읽음).
- 업로드: IndexedDB 대기열, 4MB 조각 이어 올리기(PUT files?offset=), 업로드 끝나면 자동 3D 시작(start 에 mode·kind).
- 클라우드 확인: 토큰/쿠키·원격 차단·조각 업로드 동일성·GPS·헤드리스 크롬 가짜 카메라로 촬영→업로드. 실제 폰(iOS/안드로이드) 미확인.

## 지켜야 할 것
- 사용자는 비상업 용도 → 연구용 라이선스(PGSR 등)도 사용 가능(docs/GPU_PC.md). 상업 용도로 바뀌면 빼야 함.
- 새 기법은 실사 벤치마크(eval_strecha.py)로 숫자 확인 후 채택(클라우드에서 sift_hq·refine_pp 는 효과 없어 옵션으로만 둠).
- 무거운 작업(특징점·조밀복원·스플래팅)은 동시에 돌리지 말 것(클라우드에서 메모리 부족으로 두 번 중단됨).
