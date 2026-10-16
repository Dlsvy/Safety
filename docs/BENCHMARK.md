# 상용 프로그램 벤치마크 → photo3d 적용 현황

조사 대상: DJI Terra v4.3 사용자 매뉴얼(68쪽 전체), Pix4Dmapper·PIX4Dcatch 문서, Agisoft Metashape 매뉴얼, 관련 논문.
엔젤스윙 유저가이드·업데이트 소식(Notion 85페이지)도 전부 읽고 반영(아래 '엔젤스윙' 절).

## 상용 프로그램이 하는 일 (요약)

| 단계 | DJI Terra | Pix4Dmapper | Metashape |
|---|---|---|---|
| 사진 정렬 | AT(항공삼각측량), 특징점 밀도 높음/낮음, 장면(일반·원형·전력) | Keypoint scale Full/½, 매칭 전략, Calibration Standard/Alternative | Accuracy Highest~Low, Key/Tie point limit, Adaptive camera fitting |
| 정렬 품질 | 품질 보고서(HTML), GCP 재투영오차 실시간 경고 | Quality Report(신호등: 등록률 95%, 매칭 1000, 카메라 최적화 5%) | Gradual selection(재투영오차·불확실성) → 재최적화 |
| 조밀 복원 | 해상도 높음/중간/낮음 = 원본/½/¼ | Image scale ½(기본), Min matches 3(4~6이면 노이즈↓) | Depth maps + Filtering Mild/Moderate/Aggressive |
| 메쉬·텍스처 | 메쉬 + 원본 사진 투영 텍스처, LOD(B3DM/OSGB), 모델 축소 | Max octree depth, Texture 8192, Color balancing | Mosaic 블렌딩, 8192×N, Ghosting filter |
| 정확도 | GCP 4개↑·GCP당 9장↑·고른 배치, 제어점/검사점 분리, 7파라미터 | GCP RMS ≤ 1.5×GSD | Markers, Scale bars, Check points |
| 최신 | 5.x: 가우시안 스플래팅, floater 제거, 영상 입력, 롤링셔터 보정 | PIX4Dcatch(폰): 겹침 80~90%, 마커 | Spherical(360) 카메라, 영상 프레임 |

## 적용한 것

| 항목 | 출처 | photo3d 구현 | 검증 |
|---|---|---|---|
| 원본 사진 투영 텍스처 | Terra/Metashape, Waechter ECCV14 | `texture.py` — 면별 최적 사진(정면성×해상도), Embree 가림 검사, 라벨 다듬기, 사진별 색 보정, 패치 아틀라스 8K | 분수대: 점 색 대비 확연히 선명, 80만 면 19초(CPU) |
| 깊이 필터(Moderate~Aggressive) | Metashape Depth filtering, Pix4D min matches | PatchMatch 형상 일관성 항상 켬, 융합 4장↑·1.5px·0.8%·8° | PC castle-P19 가시 원인 제거 |
| 가시·이상점 제거 | Terra floater 제거 | `pointfilter.py` 통계적 이상점 + 선형(시선 줄무늬) 필터, 메쉬 바늘 제거 | 합성 시험: 가시 94% 제거, 실제 점 99% 유지 |
| Gradual selection | Metashape | `sfm.gradual_selection` 재투영 2px·삼각측량각 1.5°·2장 트랙 → BA 재실행 | Strecha 카메라 위치 RMSE: castle-P19 67.6→46.5mm(최대 206→98mm), entry 9.0→9.1mm, fountain 4.4→4.4mm |
| 품질 보고서 | Pix4D Quality Report, Terra 보고서 | `report.py` 신호등 판정 + 겹침 지도(점마다 본 사진 수) | 앱 '품질 보고서' 버튼 |
| 영상 키프레임 | PIX4Dcatch 겹침, Terra 영상 모델링 | 광류 이동량 누적 기준(화면폭 6~20%)으로 키프레임 + 구간 최선명 | 정지 구간 60프레임 → 1장 |
| 360 카메라 | Metashape Spherical, COLMAP 파노라마 | 등장방형 → 90° 원근 10방향을 리그로 묶음(렌즈·방향 고정), 위치 순서대로만 매칭, 약한 구간 structure-less 등록 | 합성 360 걷기(30곳·45 m): 30/30 등록, 위치 RMSE 15 mm(최대 40 mm). 360 영상: 270/270 장 등록. (전수 매칭 시엔 반복무늬 오매칭으로 18 m 오등록 → 순차 매칭으로 해결) |
| 가우시안 스플래팅 | Terra 5.x | Brush(Apache-2.0) GPU 학습, 앱 '실사 보기' | 클라우드 300스텝 확인 |

## 앞으로 (우선순위)
1. LOD 출력(3D Tiles/B3DM) — 대형 현장 유니티 실시간 표시 (Terra B3DM)
2. 이음새 색 보정 고도화(현재: 패치별 오프셋, 분수대 경계 색차 2.9→1.7) → Poisson 국소 보정
3. 하늘·사람 마스크(SAM 2, Apache-2.0) — 하늘 덩어리·움직이는 사람 제거
4. 기기별 카메라 보정값 누적 재사용 (Terra: 최적화된 내부 파라미터 재사용)
5. 날짜별 같은 원점 비교(진척·변위) — Terra '맞춤 모델 원점'
6. 롤링셔터 보정 — 당분간 촬영 가이드(천천히·밝게)로 대체

## 엔젤스윙 (드론 기반 건설 디지털 트윈) → 폰·360 에 맞게 적용

| 엔젤스윙 기능·가이드 | photo3d 적용 | 상태 |
|---|---|---|
| 거리 측정(수평·표면·직선, 수직 높이) | 뷰어 '거리': 직선·수평·높이차 동시 표시 | 적용 |
| 원형 측정 | 뷰어 '반경': 중심 클릭 + 반경 숫자 입력(기본 20 m) — 크레인 선회반경·낙하 위험반경 | 적용 |
| 직육면체 모델(기본 5×5×5 m) | 뷰어 '구역': 가로·세로·높이·회전 입력 → 유니티 트리거 콜라이더(SiteZone 레이어) | 적용 |
| 마커(이름·색·메모·사진) / 작업지시 우선순위 3단계 | 뷰어 '위험 표시': 주의·위험·심각 + 메모 → 유니티 표지 | 적용 |
| 3D 모델 위 촬영 사진 보기 | 뷰어 '촬영 위치': 카메라 사각뿔, 누르면 원본 사진 | 적용 |
| GCP 최소 4개, 50m×50m 마다 1개 | 품질 보고서 '기준점 수' 판정 | 적용 |
| 소스포토 요건(원본, 카톡·편집본 불가, 흔들림 재촬영), 사선 촬영 | 앱 '촬영 가이드' | 적용 |
| 실내 360 촬영(머리 위, 타임랩스 2초·8K↑, 문틀 3~4초 멈춤, 왕복 금지, 층별) | 앱 '촬영 가이드' + 360 리그 처리 | 적용 |
| 장비 .glb 업로드(작업 반지름) | 다음 단계: 뷰어에서 크레인 GLB 배치·회전 → 배치값 유니티 전달 | 예정 |
| 종횡단면도(CSV/DXF), DTM(지장물 제거), 도면 2핀 중첩, IFC, 날짜별 비교, 타임랩스 | 다음 단계 | 예정 |

모든 표시는 작업 폴더 `annotations.json`(GLB 와 같은 Y-up·m 좌표)에 저장되고, 유니티 `Photo3D > Import Site Folder` 가 `Photo3DAnnotations` 로 씬에 만든다.
