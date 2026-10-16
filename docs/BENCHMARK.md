# 상용 프로그램 벤치마크 → photo3d 적용 현황

조사 대상: DJI Terra v4.3 사용자 매뉴얼(68쪽 전체), Pix4Dmapper·PIX4Dcatch 문서, Agisoft Metashape 매뉴얼, 관련 논문.
엔젤스윙 Notion 자료는 클라우드 네트워크 정책 때문에 아직 못 읽음(허용 도메인 추가 후 반영 예정).

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
| 360 카메라 | Metashape Spherical, COLMAP 파노라마 | 등장방형 → 90° 원근 10방향(수평 6 + 아래 35° 4), 렌즈값 고정 | 합성 360 시험 중 |
| 가우시안 스플래팅 | Terra 5.x | Brush(Apache-2.0) GPU 학습, 앱 '실사 보기' | 클라우드 300스텝 확인 |

## 앞으로 (우선순위)
1. LOD 출력(3D Tiles/B3DM) — 대형 현장 유니티 실시간 표시 (Terra B3DM)
2. 이음새 국소 색 보정(Poisson seam leveling) — 텍스처 패치 경계 더 매끄럽게
3. 하늘·사람 마스크(SAM 2, Apache-2.0) — 하늘 덩어리·움직이는 사람 제거
4. 기기별 카메라 보정값 누적 재사용 (Terra: 최적화된 내부 파라미터 재사용)
5. 날짜별 같은 원점 비교(진척·변위) — Terra '맞춤 모델 원점'
6. 롤링셔터 보정 — 당분간 촬영 가이드(천천히·밝게)로 대체
