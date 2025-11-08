# 크레인 양중분석 프로그램 (Crane Lifting Analysis Program)

크레인 작업의 안전성을 분석하고 검토하는 프로그램입니다.

## 주요 기능

- 크레인 용량 및 작업 반경 계산
- 양중물 하중 분석
- 안전율 검토 및 평가
- 상세 분석 리포트 생성
- 다양한 크레인 타입 지원 (타워크레인, 이동식크레인)

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

```python
from crane_lifting_analysis import CraneLiftingAnalysis

# 크레인 분석 객체 생성
analysis = CraneLiftingAnalysis()

# 크레인 사양 설정
analysis.set_crane_spec(
    crane_type="mobile",  # 또는 "tower"
    max_capacity=50.0,    # 최대 용량 (ton)
    boom_length=40.0,     # 붐 길이 (m)
    max_radius=35.0       # 최대 작업 반경 (m)
)

# 양중물 정보 설정
analysis.set_load_info(
    weight=30.0,          # 중량 (ton)
    radius=25.0           # 작업 반경 (m)
)

# 분석 실행
result = analysis.analyze()

# 결과 출력
analysis.print_report()
```

## 분석 항목

1. **크레인 용량 검토**: 작업 반경에 따른 크레인 정격 하중 확인
2. **안전율 계산**: 실제 하중 대비 안전율 검토
3. **작업 가능 여부**: 안전 기준 충족 여부 판정
4. **권장 사항**: 안전 작업을 위한 권장 사항 제시

## 안전 기준

- 최소 안전율: 1.25 이상
- 권장 안전율: 1.5 이상

## 라이선스

MIT License
