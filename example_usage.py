"""
크레인 양중분석 프로그램 사용 예제
다양한 시나리오에 대한 분석 예제를 제공합니다
"""

from crane_lifting_analysis import CraneLiftingAnalysis


def example_safe_operation():
    """예제 1: 안전한 작업 조건"""
    print("\n" + "="*70)
    print("예제 1: 안전한 이동식 크레인 작업")
    print("="*70)

    analysis = CraneLiftingAnalysis()

    # 50톤 이동식 크레인, 40m 붐 길이
    analysis.set_crane_spec(
        crane_type="mobile",
        max_capacity=50.0,
        boom_length=40.0,
        max_radius=35.0
    )

    # 20톤 하중, 20m 작업 반경
    analysis.set_load_info(weight=20.0, radius=20.0)

    # 분석 실행 및 결과 출력
    result = analysis.analyze()
    analysis.print_report()

    # 결과를 파일로 저장
    analysis.export_report("report_example1.txt")


def example_caution_operation():
    """예제 2: 주의가 필요한 작업 조건"""
    print("\n" + "="*70)
    print("예제 2: 주의가 필요한 타워 크레인 작업")
    print("="*70)

    analysis = CraneLiftingAnalysis()

    # 30톤 타워 크레인, 50m 붐 길이
    analysis.set_crane_spec(
        crane_type="tower",
        max_capacity=30.0,
        boom_length=50.0,
        max_radius=45.0
    )

    # 20톤 하중, 35m 작업 반경 (먼 거리 작업)
    analysis.set_load_info(weight=20.0, radius=35.0)

    # 분석 실행 및 결과 출력
    result = analysis.analyze()
    analysis.print_report()


def example_dangerous_operation():
    """예제 3: 위험한 작업 조건"""
    print("\n" + "="*70)
    print("예제 3: 위험한 작업 조건 (안전율 미달)")
    print("="*70)

    analysis = CraneLiftingAnalysis()

    # 25톤 이동식 크레인, 30m 붐 길이
    analysis.set_crane_spec(
        crane_type="mobile",
        max_capacity=25.0,
        boom_length=30.0,
        max_radius=25.0
    )

    # 20톤 하중, 24m 작업 반경 (한계 근접)
    analysis.set_load_info(weight=20.0, radius=24.0)

    # 분석 실행 및 결과 출력
    result = analysis.analyze()
    analysis.print_report()


def example_over_radius():
    """예제 4: 작업 반경 초과"""
    print("\n" + "="*70)
    print("예제 4: 최대 작업 반경 초과")
    print("="*70)

    analysis = CraneLiftingAnalysis()

    # 40톤 이동식 크레인
    analysis.set_crane_spec(
        crane_type="mobile",
        max_capacity=40.0,
        boom_length=35.0,
        max_radius=30.0
    )

    # 15톤 하중이지만 작업 반경이 최대 반경 초과
    analysis.set_load_info(weight=15.0, radius=35.0)

    # 분석 실행 및 결과 출력
    result = analysis.analyze()
    analysis.print_report()


def example_custom_analysis():
    """예제 5: 사용자 정의 분석"""
    print("\n" + "="*70)
    print("예제 5: 사용자 정의 분석")
    print("="*70)

    analysis = CraneLiftingAnalysis()

    # 사용자로부터 입력 받기 (실제 사용 시)
    print("\n크레인 사양을 입력하세요:")
    print("예제 값으로 진행합니다...")

    # 예제 값
    crane_type = "tower"
    max_capacity = 60.0
    boom_length = 60.0
    max_radius = 55.0
    load_weight = 35.0
    work_radius = 30.0

    print(f"\n입력된 값:")
    print(f"  크레인 타입: {crane_type}")
    print(f"  최대 용량: {max_capacity} ton")
    print(f"  붐 길이: {boom_length} m")
    print(f"  최대 작업 반경: {max_radius} m")
    print(f"  양중물 중량: {load_weight} ton")
    print(f"  작업 반경: {work_radius} m")

    analysis.set_crane_spec(
        crane_type=crane_type,
        max_capacity=max_capacity,
        boom_length=boom_length,
        max_radius=max_radius
    )

    analysis.set_load_info(weight=load_weight, radius=work_radius)

    # 분석 실행
    result = analysis.analyze()
    analysis.print_report()

    # 분석 결과 데이터 활용 예제
    print("\n[분석 결과 데이터 활용]")
    print(f"안전율: {result['safety_factor']}")
    print(f"안전 여부: {result['is_safe']}")
    print(f"정격 하중: {result['rated_capacity']} ton")


def example_multiple_scenarios():
    """예제 6: 여러 작업 반경에 대한 비교 분석"""
    print("\n" + "="*70)
    print("예제 6: 작업 반경별 안전율 비교")
    print("="*70)

    analysis = CraneLiftingAnalysis()

    # 동일한 크레인 사양
    analysis.set_crane_spec(
        crane_type="mobile",
        max_capacity=45.0,
        boom_length=40.0,
        max_radius=35.0
    )

    # 동일한 하중으로 다양한 작업 반경 분석
    load_weight = 25.0
    radii = [10, 15, 20, 25, 30]

    print(f"\n크레인: 45톤 이동식, 양중물: {load_weight}톤\n")
    print(f"{'작업반경(m)':<12} {'정격하중(ton)':<15} {'안전율':<10} {'상태'}")
    print("-" * 70)

    for radius in radii:
        analysis.set_load_info(weight=load_weight, radius=radius)
        result = analysis.analyze()

        status = "안전" if result['is_safe'] else "위험"
        print(f"{radius:<12} {result['rated_capacity']:<15.2f} "
              f"{result['safety_factor']:<10.2f} {status}")


def main():
    """모든 예제 실행"""
    print("\n" + "#"*70)
    print("#  크레인 양중분석 프로그램 - 사용 예제")
    print("#"*70)

    # 모든 예제 실행
    example_safe_operation()
    example_caution_operation()
    example_dangerous_operation()
    example_over_radius()
    example_custom_analysis()
    example_multiple_scenarios()

    print("\n" + "#"*70)
    print("#  모든 예제 실행 완료")
    print("#"*70)
    print()


if __name__ == "__main__":
    main()
