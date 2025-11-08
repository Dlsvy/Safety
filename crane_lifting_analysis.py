"""
크레인 양중분석 프로그램 (Crane Lifting Analysis Program)
안전한 크레인 작업을 위한 분석 및 검토 도구
"""

import math
from typing import Dict, Optional, Tuple
from datetime import datetime


class CraneLiftingAnalysis:
    """크레인 양중 분석 클래스"""

    # 안전율 기준
    MINIMUM_SAFETY_FACTOR = 1.25
    RECOMMENDED_SAFETY_FACTOR = 1.5

    def __init__(self):
        """초기화"""
        self.crane_type: Optional[str] = None
        self.max_capacity: float = 0.0
        self.boom_length: float = 0.0
        self.max_radius: float = 0.0
        self.load_weight: float = 0.0
        self.work_radius: float = 0.0
        self.analysis_result: Optional[Dict] = None

        # 크레인 타입별 용량 계수
        self.capacity_factors = {
            "mobile": 0.85,    # 이동식 크레인
            "tower": 0.90,     # 타워 크레인
        }

    def set_crane_spec(self, crane_type: str, max_capacity: float,
                       boom_length: float, max_radius: float) -> None:
        """
        크레인 사양 설정

        Args:
            crane_type: 크레인 타입 ("mobile" 또는 "tower")
            max_capacity: 최대 용량 (ton)
            boom_length: 붐 길이 (m)
            max_radius: 최대 작업 반경 (m)
        """
        if crane_type not in self.capacity_factors:
            raise ValueError(f"지원하지 않는 크레인 타입: {crane_type}. "
                           f"'mobile' 또는 'tower'를 사용하세요.")

        if max_capacity <= 0:
            raise ValueError("최대 용량은 0보다 커야 합니다.")

        if boom_length <= 0:
            raise ValueError("붐 길이는 0보다 커야 합니다.")

        if max_radius <= 0 or max_radius > boom_length:
            raise ValueError("작업 반경은 0보다 크고 붐 길이 이하여야 합니다.")

        self.crane_type = crane_type
        self.max_capacity = max_capacity
        self.boom_length = boom_length
        self.max_radius = max_radius

    def set_load_info(self, weight: float, radius: float) -> None:
        """
        양중물 정보 설정

        Args:
            weight: 양중물 중량 (ton)
            radius: 작업 반경 (m)
        """
        if weight <= 0:
            raise ValueError("양중물 중량은 0보다 커야 합니다.")

        if radius <= 0:
            raise ValueError("작업 반경은 0보다 커야 합니다.")

        self.load_weight = weight
        self.work_radius = radius

    def calculate_rated_capacity(self, radius: float) -> float:
        """
        작업 반경에 따른 정격 하중 계산

        Args:
            radius: 작업 반경 (m)

        Returns:
            정격 하중 (ton)
        """
        if self.crane_type is None:
            raise ValueError("크레인 사양을 먼저 설정하세요.")

        if radius > self.max_radius:
            return 0.0

        # 작업 반경에 따른 용량 감소 계산 (선형 근사)
        # 실제로는 크레인 제조사의 하중표를 사용해야 합니다
        capacity_factor = self.capacity_factors[self.crane_type]
        radius_factor = 1.0 - (radius / self.max_radius) * 0.5

        rated_capacity = self.max_capacity * capacity_factor * radius_factor
        return max(0.0, rated_capacity)

    def calculate_safety_factor(self, rated_capacity: float,
                                actual_load: float) -> float:
        """
        안전율 계산

        Args:
            rated_capacity: 정격 하중 (ton)
            actual_load: 실제 하중 (ton)

        Returns:
            안전율
        """
        if actual_load <= 0:
            return float('inf')

        return rated_capacity / actual_load

    def check_safety(self, safety_factor: float) -> Tuple[bool, str]:
        """
        안전성 검토

        Args:
            safety_factor: 안전율

        Returns:
            (안전 여부, 평가 내용)
        """
        if safety_factor < self.MINIMUM_SAFETY_FACTOR:
            return False, "위험 - 최소 안전율 미달"
        elif safety_factor < self.RECOMMENDED_SAFETY_FACTOR:
            return True, "주의 - 최소 안전율은 만족하나 권장 안전율 미달"
        else:
            return True, "안전 - 권장 안전율 만족"

    def analyze(self) -> Dict:
        """
        크레인 양중 분석 실행

        Returns:
            분석 결과 딕셔너리
        """
        if self.crane_type is None:
            raise ValueError("크레인 사양을 먼저 설정하세요.")

        if self.load_weight == 0:
            raise ValueError("양중물 정보를 먼저 설정하세요.")

        # 작업 반경 확인
        if self.work_radius > self.max_radius:
            self.analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "crane_type": self.crane_type,
                "max_capacity": self.max_capacity,
                "boom_length": self.boom_length,
                "max_radius": self.max_radius,
                "load_weight": self.load_weight,
                "work_radius": self.work_radius,
                "rated_capacity": 0.0,
                "safety_factor": 0.0,
                "is_safe": False,
                "safety_status": "위험 - 최대 작업 반경 초과",
                "recommendations": [
                    "작업 반경을 줄이거나 더 큰 크레인을 사용하세요.",
                    f"현재 작업 반경: {self.work_radius}m",
                    f"최대 작업 반경: {self.max_radius}m"
                ]
            }
            return self.analysis_result

        # 정격 하중 계산
        rated_capacity = self.calculate_rated_capacity(self.work_radius)

        # 안전율 계산
        safety_factor = self.calculate_safety_factor(rated_capacity, self.load_weight)

        # 안전성 검토
        is_safe, safety_status = self.check_safety(safety_factor)

        # 권장 사항 생성
        recommendations = self._generate_recommendations(
            safety_factor, rated_capacity, is_safe
        )

        self.analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "crane_type": self.crane_type,
            "max_capacity": self.max_capacity,
            "boom_length": self.boom_length,
            "max_radius": self.max_radius,
            "load_weight": self.load_weight,
            "work_radius": self.work_radius,
            "rated_capacity": round(rated_capacity, 2),
            "safety_factor": round(safety_factor, 2),
            "is_safe": is_safe,
            "safety_status": safety_status,
            "recommendations": recommendations
        }

        return self.analysis_result

    def _generate_recommendations(self, safety_factor: float,
                                  rated_capacity: float, is_safe: bool) -> list:
        """권장 사항 생성"""
        recommendations = []

        if not is_safe:
            recommendations.append("즉시 작업을 중단하고 안전 대책을 수립하세요.")
            recommendations.append(f"작업 반경을 {self.work_radius * 0.8:.1f}m 이하로 줄이거나 "
                                 f"양중물 중량을 {rated_capacity * 0.8:.1f}ton 이하로 줄이세요.")
        elif safety_factor < self.RECOMMENDED_SAFETY_FACTOR:
            recommendations.append("최소 안전율은 만족하나 추가 안전 조치를 권장합니다.")
            recommendations.append("작업 반경을 줄이거나 더 여유 있는 크레인을 사용하세요.")
        else:
            recommendations.append("안전 기준을 만족합니다.")
            recommendations.append("작업 전 최종 안전 점검을 실시하세요.")

        # 공통 권장 사항
        recommendations.extend([
            "작업 전 크레인 및 와이어로프 상태를 점검하세요.",
            "신호수를 배치하고 작업 구역을 통제하세요.",
            "기상 조건(풍속, 강우 등)을 확인하세요."
        ])

        return recommendations

    def print_report(self) -> None:
        """분석 결과 리포트 출력"""
        if self.analysis_result is None:
            print("분석을 먼저 실행하세요. (analyze() 메서드 호출)")
            return

        result = self.analysis_result

        print("\n" + "="*70)
        print("크레인 양중 분석 리포트 (Crane Lifting Analysis Report)")
        print("="*70)
        print(f"분석 일시: {result['timestamp']}")
        print()

        print("[크레인 정보]")
        crane_type_name = "이동식 크레인" if result['crane_type'] == "mobile" else "타워 크레인"
        print(f"  크레인 타입: {crane_type_name}")
        print(f"  최대 용량: {result['max_capacity']} ton")
        print(f"  붐 길이: {result['boom_length']} m")
        print(f"  최대 작업 반경: {result['max_radius']} m")
        print()

        print("[양중 작업 정보]")
        print(f"  양중물 중량: {result['load_weight']} ton")
        print(f"  작업 반경: {result['work_radius']} m")
        print(f"  정격 하중: {result['rated_capacity']} ton")
        print()

        print("[안전성 분석]")
        print(f"  안전율: {result['safety_factor']}")
        print(f"  최소 안전율 기준: {self.MINIMUM_SAFETY_FACTOR}")
        print(f"  권장 안전율 기준: {self.RECOMMENDED_SAFETY_FACTOR}")
        print(f"  안전 여부: {'안전' if result['is_safe'] else '위험'}")
        print(f"  평가: {result['safety_status']}")
        print()

        print("[권장 사항]")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")

        print("="*70)
        print()

    def export_report(self, filename: str) -> None:
        """
        분석 결과를 파일로 저장

        Args:
            filename: 저장할 파일명
        """
        if self.analysis_result is None:
            raise ValueError("분석을 먼저 실행하세요.")

        result = self.analysis_result
        crane_type_name = "이동식 크레인" if result['crane_type'] == "mobile" else "타워 크레인"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("크레인 양중 분석 리포트\n")
            f.write("="*70 + "\n")
            f.write(f"분석 일시: {result['timestamp']}\n\n")

            f.write("[크레인 정보]\n")
            f.write(f"  크레인 타입: {crane_type_name}\n")
            f.write(f"  최대 용량: {result['max_capacity']} ton\n")
            f.write(f"  붐 길이: {result['boom_length']} m\n")
            f.write(f"  최대 작업 반경: {result['max_radius']} m\n\n")

            f.write("[양중 작업 정보]\n")
            f.write(f"  양중물 중량: {result['load_weight']} ton\n")
            f.write(f"  작업 반경: {result['work_radius']} m\n")
            f.write(f"  정격 하중: {result['rated_capacity']} ton\n\n")

            f.write("[안전성 분석]\n")
            f.write(f"  안전율: {result['safety_factor']}\n")
            f.write(f"  최소 안전율 기준: {self.MINIMUM_SAFETY_FACTOR}\n")
            f.write(f"  권장 안전율 기준: {self.RECOMMENDED_SAFETY_FACTOR}\n")
            f.write(f"  안전 여부: {'안전' if result['is_safe'] else '위험'}\n")
            f.write(f"  평가: {result['safety_status']}\n\n")

            f.write("[권장 사항]\n")
            for i, rec in enumerate(result['recommendations'], 1):
                f.write(f"  {i}. {rec}\n")

            f.write("="*70 + "\n")

        print(f"리포트가 '{filename}' 파일로 저장되었습니다.")


def main():
    """예제 실행"""
    print("크레인 양중분석 프로그램 예제\n")

    # 분석 객체 생성
    analysis = CraneLiftingAnalysis()

    # 예제 1: 안전한 작업
    print("=" * 70)
    print("예제 1: 안전한 작업 조건")
    print("=" * 70)
    analysis.set_crane_spec(
        crane_type="mobile",
        max_capacity=50.0,
        boom_length=40.0,
        max_radius=35.0
    )
    analysis.set_load_info(weight=20.0, radius=20.0)
    analysis.analyze()
    analysis.print_report()

    # 예제 2: 주의가 필요한 작업
    print("\n" + "=" * 70)
    print("예제 2: 주의가 필요한 작업 조건")
    print("=" * 70)
    analysis2 = CraneLiftingAnalysis()
    analysis2.set_crane_spec(
        crane_type="tower",
        max_capacity=30.0,
        boom_length=50.0,
        max_radius=45.0
    )
    analysis2.set_load_info(weight=20.0, radius=35.0)
    analysis2.analyze()
    analysis2.print_report()

    # 예제 3: 위험한 작업
    print("\n" + "=" * 70)
    print("예제 3: 위험한 작업 조건")
    print("=" * 70)
    analysis3 = CraneLiftingAnalysis()
    analysis3.set_crane_spec(
        crane_type="mobile",
        max_capacity=25.0,
        boom_length=30.0,
        max_radius=25.0
    )
    analysis3.set_load_info(weight=20.0, radius=24.0)
    analysis3.analyze()
    analysis3.print_report()


if __name__ == "__main__":
    main()
