# 고속도로 램프 복구 시스템 (Highway Ramp Recovery System)

이 프로젝트는 고속도로 램프 주행 시 실시간 경로 복구 알고리즘을 구현합니다. 고등학교 수준의 미적분학과 기하학을 활용하여 차량의 이상적인 주행 경로를 계산하고 복구합니다.

## 주요 기능

- B-스플라인 모델링을 사용한 램프 곡선 생성
- GPS/IMU 실시간 위치 데이터와 이상적인 매개변수 곡선 비교
- 접선 방향과 오차 방향을 고려한 보정 벡터 계산
- 반복적인 경로 복구 알고리즘

## 설치 방법

1. Python 3.8 이상 설치
2. 의존성 패키지 설치:
```bash
pip install -r requirements.txt
```

## 프로젝트 구조

- `src/`: 소스 코드
  - `curve_modeling/`: B-스플라인 곡선 모델링
  - `path_recovery/`: 경로 복구 알고리즘
  - `data_processing/`: GPS/IMU 데이터 처리
- `data/`: 테스트 및 실제 주행 데이터
- `notebooks/`: 알고리즘 개발 및 시각화를 위한 Jupyter 노트북
- `tests/`: 단위 테스트
- `results/`: 실험 결과 및 시각화
