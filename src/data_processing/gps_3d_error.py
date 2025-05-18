import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class GPS3DError:
    horizontal_std: float  # 수평 방향 오차의 표준편차 (미터)
    vertical_std: float    # 수직 방향 오차의 표준편차 (미터)
    bias: np.ndarray      # 현재 시스템적 편향 (미터)

class GPS3DDistortionSimulator:
    def __init__(self, 
                 horizontal_std: float = 4.0,
                 vertical_std: float = 6.0,
                 correlation_time: float = 2.0):  # 상관 시간 (초)
        """3D GPS 왜곡 시뮬레이터 초기화
        Args:
            horizontal_std: 수평 방향 오차의 표준편차 (미터)
            vertical_std: 수직 방향 오차의 표준편차 (미터)
            correlation_time: 오차의 시간 상관관계 (초)
        """
        self.horizontal_std = horizontal_std
        self.vertical_std = vertical_std
        self.correlation_time = correlation_time
        
        # 초기 바이어스 설정
        self.current_bias = np.random.normal(0, [horizontal_std, horizontal_std, vertical_std])
        self.prev_error = np.zeros(3)
        self.dt = 0.1  # 시간 간격 (초)
        
    def update(self) -> None:
        """시간에 따른 GPS 오차 업데이트"""
        # 1차 Gauss-Markov 프로세스를 사용한 오차 업데이트
        alpha = np.exp(-self.dt / self.correlation_time)
        
        # 새로운 무작위 오차 생성 (수직 방향 오차 증가)
        new_error = np.random.normal(0, [
            self.horizontal_std,
            self.horizontal_std,
            self.vertical_std * 1.5  # 수직 방향 오차 50% 증가
        ])
        
        # 시간 상관관계를 가진 오차 계산
        self.current_bias = alpha * self.current_bias + (1 - alpha) * new_error
        
    def apply_distortion(self, true_position: np.ndarray, elevation: float = 0.0) -> np.ndarray:
        """실제 3D 위치에 GPS 왜곡 적용
        Args:
            true_position: 실제 3D 위치 [x, y, z]
            elevation: 지면으로부터의 고도 (미터)
        Returns:
            왜곡된 3D 위치
        """
        # 시간에 따른 오차 업데이트
        self.update()
        
        # 고도에 따른 오차 스케일 조정 (비선형적 증가)
        elevation_factor = 1.0 + (elevation / 10.0)**1.5  # 고도에 따른 비선형적 오차 증가
        
        # 수직 방향 오차 추가 증가
        vertical_factor = 1.0 + elevation / 5.0  # 수직 방향 오차 더 빠르게 증가
        
        # 최종 오차 계산
        error = self.current_bias.copy()
        error[:2] *= elevation_factor  # 수평 방향
        error[2] *= elevation_factor * vertical_factor  # 수직 방향
        
        # 최종 왜곡된 위치 계산
        distorted_position = true_position + error
        
        return distorted_position
        
    def get_error_estimate(self, elevation: float = 0.0) -> GPS3DError:
        """현재 GPS 오차 추정값 반환"""
        elevation_factor = 1.0 + (elevation / 10.0)**1.5
        vertical_factor = 1.0 + elevation / 5.0
        
        return GPS3DError(
            horizontal_std=self.horizontal_std * elevation_factor,
            vertical_std=self.vertical_std * elevation_factor * vertical_factor,
            bias=self.current_bias
        ) 